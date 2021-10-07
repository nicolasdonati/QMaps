import igl
import meshplot as mp
##
import numpy as np
from scipy import spatial
import scipy as sp
import matplotlib.pyplot as plt
##
from os.path import join
import time
##
from Tools.mesh import *
###
import tensorflow as tf
###############################


###show p2p map##########################################################################
def show_p2p(X,Y,T12, axis=0, axis_col=0, withuv=False, T12_gt=None):
    left = X
    right = Y
    offset = np.zeros(3)
    offset[axis] = np.max([-np.min(right.v[:,axis]), np.max(left.v[:,axis])])
    all_v = np.concatenate([left.v-offset, right.v+offset])
    all_f = np.concatenate([left.f, right.f+left.v.shape[0]])

    c = Y.v[:,axis_col]
    all_c = np.concatenate([c[T12], c])
    ###err###
    if T12_gt is not None:
        err = euc_err(Y, T12, T12_gt)
        print('euc err:', err)
    ###uv####
    uv = None
    if withuv:
        uv = Y.v[:,:2]
        uv = np.concatenate([uv[T12], uv])
    p = mp.plot(all_v, all_f, all_c, uv = uv, return_plot=True)
    i_spl = 10
    X.fps_3d(i_spl)
    spl = X.samples
    p.add_lines((left.v-offset)[spl], (right.v+offset)[T12[spl]],
               shading={'line_color': 'red'})
    return


###optimize base change#########################################################
#energies
## get C can be faster without a pinv... best is to add a bij energy to speed up the process
def get_C(X, Y, theta, k=20, vts=False):
    base_change = tf.complex(tf.math.cos(theta), tf.math.sin(theta))
    if vts:
        C = np.linalg.pinv(Y.ceig[Y.vts,:k]) @ (base_change * X.ceig[X.vts,:k])
    else: ## we have a simple p2p map
        C = np.linalg.pinv(Y.ceig[X.T12,:k]) @ (base_change * X.ceig[:,:k])
    return C

def loss(X, Y, theta, k=20, vts=False, lap_reg=0):
    C = get_C(X, Y, theta, k=k, vts=vts)
    loss =  ortho_loss(C)
    if lap_reg > 0:
         loss += lap_reg * lap_com_loss(X, Y, C)
    return loss

def ortho_loss_Q(Q):
    Q_star = tf.math.conj(tf.transpose(Q, [1,0]))
    D = Q @ Q_star - tf.eye(Q.shape[0], dtype=Q.dtype)
    return tf.cast(tf.nn.l2_loss(D), tf.float32)
    #return D
    
def lap_com_loss(X, Y, C):#, val1, val2):
    k = C.shape[0]
    D = C * X.cvals[:k] - Y.cvals[:k][:, None] * C
    return tf.cast(tf.norm(D), tf.float32)

#def bij_loss()

#main
def optimize_basechange(X,Y, theta, vts=False, k=60, lr=1e-2, n_iter=500, eps=2e-4,
                       verbose=1):
    ###
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    E_last = 0
    for i in range(int(n_iter)):
        ##energy
        with tf.GradientTape() as tape:
            E = loss(X, Y, theta, k=k, vts=vts)
        grads = tape.gradient(E, [theta])
        #stop
        if (tf.abs(E_last - E) < eps):
            print('stop criterion')
            break
        E_last = E
        #Process the gradients
        processed_grads = [g for g in grads]
        grads_and_vars = zip(processed_grads, [theta])
        ##print
        if verbose>0:
            if i%(n_iter/10)==0:
                print('step : {:03d} | E : {:0.6f}'.format(i, E.numpy()))
        if verbose==0:
            if i==0 or i==(n_iter - 1):
                print('step : {:03d} | E : {:0.6f}'.format(i, E.numpy()))
        ##
        opt.apply_gradients(grads_and_vars)
    return theta.numpy(), get_C(X, Y, theta, k=k, vts=vts).numpy()

###complex Zoomout##########################################################################
def flatten_complex(B):
    B_ = np.stack((B.real,B.imag),-1)
    B_ = B_.reshape(B.shape[0], 2 * B.shape[1])
    return B_

#with ALIGNED BASIS, simple inversion... BUT WILL NOT WORK OTHERWISE
def p2p_to_fmap(T12, T21, ceig1, ceig2, base_change=None):
    if base_change is None:
            Cc12 = np.linalg.pinv(ceig2[T12]) @ ceig1
    else:
        Cc12 = np.linalg.pinv(ceig2[T12]) @ (base_change * ceig1)
    #Cc21 = np.linalg.pinv(ceig1[T21]) @ ceig2
    return Cc12#, Cc21

## a bit useless but should be faster
def p2p_to_fmaps(X,Y, k):
    C12 = (Y.m @ Y.eig[:,:k]).T @ X.eig[Y.T21, :k]
    if X.base_change is None:
        Cc12 = (Y.m @ np.conjugate(Y.ceig[:,:k])).T @ X.ceig[Y.T21,:k]
        #Cc21 = (X.m @ np.conjugate(X.ceig)).T @ Y.ceig[X.T12]
    else:
        Cc12 = (Y.m @ np.conjugate(Y.ceig[:,:k])).T @ (X.bc * X.ceig[Y.T21,:k])
        #Cc21 = (X.m @ np.conjugate(X.ceig)).T @ Y.ceig[X.T12]
    return C12, Cc12

#knn in possibly high dimension
def fmap_to_p2p(Cc12, Cc21, ceig1, ceig2):
    #split complex in two dims
    Bc12 = ceig1 @ np.conjugate(Cc12.T)
    if np.iscomplexobj(Bc12):
        #print('complex')
        Bc12 = flatten_complex(Bc12)
        ceig2 = flatten_complex(ceig2)
    _, T12 = spatial.cKDTree(ceig2).query(Bc12, n_jobs=-1)
    #_, T21 = spatial.cKDTree(ceig1).query(Bc21, n_jobs=-1)
    return T12#, T21

def ZO(Cc12, X, Y, ks=[6,8,10, 12, 15, 20, 25, 30, 40, 50, 70], cplx=True,
       T12_gt = None):
    ceig1 = X.eig; ceig2 = Y.eig;
    if cplx:
        theta = tf.Variable(np.zeros((X.v.shape[0],1)))
        ceig1 = X.ceig; ceig2 = Y.ceig;
    ###
    assert (ceig1.shape[1] >= ks[-1] and ceig2.shape[1] >= ks[-1])
    assert (Cc12.shape[0] == ks[0] and Cc12.shape[1] == ks[0])
    ###
    errs = []
    B1 = ceig1[:,:ks[0]]
    B2 = ceig2[:,:ks[0]]
    T12 = fmap_to_p2p(Cc12, None, B1, B2)
    for k in ks:
        print(k)
        B1 = ceig1[:,:k]
        B2 = ceig2[:,:k]
        ##compute base_change
        if cplx and X.base_change:
            X.T12 = T12
            ##possible way to get complex fmap from p2p
            th, Cc12 = optimize_basechange(X,Y,theta, k=k, n_iter=200, verbose=0)
            theta = tf.Variable(th)
        else:
            Cc12 = p2p_to_fmap(T12, None, B1, B2, base_change=None)
        T12 = fmap_to_p2p(Cc12, None, B1, B2)
        ###
        if T12_gt is not None:
            d = Y.v[T12] - Y.v[T12_gt]
            err = np.mean(np.linalg.norm(d, axis=1))#/np.mean(np.linalg.norm(Y.v, axis=1))
            errs += [err]
    return T12, Cc12, errs


#### jacobian tools
def to_rot(J):
    u, _, v = np.linalg.svd(J)
    #R = (u @ v).transpose([0,2,1])
    R = u@v
    dets = (np.linalg.det(R) < 0)
    R[dets,:,0] *= -1
    return R


#### tools to get a complex fmap using D_fi operators

#this one does not work (probably because the Ceig_real do not form a basis as real functions)
def optimal_spectral_2d_tensor(X,Y,C,k2, step_spec=10):
    k1 = C.shape[0]
    Q = np.zeros((2*k2,2*k2))
    ll = range(1, k1, step_spec)
    for i in ll:
        print(i)
        Dfi = grad_fun_scal_op(X,X.eig[:,i],k1,k2)
        CDfi = C @ Dfi
        DCfi = grad_fun_scal_op(Y,Y.eig[:,:k1] @ C[:k1,i],k1,k2)
        Q += np.linalg.lstsq(DCfi,CDfi,rcond=None)[0]
    Q /= len(ll)
    return Q

#tf conversion from complex to 2d real block
# a + ib is represented by 2d tensor
# (a , -b)
# (b ,  a)
def comp_to_real(Q_):
    Qq1 = tf.stack([tf.math.real(Q_), tf.math.imag(Q_)], axis=1)
    Qq1 = tf.reshape(Qq1, [2 * Q_.shape[0], Q_.shape[1]])
    Qq2 = tf.stack([-tf.math.imag(Q_), tf.math.real(Q_)], axis=1)
    Qq2 = tf.reshape(Qq2, [2 * Q_.shape[0], Q_.shape[1]])
    Qq = tf.reshape(tf.stack([Qq1,Qq2], -1), [2*Q_.shape[0], 2*Q_.shape[1]])
    return Qq

# we use this energy to fit the equation
#<X, grad f> o T = <dT . X, grad (f o T)>
def grad_fun_scal_loss(Q, op_l, op_r, lam_ortho=-1):
    Q_ = comp_to_real(Q)
    #Q_ = Q
    E = 0
    for i in range(len(op_l)):
        #print(op_l[i].shape, op_r[i].shape, Q_.shape)
        ddd = tf.nn.l2_loss(op_l[i] - op_r[i] @ Q_)
        E += tf.cast(ddd, tf.float32)
        #print((op_l[i] - op_r[i] @ Q_).shape)
    if lam_ortho>0:
        E = E / len(op_l) + lam_ortho * ortho_loss(Q_)
    return E


def optimize_cfmap(X,Y,C, k2, n_iter = 300, lr = 1e-4, eps = 1e-5, verbose=1,
                  recompute_ops=False, step_spec=7, lam_ortho=-1):
    ##init Q
    #Q = tf.eye(k2, dtype=np.complex128)
    #Q = tf.Variable(np.eye(2 * k2))
    Q = tf.Variable(np.eye(k2, dtype=np.complex128))
    #Q = tf.Variable(Cc12[:k2,:k2])

    #opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    E_last = 0
    k1 = C.shape[0]

    op_l = []
    op_r = []
    if verbose >= 0: print("preshot ops")
    if not hasattr(X, 'op_l') or recompute_ops:
        for i in range(1, C.shape[0], step_spec):
            Dfi = X.grad_fun_scal_op(X.eig[:,i],k1,k2)
            CDfi = C @ Dfi
            DCfi = Y.grad_fun_scal_op(Y.eig[:,:k1] @ C[:k1,i],k1,k2)
            op_l += [CDfi]
            op_r += [DCfi]
        X.op_l = op_l
        Y.op_r = op_r
    else:
        op_l = X.op_l
        op_r = Y.op_r
    if verbose >= 0: print("done")
    ####

    for i in range(int(n_iter)):
        ##energy
        with tf.GradientTape() as tape:
            E = grad_fun_scal_loss(Q, op_l, op_r, lam_ortho=lam_ortho)
            #Qq = comp_to_real(Q)
        #print(E)
        grads = tape.gradient(E, [Q])
        #grads2 = tape.gradient(Qq, [Q])
        #print(grads)
        #stop
        if (tf.abs(E_last - E) < eps):
            if verbose >= 0:
                print('stop criterion')
                print('step : {:03d} | E : {:0.6f}'.format(i, E.numpy()))
            break
        E_last = E
        #Process the gradients
        processed_grads = [g for g in grads]
        grads_and_vars = zip(processed_grads, [Q])
        #print(Q[:3,:3])
        ##print
        if verbose>0:
            if i%(n_iter/10)==0:
                print('step : {:03d} | E : {:0.6f}'.format(i, E.numpy()))
        if verbose==0:
            if i==0 or i==(n_iter - 1):
                print('step : {:03d} | E : {:0.6f}'.format(i, E.numpy()))
        ##
        opt.apply_gradients(grads_and_vars)
    return Q.numpy()

###

def TBC_loss(C, op_l, op_r, lam_ortho=1e1):
    E = 0
    for i in range(len(op_l)):
        #print(op_l[i].shape, op_r[i].shape, Q_.shape)
        E += tf.cast(tf.nn.l2_loss(C @ op_l[i] - op_r[i] @ C), tf.float32)
    E = E / len(op_l) + lam_ortho * ortho_loss(C)
    return E

def ortho_loss(C):
    C_star = tf.transpose(C, [1,0])
    D = C @ C_star - tf.eye(C.shape[0], dtype=C.dtype)
    return tf.cast(tf.nn.l2_loss(D), tf.float32)

def optimize_fmap(X,Y,Q, k1, n_iter=300, lr=1e-4, eps=1e-5, verbose=1,
                  recompute_ops=False, step_spec=7, C_init=None, lam_ortho=1e1):
    ##init C
    C_ = 1 * np.eye(k1, dtype=np.float32)
    if C_init is not None:
        k0 = C_init.shape[0]
        C_[:k0,:k0] = C_init
    C = tf.Variable(C_)

    #opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    E_last = 0
    k2 = Q.shape[0]

    opVF_l = []
    opVF_r = []
    print("preshot ops")
    if not hasattr(X, 'opVF_l') or recompute_ops:
        for i in range(1, Q.shape[0], step_spec):
            DXi = X.VF_fun_scal_op(X.ceig[:,i],k1)
            #DXi = C @ Dfi
            DQXi = Y.VF_fun_scal_op(Y.ceig[:,:k2] @ Q[:k2,i],k1)
            opVF_l += [DXi]
            opVF_r += [DQXi]
        X.opVF_l = opVF_l
        Y.opVF_r = opVF_r
    else:
        opVF_l = X.opVF_l
        opVF_r = Y.opVF_r
    print("done. we have", len(opVF_l), "operators")
    ####
    flag_lr = False
    for i in range(int(n_iter)):
        ##energy
        with tf.GradientTape() as tape:
            E = TBC_loss(C, opVF_l, opVF_r, lam_ortho=lam_ortho)
        grads = tape.gradient(E, [C])
        
        #if (tf.abs(E_last - E) <  eps) and not flag_lr:
        #    lr /=1e2
        #    eps /=1e2
        #    print('step : {:03d} | lr down to : {:0.5f}'.format(i, lr))
        #    opt = tf.keras.optimizers.Adam(learning_rate=lr)
        #    flag_lr = True
        
        #stop
        if (tf.abs(E_last - E) < eps):
            print('stop criterion')
            print('step : {:03d} | E : {:0.6f}'.format(i, E.numpy()))
            break
        E_last = E
        #Process the gradients
        processed_grads = [g for g in grads]
        grads_and_vars = zip(processed_grads, [C])
        ##
        if verbose>0:
            if i%(n_iter/10)==0:
                print('step : {:03d} | E : {:0.6f}'.format(i, E.numpy()))
        if verbose==0:
            if i==0 or i==(n_iter - 1):
                print('step : {:03d} | E : {:0.6f}'.format(i, E.numpy()))
        ##
        opt.apply_gradients(grads_and_vars)
    return C.numpy()

##### join optimisation of Q and C

def ortho_loss_Q(Q):
    Q_star = tf.math.conj(tf.transpose(Q, [1,0]))
    D = Q @ Q_star - tf.eye(Q.shape[0], dtype=Q.dtype)
    D = tf.abs(D)
    E = tf.cast(tf.nn.l2_loss(D), tf.float32)
    return E

def lapcom_loss(C, X, Y):
    k1 = C.shape[0]
    L1 = np.diag(X.vals[:k1])
    L2 = np.diag(Y.vals[:k1])
    E = tf.cast(tf.nn.l2_loss(C@L1 - L2@C), tf.float32)
    return E

def C_loss(X,Y,C, lam_ortho=-1, lam_lapcom=-1):
    k1 = C.shape[0]
    F = X.eig_trans[:k1] @ X.hks
    G = Y.eig_trans[:k1] @ Y.hks
    E = tf.cast(tf.nn.l2_loss(C @ F - G), tf.float32)
    if lam_ortho > 0: E += lam_ortho * ortho_loss(C)
    if lam_lapcom > 0: E += lam_lapcom * lapcom_loss(C,X,Y)
    return E

def mixed_CQ_loss(X,Y,C,Q):
    ## we store D_f operators in k1 x k1 x 2k2 operators
    ## the loss is then sum_i | C D_fi - sum_j C_ij D_fj Q |
    k1 = C.shape[0]
    k2 = Q.shape[0]
    Q_ = comp_to_real(Q)
    CD  = tf.einsum('ij,bjk->bik', C, X.Df[:k1,:k1,:2*k2])
    DQ  = tf.einsum('bij,jk->bik', Y.Df[:k1,:k1,:2*k2], Q_)
    CDQ = tf.einsum('bi,bjk->ijk', C, DQ)
    #
    E = tf.cast(tf.nn.l2_loss(CD - CDQ), tf.float32)
    return E

def total_loss(X,Y,C,Q, lam_Q=1e0, lam_ortho=1e0, lam_lapcom=1e0, lam_mixed=1e0):
    EC = C_loss(X,Y,C,lam_ortho=lam_ortho, lam_lapcom=lam_lapcom)
    #EC = 0
    if lam_ortho > 0 and lam_Q > 0: EC += lam_Q * lam_ortho * ortho_loss_Q(Q)
    if lam_mixed > 0: EC += lam_mixed * mixed_CQ_loss(X,Y,C,Q)
    return EC

###########################
###########################
####### bij ZO from random

def to_rl(A):
    a = A.real; b = A.imag;
    c = np.stack([a,b], -1)
    #print(c.shape)
    c = c.reshape(A.shape[0], 2*A.shape[1])
    return c

def op_cpl(op):
    idv = np.arange(op.shape[1]//2)
    a = op[:, 2 * idv]
    b = op[:, 2 * idv + 1]
    return a + 1j * b

#random T12
def initialize_pMap(nX, nY):
    #rng = np.random.default_rng(42)
    #T12 = rng.random(nY, size=nX)
    T12 = np.random.randint(nY, size=nX)
    return T12

def fMap2pMap(L1, L2, fmap):
    dim1 = fmap.shape[1]
    dim2 = fmap.shape[0]

    B1 = L1[:,:dim1]
    B2 = L2[:,:dim2]
    C12 = fmap
    _, T21 = spatial.cKDTree(B1@C12.T).query(B2, n_jobs=-1)
    #T21 = knnsearch(B1*C12', B2);
    return T21

def pMap2fMap(L1, L2, pmap):
    C21 = np.linalg.pinv(L1) @ L2[pmap]
    return C21
####
def bij_fMap2pMap(B1, B2, C12, C21):
    _, T12 = spatial.cKDTree(B2@C21.T).query(B1, n_jobs=-1)
    _, T21 = spatial.cKDTree(B1@C12.T).query(B2, n_jobs=-1)

    #bijective modification for C12, C21
    B2_aug = np.concatenate([B2, B2[T12]], axis=0)
    B1_aug = np.concatenate([B1[T21], B1], axis=0)
    C12 = np.linalg.pinv(B2_aug) @ B1_aug
    #
    B2_aug = np.concatenate([B2[T12], B2], axis=0)
    B1_aug = np.concatenate([B1, B1[T21]], axis=0)
    C21 = np.linalg.pinv(B1_aug) @ B2_aug
    
    #use both maps to get point to point
    B2_aug = np.concatenate([B2@C21.T, B2@C12], axis=1)
    B1_aug = np.concatenate([B1, B1], axis=1)
    _, T12 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    #
    B2_aug = np.concatenate([B1@C12.T, B1@C21], axis=1)
    B1_aug = np.concatenate([B2, B2], axis=1)
    _, T21 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    return T12, T21
####

def bij3_fMap2pMap(B1, B2, C12, C21, L1, L2, w1=1, w2=1, w3=1):
    _, T12 = spatial.cKDTree(B2@C21.T).query(B1, n_jobs=-1)
    _, T21 = spatial.cKDTree(B1@C12.T).query(B2, n_jobs=-1)

    #bijective modification for C12, C21
    B2_aug = np.concatenate([B2, B2[T12]], axis=0)
    B1_aug = np.concatenate([B1[T21], B1], axis=0)
    C12 = np.linalg.pinv(B2_aug) @ B1_aug
    #
    B2_aug = np.concatenate([B2[T12], B2], axis=0)
    B1_aug = np.concatenate([B1, B1[T21]], axis=0)
    C21 = np.linalg.pinv(B1_aug) @ B2_aug
    
    #use triple energy to get T12, T21  
    B2_aug = np.concatenate([w1 * B2@C21.T,
                             w2 * B2@L2@C21.T,
                             w3 * B2@C12], axis=1)
    B1_aug = np.concatenate([w1 * B1,
                             w2 * B1@L1,
                             w3 * B1], axis=1)
    _, T12 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    #
    B2_aug = np.concatenate([w1 * B1@C12.T,
                             w2 * B1@L1@C12.T,
                             w3 * B1@C21], axis=1)
    B1_aug = np.concatenate([w1 * B2,
                             w2 * B2@L2,
                             w3 * B2], axis=1)
    _, T21 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    return T12, T21

def CMap2QMap(X,Y,C,k2, verbose=0):
    k1 = C.shape[0]
    ### we can consider only a subspace of the eigenspace (crop)
    if not hasattr(X, 'Df'):
        if verbose > 0: print('recomputing ops')
        X.fun_scal_op_basis(k1,k2)
        Y.fun_scal_op_basis(k1,k2)
    #print(X.Df.shape, Y.Df.shape, C.shape)
    #
    k1_ = X.Df.shape[1]
    k2_ = X.Df.shape[2]//2
    if k1>k1_ or k2>k2_:
        X.fun_scal_op_basis(k1,k2)
        Y.fun_scal_op_basis(k1,k2)
    #
    CD = np.einsum('ij,bjk->bik', C, X.Df[:k1,:k1,:2*k2])  #size b x k1 x k2
    DC = np.einsum('bi,bjk->ijk', C, Y.Df[:k1,:k1,:2*k2])  #same size
    #
    CD = op_cpl(np.concatenate(list(CD), axis=0))
    DC = op_cpl(np.concatenate(list(DC), axis=0))
    #
    Q = np.linalg.lstsq(DC,CD,rcond=None)[0]
    #print(Q_flat.shape)
    return np.conjugate(Q)
    

def bij3_fMap2pMap_withQor(X, Y, B1, B2, C12, C21, L1, L2,
                         w1=1, w2=1, w3=1, Qdes=30, verbose=0,
                         lam_lr=0.5):
    
    ### first convert to Q to "get rid" of symmetries
    k = C12.shape[0]
    lr = lam_lr * 1/k**2
    
    #Q12 = CMap2QMap(X,Y,C12,k)
    #Q21 = CMap2QMap(Y,X,C21,k)
    
    Q12 = optimize_cfmap(X,Y,C12, k, n_iter = Qdes, lr = lr,
                        recompute_ops=True, step_spec=k//5, eps=0.05 * lr,
                        verbose=verbose, lam_ortho=1e0)
    Q21 = optimize_cfmap(Y,X,C21, k, n_iter = Qdes, lr = lr,
                        recompute_ops=True, step_spec=k//5, eps=0.05 * lr,
                        verbose=verbose, lam_ortho=1e0)
    
    B1_ = to_rl(X.ceig[X.samples,:k])
    B1_Q12 = to_rl(X.ceig[X.samples,:k]@np.conjugate(Q12.T))
    B2_ = to_rl(Y.ceig[Y.samples,:k])
    B2_Q21 = to_rl(Y.ceig[Y.samples,:k]@np.conjugate(Q21.T))
    
    _, T12 = spatial.cKDTree(B2_Q21).query(B1_, n_jobs=-1)
    _, T21 = spatial.cKDTree(B1_Q12).query(B2_, n_jobs=-1)

    #bijective modification for C12, C21
    B2_aug = np.concatenate([B2, B2[T12]], axis=0)
    B1_aug = np.concatenate([B1[T21], B1], axis=0)
    C12 = np.linalg.pinv(B2_aug) @ B1_aug
    #
    B2_aug = np.concatenate([B2[T12], B2], axis=0)
    B1_aug = np.concatenate([B1, B1[T21]], axis=0)
    C21 = np.linalg.pinv(B1_aug) @ B2_aug
    
    #use triple energy to get T12, T21  
    B2_aug = np.concatenate([w1 * B2@C21.T,
                             w2 * B2@L2@C21.T,
                             w3 * B2@C12], axis=1)
    B1_aug = np.concatenate([w1 * B1,
                             w2 * B1@L1,
                             w3 * B1], axis=1)
    _, T12 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    #
    B2_aug = np.concatenate([w1 * B1@C12.T,
                             w2 * B1@L1@C12.T,
                             w3 * B1@C21], axis=1)
    B1_aug = np.concatenate([w1 * B2,
                             w2 * B2@L2,
                             w3 * B2], axis=1)
    _, T21 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    return T12, T21

def bij3_fMap2pMap_withQ(X, Y, B1, B2, C12, C21, L1, L2,
                         w1=1, w2=1, w3=1):
    
    ### first convert to Q to "get rid" of symmetries
    k = C12.shape[0]
    
    Q12 = CMap2QMap_procrustes(X,Y,C12,k)
    Q21 = CMap2QMap_procrustes(Y,X,C21,k)
    
    B1_ = to_rl(X.ceig[X.samples,:k])
    B1_Q12 = to_rl(X.ceig[X.samples,:k]@np.conjugate(Q12.T))
    B2_ = to_rl(Y.ceig[Y.samples,:k])
    B2_Q21 = to_rl(Y.ceig[Y.samples,:k]@np.conjugate(Q21.T))
    
    _, T12 = spatial.cKDTree(B2_Q21).query(B1_, n_jobs=-1)
    _, T21 = spatial.cKDTree(B1_Q12).query(B2_, n_jobs=-1)
    
    #print(T12[:10], T21[:10])
    #print(np.min(T12), np.min(T21))
    #print(np.max(T12), np.max(T21), B1.shape, B2.shape)
    #print(B2[T12].shape)
    #print(T12.shape, T21.shape, np.max(T12), np.max(T21))
    #print(B1.shape, B2.shape)
    
    #_, T12 = spatial.cKDTree(B2@C21.T).query(B1, n_jobs=-1)
    #_, T21 = spatial.cKDTree(B1@C12.T).query(B2, n_jobs=-1)

    #bijective modification for C12, C21
    B2_aug = np.concatenate([B2, B2[T12]], axis=0)
    B1_aug = np.concatenate([B1[T21], B1], axis=0)
    C12 = np.linalg.pinv(B2_aug) @ B1_aug
    #
    B2_aug = np.concatenate([B2[T12], B2], axis=0)
    B1_aug = np.concatenate([B1, B1[T21]], axis=0)
    C21 = np.linalg.pinv(B1_aug) @ B2_aug
    
    #use triple energy to get T12, T21  
    B2_aug = np.concatenate([w1 * B2@C21.T,
                             w2 * B2@L2@C21.T,
                             w3 * B2@C12], axis=1)
    B1_aug = np.concatenate([w1 * B1,
                             w2 * B1@L1,
                             w3 * B1], axis=1)
    _, T12 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    #
    B2_aug = np.concatenate([w1 * B1@C12.T,
                             w2 * B1@L1@C12.T,
                             w3 * B1@C21], axis=1)
    B1_aug = np.concatenate([w1 * B2,
                             w2 * B2@L2,
                             w3 * B2], axis=1)
    _, T21 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    return T12, T21

def CMap2QMap_procrustes(X,Y,C,k2, verbose=0):
    k1 = C.shape[0]
    ### we can consider only a subspace of the eigenspace (crop)
    if not hasattr(X, 'Df'):
        if verbose > 0: print('recomputing ops')
        X.fun_scal_op_basis(k1,k2)
        Y.fun_scal_op_basis(k1,k2)
    #print(X.Df.shape, Y.Df.shape, C.shape)
    #
    k1_ = X.Df.shape[1]
    k2_ = X.Df.shape[2]//2
    if k1>k1_ or k2>k2_:
        X.fun_scal_op_basis(k1,k2)
        Y.fun_scal_op_basis(k1,k2)
    #
    CD = np.einsum('ij,bjk->bik', C, X.Df[:k1,:k1,:2*k2])  #size b x k1 x k2
    DC = np.einsum('bi,bjk->ijk', C, Y.Df[:k1,:k1,:2*k2])  #same size
    #
    CD = op_cpl(np.concatenate(list(CD), axis=0))
    DC = op_cpl(np.concatenate(list(DC), axis=0))
    #
    #Q = np.linalg.lstsq(DC,CD,rcond=None)[0]
    M = np.conjugate(DC).T @ CD
    u, _, v = np.linalg.svd(M)
    R = u@v
    #print(Q_flat.shape)
    return np.conjugate(R)

def bij3_fMap2pMap_withQ_v2(X, Y, B1, B2, C12, C21, L1, L2,
                         w1=1, w2=1, w3=1, w4=1):
    
    ### first convert to Q to "get rid" of symmetries
    k = C12.shape[0]
    
    ### try with procrustes
    Q12 = CMap2QMap_procrustes(X,Y,C12,k)
    Q21 = CMap2QMap_procrustes(Y,X,C21,k)
    
    B1_ = to_rl(X.ceig[X.samples,:k])
    B1_Q12 = to_rl(X.ceig[X.samples,:k]@np.conjugate(Q12.T))
    B2_ = to_rl(Y.ceig[Y.samples,:k])
    B2_Q21 = to_rl(Y.ceig[Y.samples,:k]@np.conjugate(Q21.T))
    
    _, T12 = spatial.cKDTree(B2_Q21).query(B1_, n_jobs=-1)
    _, T21 = spatial.cKDTree(B1_Q12).query(B2_, n_jobs=-1)

    #bijective modification for C12, C21
    B2_aug = np.concatenate([B2, B2[T12]], axis=0)
    B1_aug = np.concatenate([B1[T21], B1], axis=0)
    C12 = np.linalg.pinv(B2_aug) @ B1_aug
    #
    B2_aug = np.concatenate([B2[T12], B2], axis=0)
    B1_aug = np.concatenate([B1, B1[T21]], axis=0)
    C21 = np.linalg.pinv(B1_aug) @ B2_aug
    
    #use triple energy to get T12, T21  
    B2_aug = np.concatenate([w1 * B2@C21.T,
                             w2 * B2@L2@C21.T,
                             w3 * B2@C12], axis=1)
    if w4>0:B2_aug = np.concatenate([B2_aug, w4 * B2_Q21], axis=1)
    B1_aug = np.concatenate([w1 * B1,
                             w2 * B1@L1,
                             w3 * B1], axis=1)
    if w4>0:
        #print('wol')
        B1_aug = np.concatenate([B1_aug, w4 * B1_], axis=1)
    _, T12 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    #
    B2_aug = np.concatenate([w1 * B1@C12.T,
                             w2 * B1@L1@C12.T,
                             w3 * B1@C21], axis=1)
    if w4>0:B2_aug = np.concatenate([B2_aug, w4 * B1_Q12], axis=1)
    B1_aug = np.concatenate([w1 * B2,
                             w2 * B2@L2,
                             w3 * B2], axis=1)
    if w4>0:B1_aug = np.concatenate([B1_aug, w4 * B2_], axis=1)
    _, T21 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    return T12, T21


# Complete bij ZO algo 
def func_bijective_zm_fmap(X, Y, C12_ini, C21_ini, k_init=10,k_step=1, k_final=30, 
                           N_inter=5, method='bij',
                           w1=1, w2=2, w3=1, w4=1, 
                           Qdes=30, lam_lr=0.5,
                           verbose=-1):
    X.fps_3d(500)
    Y.fps_3d(500)

    B1_all = X.eig[X.samples]
    B2_all = Y.eig[Y.samples]

    T12 = fMap2pMap(B2_all, B1_all, C21_ini)
    T21 = fMap2pMap(B1_all, B2_all, C12_ini)

    for k in range(k_init, k_final, k_step):
        
        print("step:", k)
        for n in range(N_inter):
        
            B1 = B1_all[:, :k]
            B2 = B2_all[:, :k]
            #classic ZO step
            #print(B2.shape, B1[T21].shape)
            C12 = np.linalg.pinv(B2) @ B1[T21]
            C21 = np.linalg.pinv(B1) @ B2[T12]
            ##
            L1 = np.diag(X.vals[:k]); L2 = np.diag(Y.vals[:k]);
            #print(L1.shape, L2.shape)
            if method=='bij':
                #original bijective zoomout
                T12, T21 = bij_fMap2pMap(B1, B2, C12, C21)
            elif method=='bij3':
                #same as before but with isometry
                T12, T21 = bij3_fMap2pMap(B1, B2, C12, C21, L1, L2,
                                          w1=w1, w2=w2, w3=w3)
            elif method=='bijQ':
                #adding 1 Q projection at the beginning
                T12, T21 = bij3_fMap2pMap_withQ(X,Y,B1, B2, C12, C21, L1, L2,
                                                w1=w1, w2=w2, w3=w3)
            elif method=='bijQ2':
                #with procrustes problem and using Q also at the last p2p step
                T12, T21 = bij3_fMap2pMap_withQ_v2(X,Y,B1, B2, C12, C21, L1, L2,
                                                   w1=w1, w2=w2, w3=w3, w4=w4)
            elif method=='bijQo':
                T12, T21 = bij3_fMap2pMap_withQor(X,Y,B1, B2, C12, C21, L1, L2,
                                        w1=w1, w2=w2, w3=w3,
                                        Qdes=Qdes,verbose=verbose,lam_lr=lam_lr)
            else:
                raise NameError('this method does not exist')
                

    B1 = B1_all[:, :k_final]
    B2 = B2_all[:, :k_final]
    C21 = np.linalg.pinv(B1) @ B2[T12]
    C12 = np.linalg.pinv(B2) @ B1[T21]
    return C12, C21

def QC_to_pMap(X, Y, Q):
    k = Q.shape[0]
    B1 = to_rl(X.ceig[:,:k] @ np.conjugate(Q.T))
    B2 = to_rl(Y.ceig[:,:k])
    _, T12 = spatial.cKDTree(B2).query(B1, n_jobs=-1)
    return T12

def euc_err(Y, T, Tgt):
    area = np.sum(igl.doublearea(Y.v, Y.f)/2)
    #print(np.sqrt(area))
    d = Y.v[T] - Y.v[Tgt]
    err = np.mean(np.linalg.norm(d, axis=1)) / np.sqrt(area)
    return err


########## to write ply ##########
def write_ply(filename, verts, faces, color=['0', '0', '0']):
    nv = verts.shape[0]
    nf = faces.shape[0]
    file = open(filename, 'w')
    file.write("ply\n")
    file.write("format ascii 1.0\n")
    file.write("comment made by Nicolas Donati\n")
    file.write("comment this file is a FAUST shape with mapping colormap\n")
    file.write("element vertex " + str(nv) + "\n")
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    file.write("property uchar red\n")
    file.write("property uchar green\n")
    file.write("property uchar blue\n")
    file.write("element face " + str(nf) + "\n")
    file.write("property list uchar int vertex_index\n")
    file.write("end_header\n")
    #color = ['0', '0', '1']
    i = 0
    for item in verts:
        file.write("{0} {1} {2} {3} {4} {5}\n".format(item[0],item[1],item[2],
                           color[i,0], color[i,1], color[i,2]))
        i+=1
    #for item in normals:
    # file.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))
    for item in faces:
        file.write("3 {0} {1} {2}\n".format(item[0],item[1],item[2]))  
    file.close()
    return