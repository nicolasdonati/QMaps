import igl
##
import numpy as np
from scipy import spatial
import scipy as sp
##
from os.path import join, exists

#################################
#################################
#   Nicolas Donati on 01/2021   #
#################################
#################################


class mesh:
    #fields
    v = None
    f = None
    e = None
    ##
    n = None
    nv = None
    ##
    l = None
    cl = None
    eig = None
    ceig = None
    #
    vts = None
    samples = None
    
    
    def __init__(self, path, normalized = False, normals=True,
                 neighbors=True, spectral=30, verbose=0):
        self.path = path
        self.name = path.split('/')[-1]
        self.folder = self.path[:-len(self.name)-1]
        self.verbose = verbose
        
        if verbose > 0:
            print("loading ", self.name, "with init spectral =", spectral)
        
        self.v, self.f = igl.read_triangle_mesh(path)
    
        if normalized: self.center_and_scale(scale=True)
        if normals:
            self.normals_on_faces()
            self.normals_on_vertices()
        if spectral>0: self.spectral(k=spectral)
        if neighbors: self.neighbors()
    
    def get_vts(self, cor_folder='cor'):
        if cor_folder is None:
            self.vts = np.arange(self.v.shape[0])
            return
        vts_file = join(self.folder, cor_folder , self.name[:-4]+'.vts')
        self.vts  = np.loadtxt(vts_file, dtype=np.int32) - 1
    
    def center_and_scale(self, scale=False):
        self.v -= np.mean(self.v, axis=0)
        if scale:
            area = np.sum(igl.doublearea(self.v,self.f))/2
            print('area was', area)
            self.v /= np.sqrt(area)
    
    def neighbors(self):
        self.neigh = igl.adjacency_matrix(self.f)
        if self.samples is not None:
            self.neigh_samples = self.neigh[self.samples]
    
    ## recover halfedge data structure
    def halfedge(self):
        self.e, self.ue, self.emap, self.ue2e = igl.unique_edge_map(self.f)
        #_, _, ef, ei = igl.edge_flaps(self.f)
        e2ue2e = np.array(self.ue2e)[self.emap]
        ee = np.tile(np.arange(self.e.shape[0]), [2, 1]).T
        #print(self.e.shape[0], ee.shape)
        op_mask = (e2ue2e != ee)
        self.op = e2ue2e[op_mask]
        
        e2f = np.remainder(np.arange(self.e.shape[0]), self.f.shape[0])
        e2f_in = np.arange(self.e.shape[0]) // self.f.shape[0]
        nex_in = (e2f_in + 1) % 3
        self.nex = nex_in * self.f.shape[0] + e2f
        
        
    def normals_on_faces(self, with_def=False):
        if with_def:  #there is a deformation affecting the embedding
            #self.n_def = igl.per_face_normals(self.v_def_total,self.f,0*self.v[0])
            self.n_def = igl.per_face_normals(self.v_def,self.f,0*self.v[0])
            return  # no need to recompute n
        self.n = igl.per_face_normals(self.v,self.f,0*self.v[0])
    
    def frames_on_faces(self, smooth=True):
        if self.n is None: self.normals_on_faces()
        v = self.v
        if smooth:
            v = self.v_smooth
        v1 = v[self.f[:,0],:]
        v2 = v[self.f[:,1],:]
        v3 = v[self.f[:,2],:]
        #v4 = self.n + v1
        v4 = np.cross((v2-v1),(v3-v1),1); print('compute normal with cross product')
        self.v4 = np.stack([v2 - v1, v3 - v1, v4 - v1], -1)
    
    def normals_on_vertices(self, with_def=False):
        if with_def:  #there is a deformation affecting the embedding
            #self.nv_def = igl.per_vertex_normals(self.v_def_total,self.f)
            self.nv_def = igl.per_vertex_normals(self.v_def,self.f)
            return  # no need to recompute n
        self.nv = igl.per_vertex_normals(self.v,self.f)
    
    ######
    def spectral(self, k = 30, save=True):
        save_spectral = join(self.folder, 'spectral', self.name[:-4]+'.npz')
        load = exists(save_spectral)
        if load:
            F =  np.load(save_spectral)
            if np.isin('eig', F.files):
                if F['eig'].shape[1] >= k:
                    save = False;
                    if self.verbose > 0: print('from file')
                    self.eig = F['eig'][:,:k]; self.vals = F['val'][:k];            
        ###
        l = -igl.cotmatrix(self.v, self.f)
        m = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
        self.l = l
        self.m = m
        if not save: return  ## l and m should still be computed
        self.vals, self.eig = sp.sparse.linalg.eigsh(l, k, m, sigma=0, which="LM") #eig ~ n*k
        #self.eig_trans = (m @ self.eig).T
        ###
        if save:
            if load and np.isin('ceig', F.files):
                np.savez(save_spectral, eig=self.eig, val=self.vals, ceig=F['ceig'], cval=F['cval'])
                return
            np.savez(save_spectral, eig=self.eig, val=self.vals)

    
    def complex_spectral(self, k = 30, save=True):
        if self.verbose>0: print("loading complex spectral =", k)
        ###
        save_spectral = join(self.folder, 'spectral', self.name[:-4]+'.npz')
        load = exists(save_spectral)
        if load:
            F =  np.load(save_spectral)
            if np.isin('ceig', F.files):
                if F['ceig'].shape[1] >= k:
                    save = False;
                    if self.verbose>0: print('from file')
                    self.ceig = F['ceig'][:,:k]; self.cvals = F['cval'][:k];
        ###
        self.cl = self.complex_laplacian()
        if not save: return  ## cl should still be computed
        self.cvals, self.ceig = sp.sparse.linalg.eigsh(self.cl, k, self.m, sigma=0, which="LM")
        ###
        if save:
            print('saving complex spectral')
            if load and np.isin('eig', F.files):
                np.savez(save_spectral, eig=F['eig'], val=F['val'], ceig=self.ceig, cval=self.cvals)
                return
            np.savez(save_spectral, ceig=self.ceig, cval=self.cvals)
    
    ######
    def smooth(self, k=30, affect_shape=False):
        if self.eig is None or self.eig.shape[1] < k:
            print("need (more) spectral for smooth [k =",k,"]")
            self.spectral(k)
        X = self.eig_trans[:k] @ self.v
        v_smooth = self.eig[:,:k] @ X
        #v_smooth -= np.mean(v_smooth, axis=0)
        if affect_shape: self.v = v_smooth
        return v_smooth, X
    
    def smooth_sigmoid(self,k=30,trunc_thresh=1e-2,smooth_thresh=10, affect_shape=False):
        #compute the sigmoid weights
        if k == 1: weights = 1
        elif k >= smooth_thresh:
            t = -1/(smooth_thresh-1) * np.log(1/(1-trunc_thresh)-1)
            weights = 1/(1+np.exp(t*np.arange(1-k,smooth_thresh)))
        else:
            t = -1/(k-1) * np.log(1/(1-trunc_thresh)-1)
            weights = 1/(1+np.exp(t*np.arange(1-k,k)))
        
        k = weights.shape[0]; #print('k_smoothshells:', k)
        #recompute eigs if necessary
        if self.eig is None or self.eig.shape[1] < k:
            print("need (more) spectral for smooth [k =",k,"]")
            self.spectral(k)
        #project the geometry on the eigenfunctions
        X = weights[:, None] * (self.eig_trans[:k] @ self.v)
        v_smooth = self.eig[:,:k] @ X
        
        self.v_smooth = v_smooth  ## store smooth shape as different embedding
        if affect_shape:
            self.v = v_smooth
            self.normals_on_faces();
        return v_smooth, X
    
    def deform(self, tau, J=None):
        ##deformation affecting v_smooth (but maybe also v in order to compute normals)
        k = tau.shape[0]
        def_field = self.eig[:,:k] @ tau
        if J is not None:
            J_def_field = J @ def_field[:,:,None]
            self.v_def = self.v_smooth + J_def_field
            self.v_def_total = self.v + J_def_field
        else:
            self.v_def = self.v_smooth + def_field
            self.v_def_total = self.v + def_field
        self.normals_on_faces(with_def=True)
        self.normals_on_vertices(with_def=True)
                            
    
    def fps_3d(self, n, seed=42):
        if n > self.v.shape[0]: n = self.v.shape[0]

        S = np.zeros(n, dtype=int);
        S[0] = seed;
        d = np.linalg.norm(self.v - self.v[S[0]], axis=1)
        # avoiding loop would be nice .. but this is already quite fast
        for i in range(1, n):
            m = np.argmax(d);
            S[i] = m;
            new_d = np.linalg.norm(self.v - self.v[S[i]], axis=1)
            d = np.min(np.stack([new_d,d], axis=-1),axis=-1)
        self.samples = S
        return S
    
    ######
    def local_basis(self):
        self.halfedge()
        _, self.he_start = np.unique(self.e[:,0], return_index=1)  #starting he for each vertex
        return
    
    def embed_he(self, he):
        he_emb = self.v[self.e[he][:,1]] - self.v[self.e[he][:,0]]
        return he_emb
    
    def embed_vector_field(self, VF, reverse=False):
        ## VF is a complex per-vertex, we have to translate this into 3d vectors
        ## first find the two halfedges concerned (in the right order)
        ## also keep the angle within
        if self.e is None: self.local_basis()
        if self.cl is None: self.complex_laplacian()
        ##
        angles = np.angle(VF)
        angles[angles<0] += 2 * np.pi ## reposition in [0, 2pi]
        ##
        rotate = np.zeros(self.v.shape[0], dtype=int)  #will keep in degree of VF
        he = np.copy(self.he_start)
        last_he = he
        he_VF = np.zeros((self.v.shape[0],2), dtype=int)
        in_ang_VF = np.zeros(self.v.shape[0])
        i = 0
        while np.any(rotate==0):
            ##circulate CW or CCW (something seems off on CCW, results are awkward..)
            if not reverse:
                he = self.op[he]
                he = self.nex[he]
            else:
                he = self.nex[he]
                he = self.nex[he]
                he = self.op[he]
            i+=1
            ##compare angles
            he_cur_angle = self.he_angles_norm[he]
            ang_mask = (he_cur_angle > angles)
            rot_mask = (rotate == 0) * ang_mask
            ##check if we reached end
            rot_mask = rot_mask | ((rotate==0) * (self.he_start == he))
            #print('size', np.sum(rot_mask))
            rotate[rot_mask] = i
            ##
            he_VF[rot_mask,0] = last_he[rot_mask]
            he_VF[rot_mask,1] = he[rot_mask]
            in_ang_VF[rot_mask] = angles[rot_mask] - self.he_angles_norm[last_he[rot_mask]]
            last_he = he
        in_ang_VF *= (1 - self.K/(2*np.pi))
        #print('rot', np.mean(rotate))
        b1 = self.embed_he(he_VF[:, 0]); b2 = self.embed_he(he_VF[:, 1]);
        b1 /= np.linalg.norm(b1, axis=1)[:,None]
        b2 -= np.sum(b1*b2, axis=1)[:,None] * b1
        b2 /= np.linalg.norm(b2, axis=1)[:,None]
        VF_embed = np.cos(in_ang_VF)[:,None] * b1 + np.sin(in_ang_VF)[:,None] * b2
        VF_embed *= np.abs(VF)[:,None]
        return VF_embed
    
    def complex_laplacian(self):
        angles = igl.internal_angles(self.v, self.f)
        ## here we have an angle deficit on verts
        ## we need to flatten it for our vert Laplacian
        self.he_angles = np.concatenate([angles[:,1], angles[:,2], angles[:,0]])
        self.local_basis()
        self.he_angles[self.he_start] = 0  # angle basis
        self.K = igl.gaussian_curvature(self.v, self.f) ##angle deficit
        ## (we already computed the angles but how to sum them quickly around vertices ?)
        vert_angle_sum = 2 * np.pi - self.K
        self.he_angles_norm = 2 * np.pi * self.he_angles/vert_angle_sum[self.e[:,0]]
        #print(self.he_angles_norm[self.e[:,0]==4] *180/np.pi) ## precision 1e-12
        
        ## now we have to circle around vertices just to get cumulative angle sums
        rotate = np.zeros(self.v.shape[0], dtype=int)  #will keep in degree of vertex
        he = np.copy(self.he_start)
        last_he = he
        i = 0
        while np.any(rotate==0):
            ##circulate CW
            he = self.op[he]
            he = self.nex[he]
            i+=1
            rot_mask = (self.he_start == he) * (rotate == 0)
            rotate[rot_mask] = i
            self.he_angles_norm[he[rotate==0]] += self.he_angles_norm[last_he[rotate==0]]
            last_he = he
        #print(self.he_angles_norm[self.e[:,0]==4] *180/np.pi) ## precision 1e-12
        
        ## then simply  get the rho
        self.rho = (self.he_angles_norm[self.op] + np.pi) - self.he_angles_norm
        r = np.cos(self.rho) + np.sin(self.rho) * 1j
        r_op = r[self.op]
        ##
        self.r = r.reshape(3, self.f.shape[0]).T
        self.r_op = r_op.reshape(3, self.f.shape[0]).T
        cot_ = 0.5 / np.tan(angles)
        cot = cot_ * self.r
        cot_op = cot_ * self.r_op
        S_ = np.concatenate([cot_[:,2], cot_[:,0], cot_[:,1]])
        S = np.concatenate([cot[:,2], cot[:,0], cot[:,1]])
        S_op = np.concatenate([cot_op[:,2], cot_op[:,0], cot_op[:,1]])
        
        ##
        I = np.concatenate([self.f[:,0], self.f[:,1], self.f[:,2]])
        J = np.concatenate([self.f[:,1], self.f[:,2], self.f[:,0]])
        In = np.concatenate([I, J, I, J])
        Jn = np.concatenate([J, I, I, J])
        Sn = np.concatenate([-S_op,-S,S_, S_])
        ##
        A = sp.sparse.csr_matrix((Sn, (In, Jn)), shape=(self.v.shape[0],self.v.shape[0]))
        return A
    
    
    ###### additional tools for Jacobian-based deformation
    def divergence_1_forms(self):
        nv = self.v.shape[0]
        ne = self.e.shape[0]
        ###
        a = -igl.cotmatrix_entries(self.v, self.f)
        er = np.arange(ne) #edge range
        op = self.op #opposite edge
        orv = self.e[:,0] #orig vertex
        exv = self.e[:,1] #extr vertex
        ags = np.concatenate([a[:,0], a[:,1], a[:,2]]) #flatten angles
        I = np.concatenate([orv, exv])
        J = np.concatenate([er, op])
        V = np.concatenate([ags, ags])
        #print(I[:10], J[:10], V[:10])
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(nv,ne))
        self.div1form = A
        return A
     
    def edge_jacobian_poisson(self, jac):
        l = self.l
        if not hasattr(self, 'div1form'): self.divergence_1_forms()
        ###
        e_tilde = jac @ self.embed_he(np.arange(self.e.shape[0]))[:,:,None]
        e_tilde = e_tilde[:,:,0]
        #print(e_tilde.shape)
        b = self.div1form @ e_tilde
        Y = sp.sparse.linalg.spsolve(l.T @ l, l.T @ b)
        return Y
    
    #a bit slow to build ... needs v4
    def face_jacobian_poisson(self, jac):
        nf = self.f.shape[0]
        nv = self.v.shape[0]
        #
        if not hasattr(self, 'v4'): self.frames_on_faces(smooth=False)
        #
        F = np.zeros((3*nf,3))
        Ir = [];
        Ic = [];
        Val = [];
        for t in range(nf):
            F[3*t:3*(t+1),:] = jac[t].T

            q, r = np.linalg.qr(self.v4[t, : ,:2])
            alphaT = np.linalg.inv(r)@(q.T)
            alphaT = np.vstack([alphaT,-np.sum(alphaT,0)]).T

            Ir += [3*t, 3*t, 3*t, 3*t+1, 3*t+1, 3*t+1, 3*t+2, 3*t+2, 3*t+2]
            Ic += [self.f[t,[1,2,0]], self.f[t,[1,2,0]], self.f[t,[1,2,0]]]
            Val += [alphaT.flatten()]

        Ic = np.concatenate(Ic)
        Val = np.concatenate(Val)
        A = sp.sparse.csr_matrix((Val, (Ir, Ic)), shape=(3*nf,nv))
        self.face_jac_A = A
        self.face_jac_F = F
        return A, F

    def solve_face_poisson(self):
        if not hasattr(self, 'face_jac_A'): raise NameError('face jac not defined')
        A = self.face_jac_A
        F = self.face_jac_F
        #
        Y = sp.sparse.linalg.spsolve((A.T@A), (A.T @ F))
        Y -= np.mean(Y,0)
        self.v_def_ = Y

    def avg_on_face(self, Fv):
        f = self.f
        Fv1 = Fv[f[:,0]]; Fv2 = Fv[f[:,1]]; Fv3 = Fv[f[:,2]]
        return (Fv1 + Fv2 + Fv3) / 3

    def avg_on_edge(self, Fv):
        e = self.e
        Fv1 = Fv[e[:,0]]; Fv2 = Fv[e[:,1]];
        return (Fv1 + Fv2) / 2

    def smooth_jac(self, jac, k=50):
        nv = self.v.shape[0]
        if not hasattr(self, 'eig_trans'): self.eig_trans = (self.m @ self.eig).T
        return (self.eig[:,:k] @ self.eig_trans[:k] @ jac.reshape(nv,9)).reshape(nv,3,3)
    
##### D_fi operators (transfer from TVF to Fun space)
    def grad_vert_op(self):
        I = []
        J = []
        V = []
        #
        idv = np.arange(self.v.shape[0])
        jdv = self.e[self.he_start][:,1]
        kdv = self.e[self.nex[self.op[self.he_start]]][:,1]
        #
        eij = self.v[jdv] - self.v[idv]
        eik = self.v[kdv] - self.v[idv]
        lij = np.linalg.norm(eij, axis=1)
        lik = np.linalg.norm(eik, axis=1)
        #phi = X.he_angles[X.nex[X.op[X.he_start]]]
        phi = self.he_angles_norm[self.nex[self.op[self.he_start]]]
        L = np.linalg.norm(eik - (np.cos(phi)*lik/lij)[:,None]*eij, axis=1)
        #
        I = np.concatenate([2*idv, 2*idv, 2*idv+1, 2*idv+1, 2*idv+1])
        J = np.concatenate([idv, jdv, idv, jdv, kdv])
        V = np.concatenate([-1/lij, 1/lij, -1/L+lik/(L*lij)*np.cos(phi),
                            -lik/(L*lij)*np.cos(phi), 1/L])
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(2 * self.v.shape[0],self.v.shape[0]))
        self.gradv = A
        return A
    
    def grad_vert(self,f):
        if not hasattr(self, 'gradv'): self.grad_vert_op()
        ##
        idv = np.arange(self.v.shape[0])
        gf = self.gradv @ f
        #print(gf)
        gf1 = gf[2*idv]
        gf2 = gf[2*idv+1]
        return gf1 + 1j * gf2

    def grad_fun_scal(self,f):
        I = []
        J = []
        V = []
        #
        if not hasattr(self, 'gradv'): self.grad_vert_op()
        gf = self.gradv @ f
        idv = np.arange(self.v.shape[0])
        gf1 = gf[2*idv]
        gf2 = gf[2*idv+1]
        #
        I = np.concatenate([idv, idv])
        J = np.concatenate([2*idv, 2*idv+1])
        V = np.concatenate([gf1, gf2])
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(self.v.shape[0], 2 * self.v.shape[0]))
        return A

    def grad_fun_scal_op(self,f,k1,k2):
        #
        if not hasattr(self, 'eig_trans'):
            self.eig_trans = (self.m @ self.eig).T
        #conversion from complex to 2nv * 2k2 matrix
        if not hasattr(self, 'ceig_real'):
            a = self.ceig.real; b = self.ceig.imag;
            c1 = np.stack([a,b], 1)
            d1 = c1.reshape(2*self.v.shape[0], self.ceig.shape[1])
            c2 = np.stack([-b,a], 1)
            d2 = c2.reshape(2*self.v.shape[0], self.ceig.shape[1])
            d = np.stack([d1,d2], -1).reshape(2*self.v.shape[0], 2*self.ceig.shape[1])
            #
            self.ceig_real = d
        #
        Df_spec = self.eig_trans[:k1] @ self.grad_fun_scal(f) @ self.ceig_real[:, :2*k2]
        return Df_spec
    
    def spec_grad(self,k):
        if not hasattr(self, 'ceig_trans_real'):
            self.ceig_trans = np.conjugate(self.m @ self.ceig).T
            a = self.ceig.real; b = self.ceig.imag;
            c1 = np.stack([a,b], 0)
            d1 = c1.reshape(self.ceig.shape[1], 2*self.v.shape[0])
            c2 = np.stack([-b,a], 0)
            d2 = c2.reshape(self.ceig.shape[1], 2*self.v.shape[0])
            d = np.stack([d1,d2], 1).reshape(2*self.ceig.shape[1], 2*self.v.shape[0])
            #
            self.ceig_trans_real = d
        #
        sg = self.ceig_trans_real[:k] @ self.gradv
        self.spec_gradv = sg
        return sg
    
    # same operators but with the VFs
    # we use this energy to fit the equation
    # <X, grad f> o T = <dT . X, grad (f o T)>
    # here that translates as C D_X = D_QX C, for all X

    def VF_fun_scal(self, X):
        I = []
        J = []
        V = []
        #
        if not hasattr(self, 'gradv'): self.grad_vert_op()
        #
        idv = np.arange(self.v.shape[0])
        #
        I = np.concatenate([idv, idv])
        J = np.concatenate([2*idv, 2*idv+1])
        V = np.concatenate([X.real, X.imag])
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(self.v.shape[0], 2 * self.v.shape[0]))
        return A @ self.gradv

    def VF_fun_scal_op(self, X, k1):
        #
        if not hasattr(self, 'eig_trans'): self.eig_trans = (self.m @ self.eig).T
        #
        Df_spec = self.eig_trans[:k1] @ self.VF_fun_scal(X) @ self.eig[:, :k1]
        return Df_spec
    
    ## grouping operators for basis function
    def fun_scal_op_basis(self, k1=10, k2=10):
        Df = []
        for i in range(k1):
            Df += [self.grad_fun_scal_op(self.eig[:,i],k1,k2)]
        Df = np.stack(Df, axis=0)
        self.Df = Df
        return Df
    
    ## hks
    def hks_desc(self, Nhks = 10, K = 300, t = None):#Mhks = 200
        D = self.eig[:,:K]**2
        abs_val = np.abs(self.vals)
        ## ts pre-shot
        if t is None:
            log_ts = np.linspace(np.log(0.005), np.log(0.2), Nhks)
            t = np.exp(log_ts);
        T = np.exp(-abs_val[:K, None] * t[None,:])
        hks = D @ T
        self.hks = hks
        return hks
    
    ###### new vertex operators
    def grad_vert_op2(self):
        I = []
        J = []
        V = []

        #
        Vjs = []
        rotate = np.zeros(self.v.shape[0], dtype=int)  #will keep in degree of vertex
        he = np.copy(self.he_start)
        i = 0
        while np.any(rotate==0):
            #-- get [vj] and store it in X; same for fj - fi in f
            lij = np.linalg.norm(self.v[self.e[he][:,1]] - self.v[self.e[he][:,0]], axis=1)
            aij = self.he_angles_norm[he]
            vj = lij[:, None] * np.cos(np.stack([aij, np.pi/2 - aij], axis=-1))
            vj[rotate>0]=0 # do not add values if cycle done
            Vjs+=[vj]

            #-- circulate CW
            he = self.op[he]
            he = self.nex[he]
            i+=1
            #-- update rot mask once cycle is done
            rot_mask = (self.he_start == he) * (rotate == 0)
            rotate[rot_mask] = i

        #-- build and invert local systems
        Vjs = np.stack(Vjs, axis=1)
        Vjs_inv = np.linalg.pinv(Vjs)

        #-- new rotation around the vertex to add in the coefficients to the sparse matrix
        rotate = np.zeros(self.v.shape[0], dtype=int)  #will keep in degree of vertex
        he = np.copy(self.he_start)
        i = 0
        while np.any(rotate==0):
            #-- fill in the values
            jdv = self.e[he][:,1]; idv = self.e[he][:,0];
            I += [2*idv, 2*idv, 2*idv+1, 2*idv+1]
            J += [idv, jdv, idv, jdv]
            V += [-Vjs_inv[:,0,i], Vjs_inv[:,0,i], -Vjs_inv[:,1,i], Vjs_inv[:,1,i]]

            #-- circulate CW
            he = self.op[he]
            he = self.nex[he]
            i+=1
            #-- update rot mask
            rot_mask = (self.he_start == he) * (rotate == 0)
            rotate[rot_mask] = i


        I = np.concatenate(I)
        J = np.concatenate(J)
        V = np.concatenate(V)
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(2 * self.v.shape[0],self.v.shape[0]))
        self.degv = rotate
        #self.gradv = A
        return A


    def div_c_vert_op(self):
        I = []
        J = []
        V = []

        #
        rotate = np.zeros(self.v.shape[0], dtype=int)  #will keep in degree of vertex
        he = np.copy(self.he_start)
        i = 0
        while np.any(rotate==0):
            #-- get [vj] and store it in X; same for fj - fi in f
            lij = np.linalg.norm(self.v[self.e[he][:,1]] - self.v[self.e[he][:,0]], axis=1)
            aij = self.he_angles_norm[he]
            d = self.degv
            alpha = 1/lij * 1/d
            vj = alpha * (np.cos(aij) - 1j * np.sin(aij))  # conjugate vj ?
            vj[rotate>0]=0 # do not add values if cycle done

            #-- fill in the values
            jdv = self.e[he][:,1]; idv = self.e[he][:,0];
            #
            #cc = np.cos(self.rho[he]); ss = np.sin(self.rho[he]);
            rr = (np.cos(self.rho) - np.sin(self.rho) * 1j)[he]

            J += [jdv, idv]
            I += [idv, idv]
            V += [rr * vj, -vj]

            #-- circulate CW
            he = self.op[he]
            he = self.nex[he]
            i+=1
            #-- update rot mask once cycle is done
            rot_mask = (self.he_start == he) * (rotate == 0)
            rotate[rot_mask] = i
            #print(i)


        I = np.concatenate(I)
        J = np.concatenate(J)
        V = np.concatenate(V)
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(self.v.shape[0], self.v.shape[0]))
        #self.gradv = A
        return A
    
    def div_c_vert_op2(self):
        I = []
        J = []
        V = []

        #
        rotate = np.zeros(self.v.shape[0], dtype=int)  #will keep in degree of vertex
        he = np.copy(self.he_start)
        i = 0
        while np.any(rotate==0):
            #-- get [vj] and store it in X; same for fj - fi in f
            #he2 = self.nex[he]
            lij = np.linalg.norm(self.v[self.e[he][:,1]] - self.v[self.e[he][:,0]], axis=1)
            aij = self.he_angles_norm[he]
            he2 = self.nex[self.op[he]]
            lij2 = np.linalg.norm(self.v[self.e[he2][:,1]] - self.v[self.e[he2][:,0]], axis=1)
            aij2 = self.he_angles_norm[he2]

            vj = lij * (np.cos(aij) - 1j * np.sin(aij))
            vj2 = lij2 * (np.cos(aij2) - 1j * np.sin(aij2))
            vj = -1j * (vj2 - vj)
            vj[rotate>0]=0 # do not add values if cycle done

            #-- fill in the values
            jdv = self.e[he][:,1]; idv = self.e[he][:,0];
            #
            #cc = np.cos(self.rho[he]); ss = np.sin(self.rho[he]);
            rr = (np.cos(self.rho) - np.sin(self.rho) * 1j)[he]

            J += [jdv]
            I += [idv]
            V += [rr * vj]

            #-- circulate CW
            he = self.op[he]
            he = self.nex[he]
            i+=1
            #-- update rot mask once cycle is done
            rot_mask = (self.he_start == he) * (rotate == 0)
            rotate[rot_mask] = i
            #print(i)


        I = np.concatenate(I)
        J = np.concatenate(J)
        V = np.concatenate(V)
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(self.v.shape[0], self.v.shape[0]))
        inv_m = sp.sparse.diags(1/self.m.diagonal())
        return inv_m @ A

#
##
######
#################
#######################################        