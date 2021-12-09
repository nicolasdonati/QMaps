# Complex Functional Maps: a Conformal Link between Tangent Bundles

This repository contains Python implementation for complex functional maps and additionally provides code to reproduce the main figures of the paper

> **Complex Functional Maps**<br/>
> Nicolas Donati, Etienne Corman, Simone Melzi, Maks Ovsjanikov<br/>
> In *CGF 2022*<br/>
<!--
> [PDF](),
> [Video](https://www.youtube.com/watch?v=U6wtw6W4x3I),
> [Project page](http://igl.ethz.ch/projects/instant-meshes/)
-->


<p align="center">
<img src="images/TEASER.png" width="600">
</p>

## Use Complex Functional Maps
All the necessary code for complex functional maps is in the ``Tools`` Folder. Apart from the standard libraries (numpy, scipy), you will need python bindings for ``libigl``, which can be installed for instance with:

    conda install -c conda-forge igl

Also, to run some of the scripts we provide to reproduce some of the figures and tables of the paper, you will need meshplot, which can also be fetched with conda:

    conda install -c conda-forge meshplot

We provide jupyter notebooks and their corresponding python scripts to show how to use complex functional maps in the scenarii we propose in the paper. 

Namely, we show:
* how to perform vector field transfer using complex functional maps, and a visualization of the transfer
* how to use bijective Zoomout with discrete Optimisation (from Ren et al., SGP 2021) with our complex functional maps modification

## Citation
If you use our work, please cite our paper.
```
@article{donati2022CFMaps,
  title={Complex Functional Maps: a Conformal Link between Tangent Bundles},
  author={Donati, Nicolas and Corman, Etienne and Melzi, Simone and Ovsjanikov, Maks},
  journal={CGF},
  year={2022}
}
```

## Contact
If you have any problem about this implementation, please feel free to contact via:

nicolas DOT donati AT polytechnique DOT edu