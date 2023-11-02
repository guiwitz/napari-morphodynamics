# napari-morphodynamics

napari-morphodynamics is a plugin that allows to study the correlated dynamics of cell shape and molecular events in 2D+time fluorescence microscopy. It offers and interface to the [MorphoDynamics](https://github.com/guiwitz/MorphoDynamics) package in napari.

Head to the [Getting Started section](https://guiwitz.github.io/napari-morphodynamics/book/Description.html) to learn how to install the plugin and use its basics functionalities.

Head to the [Examples section](https://guiwitz.github.io/napari-morphodynamics/book/Synthetic.html) to discover how to use the plugin on a few examples.

## Alternative: napari-roidynamics

As some of the processes in this plugin are computationally heavy, we have developed a spin-off plugin called napari-roidynamics, which allows to generate fixed geometries instead of precise segmentation and analyses intensity fluctuations in these regions of interest. Please refer to [napari-roidynamics](https://stojiljkovicvetana.github.io/napari-roidynamics/README.html) if you wish to apply intensity dynamics measurements in fixed geometries.

```{image} images/napari-roidynamics_logo.png
:alt: napari-roidynamics logo
:class: bg-primary
:width: 300px
:align: center
```

## Authors

The [Morphodynamics](https://github.com/guiwitz/MorphoDynamics) package on which this plugin relies was inspired by the [Windowing-Protrusion](https://github.com/DanuserLab/Windowing-Protrusion) Matlab package developed by the [Danuser Lab](https://www.danuserlab-utsw.org/). It was developed by Guillaume Witz and Cédric Vonesch at the Data Science Lab (DSL), University of Bern and Jakobus van Unen, Lucien Hinderling and Olivier Pertz from the [Pertz Lab](https://www.pertzlab.net/), Institute of Cell Biology, University of Bern. This plugin was developed thanks to a [Chan Zuckerberg Initiative grant](https://chanzuckerberg.com/science/programs-resources/imaging/napari/napari-morphodynamics-a-plugin-to-quantify-cellular-dynamics/), by Guillaume Witz and Ana Stojiljkovic at DSL, again in collaboration with the Pertz lab.