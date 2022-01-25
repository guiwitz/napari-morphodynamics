# napari-morphodynamics

[![License](https://img.shields.io/pypi/l/napari-morphodynamics.svg?color=green)](https://github.com/guiwitz/napari-morphodynamics/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-morphodynamics.svg?color=green)](https://pypi.org/project/napari-morphodynamics)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-morphodynamics.svg?color=green)](https://python.org)
[![tests](https://github.com/guiwitz/napari-morphodynamics/workflows/tests/badge.svg)](https://github.com/guiwitz/napari-morphodynamics/actions)
[![codecov](https://codecov.io/gh/guiwitz/napari-morphodynamics/branch/main/graph/badge.svg)](https://codecov.io/gh/guiwitz/napari-morphodynamics)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-morphodynamics)](https://napari-hub.org/plugins/napari-morphodynamics)

**!!!!!!!Plugin under development!!!!!!**

This plugin offers an interface for the [Morphodynamics](https://github.com/guiwitz/MorphoDynamics) package. It allows to select and visualize time-lapses, choose segmentation algorithm and parameters, train a pixel-classifier based segmentation (via [napari-convpaint](https://github.com/guiwitz/napari-convpaint)) and finally run the morphodynamics analysis (breaking up the segmentation into layered windows, fitting a spline on the contour).
## Installation

You can install the plugin via pip with:

```
pip install pip install git+https://github.com/guiwitz/napari-morphodynamics
```

**Not yet on pypi**
You can install `napari-morphodynamics` via [pip]:

    pip install napari-morphodynamics



To install latest development version :

    pip install git+https://github.com/guiwitz/napari-morphodynamics.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-morphodynamics" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/guiwitz/napari-morphodynamics/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
