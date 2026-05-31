![#](docs/source/_static/img/neural-hyd-logo-black.png)

cff-version: 1.2.0
message: "If you use this Persistent LSTM extension, please cite it using the metadata below and also cite the original NeuralHydrology software."
title: "Persistent LSTM Extension for NeuralHydrology"
authors:
  - family-names: "Shah"
    given-names: "Majid Hussain"
year: 2026
repository-code: "https://github.com/majid828/neuralhydrology-persistent"
license: "MIT"
abstract: "Persistent LSTM extension of the NeuralHydrology framework for sequence-free hydrological training with persistent hidden-state handling."
keywords:
  - hydrology
  - LSTM
  - persistent states
  - NeuralHydrology
  - rainfall-runoff modeling
references:
  - type: article
    authors:
      - family-names: "Kratzert"
        given-names: "Frederik"
      - family-names: "Gauch"
        given-names: "Martin"
      - family-names: "Nearing"
        given-names: "Grey"
      - family-names: "Klotz"
        given-names: "Daniel"
    title: "NeuralHydrology -- A Python library for Deep Learning research in hydrology"
    journal: "Journal of Open Source Software"
    year: 2022
    volume: 7
    issue: 71
    doi: "10.21105/joss.04050"



Python library to train neural networks with a strong focus on hydrological applications.

This package has been used extensively in research over the last years and was used in various academic publications. 
The core idea of this package is modularity in all places to allow easy integration of new datasets, new model 
architectures or any training-related aspects (e.g. loss functions, optimizer, regularization). 
One of the core concepts of this code base are configuration files, which let anyone train neural networks without
touching the code itself. The NeuralHydrology package is built on top of the deep learning framework 
[PyTorch](https://pytorch.org/), since it has proven to be the most flexible and useful for research purposes.

We (the AI for Earth Science group at the Institute for Machine Learning, Johannes Kepler University, Linz, Austria) are using
this code in our day-to-day research and will continue to integrate our new research findings into this public repository.

- Documentation: [neuralhydrology.readthedocs.io](https://neuralhydrology.readthedocs.io)
- Research Blog: [neuralhydrology.github.io](https://neuralhydrology.github.io)
- Bug reports/Feature requests [https://github.com/neuralhydrology/neuralhydrology/issues](https://github.com/neuralhydrology/neuralhydrology/issues)

# Cite NeuralHydrology

In case you use NeuralHydrology in your research or work, it would be highly appreciated if you include a reference to our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.04050#) in any kind of publication.

```bibtex
@article{kratzert2022joss,
  title = {NeuralHydrology --- A Python library for Deep Learning research in hydrology},
  author = {Frederik Kratzert and Martin Gauch and Grey Nearing and Daniel Klotz},
  journal = {Journal of Open Source Software},
  publisher = {The Open Journal},
  year = {2022},
  volume = {7},
  number = {71},
  pages = {4050},
  doi = {10.21105/joss.04050},
  url = {https://doi.org/10.21105/joss.04050},
}
```

# Contact

For questions or comments regarding the usage of this repository, please use the [discussion section](https://github.com/neuralhydrology/neuralhydrology/discussions) on Github. For bug reports and feature requests, please open an [issue](https://github.com/neuralhydrology/neuralhydrology/issues) on GitHub.
In special cases, you can also reach out to us by email: neuralhydrology(at)googlegroups.com
