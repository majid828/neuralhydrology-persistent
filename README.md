![#](docs/source/_static/img/neural-hyd-logo-black.png)

# Persistent LSTM Extension for NeuralHydrology

This repository contains the Persistent LSTM implementation developed for the manuscript:

**Sequence-Free LSTM Training for Long-Term Hydrological States**

This code is derived from the open-source NeuralHydrology framework and extends its training workflow by adding persistent hidden-state handling across sequential hydrological batches. The main goal is to allow the LSTM to carry hydrological memory across non-overlapping sequences, instead of resetting the hidden and cell states for every training sequence.

## Main modifications

The main changes relative to the original NeuralHydrology code include:

1. Addition of a Persistent LSTM model:
   - `neuralhydrology/modelzoo/persistentlstm.py`

2. Modifications to the training workflow:
   - persistent hidden and cell state handling
   - state detaching between batches
   - reset of states at basin boundaries
   - optional reset of states after specified temporal intervals

3. Modifications to dataset handling:
   - non-overlapping sequence support
   - basin index tracking for persistent training

4. Example configuration file:
   - `test/test_configs/persistent_hourly_camels.test.yml`

## Relationship to NeuralHydrology

This repository is a derivative of the original NeuralHydrology package. The original NeuralHydrology software should be cited as:

```bibtex
@article{kratzert2022joss,
  title = {NeuralHydrology -- A Python library for Deep Learning research in hydrology},
  author = {Frederik Kratzert and Martin Gauch and Grey Nearing and Daniel Klotz},
  journal = {Journal of Open Source Software},
  publisher = {The Open Journal},
  year = {2022},
  volume = {7},
  number = {71},
  pages = {4050},
  doi = {10.21105/joss.04050},
  url = {https://doi.org/10.21105/joss.04050}
}



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
