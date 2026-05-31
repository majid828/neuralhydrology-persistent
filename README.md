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

