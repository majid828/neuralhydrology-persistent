
# Persistent LSTM Extension for NeuralHydrology

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20479712.svg)](https://doi.org/10.5281/zenodo.20479712)


This code is derived from the open-source NeuralHydrology framework and extends its training workflow by adding persistent hidden-state handling across sequential hydrological batches. The main goal is to allow the LSTM to carry hydrological memory across non-overlapping sequences, instead of resetting the hidden and cell states for every training sequence.

---

## Main modifications

The main changes relative to the original NeuralHydrology code include:

1. Addition of a Persistent LSTM model:
   - `neuralhydrology/modelzoo/persistentlstm.py`

2. Modifications to the training workflow:
   - persistent hidden and cell state handling
   - state detaching between batches/training segments
   - reset of states at basin boundaries
   - optional reset of states after specified temporal intervals
   - sequential/non-shuffled training support for Persistent LSTM experiments

3. Modifications to dataset handling:
   - non-overlapping sequence support
   - basin index tracking for persistent-state training

4. Manuscript configuration files:
   - `test/test_configs/persistent_lstm_hourly.test.yml`
   - `test/test_configs/persistent_lstm_15_mint.test.yml`
   - `test/test_configs/persistent_lstm_for_long_memory_basins.test.yml`
   - `test/test_configs/persistent_lstm_nonstationary_basins.test.yml`
   - `test/test_configs/cuda_lstm_for_long_memory_basins.test.yml`
   - `test/test_configs/cuda_lstm_nonstationary_basins.test.yml`
   - `test/test_configs/mts_lstm_hourly.test.yml`

---

## Relationship to NeuralHydrology

This repository is a derivative of the original NeuralHydrology package. The original NeuralHydrology code was used as the base hydrological deep-learning framework, and this repository extends it by adding a Persistent LSTM model and modifying the training and dataset workflow to support persistent hidden-state handling across sequential hydrological batches.

The original NeuralHydrology software should be cited as:

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
```

---

## Citation for this repository

The release version of this repository is archived on Zenodo with DOI:

https://doi.org/10.5281/zenodo.20479712

Please cite this repository as:

```bibtex
@software{shah_persistent_lstm_2026,
  author = {Shah, Majid Hussain},
  title = {Persistent LSTM Extension for NeuralHydrology},
  year = {2026},
  publisher = {Zenodo},
  version = {v1.0.0},
  doi = {10.5281/zenodo.20479712},
  url = {https://doi.org/10.5281/zenodo.20479712}
}
```

---

## Manuscript configuration files

The configuration files used to generate the manuscript results are provided in:

`test/test_configs/`

The following configuration files were used for the experiments reported in the manuscript:

| Configuration file | Purpose |
|---|---|
| `persistent_lstm_hourly.test.yml` | Persistent LSTM experiment using hourly hydrological data |
| `persistent_lstm_15_mint.test.yml` | Persistent LSTM experiment using 15-minute hydrological data |
| `persistent_lstm_for_long_memory_basins.test.yml` | Persistent LSTM experiment for long-memory basin analysis |
| `persistent_lstm_nonstationary_basins.test.yml` | Persistent LSTM experiment for nonstationary basin analysis |
| `cuda_lstm_for_long_memory_basins.test.yml` | Standard CUDA LSTM comparison for long-memory basins |
| `cuda_lstm_nonstationary_basins.test.yml` | Standard CUDA LSTM comparison for nonstationary basins |
| `mts_lstm_hourly.test.yml` | Multi-timescale LSTM experiment using hourly hydrological data |

These files define the model settings used in the manuscript, including the model type, basin selection, input features, training period, validation period, test period, sequence length, batch size, and other training options.

---
