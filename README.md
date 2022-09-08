<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/clairvoyance2.svg?branch=main)](https://cirrus-ci.com/github/<USER>/clairvoyance2)
[![ReadTheDocs](https://readthedocs.org/projects/clairvoyance2/badge/?version=latest)](https://clairvoyance2.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/clairvoyance2/main.svg)](https://coveralls.io/r/<USER>/clairvoyance2)
[![PyPI-Server](https://img.shields.io/pypi/v/clairvoyance2.svg)](https://pypi.org/project/clairvoyance2/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/clairvoyance2.svg)](https://anaconda.org/conda-forge/clairvoyance2)
[![Monthly Downloads](https://pepy.tech/badge/clairvoyance2/month)](https://pepy.tech/project/clairvoyance2)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/clairvoyance2)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)



# clairvoyance2

> clairvoyance2: a Unified Toolkit for Medical Time Series

**⚠️ The library is in pre-alpha / dev. API will change without warning.**

**clairvoyance2** is a library that unifies time series tasks for the medicine and healthcare use case.  It provides tools for manipulating multi-dimensional time series, as well as static data, and implements models for: time series prediction, individualized treatment effects estimation (*upcoming*), time-to-event analysis (*upcoming*), and model interpretability (*upcoming*).  **clairvoyance2** is primarily focussed on machine learning (ML) models.




## Installation

`pip install git+https://github.com/vanderschaarlab/clairvoyance2.git`



## Models

| Model | Status |
|-|-|
| **Prediction (Forecasting)** |
| RNN | ✅ Implemented |
| Seq2Seq | ✅ Implemented |
| [NeuralLaplace](https://github.com/samholt/NeuralLaplace) | 🔲 Planned |
| **Imputation** |
| {f,b}fill & Mean | ✅ Implemented |
| [HyperImpute](https://proceedings.mlr.press/v162/jarrett22a/jarrett22a.pdf) | 🔲
| **Individualized Treatment Effects** |
| [CRN](https://openreview.net/forum?id=BJg866NFvB) | ✅ Implemented |
| [SyncTwin](https://proceedings.neurips.cc/paper/2021/hash/19485224d128528da1602ca47383f078-Abstract.html) | ⚙️ Experimental |
| [TE-CDE](https://proceedings.mlr.press/v162/seedat22b/seedat22b.pdf) | 🔲 Planned |
| **Time-to-event Analysis** |
| [Dynamic DeepHit](https://pubmed.ncbi.nlm.nih.gov/30951460/) | 🔲 Planned |
| **Interpretability** |
| [DynaMask](http://proceedings.mlr.press/v139/crabbe21a/crabbe21a.pdfsa) | 🔲 Planned |



## Tutorials

* [📔 Basic Usage Notebook](tutorials/basic_usage.ipynb)
* [📔 Example: CRN](tutorials/crn.ipynb)
* [📔 Example: SyncTwin](tutorials/synctwin.ipynb)
