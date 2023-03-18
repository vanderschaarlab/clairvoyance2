# clairvoyance2

> clairvoyance2: a Unified Toolkit for Medical Time Series

**⚠️ The library is in pre-alpha / dev. API will change without warning.**

**clairvoyance2** is a library that unifies time series tasks for the medicine and healthcare use case.  It provides tools for manipulating multi-dimensional time series, as well as static data, and implements models for: time series prediction, individualized treatment effects estimation, time-to-event analysis (*upcoming*), and model interpretability (*upcoming*).  **clairvoyance2** is primarily focussed on machine learning models.




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

* [📔 Basic Usage](tutorials/basic_usage.ipynb)
* [📔 Example: CRN](tutorials/crn.ipynb)
* [📔 Example: SyncTwin](tutorials/synctwin.ipynb)


## Contact

If you wish to reach about to us specifically about `clairvoyance2` (bugs, suggestions, problems, ...) please message
Evgeny on [LinkedIn](https://www.linkedin.com/in/e-s-saveliev/) for now, until we set up an official communication channel.
