# easybird
![PyPI](https://img.shields.io/pypi/v/easybird?color=df&style=flat)&nbsp;
![GitHub](https://img.shields.io/github/license/realzza/easybird?color=%23FFB6C1&style=flat)&nbsp;
![GitHub last commit](https://img.shields.io/github/last-commit/realzza/easybird?color=orange&style=flat)&nbsp;
![GitHub top language](https://img.shields.io/github/languages/top/realzza/easybird?color=%236495ed&style=flat)&nbsp;
[![CodeFactor](https://www.codefactor.io/repository/github/realzza/easybird/badge)](https://www.codefactor.io/repository/github/realzza/easybird)&nbsp;

---

**Easybird** is python toolkit for Bird Activity Detection (BAD).

## Setup and Install
We recommend using **conda** for environment management, since we use conda-forge to install an essential library `libsndfile`. To setup, copy the following command to your terminal.
```bash
conda create -n bird
conda activate bird
conda install -c conda-forge libsndfile -y
```

After setup the virtual env, install with pip.
```bash
pip install easybird
```

## Single Wav
Identify bird activities for single waveform.
```python
from easybird import detection

hasBird, confidence = detection.from_wav('bird.wav', noise_thres=0.5, device='cpu') # device=option('cuda','cpu')
```
Output
```python
print(hasBird)
>>> True
print(confidence)
>>> 0.9996312260627747
```

## Multiple Wavs
Identify bird activities for multiple wavforms.
```python
from easybird import detection

results = detection.from_wavs(['bird1.wav','bird2.wav','bird3.wav'], noise_thres=0.5, device='cpu')
```
Output
```python
print(results)
>>> [(bird1, True, 0.99963122), (bird2, False, 0.37834975), (bird3, True, 0.87340939)]
```
