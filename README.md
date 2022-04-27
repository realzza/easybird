# easybird
**easybird** is python toolkit for Bird Activity Detection (BAD).

## Install
```bash
pip install easybird
```

## Single Wav
Identify bird activities for single waveform.
```python
from easybird import detection

result = detection.from_wav('bird.wav')
```
Output
```python
print(result)
# (True, 0.9996312260627747)
```

## Multiple Wavs
Identify bird activities for multiple wavs
```python
from easybird import detection

results = detection.from_wavs(['bird1.wav','bird2.wav','bird3.wav'])
```
Output
```python
print(results)
# [(bird1, True, 0.99963122), (bird2, True, 0.37834975), (bird3, True, 0.87340939)]
```