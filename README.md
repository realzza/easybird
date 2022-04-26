# easyBAD
**easyBAD** is clean implementation of Bird Activity Detection (BAD) in pyTorch.

## Single Wav
Identify bird activities for single waveform.
```bash
python infer-bad-single.py --data example_bird.wav --model-bad easyBAD.pt --device cpu
```

## Wav Directory
Identify bird activities from wavform directory
```bash
python infer-bad-dir.py --data dir/to/wavs --model-bad easyBAD.pt --device cpu -o out.txt
```

