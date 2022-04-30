import os
import soxr
import torch
import soundfile as sf

from tqdm import tqdm
from scipy.special import softmax
from python_speech_features import logfbank


mdl_bad_kwargs = {
    "channels": 16, 
    "block": "BasicBlock", 
    "num_blocks": [2,2,2,2], 
    "embd_dim": 1024, 
    "drop": 0.3, 
    "n_class": 2
}

logfbank_kwargs = {
    "winlen": 0.025, 
    "winstep": 0.01, 
    "nfilt": 80, 
    "nfft": 2048, 
    "lowfreq": 50, 
    "highfreq": None, 
    "preemph": 0.97    
}

int2label = {0:"0",1:"1"}
model_dir = os.path.join(os.path.dirname(__file__), 'models/jit_bad.pt')


def extract_feat(wav_path, samplerate=16000, cmn=True):
    kwargs = {
        "winlen": 0.025,
        "winstep": 0.01,
        "nfilt": 80,
        "nfft": 2048,
        "lowfreq": 50,
        "highfreq": 8000,
        "preemph": 0.97
    }
    y, sr = sf.read(wav_path)
    if sr!=samplerate:
        y = soxr.resample(y, sr, samplerate)
        sr = samplerate
    logfbankFeat = logfbank(y, sr, **kwargs)
    if cmn:
        logfbankFeat -= logfbankFeat.mean(axis=0, keepdims=True)
    return logfbankFeat.astype('float32')
    

class SVExtractor():
    def __init__(self, mdl_kwargs, model_path, device):
        self.model = self.load_model(mdl_kwargs, model_path, device)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)

    def load_model(self, mdl_kwargs, model_path, device):
        model = torch.jit.load(model_path, map_location=device)
        return model

    def __call__(self, frame_feats):
        feat = torch.from_numpy(frame_feats).unsqueeze(0)
        feat = feat.float().to(self.device)
        with torch.no_grad():
            embd = self.model(feat)
        embd = embd.squeeze(0).cpu().numpy()
        return embd
    

def from_wav(wav, noise_thres=0.5, device="cpu"):
    """
    params:
        wav:             string; wave file for bird activity detection
        noise_thres:     float; noise threshold [default=0.5]
        device:          string; device for calculation [default="cpu"] [options: ['cpu','cuda:0']]
        
    return:
        hasBird:         0 or 1 representing the presence of bird
        logits:      confidence for the result
    """
    wav_feats = extract_feat(wav)
    detector = SVExtractor(mdl_bad_kwargs, model_dir, device=device)
    logits = softmax(detector(wav_feats)).tolist()
    hasBird = (logits[1] >= noise_thres)
    return hasBird, logits[1]
    
def from_wavs(wavs, noise_thres=0.5, device="cpu"):
    """
    params:
        wav:             wave file for bird activity detection
        noise_thres:     noise threshold [default=0.5]
        device:          device for calculation [default="cpu"] [options: ['cpu','cuda:0']]
    
    return:
        results:         List(Tuples(utt_name, hasBird, logits))
    """
    detector = SVExtractor(mdl_bad_kwargs, model_dir, device=device)
    results = []
    for wav_ in tqdm(wavs, desc="Detecting noises"):
        utt_ = wav_.split('/')[-1].split('.')[0]
        wav_feats = extract_feat(wav_)
        wav_logits = softmax(detector(wav_feats)).tolist()
        wav_hasBird = (wav_logits[1] >= noise_thres)
        results.append((utt_, wav_hasBird, wav_logits[1]))
    return results