import os
import sys
import sox
import json
import h5py
import time
import torch
import shutil
import GPUtil
import argparse
import numpy as np
import pandas as pd
import audiofile as af

from tqdm import tqdm
from scipy.special import softmax
from python_speech_features import logfbank

from module.model import Gvector


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


# parse args
def parse_args():
    desc="infer labels"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument('--model-bad', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda:0")
    return parser.parse_args()

def extract_feat(wav_path, cmn=True):
    kwargs = {
        "winlen": 0.025,
        "winstep": 0.01,
        "nfilt": 80,
        "nfft": 2048,
        "lowfreq": 50,
        "highfreq": 8000,
        "preemph": 0.97
    }
    y, sr = af.read(wav_path)
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
        model = Gvector(**mdl_kwargs)
        state_dict = torch.load(model_path, map_location=device)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        return model

    def __call__(self, frame_feats):
        feat = torch.from_numpy(frame_feats).unsqueeze(0)
        feat = feat.float().to(self.device)
        with torch.no_grad():
            embd = self.model(feat)
        embd = embd.squeeze(0).cpu().numpy()
        return embd
    
def most_common(lst):
    return max(set(lst), key=lst.count)
    
def infer_bad(wav, detector, int2label_dict):
    wav_feats = extract_feat(wav)
    logits = softmax(detector(wav_feats))
    hasBird = np.argmax(logits)
    return hasBird, logits[1]
    
if __name__ == "__main__":
    
    with open('kill.sh','w') as f:
        f.write('')
        
    args = parse_args()
    model_bad_path = args.model_bad
    label2int = {"0":0,"1":1}
    int2label = {v:k for k,v in label2int.items()}
        
    print('... loading activity detector ...')
    bad_extractor = SVExtractor(mdl_bad_kwargs, model_bad_path, device=args.device)
    print('... loaded ...')
        
    pred_dict = {}
    
    wav_ = args.data
    hasBird, confidence = infer_bad(wav_, bad_extractor, int2label)

    result = ["noise","bird"]
    print(result[hasBird],confidence)