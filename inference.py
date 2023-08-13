#matplotlib inline
import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import json
import math
import torch
import sys
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import logging


def get_text(text, hps):
    
    text_norm = text_to_sequence(text, hps.data.text_cleaners)

    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)

    text_norm = torch.LongTensor(text_norm)

    return text_norm


hps = utils.get_hparams_from_file(f"./models/{sys.argv[1]}/config.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()

_ = utils.load_checkpoint(f"./models/{sys.argv[1]}/G_{sys.argv[2]}.pth", net_g, None)
_ = net_g.eval()

output_dir = f"output/{sys.argv[1]}"
os.makedirs(output_dir, exist_ok=True)

speakers = len([dirs for dirs in os.listdir("models") if os.path.isdir(os.path.join("models", dirs))])

text = "안녕하세요"
speed = 1

for idx in range(speakers):
    sid = torch.LongTensor([idx]).cuda()
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda() 
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][0,0].data.cpu().float().numpy()
    write(f'{output_dir}/output{idx}_{sys.argv[2]}.wav', hps.data.sampling_rate, audio)
    print(f'{output_dir}/output{idx}_{sys.argv[2]}.wav 생성완료!')
