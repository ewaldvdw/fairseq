import sys
import os
import glob
from pprint import pprint

import fairseq
import torch

import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


#audio_fn = "/media/data1/ewaldvdw/projects/unglobalpulse/bam/transcriptions/ungp_build_speech_corpus_bam/ungp_bambara_asrfree_corpus_v0.3/speech_data/set1/audio/ungp_bam_101_6_2019-06-14T19_05_05_00000043-00000340.wav"
audio_flist = glob.glob("/media/data1/ewaldvdw/projects/unglobalpulse/bam/audio/keywords_isolated_recordings/ungp_build_kws_corpus_bam/v0.5/6_keywords/jatigɛwale-gurupu-sahara-kɔnɔ/*.wav")

#audio_fn = "/media/data1/ewaldvdw/projects/unglobalpulse/bam/audio/keywords_isolated_recordings/ungp_build_kws_corpus_bam/v0.5/6_keywords/jatigɛwale-gurupu-sahara-kɔnɔ/SPK000001_jatigɛwale-gurupu-sahara-kɔnɔ_37.wav"


if False:
    # FOR XLSR
    ############################################################################
    # From https://github.com/pytorch/fairseq/issues/3134#issuecomment-761110102
    wav2vec2_checkpoint_path = "/media/data1/ewaldvdw/projects/fairseq/examples/wav2vec/xlsr_53_56k.pt"
    checkpoint = torch.load(wav2vec2_checkpoint_path)
    #input("Enter to continue.")
    wav2vec2_encoder = fairseq.models.wav2vec.Wav2Vec2Model.build_model(checkpoint['cfg']['model'])
    wav2vec2_encoder.load_state_dict(checkpoint['model'])
    # From https://github.com/pytorch/fairseq/issues/3134#issuecomment-761110102
    ############################################################################
else:
    wav2vec2_checkpoint_path = "/media/data1/ewaldvdw/projects/fairseq/examples/wav2vec/wav2vec_small.pt"

    #model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec2_checkpoint_path])
    #model = model[0]
    #model.eval()


    checkpoint = torch.load(wav2vec2_checkpoint_path)

    cfg = fairseq.dataclass.utils.convert_namespace_to_omegaconf(checkpoint['args'])
    wav2vec2_encoder = fairseq.models.wav2vec.Wav2Vec2Model.build_model(cfg.model)
    wav2vec2_encoder.load_state_dict(checkpoint['model'])




audio_list = []
for audio_fn in audio_flist[0:5]:
    audioinfo = torchaudio.info(audio_fn)
    print(audioinfo)
    audiodata, samplerate = torchaudio.load(audio_fn)
    print(audiodata.size())
    #audiodata = torch.mean(torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000).forward(audiodata), dim=0).unsqueeze(0)
    #print(audiodata.size())
    audio_list.append(audiodata)


############################################################################
#import fairseq

##cp = '/content/pretrain_model/wav2vec_vox_new.pt'
#cp = "/media/data1/ewaldvdw/projects/fairseq/examples/wav2vec/xlsr_53_56k.pt"
#print("Loading model:", cp)
#model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
#print("Done loading model:", cp)
#model = model[0]
#print("Done model = model[0]")
############################################################################


fig, ax = plt.subplots(nrows=5, ncols=1)

#audio = torch.randn(1,10000)
for audcnt, audio in enumerate(audio_list):

    features = wav2vec2_encoder(audio, features_only=True, mask=False)['x']

    print(features.size())

    ax = plt.subplot(5, 1, audcnt+1)
    ax.imshow(features.detach().numpy()[0, :, :])

plt.tight_layout()
#plt.savefig(os.path.join("./examples/wav2vec", os.path.basename(audio_fn)+".pdf"))
#print("Saving plot to", os.path.join("./examples/wav2vec", os.path.basename(audio_fn)+".pdf"))
print("Saving plot to ./examples/wav2vec/features.pdf")
plt.savefig(os.path.join("./examples/wav2vec", "features.pdf"))

