## Extract and visualize MFCCs in 2 dimensions ##

import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import shutil
import subprocess
import os
import matplotlib.pyplot as plt
import plotly.plotly as py
import random


os.chdir(os.path.expanduser('~/Dropbox/smacpy'))
from smacpy import Smacpy

child="/Volumes/U/AAPB_Corpus_May_2017/test_set_616_clips/Child, Julia|cpb-aacip-15-wp9t14v189__Barcode260268_.h264|124|1641s.16000.wav"

nixon="/Volumes/U/AAPB_Corpus_May_2017/test_set_616_clips/Nixon, Richard|cpb-aacip-15-hx15m62g43__ABC11913_.h264|1|135s.16000.wav"

steinem="/Volumes/U/AAPB_Corpus_May_2017/test_set_616_clips/Steinem, Gloria|cpb-aacip-43-0c4sj19t4b__8379971_|2042|593s.16000.wav"

bill_clinton="/Volumes/U/AAPB_Corpus_May_2017/test_set_616_clips/Clinton, Bill|cpb-aacip-503-3b5w669n0k__NHPR95188|344|1181s.16000.wav"

johnson="/Volumes/U/AAPB_Corpus_May_2017/test_set_616_clips/Johnson, Lyndon|cpb-aacip-15-78gf2967|5|902s.16000.wav"

baldwin="/Volumes/U/AAPB_Corpus_May_2017/test_set_616_clips/Baldwin, James|cpb-aacip-15-687h4jgd|25|2411s.16000.wav"


speakers=[child,nixon,steinem,bill_clinton,johnson,baldwin]



def plot_mfccs(speaker_a_path,speaker_b_path):
    model = Smacpy("", {speaker_a_path:'a',speaker_b_path:'b'})
    speaker_a_feats=model.file_to_features(speaker_a_path)
    speaker_b_feats=model.file_to_features(speaker_b_path)
    sc = StandardScaler()
    speaker_b_feats_std = sc.fit_transform(speaker_b_feats)[:2000]
    pca = decomposition.PCA(n_components=2)
    speaker_b_feats_std_pca = pca.fit_transform(speaker_b_feats_std)
    x_a=speaker_b_feats_std_pca[:,0]
    y_a=speaker_b_feats_std_pca[:,1]
    speaker_a_feats_std = sc.fit_transform(speaker_a_feats)[:2000]
    pca = decomposition.PCA(n_components=2)
    speaker_a_feats_std_pca = pca.fit_transform(speaker_a_feats_std)
    x_b=speaker_a_feats_std_pca[:,0]
    y_b=speaker_a_feats_std_pca[:,1]
    fig, ax = plt.subplots()
    a_legend = ' '.join(speaker_a_path.split('/')[-1].split('|')[0].split(', ')[::-1])
    b_legend = ' '.join(speaker_b_path.split('/')[-1].split('|')[0].split(', ')[::-1])
    ax.scatter(x_a, y_a, alpha=0.47, color="#6495ed", label=a_legend)  #blue
    ax.scatter(x_b, y_b, alpha=0.27, color="#991525", label=b_legend)  #red
    plt.legend(loc=4)
    plt.show()


speaker_a_path, speaker_b_path = random.sample(speakers,2)

plot_mfccs(speaker_a_path,speaker_b_path)




