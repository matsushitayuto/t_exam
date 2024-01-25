#クラスごとの平均点に有意差の有無を算出する

import numpy as np
import pandas as pd
import math
import statistics
import argparse
from scipy import stats
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import random
random.seed(0)

#引数ディレクトリ解析
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="csv_path")
args = parser.parse_args()

csv_path = args.input
df = pd.read_csv(csv_path, header=0)

score1 = []
score2 = []
for i, group_df in df.groupby("class"):
    score1_list = group_df["score1"].tolist()
    score2_list = group_df["score2"].tolist()
    score1_1 = score1.copy()
    score2_1 = score2.copy()
    score1 = []
    score2 = []
    for j in range(len(score1_list)):
        score1.append(score1_list[j])
    for k in range(len(score2_list)):
        score2.append(score2_list[k])

score1_1 = np.array(score1_1).astype(float)
score1 = np.array(score1).astype(float)
score2_1 = np.array(score2_1).astype(float)
score2 = np.array(score2).astype(float)



#F検定
def FValue(score1, score2, score1_1, score2_1):
    f1_val, p1_val = f_oneway(score1, score1_1)
    f2_val, p2_val = f_oneway(score2, score2_1)
    print(f"score1のF値:{f1_val}")
    print(f"score1のP値:{p1_val}")
    print(f"score2のF値:{f2_val}")
    print(f"score2のP値:{p2_val}")
    print()
    return p1_val, p2_val

#Studentのt検定
def Student(score_1, score):
    t, tp = stats.ttest_ind(score_1, score, alternative="two-sided")

    if tp < 0.05:
        print("2クラス間の平均点に有意差がある。\n")
    else:
        print("2クラス間の平均点に有意差はない。\n")

#Welchのt検定
def Welch(score_1, score):
    t, tp = stats.ttest_ind(score_1, score, alternative="two-sided", equal_var=False)

    if tp < 0.05:
        print("2クラス間の平均点に有意差がある。\n")
    else:
        print("2クラス間の平均点に有意差はない。\n")

def barplot_annotate_brackets(num1, num2, data, center, 
                              height, yerr=None, dh=.05, 
                              barh=.05, fs=None, maxasterix=None):
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05
  
        while data < p:
            text += '*'
            p /= 10.
  
            if maxasterix and len(text) == maxasterix:
                break
  
        # if len(text) == 0:
        #     text = 'n. s.'
  
    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]
  
    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]
  
    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)
  
    y = max(ly, ry) + dh
  
    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)
  
    # plt.plot(barx, bary, c='black')
  
    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
  
    plt.text(*mid, text, **kwargs)
 


#有意水準0.05でのクラスの平均点の有意差の検定
def main():
    p1_val, p2_val = FValue(score1, score2, score1_1, score2_1)

    if p1_val <0.05:
        print("Score1において、\n")
        Student(score1_1, score1)
        print()
    else:
        print("Score1において、\n")
        Welch(score1_1, score1)
        print()

    if p2_val <0.05:
        print("Score2において、\n")
        Student(score2_1, score2)
        print()
    else:
        print("Score2において、\n")
        Welch(score2_1, score2)
        print()

    #標準偏差
    sd_score1_1 = np.std(score1_1)
    sd_score1 = np.std(score1)
    sd_score2_1 = np.std(score2_1)
    sd_score2 = np.std(score2)


    """ plot """
    heights = [np.mean(score1_1), np.mean(score1), np.mean(score2_1), np.mean(score2)]
    label = ["class1_score1", "class1_score2", "class2_score1", "class2_score2"]
    width = 0.8# the width of the bars
    bars = np.arange(len(heights))
    std = [sd_score1_1, sd_score1, sd_score2_1, sd_score2]

    plt.figure(figsize=(8, 5))
    plt.bar(bars, heights, width, tick_label=label, yerr=std,
            align='center', alpha=0.5, ecolor='black', capsize=5)
    plt.ylim(0, 100)
    barplot_annotate_brackets(0, 1, p1_val, bars,
                            heights, yerr=std)
    # barplot_annotate_brackets(0, 2, 'p < 0.001', bars,
    #                         heights, yerr=std)
    # barplot_annotate_brackets(1, 2, p2_val, bars,
    #                         heights, yerr=std,dh=0.2)
    # barplot_annotate_brackets(2, 2, 'p < 0.001', bars,
    #                         heights, yerr=std,dh=0.2)
    plt.tight_layout()
    plt.savefig("barplot_sig.png")
    plt.show()

if __name__ == "__main__":
    main()
