
import re
import torch
import random
import numpy as np
import pandas as pd


def pos_neg(data: pd.DataFrame,
            context: str,
            n_pairing: int):
    data = data.dropna(axis=0).reset_index(drop=True)

    positive = data['warning'][data['warning'] != 0].index
    negative = data['warning'][data['warning'] == 0].index

    pos = data.iloc[positive, :]
    neg = data.iloc[negative, :]

    pos = pos.loc[:, [context, 'warning']]
    neg = neg.loc[:, [context, 'warning']]

    pair_set = {'First': [],
                'Second': [],
                'Is_Same': []}

    for i in range(pos.shape[0]):
        for j in range(n_pairing):
            pair_set['First'].append(re.sub('[^A-Za-z가-힣0-9 ]', '', pos.iloc[i, 0]).replace('\n', ' '))
            pair_set['Second'].append(
                re.sub('[^A-Za-z가-힣0-9 ]', '', neg.iloc[random.randrange(0, neg.shape[0]), 0]).replace('\n', ' '))
            pair_set['Is_Same'].append(0)

    for i in range(int(pos.shape[0] * n_pairing / 2)):
        nums1 = random.sample(range(0, pos.shape[0]), 2)
        nums2 = random.sample(range(0, neg.shape[0]), 2)

        pair_set['First'].append(re.sub('[^A-Za-z가-힣0-9 ]', '', pos.iloc[nums1[0], 0]).replace('\n', ' '))
        pair_set['First'].append(re.sub('[^A-Za-z가-힣0-9 ]', '', neg.iloc[nums2[0], 0]).replace('\n', ' '))

        pair_set['Second'].append(re.sub('[^A-Za-z가-힣0-9 ]', '', pos.iloc[nums1[1], 0]).replace('\n', ' '))
        pair_set['Second'].append(re.sub('[^A-Za-z가-힣0-9 ]', '', neg.iloc[nums2[1], 0]).replace('\n', ' '))

        pair_set['Is_Same'].append(1)
        pair_set['Is_Same'].append(1)

    pair_df = pd.DataFrame(pair_set)
    pair_df = pair_df.sample(frac=1).reset_index(drop=True)

    return pair_df