
import re
import torch
import numpy as np
import torch.functional as F
from konlpy.tag import Okt


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin: int):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euc_dist = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean(
            (1 - label) * torch.pow(euc_dist, 2) + label * torch.pow(torch.clamp(self.margin - euc_dist, min=0.0), 2))

        return loss


def make_weights(labels, nclasses):
    labels = np.array(labels)
    weight_arr = np.zeros_like(labels)

    _, counts = np.unique(labels, return_counts=True)
    for cls in range(nclasses):
        weight_arr = np.where(labels == cls, 1 / counts[cls], weight_arr)

    return weight_arr


def stalking_cat(glove, number,
                 last, reason, action, try_, reaction, valuable, start, charmingLover, charmingCustomer, relation,
                 event):

    columns = [reason, action, try_, reaction, valuable, start, charmingLover, charmingCustomer, relation, event]
    text = last

    for i in columns:
        text = f"{text} {i}"

    text = re.sub(f'[^A-Za-z가-힣 ]', '', text)

    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    num_5 = 0
    num_6 = 0
    num_7 = 0
    num_8 = 0
    num_9 = 0
    num_10 = 0
    num_11 = 0
    num_12 = 0
    num_13 = 0

    # 유형 1 : 찾아가기
    home = dict(glove.most_similar('방문', number+1))

    for i in home.keys():
        k = text.count(i) * home[i]
        num_1 += k

    # 유형 2: 접근하기
    near = dict(glove.most_similar('손편지', number+1))

    for i in near.keys():
        k = text.count(i) * near[i]
        num_2 += k

    # 유형 3: 기다리기
    wait = dict(glove.most_similar('기다림', number+1))

    for i in wait.keys():
        k = text.count(i) * wait[i]
        num_3 += k

    # 유형 4: 미행하기
    follow = dict(glove.most_similar('미행', number+1))

    for i in follow.keys():
        k = text.count(i) * follow[i]
        num_4 += k

    # 유형 5: 지켜보기
    watch = dict(glove.most_similar('몰래', number+1))

    for i in watch.keys():
        k = text.count(i) * watch[i]
        num_5 += k

    # 유형 6: 연락 도달하게 하기
    phone = dict(glove.most_similar('카톡', number+1))

    for i in phone.keys():
        k = text.count(i) * phone[i]
        num_6 += k

    # 유형 7: 두드리기
    knock = dict(glove.most_similar('현관', number+1))

    for i in knock.keys():
        k = text.count(i) * knock[i]
        num_7 += k

    # 유형 8: 물건
    letter = dict(glove.most_similar('쪽지', number+1))

    for i in letter.keys():
        k = text.count(i) * letter[i]
        num_8 += k

    # 유형 9: 진로방해
    stop = dict(glove.most_similar('퇴근', number+1))

    for i in stop.keys():
        k = text.count(i) * stop[i]
        num_9 += k

    # 유형 10: 배회하기
    around = dict(glove.most_similar('직장', number+1))

    for i in around.keys():
        k = text.count(i) * around[i]
        num_10 += k

    # 유형 11: 지인 연락
    neigh = dict(glove.most_similar('가족', number+1))

    for i in neigh.keys():
        k = text.count(i) * neigh[i]
        num_11 += k

    # 유형 12: 침입
    broke = dict(glove.most_similar('침입', number+1))

    for i in broke.keys():
        k = text.count(i) * broke[i]
        num_12 += k

    # 유형 13: 기타
    etc = dict(glove.most_similar('접근', number+1))

    for i in etc.keys():
        k = text.count(i) * etc[i]
        num_13 += k

    # 확률 결과
    rate = [num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8, num_9, num_10, num_11, num_12, num_13]
    percentage = rate / sum(rate)

    return percentage
