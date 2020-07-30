# -*- coding:utf-8 -*-


from collections import Counter


def positive_sample_percentage(filename):
    tot, pos = 0, 0
    for line in open(filename, 'r'):
        line = line.strip().split('\t')
        if line[-1] == '1':
            pos += 1
        tot += 1
    print(f"positive sample percentage:{pos/tot}")


def title_length_distribution(filename):
    tot, pos = 0, 0
    pos_sen_len, neg_sen_len, tot_sen_len = [], [], []
    for line in open(filename, 'r'):
        line = line.strip().split('\t')
        s1, s2, label = line[1], line[2], line[-1]
        tot_sen_len.extend([len(s1), len(s2)])
        tot += 2
        if label == '1':
            pos_sen_len.extend([len(s1), len(s2)])
            pos += 2
        else:
            neg_sen_len.extend([len(s1), len(s2)])
    tot_counter = Counter(tot_sen_len)
    pos_counter = Counter(pos_sen_len)
    neg_counter = Counter(neg_sen_len)
    tot_freq = sorted(map(lambda x: (x[0], round(x[1] / tot, 4)), tot_counter.items()))
    pos_freq = sorted(map(lambda x: (x[0], round(x[1] / pos, 4)), pos_counter.items()))
    neg_freq = sorted(map(lambda x: (x[0], round(x[1] / (tot - pos), 4)), neg_counter.items()))
    print('Total sample length distribution: {}'.format(tot_freq))
    print('Positive sample length distribution: {}'.format(pos_freq))
    print('Negetive sample length distribution: {}'.format(neg_freq))


if __name__ == '__main__':
    filename = "../data/atec_nlp_sim_train.csv"
    positive_sample_percentage(filename)
    title_length_distribution(filename)
