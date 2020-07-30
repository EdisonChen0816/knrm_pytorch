# -*- coding:utf-8 -*-


import random


def data_split(filename, test_rate=0.1):
    with open('data/train.csv', 'w') as tn, open('data/test.csv', 'w') as tt:
        for line in open(filename):
            if random.random() > test_rate:
                tn.write(line)
            else:
                tt.write(line)


if __name__ == "__main__":
    data_split('data/atec_nlp_sim_train.csv')
