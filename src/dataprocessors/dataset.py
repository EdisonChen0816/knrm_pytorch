# encoding=utf-8
import os
import numpy as np
import torch


class Dataset:

    def __init__(self, filename, seg, word2id, config):
        self.config = config
        query, target, labels = [], [], []
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as fe:
                for line in fe:
                    line = line.strip().split('\t')
                    q, t, label = line[1], line[2], line[-1]
                    query.append([word2id.get(qw, word2id['UNK']) for qw in seg(q, ifremove=False)['tokens']])
                    target.append([word2id.get(tw, word2id['UNK']) for tw in seg(t, ifremove=False)['tokens']])
                    labels.append(1) if label == '1' else labels.append(0)
                # print(query)
                self.query = query
                self.target = target
                self.labels = labels
        else:
            raise FileNotFoundError(f"train data file not found in {filename}")

    def batch(self, index):
        s1_ids, s2_ids, s_labels = self.query[index], self.target[index], self.labels[index]
        s1_data, s1_mask = self.pad2longest(s1_ids, self.config['query_max_len'])
        s2_data, s2_mask = self.pad2longest(s2_ids, self.config['target_max_len'])
        return s1_data, s2_data, s1_mask, s2_mask, s_labels

    def __getitem__(self, index):
        s1_ids, s2_ids, s_labels = self.query[index], self.target[index], self.labels[index]
        s1_data, s1_mask = self.pad2longest(s1_ids, self.config['query_max_len'])
        s2_data, s2_mask = self.pad2longest(s2_ids, self.config['target_max_len'])
        # return torch.LongTensor(s1_data).to(self.config['device']), torch.LongTensor(s2_data).to(self.config['device']), torch.FloatTensor(s1_mask).to(self.config['device']), torch.FloatTensor(s2_mask).to(self.config['device']), torch.LongTensor([s_labels]).to(self.config['device'])
        return torch.LongTensor(s1_data), torch.LongTensor(s2_data), torch.FloatTensor(s1_mask), torch.FloatTensor(s2_mask), torch.LongTensor([s_labels])

    @staticmethod
    def pad2longest(data_ids, max_len):
        if isinstance(data_ids[0], list):
            s_data = np.array([s[:max_len] + [0] * (max_len - len(s[:max_len])) for s in data_ids])
            s_mask = np.array([[1] * m[:max_len] + [0] * (max_len - len(m[:max_len])) for m in data_ids])
        elif isinstance(data_ids[0], int):
            s_data = np.array(data_ids[:max_len] + [0] * (max_len - len(data_ids[:max_len])))
            s_mask = np.array([1] * len(data_ids[:max_len]) + [0] * (max_len - len(data_ids[:max_len])))
        else:
            raise TypeError("list type is required")
        return (s_data, s_mask)

    def __len__(self):
        return len(self.labels)
