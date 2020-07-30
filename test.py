# -*- coding:utf-8 -*-


import os

import torch

from utils.utils import loadyaml
from dataprocessors.tokenizer import Segment_jieba
from dataprocessors.w2v import Embedding
from dataprocessors.vocab import Vocab
from models.knrm import KNRM
from models.cknrm import CKNRM
from utils.logger import setlogger
from dataprocessors.dataset import Dataset
from torch.utils.data import DataLoader


def eval(eval_iter, model, config, logger):
    scores = []
    batch = 0
    model.eval()
    for tbatch in eval_iter:
        q, t, q_mask, t_mask, labels = tbatch
        score = model(q, t, q_mask, t_mask)
        labels = torch.squeeze(labels, 1)
        labels = labels > 0.5
        y_predict = score > 0.5
        correct = y_predict.eq(labels).long().sum()
        accu = correct.detach().numpy() / y_predict.size()[0]
        scores.append(accu)
        batch += 1
        # logger.info(f"accu:{accu}")
    print(f"Average accu: {sum(scores) / batch}")


if __name__ == '__main__':
    config = loadyaml('data/config/cknrm.yaml')
    # config = loadyaml('data/config/knrm.yaml')
    logger = setlogger(config)
    # print(f"config:{config}")
    # set the random seed manually for reproducibility.
    torch.manual_seed(config['seed'])
    # config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = 'cpu'
    print(f"Begin to load the embeding")
    embedding = Embedding(config['embed'], logger=logger)
    print(f"Begin to build segment")
    segment = Segment_jieba(user_dict=config['user_dict'])
    print(f"Begin to build vocab")
    vocab = Vocab(config['datapath'], segment, embedding)
    print(f"Begin to build dataset")
    test_dataset = Dataset(config['evalpath'], segment, vocab.word2idx, config)
    # print(train_dataset[3])
    print(f"Begin to buidl train_loader")
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    print(f"Init the model")
    model = CKNRM(config, embedding)
    #model = KNRM(config, embedding)
    if os.path.exists(config['saved_model']):
        checkpoint = torch.load(config['saved_model'])
        model.load_state_dict(checkpoint['model'])
    eval(test_loader, model, config, logger)
