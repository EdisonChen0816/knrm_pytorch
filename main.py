# encoding=utf-8
import torch
from src.utils.utils import loadyaml
from src.dataprocessors.tokenizer import Segment_jieba
from src.dataprocessors.w2v import Embedding
from src.dataprocessors.vocab import Vocab
from src.models.knrm import KNRM
from src.utils.logger import setlogger
from src.dataprocessors.dataset import Dataset
from torch.utils.data import DataLoader
from src.train import train
from src.test import test
import os


def train_model():
    config = loadyaml('./conf/knrm.yaml')
    logger = setlogger(config)
    # config = loadyaml('data/config/cknrm.yaml')
    torch.backends.cudnn.benchmark = True
    # print(f"config:{config}")
    # set the random seed manually for reproducibility.
    torch.manual_seed(config['seed'])
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # config['device'] = 'cpu'
    print(f"device:{config['device']}")
    print(f"Begin to load the embeding")
    embedding = Embedding(config['embed'], logger=logger)
    print(f"Begin to build segment")
    segment = Segment_jieba(user_dict=config['user_dict'])
    print(f"Begin to build vocab")
    vocab = Vocab(config['datapath'], segment, embedding)
    print(f"vocab length: {len(vocab)}")
    print(f"Begin to build dataset")
    train_dataset = Dataset(config['trainpath'], segment, vocab.word2idx, config)
    test_dataset = Dataset(config['evalpath'], segment, vocab.word2idx, config)
    # print(train_dataset[3])
    print(f"Begin to buidl train_loader")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    print(f"Init the model")
    model = KNRM(config, embedding).to(config['device'])
    # model = CKNRM(config, embedding).to(config['device'])
    print(f"Begin to train ......")
    train(train_loader, test_loader, model, config, logger)


def test_model():
    config = loadyaml('./conf/knrm.yaml')
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
    # model = CKNRM(config, embedding)
    model = KNRM(config, embedding)
    if os.path.exists(config['saved_model']):
        checkpoint = torch.load(config['saved_model'])
        model.load_state_dict(checkpoint['model'])
    test(test_loader, model, logger)


if __name__ == '__main__':
    train_model()
    #test_model()