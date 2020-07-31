# encoding=utf-8
import torch
import torch.nn as nn
from src.utils.utils import loadyaml
from src.dataprocessors.tokenizer import Segment_jieba
from src.dataprocessors.w2v import Embedding
from src.dataprocessors.vocab import Vocab
from src.models.knrm import KNRM
from src.utils.logger import setlogger
from src.dataprocessors.dataset import Dataset
from torch.utils.data import DataLoader


def train(train_iter, eval_iter, model, config, logger):
    # config
    # model = model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    # certerion = nn.CrossEntropyLoss()
    certerion = nn.MSELoss()
    # change model mode
    step = 0
    f_best = 0
    model.train()
    for epoch in range(1, config['epochs'] + 1):
        for batch in train_iter:
            q, t, q_mask, t_mask, label = batch
            y = torch.squeeze(label, 1).float().to(config['device'])
            optimizer.zero_grad()
            score = model(q.to(config['device']), t.to(config['device']), q_mask.to(config['device']), t_mask.to(config['device']))
            # print(f"shape | score -> {score.size()} | y -> {y.size()}")
            # loss = F.binary_cross_entropy(score, y)
            loss = certerion(score, y)
            loss.backward()
            # print(loss.item())
            optimizer.step()
            step += 1
            if step % 10 == 0:
                label = label > 0.5
                y_predict = y > 0.5
                label, y_predict = label.to(config['device']), y_predict.to(config['device'])
                acc, p, r, F1 = metrics(label, y_predict)
                if F1 > f_best:
                    f_best = F1
                    model_state_dict = model.state_dict()
                    optimizer_state_dict = optimizer.state_dict()
                    checkpoint = {
                            'model': model_state_dict,
                            'optimizer': optimizer_state_dict,
                            }
                    torch.save(checkpoint, config['saved_model'])
            if step % 100 == 0:
                ave_accu = test(eval_iter, model, config)
                logger.info(f"Step: {step} |Epoch:{epoch}, | train loss: {loss.detach().cpu().numpy():.{4}} | ave_accu:{ave_accu} | F1: {F1}")


def test(eval_iter, model, config):
    scores = []
    batch = 0
    model.eval()
    for tbatch in eval_iter:
        q, t, q_mask, t_mask, labels = tbatch
        # score = model(q, t, q_mask, t_mask)
        score = model(q.to(config['device']), t.to(config['device']), q_mask.to(config['device']), t_mask.to(config['device']))
        labels = torch.squeeze(labels, 1).to(config['device'])
        labels = labels > 0.5
        # y = torch.squeeze(score, 1).float()
        y_predict = score > 0.5
        correct = y_predict.eq(labels).long().sum()
        # accu = torch.sum(y_score) / labels.size()[0]
        accu = correct.detach().cpu().numpy()/ y_predict.size()[0]
        scores.append(accu)
        batch += 1
    model.train()
    return sum(scores)/batch


def metrics(labels, y_pred):
    # labels and y_pred dtype is 8-bit integer
    TP = ((y_pred == 1) & (labels == 1)).sum().float()
    TN = ((y_pred == 0) & (labels == 0)).sum().float()
    FN = ((y_pred == 1) & (labels == 0)).sum().float()
    FP = ((y_pred == 0) & (labels == 1)).sum().float()
    p = TP / (TP + FP).clamp(min=1e-8)
    r = TP / (TP + FN).clamp(min=1e-8)
    F1 = 2 * r * p / (r + p).clamp(min=1e-8)
    acc = (TP + TN) / (TP + TN + FP + FN).clamp(min=1e-8)
    return acc, p, r, F1


if __name__ == '__main__':
    config = loadyaml('../conf/knrm.yaml')
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
    train(train_loader, test_loader, model, config)