# encoding=utf-8
import torch


def test(eval_iter, model, logger):
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
        logger.info(f"accu:{accu}")
    print(f"Average accu: {sum(scores) / batch}")


if __name__ == '__main__':
    pass
