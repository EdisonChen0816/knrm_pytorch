# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import kernel_mu
from models.model_utils import kernel_sigma


class CKNRM(nn.Module):

    def __init__(self, config, weights):
        super(CKNRM, self).__init__()
        self.weights = weights
        self.config = config
        self.embed_layer = nn.Embedding(len(self.weights.w2v.index2word), self.weights.vector_size, padding_idx=0)
        self.embed_layer.weight.requires_grad = False
        self.conv_uni = nn.Sequential(nn.Conv2d(1, self.config['out_kernel'], (1, self.weights.vector_size)), nn.ReLU())
        self.conv_bi = nn.Sequential(nn.Conv2d(1, self.config['out_kernel'], (2, self.weights.vector_size)), nn.ReLU())
        self.conv_tri = nn.Sequential(nn.Conv2d(1, self.config['out_kernel'], (3, self.weights.vector_size)), nn.ReLU())
        self.conv_grams = [self.conv_uni, self.conv_bi, self.conv_tri]
        self.dense = nn.Linear(self.config['n_kernels'] * self.config['max_grams'] ** 2, 1)
        self.get_param()

    def get_param(self):
        self.embed_layer.weight.data.copy_(torch.from_numpy(self.weights.get_vectors).to(self.config['device']))
        self.mus = torch.FloatTensor(kernel_mu(self.config['n_kernels']))
        self.sigmas = torch.FloatTensor(kernel_sigma(self.config['n_kernels']))
        self.mus = self.mus.view(self.config['n_kernels']).to(self.config['device'])
        self.sigmas = self.sigmas.view(self.config['n_kernels']).to(self.config['device'])

    def get_match_features(self):
        pass

    def conv_and_norm(self, q_emb, t_emb, q_mask, t_mask):
        q_emb_conv = [torch.transpose(torch.squeeze(self.conv_grams[i](q_emb.unsqueeze(1))), 1, 2) for i in range(self.config['max_grams'])]
        t_emb_conv = [torch.squeeze(self.conv_grams[j](t_emb.unsqueeze(1))) for j in range(self.config['max_grams'])]
        q_emb_norm = [F.normalize(q_item, p=2, dim=2, eps=1e-10) for q_item in q_emb_conv]
        t_emb_norm = [F.normalize(t_item, p=2, dim=1, eps=1e-10) for t_item in t_emb_conv]
        q_masks = [q_mask[:, :q_mask.size()[1] - m].unsqueeze(2) for m in range(self.config['max_grams'])]
        t_masks = [t_mask[:, :t_mask.size()[1] - n].unsqueeze(1).unsqueeze(3) for n in range(self.config['max_grams'])]
        return q_emb_norm, t_emb_norm, q_masks, t_masks

    def interaction_matrix(self, q_emb_norm, t_emb_norm, q_mask, t_mask):
        match_matrix = torch.unsqueeze(torch.bmm(q_emb_norm, t_emb_norm), 3)
        kernel_pooling = torch.exp(-((match_matrix - self.mus) ** 2) / (2 * (self.sigmas ** 2)))
        kernel_row = kernel_pooling * t_mask
        pooling_row_sum = torch.sum(kernel_row, 2)
        log_pooling = torch.log(torch.clamp(pooling_row_sum, min=1e-10)) * q_mask * 0.01  # scale down the data
        log_pooling_sum = torch.sum(log_pooling, 1)
        return log_pooling_sum

    def forward(self, q, t, q_mask, t_mask):
        q_emb = self.embed_layer(q)
        t_emb = self.embed_layer(t)
        # q_mask = q_mask.unsqueeze(2)
        # t_mask = t_mask.unsqueeze(1).unsqueeze(2)
        # import ipdb; ipdb.set_trace()
        q_emb_norm, t_emb_norm, q_masks, t_masks = self.conv_and_norm(q_emb, t_emb, q_mask, t_mask)
        mm_total = []
        for q_ix in range(self.config['max_grams']):
            for t_ix in range(self.config['max_grams']):
                # pass  # t_mask 和 q_mask 的数据还要调整；
                mm_total.append(self.interaction_matrix(q_emb_norm[q_ix], t_emb_norm[t_ix], q_masks[q_ix], t_masks[t_ix]))
        log_pooling_sums = torch.cat(mm_total, 1)
        output = torch.sigmoid(self.dense(log_pooling_sums))
        output = torch.squeeze(output)
        return output

    @classmethod
    def _kernel_layer(cls, mu, sigma):
        pass


if __name__ == "__main__":
    pass
