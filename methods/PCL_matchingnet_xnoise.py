import torch
import torch.nn as nn
import numpy as np
from .PCL_xnoisemeta_template import PCL_xNoiseMetaTemplate, FiLM
from utils import utils


class PCL_Matching_xnoise(PCL_xNoiseMetaTemplate):
    def __init__(self, model_func, n_way, n_support, image_size=None,
                 noise_type='extreme', noise_rate=0.5, merge_rate=1.0, tao=64, device='cuda:0'):
        super(PCL_Matching_xnoise, self).__init__(model_func, n_way, n_support, image_size=image_size,
                                                  noise_type=noise_type, noise_rate=noise_rate,
                                                  merge_rate=merge_rate, device=device)
        self.f1 = FiLM(self.feat_dim)
        self.FCE1 = FullyContextualEmbedding(self.feat_dim, device)
        self.G_encoder1 = nn.LSTM(input_size=self.feat_dim, hidden_size=self.feat_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)
        self.f2 = FiLM(self.feat_dim)
        self.FCE2 = FullyContextualEmbedding(self.feat_dim, device)
        self.G_encoder2 = nn.LSTM(input_size=self.feat_dim, hidden_size=self.feat_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)
        self.loss_fn = nn.NLLLoss(reduction='none')
        self.tao = tao
        self.to(self.device)

    def set_forward(self, x):
        y_s = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))  # [N*S]
        Y_S = utils.one_hot(y_s, self.n_way)  # [N*S,N]
        Y_S = Y_S.to(self.device)
        (z_support_1, z_query_1), (z_support_2, z_query_2) = self.parse_feature(x)

        z_support_1 = z_support_1.reshape(self.n_way * self.n_support, -1)  # [N*S,d]
        z_query_1 = z_query_1.reshape(-1, z_support_1.shape[-1])  # [N*Q,d]
        out_G_1 = self.G_encoder1(z_support_1.unsqueeze(0))[0].squeeze()  # S.unsqueeze(0):[1,N*S,d], out_G:[N*S,2d]
        G_1 = z_support_1 + out_G_1[:, :z_support_1.size(1)] + out_G_1[:, z_support_1.size(1):]  # [N*S,d]
        G_normalized_1 = torch.nn.functional.normalize(G_1, p=2, dim=1)
        F_1 = self.FCE1(z_query_1, G_1)  # [N*Q,d]
        F_normalized_1 = torch.nn.functional.normalize(F_1, p=2, dim=1)  # [N*Q,d]
        scores_1 = torch.relu(F_normalized_1.mm(G_normalized_1.transpose(0, 1))) * 100 / self.tao  # [N*Q,N*S]
        logprobs_1 = (torch.softmax(scores_1, dim=1).mm(Y_S) + 1e-6).log()  # [N*Q,N]

        z_support_2 = z_support_2.reshape(self.n_way * self.n_support, -1)  # [N*S,d]
        z_query_2 = z_query_2.reshape(-1, z_support_2.shape[-1])  # [N*Q,d]
        out_G_2 = self.G_encoder2(z_support_2.unsqueeze(0))[0].squeeze()  # S.unsqueeze(0):[1,N*S,d], out_G:[N*S,2d]
        G_2 = z_support_2 + out_G_2[:, :z_support_2.size(1)] + out_G_2[:, z_support_2.size(1):]  # [N*S,d]
        G_normalized_2 = torch.nn.functional.normalize(G_2, p=2, dim=1)
        F_2 = self.FCE2(z_query_2, G_2)  # [N*Q,d]
        F_normalized_2 = torch.nn.functional.normalize(F_2, p=2, dim=1)  # [N*Q,d]
        scores_2 = torch.relu(F_normalized_2.mm(G_normalized_2.transpose(0, 1))) * 100 / self.tao  # [N*Q,N*S]
        logprobs_2 = (torch.softmax(scores_2, dim=1).mm(Y_S) + 1e-6).log()  # [N*Q,N]

        return logprobs_1, logprobs_2

    def set_forward_loss(self, x):
        y_s = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))  # [N*S]
        Y_S = utils.one_hot(y_s, self.n_way)  # [N*S,N]
        Y_S = Y_S.to(self.device)
        (z_support_1, z_query_1), (z_support_2, z_query_2) = self.parse_feature(x)

        z_support_1 = z_support_1.reshape(self.n_way * self.n_support, -1)  # [N*S,d]
        z_query_1 = z_query_1.reshape(-1, z_support_1.shape[-1])  # [N*Q,d]
        out_G_1 = self.G_encoder1(z_support_1.unsqueeze(0))[0].squeeze()  # S.unsqueeze(0):[1,N*S,d], out_G:[N*S,2d]
        G_1 = z_support_1 + out_G_1[:, :z_support_1.size(1)] + out_G_1[:, z_support_1.size(1):]  # [N*S,d]
        G_normalized_1 = torch.nn.functional.normalize(G_1, p=2, dim=1)
        F_1 = self.FCE1(z_query_1, G_1)  # [N*Q,d]
        F_normalized_1 = torch.nn.functional.normalize(F_1, p=2, dim=1)  # [N*Q,d]
        scores_1 = torch.relu(F_normalized_1.mm(G_normalized_1.transpose(0, 1))) * 100 / self.tao  # [N*Q,N*S]
        scores_1 = (torch.softmax(scores_1, dim=1).mm(Y_S) + 1e-6).log()  # [N*Q,N]

        z_support_2 = z_support_2.reshape(self.n_way * self.n_support, -1)  # [N*S,d]
        z_query_2 = z_query_2.reshape(-1, z_support_2.shape[-1])  # [N*Q,d]
        out_G_2 = self.G_encoder2(z_support_2.unsqueeze(0))[0].squeeze()  # S.unsqueeze(0):[1,N*S,d], out_G:[N*S,2d]
        G_2 = z_support_2 + out_G_2[:, :z_support_2.size(1)] + out_G_2[:, z_support_2.size(1):]  # [N*S,d]
        G_normalized_2 = torch.nn.functional.normalize(G_2, p=2, dim=1)
        F_2 = self.FCE2(z_query_2, G_2)  # [N*Q,d]
        F_normalized_2 = torch.nn.functional.normalize(F_2, p=2, dim=1)  # [N*Q,d]
        scores_2 = torch.relu(F_normalized_2.mm(G_normalized_2.transpose(0, 1))) * 100 / self.tao  # [N*Q,N*S]
        scores_2 = (torch.softmax(scores_2, dim=1).mm(Y_S) + 1e-6).log()  # [N*Q,N]

        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda().long()
        loss_1, loss_2 = self.loss_fn(scores_1, y_query), self.loss_fn(scores_2, y_query)
        return self.loss_PCL(scores_1, scores_2, y_query, loss_1, loss_2,
                             z_query_1.reshape(-1, self.feat_dim),
                             z_query_2.reshape(-1, self.feat_dim), )


class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim, device):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim * 2, feat_dim)
        self.device = device
        self.c_0 = torch.zeros(1, feat_dim).to(self.device)
        self.c_0 = self.c_0

    def forward(self, f, G):
        h = f  # [N*Q,d]
        c = self.c_0.expand_as(f)  # [N*Q,d]
        G_T = G.transpose(0, 1)  # [d,N*S]
        for k in range(G.size(0)):
            logit_a = h.mm(G_T)  # [N*Q,N*S]
            a = torch.softmax(logit_a, dim=1)  # [N*Q,N*S]
            r = a.mm(G)  # [N*Q, d]
            x = torch.cat((f, r), 1)  # [N*Q,2d]
            h, c = self.lstmcell(x, (h, c))
            h = h + f
        return h
