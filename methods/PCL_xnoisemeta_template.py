import torch
import torch.nn as nn
import numpy as np
from .xnoisemeta_template import xNoiseMetaTemplate
from torch.nn.functional import cross_entropy


class FiLM_original(nn.Module):
    def __init__(self, n_in, n_hidden=100):
        super(FiLM_original, self).__init__()
        self.linear1 = nn.Linear(n_in, n_hidden)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.f = nn.Linear(n_hidden, n_in)
        self.h = nn.Linear(n_hidden, n_in)
        self.linear2 = nn.Linear(n_in, n_in)

    def forward(self, x):
        xx = self.bn(self.linear1(torch.relu(x)))
        return x + self.linear2(torch.relu((self.f(xx))) * x + self.h(xx))


# To reduce the consumption, we adopt a more efficient way
class FiLM(nn.Module):
    def __init__(self, n_in, n_hidden=100):
        super(FiLM, self).__init__()
        self.linear1 = nn.Linear(n_in, n_hidden)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.f = nn.Linear(n_hidden, n_hidden)
        self.h = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_in)

    def forward(self, x):
        x = torch.relu(x)
        xx = self.bn(self.linear1(x))
        xx = torch.relu(self.f(xx) * xx + self.h(xx))
        return x + self.linear2(xx)


class PCL_xNoiseMetaTemplate(xNoiseMetaTemplate):
    def __init__(self, model_func, n_way, n_support, image_size=None, hidden_dim=20, n_query=16,
                 noise_type='extreme', noise_rate=0.5, merge_rate=0.5, device='cuda:0'):
        super(PCL_xNoiseMetaTemplate, self).__init__(model_func, n_way, n_support, image_size=image_size,
                                                     noise_type=noise_type, noise_rate=noise_rate,
                                                     merge_rate=merge_rate, device=device)
        if type(self.feat_dim) == int:
            feat_dim = self.feat_dim
        else:
            feat_dim = self.feat_dim[0]
            for i in range(1, len(self.feat_dim)):
                feat_dim *= self.feat_dim[i]
        self.loss_attention_1 = LossAttention(feat_dim, hidden_dim=hidden_dim, n_way=n_way, n_query=n_query)
        self.loss_attention_2 = LossAttention(feat_dim, hidden_dim=hidden_dim, n_way=n_way, n_query=n_query)

    def parse_feature(self, x):
        x = x.requires_grad_(True).cuda()
        x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        x = self.feature_extractor.forward(x)
        z_all_1 = self.f1.forward(x)  # [N*(S+Q), d]
        z_all_2 = self.f2.forward(x)  # [N*(S+Q), d]
        z_all_1 = z_all_1.reshape(self.n_way, self.n_support + self.n_query, -1)  # [N, S+Q, d]
        z_all_2 = z_all_2.reshape(self.n_way, self.n_support + self.n_query, -1)  # [N, S+Q, d]
        z_support_1, z_query_1 = z_all_1[:, :self.n_support], z_all_1[:, self.n_support:]  # [N, S, d], [N, Q, d]
        z_support_2, z_query_2 = z_all_2[:, :self.n_support], z_all_2[:, self.n_support:]  # [N, S, d], [N, Q, d]
        return (z_support_1, z_query_1), (z_support_2, z_query_2)

    def correct(self, x):
        scores_1, scores_2 = self.set_forward(x)
        scores = scores_1 + scores_2
        y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
        topk_ind = topk_labels.cpu().numpy()  # index of topk
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def loss_PCL(self, pred_1, pred_2, true, loss_1, loss_2, z_query_1, z_query_2):
        ind_1_sorted = torch.argsort(loss_1.data)
        ind_2_sorted = torch.argsort(loss_2.data)
        attention_1 = self.loss_attention_1(z_query_1, ind_1_sorted)
        attention_2 = self.loss_attention_2(z_query_2, ind_2_sorted)
        loss_1_update = torch.sum(cross_entropy(pred_1[ind_2_sorted], true[ind_2_sorted], reduce=False) * attention_2)
        loss_2_update = torch.sum(cross_entropy(pred_2[ind_1_sorted], true[ind_1_sorted], reduce=False) * attention_1)
        return loss_1_update + loss_2_update

    def train_loop(self, epoch, train_loader, optimizer):
        self.train()
        self.epoch = epoch
        if self.epoch < 200:
            self.loss_attention_1.threshold = 0.2
            self.loss_attention_2.threshold = 0.2
        else:
            self.loss_attention_1.threshold = 0
            self.loss_attention_2.threshold = 0
        print_freq = 10
        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            if self.image_size is None: self.image_size = x.shape[-2:]
            x = self.add_feature_noise(x)
            x = x.to(self.device)
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return avg_loss


class LossAttention(nn.Module):
    def __init__(self, fin, hidden_dim, n_way, n_query, threshold=0):
        super(LossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_way = n_way
        self.layers11 = nn.Linear(fin, hidden_dim)
        self.G_encoder1 = nn.LSTM(input_size=hidden_dim, hidden_size=20, num_layers=2, batch_first=True,
                                  bidirectional=True)
        self.layers12 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.layers21 = nn.Linear(fin, hidden_dim)
        self.G_encoder2 = nn.Transformer(d_model=n_query, nhead=n_query, dim_feedforward=2 * hidden_dim)
        self.layers22 = nn.Sequential(
            nn.Linear(n_query * n_query, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_query)
        )
        self.threshold = threshold  # To prevent ignoring the correct samples in early training

    def forward(self, x, idx):
        # ---------- global attention ----------
        x1 = self.layers11(x)  # [80, hidden_dim]
        out_G = self.G_encoder1(x1[idx].unsqueeze(0))[0].squeeze()  # [80, hidden_dim*2]
        # out = x + out_G[:, :20] + out_G[:, 20:]
        out = self.layers12(out_G).squeeze()  # [80]
        p_global = torch.softmax(out, dim=0)
        # ---------- local attention ----------
        x2 = self.layers21(x)
        x2 = x2.reshape(self.n_way, -1, x2.shape[-1])  # [N,Q,hidden_dim]
        x2_norm = x2 / torch.norm(x2, 2, 2, keepdim=True)  # [N,Q,hidden_dim]
        correlation = torch.bmm(x2_norm, x2_norm.permute(0, 2, 1))  # [N,Q,Q]
        out = self.G_encoder2(correlation, correlation)  # [N,Q,Q]
        out = self.layers22(out.reshape(self.n_way, -1))  # [N,Q]
        p_local = torch.softmax(out, dim=1).reshape(-1)[idx]
        return p_global + p_local + self.threshold
