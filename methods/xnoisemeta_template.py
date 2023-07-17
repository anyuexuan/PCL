import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from torchvision.transforms import Resize, CenterCrop, ToTensor, RandomResizedCrop, RandomHorizontalFlip, Normalize
from PIL import ImageEnhance, Image


class xNoiseMetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, verbose=False, image_size=None,
                 noise_type='extreme', noise_rate=0.5, merge_rate=0.5, device='cuda:0'):
        super(xNoiseMetaTemplate, self).__init__()
        self.n_way = n_way  # N, n_classes
        self.n_support = n_support  # S, sample num of support set
        self.n_query = -1  # Q, sample num of query set(change depends on input)
        self.feature_extractor = model_func()  # feature extractor
        self.feat_dim = self.feature_extractor.final_feat_dim
        self.verbose = verbose
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.merge_rate = merge_rate
        self.image_size = image_size
        self.device = device
        self.epoch = 0

    @abstractmethod
    def set_forward(self, x):
        # x -> predicted score
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        # x -> loss value
        pass

    def forward(self, x):
        # x-> feature embedding
        out = self.feature_extractor.forward(x)
        return out

    def parse_feature(self, x):
        x = x.requires_grad_(True)
        x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        z_all = self.feature_extractor.forward(x)
        z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
        z_support = z_all[:, :self.n_support]  # [N, S, d]
        z_query = z_all[:, self.n_support:]  # [N, Q, d]
        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
        topk_ind = topk_labels.cpu().numpy()  # index of topk
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer):
        self.train()
        self.epoch = epoch
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

    def test_loop(self, test_loader, return_std=False):
        self.eval()
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            x = x.to(self.device)
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if self.verbose:
            # Confidence Interval   90% -> 1.645      95% -> 1.96     99% -> 2.576
            print('%d Test Acc = %4.2f±%4.2f' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def add_feature_noise(self, x):
        noise_mask = torch.from_numpy(
            np.random.uniform(0, 1, size=(x.shape[0], x.shape[1], 1, 1, 1)) < self.noise_rate).float()
        x_clean = (1 - noise_mask) * x
        x_noise = noise_mask * (
                x * (1 - self.merge_rate) + np.random.uniform(-2.7, 2.7, size=x.shape) * self.merge_rate)
        x = (x_clean + x_noise).float()
        return x

    def cosine_similarity(self, x, y):
        '''
        Cosine Similarity of two tensors
        Args:
            x: torch.Tensor, m x d
            y: torch.Tensor, n x d
        Returns:
            result, m x n
        '''
        assert x.size(1) == y.size(1)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        return x @ y.transpose(0, 1)

    def mahalanobis_dist(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        cov = torch.cov(x)  # [m,m]
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        delta = x - y  # [m,n,d]
        return torch.einsum('abc,abc->ab', torch.einsum('abc,ad->abc', delta, torch.inverse(cov)), delta)

    def euclidean_dist(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        return torch.pow(x - y, 2).sum(2)


class ImageJitter(object):
    def __init__(self, transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                                              Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color),
                 transformdict=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out
