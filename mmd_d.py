import torch
import numpy as onp
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter


# Define the deep network for MMD-D
class Featurizer(nn.Module):
    def __init__(self, channels=1, img_size=32):
        super(Featurizer, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 100))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature


def Pdist2(x, y):
    """Cool way of computing the paired distance between x and y
       x[m1,d] y[m2,d] -> [m1, m2]"""
    x_norm = (x ** 2).sum(axis=1).reshape([-1, 1])
    if y is not None:
        y_norm = (y ** 2).sum(axis=1).reshape([1, -1])
    else:
        y = x
        y_norm = x_norm.reshape([1, -1])
    Pdist = x_norm + y_norm - 2.0 * onp.matmul(x, y.T)
    Pdist[Pdist<0]=0
    return Pdist


def compute_gaussian_K(S_P, S_Q, batch_size):
    return onp.exp(-Pdist2(S_P, S_Q)/2)


def compute_deep_K(S_P, S_Q, batch_size, model=None):
    S_P = S_P.reshape([-1, 1, 32, 32])
    S_Q = S_Q.reshape([-1, 1, 32, 32])
    if model is None:
        model = Featurizer()
        model = model.cuda()
    S_P = torch.from_numpy(S_P)
    S_Q = torch.from_numpy(S_Q)
    S_P = Variable(S_P.type(torch.FloatTensor)).cuda()
    S_Q = Variable(S_Q.type(torch.FloatTensor)).cuda()
    S_P_feat = (model(S_P)).detach().cpu().numpy()
    S_Q_feat = (model(S_Q)).detach().cpu().numpy()
    K = onp.exp(-Pdist2(S_P_feat, S_Q_feat)/2)
    return K, model

