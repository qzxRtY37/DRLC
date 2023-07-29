from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import math


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # out = identity * a_w * a_h
        out = a_w * a_h
        return out
    

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)  


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1

    

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class DomainDiscriminator(nn.Module):
    r"""Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Parameters:
        - **in_feature** (int): dimension of the input feature
        - **hidden_size** (int): dimension of the hidden features

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr_mult": 1.}]  


    
    
class ClassifierBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None):
        super(ClassifierBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f = f.view(-1, self.backbone.out_features)
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params


    

class ImageClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = 256, head: Optional[nn.Module] = None):
        super(ImageClassifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self._features_dim = bottleneck_dim
        if head is None:
            self.head1 = nn.Linear(self._features_dim, num_classes)
            self.head2 = nn.Linear(self._features_dim, num_classes)
        else:
            self.head1 = head
            self.head2 = head
        self.ca = CoordAtt(backbone.out_features, backbone.out_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        bottleneck_F = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        bottleneck_B = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        bottleneck_E = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.bottleneck = nn.ModuleDict({
            'f': bottleneck_F,
            'b': bottleneck_B,
            'e': bottleneck_E,
        })

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x) 
        weight = self.ca(f)
        b = weight * f
        e = (1-weight) * f
        f = self.avgpool(f).view(-1, self.backbone.out_features)
        b = self.avgpool(b).view(-1, self.backbone.out_features)
        e = self.avgpool(e).view(-1, self.backbone.out_features)
        f = self.bottleneck['f'](f)
        b = self.bottleneck['b'](b)
        e = self.bottleneck['e'](e)
        predictions1 = self.head1(f)
        predictions2 = self.head2(b)
        # return predictions, f, b, e
        return predictions1, predictions2, f, b, e

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.ca.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.0},
            {"params": self.head1.parameters(), "lr_mult": 1.0},
            {"params": self.head2.parameters(), "lr_mult": 1.0},
        ]
        return params

    




