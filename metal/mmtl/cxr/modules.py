import torch
from torchvision import models
import torch.nn.init as init
import torch.nn as nn
from metal.end_model import IdentityModule

class TorchVisionEncoder(nn.Module):
    def __init__(
        self, net_name, freeze_cnn=False, pretrained=False, drop_rate=0.2):
        super().__init__()
        self.model = self.get_tv_encoder(net_name, pretrained,
            drop_rate)
        if freeze_cnn:
            for param in self.parameters():
                param.requires_grad = False

    def get_tv_encoder(self, net_name, pretrained, drop_rate):
        # HACK: replace linear with identity -- ideally remove this
        net = getattr(models, net_name, None)
        if net is None:
            raise ValueError(f'Unknown torchvision network {net_name}')
        if 'densenet' in net_name.lower():
            model = net(pretrained=pretrained, drop_rate=drop_rate)
            self.encode_dim=int(model.classifier.weight.size()[1])
            model.classifier=IdentityModule()
            #model = torch.nn.Sequential(*(list(model.children())[:-1]))
        elif 'resnet' in net_name.lower():
            model = net(pretrained=pretrained)
            self.encode_dim=int(model.fc.weight.size()[1])
            model.fc=IdentityModule()
            #model = torch.nn.Sequential(*(list(model.children())[:-1]))
        else:
            raise ValueError('Network {net_name} not supported')
        return model

    def forward(self, X):
        out = self.model(X)
        return out
