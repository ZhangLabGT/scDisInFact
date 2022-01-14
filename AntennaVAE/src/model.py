import torch
import torch.nn as nn
import torch.nn.functional as F

import collections
from typing import Iterable, List

from torch.nn.parameter import Parameter

class FC(nn.Module):
    def __init__(self, features = [1000, 500, 500], use_batch_norm = True, dropout_rate = 0.0, negative_slope = 0.0, use_bias = True, act_func_type='relu'):
        super().__init__()
        self.fc_layers = []

        self.features = features
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.negative_slope = negative_slope
        self.use_bias = use_bias
        self.act_func_type = act_func_type
        self.act_func = self.__act_func_selector__()

        # create fc layers according to the layers_dim
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            collections.OrderedDict(
                                [
                                    ("linear", nn.Linear(n_in, n_out) if self.use_bias else nn.Linear(n_in, n_out, bias = False),),
                                    ("batchnorm", nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if self.use_batch_norm else None,),
                                    ("act", self.act_func),
                                    ("dropout", nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else None,),
                                ]
                            )
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(self.features[:-1], self.features[1:]))
                ]
            )
        )
    # Selecting activation function
    def __act_func_selector__(self):
        # TODO: find a more elegant way
        if self.act_func_type == 'relu':
            return nn.ReLU() if self.negative_slope <= 0 else nn.LeakyReLU(negative_slope = self.negative_slope)
        elif self.act_func_type == 'sigmoid':
            return nn.Sigmoid()

    def forward(self, x):
        # loop through all layers
        for layers in self.fc_layers:
            # loop through linear, batchnorm, relu, dropout, etc
            for layer in layers:
                if layer is not None:
                    x = layer(x)
        return x    

# Encoder
class Encoder(nn.Module):
    def __init__(self, features = [1024, 256, 32, 8], dropout_rate = 0.1, negative_slope = 0.2):
        super(Encoder,self).__init__()
        
        self.features = features
        if len(features) > 2:
            self.fc = FC(
                features = features[:-1],
                dropout_rate = dropout_rate,
                negative_slope = negative_slope,
                use_bias = True
            )
        self.output = nn.Linear(features[-2], features[-1])

    def forward(self, x):
        if len(self.features) > 2:
            x = self.fc(x)
        x = self.output(x)
        return x


# Decoder
class Decoder(nn.Module):
    def __init__(self, features = [8, 32, 256, 1024], dropout_rate = 0.0, negative_slope = 0.2):
        super(Decoder, self).__init__()
        self.fc = FC(
            features = features,
            dropout_rate = dropout_rate,
            negative_slope = negative_slope,
            use_bias = True
        )



    def forward(self, z):
        # The decoder returns values for the parameters of the ZINB distribution
        x_mean = self.fc(z)
        return x_mean


class gene_act(nn.Module):
    def __init__(self, features = [1000, 500, 500], use_batch_norm = True, dropout_rate = 0.0, negative_slope = 0.0):
        super(gene_act, self).__init__()

        self.features = features
        self.fc_layers = []

        # create fc layers according to the layers_dim
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            collections.OrderedDict(
                                [
                                    ("linear", nn.Linear(n_in, n_out, bias = False),),
                                    ("batchnorm", nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,),
                                    ("act", nn.ReLU() if negative_slope <= 0 else nn.LeakyReLU(negative_slope = negative_slope),),
                                    ("dropout", nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,),
                                ]
                            )
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(self.features[:-1], self.features[1:]))
                ]
            )
        )

    def forward(self, x):
        # loop through all layers
        for layers in self.fc_layers:
            # loop through linear, batchnorm, relu, dropout, etc
            for layer in layers:
                if layer is not None:
                    x = layer(x)
        
        return x

# The three output layers of DCA method
class ACT_EXP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.exp(x)

class OutputLayer(nn.Module):
    def __init__(self, features=[256, 1024]) -> None:
        super().__init__()
        self.output_size = features[1]
        self.last_hidden = features[0]
        self.mean_layer = nn.Sequential(nn.Linear(self.last_hidden, self.output_size), ACT_EXP())
        # ! Parameter Pi needs Sigmoid as activation func 
        self.pi_layer = nn.Sequential(nn.Linear(self.last_hidden, self.output_size), nn.Sigmoid())
        self.theta_layer = nn.Sequential(nn.Linear(self.last_hidden, self.output_size), ACT_EXP())

    def forward(self, decodedData):
        Miu = self.mean_layer(decodedData)
        Pi = self.pi_layer(decodedData)
        Theta = self.theta_layer(decodedData)
        return Miu, Pi, Theta
