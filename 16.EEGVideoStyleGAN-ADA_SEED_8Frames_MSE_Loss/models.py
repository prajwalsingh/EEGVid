import torch.nn as nn
import torch.nn.init as init
from einops.layers.torch import Rearrange
from torch import Tensor

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        return x

def my_conv_block(in_channels, out_channels, kernel=3,padding=1,stride=1):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, stride=stride), 
              nn.BatchNorm2d(out_channels), 
              nn.ELU()]
    return nn.Sequential(*layers)

class ConvEmbedder(nn.Module):

    def __init__(self, in_channels=250, embedding_dim=768, dropout=0.3):
        super().__init__()
        
        self.cnn = nn.Sequential(
            my_conv_block(in_channels, 200, 3, 1, 2),
            nn.Dropout(p=dropout),
            my_conv_block(200, 150, 3, 1, 1),
            nn.Dropout(p=dropout),
            my_conv_block(150, 150, 3, 1, 2),
            nn.Dropout(p=dropout),
            my_conv_block(150, 90, 3, 1, 1),
            nn.Dropout(p=dropout),
            my_conv_block(90, 90, 3, 1, 1),
            nn.Dropout(p=dropout),
            my_conv_block(90, 90, 3, 1, 2),
        )
    
    def forward(self, x):
        out = self.cnn(x)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x



class Enc_eeg_topo(nn.Sequential):
    def __init__(self, emb_size=768, **kwargs):
        super().__init__(
            ConvEmbedder(embedding_dim=emb_size, **kwargs),
            FlattenHead()
        )

from eeg_encoder import EEG_GAT
class Enc_eeg_gat(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    EEG_GAT(),
                    nn.Dropout(0.3),
                )
            ),
            PatchEmbedding(emb_size),
            FlattenHead()
        )
    
class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=512, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 