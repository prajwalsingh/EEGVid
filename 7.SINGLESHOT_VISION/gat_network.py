import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch_geometric.nn import GATConv

from torch.autograd import Variable
from einops.layers.torch import Rearrange

# Safe Gumbel noise generation with clamping to prevent NaNs
def safe_gumbel_noise(shape, epsilon=1e-10):
    U = torch.rand(shape).clamp(epsilon, 1 - epsilon).to('cuda')
    return -torch.log(-torch.log(U))

# Safe sigmoid function to avoid exact 0 or 1 probabilities
def safe_sigmoid(x, epsilon=1e-10):
    return torch.sigmoid(x).clamp(epsilon, 1 - epsilon)

# Utility function to replace NaNs with a specified value
def replace_nan(tensor, value=0.0):
    return torch.where(torch.isnan(tensor), torch.tensor(value, device=tensor.device), tensor)

# Gumbel-Softmax soft rounding with NaN handling
def gumbel_soft_round(x, temperature=1.0, epsilon=1e-10):
    # Clamp temperature to avoid division by very small numbers
    temperature = max(temperature, epsilon)
    
    # Step 1: Separate integer and fractional parts
    floor_val = torch.floor(x)
    frac_part = x - floor_val

    # Step 2: Calculate the probability to round up using safe sigmoid
    probs = safe_sigmoid(frac_part - 0.5, epsilon=epsilon)

    # Step 3: Generate Gumbel noise with clamping
    gumbel_noise = safe_gumbel_noise(x.shape, epsilon=epsilon)

    # Step 4: Calculate the soft rounding with temperature
    soft_values = floor_val + (gumbel_noise + probs) / temperature
    
    # Step 5: Replace any NaNs in the result (as a final safeguard)
    soft_values = replace_nan(soft_values, value=0.0)
    
    return soft_values

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()

        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.MaxPool2d((1, 35), (1, 5)),
            nn.BatchNorm2d(40),
            # nn.ELU(),
            nn.LeakyReLU(),
            nn.Conv2d(40, 80, (1, 1), (1, 1)),
            # nn.MaxPool2d((1, 25), (1, 5)),
            nn.MaxPool2d((1, 15), (1, 5)),
            nn.BatchNorm2d(80),
            # nn.ELU(),
            nn.LeakyReLU(),
            nn.Conv2d(80, 120, (1, 1), (1, 1)),
            nn.MaxPool2d((1, 5), (1, 5)),
            nn.BatchNorm2d(120),
            # nn.ELU(),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(120, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x) # B, 40, 62, 105
        x = self.projection(x) # B, (62*105), 40
        return x



class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250, num_eeg_channels=62):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)
        self.num_channels = num_eeg_channels
        # Create a list of tuples representing all possible edges between channels
        self.edge_index_list = torch.Tensor([(i, j) for i in range(self.num_channels) for j in range(self.num_channels) if i != j]).cuda()
        # Convert the list of tuples to a tensor
        self.edge_index = torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous().cuda()

    def forward(self, x):
        batch_size, _, num_channels, num_features = x.size()
        # Reshape x to (batch_size*num_channels, num_features) to pass through GATConv
        if num_channels == 1:
            return x
        x = x.view(batch_size*num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)
        return x

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

class EEGEncoder(nn.Sequential):
    def __init__(self, emb_size=40, in_timesteps=400, eeg_channels=62, **kwargs):
        super().__init__()
        self.OUT_DIM = {
                        'all': 5120, # 62
                        'noseback_left': 2160, # 27
                        'noseback_right': 2160, # 27
                        'noseback_front': 2080, # 26
                        'noseback_back': 2320, # 29
                        'noseback_Q2_left': 880, # 11
                        'noseback_Q3_left': 960, # 12
                        'noseback_Q1_right': 880, # 11
                        'noseback_Q4_right': 960, # 12
                        'noseback_center': 800, # 10
                        'leftrightear_center': 720, # 9
                        'per_channel': 80, # 1
                        'lobes_frontal_left': 800, # 10
                        'lobes_frontal_right': 800, # 10
                        'lobes_parietal_left': 640, # 8
                        'lobes_parietal_right': 640, # 8
                        'lobes_occipital_left': 240, # 3
                        'lobes_occipital_right': 240, # 3
                        'lobes_temporal_left': 240, # 3
                        'lobes_temporal_right': 240, # 3
                        'lobes_frontal': 1920, # 24
                        'lobes_parietal': 1440, # 18
                        'lobes_occipital': 640, # 8
                        'lobes_temporal': 480, # 6
                        }
        self.gat_layer = ResidualAdd(nn.Sequential( 
                        EEG_GAT(in_channels=in_timesteps, out_channels=in_timesteps, num_eeg_channels=eeg_channels), 
                        nn.Dropout(0.3),
                        ))
        self.patch_emb  = PatchEmbedding(emb_size)
        self.flatten    = FlattenHead()
        self.proj       = nn.Sequential(
                            # nn.Linear(62 *120, 4096),
                            nn.Linear(self.OUT_DIM[kwargs['exp_type']], 2048),
                                ResidualAdd(nn.Sequential(
                                    # nn.GELU(),
                                    nn.LeakyReLU(),
                                    nn.Linear(2048, 2048),
                                    nn.Dropout(0.3),
                                )),
                                nn.LayerNorm(2048),
                                # nn.GELU(),
                                nn.LeakyReLU(),
                                nn.Linear(2048, 1024),
                                ResidualAdd(nn.Sequential(
                                    # nn.GELU(),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 1024),
                                    nn.Dropout(0.3),
                                )),
                                nn.LayerNorm(1024),
                                # nn.Linear(2048, 1024, bias=False),
                                # nn.Sigmoid(),
                                nn.Linear(1024, 1024, bias=False),
                                # nn.ReLU()
                            )
            
    def forward(self, x):
        x = self.gat_layer(x)
        x = self.patch_emb(x)
        x = self.flatten(x)#.unsqueeze(dim=1)
        # x = gumbel_soft_round(self.proj(x))
        x = self.proj(x)
        x = x / x.norm(dim=1, keepdim=True)
        return x