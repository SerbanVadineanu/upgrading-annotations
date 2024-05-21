from msd_pytorch import MSDSegmentationModel
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict

class MSDClassificationModel(MSDSegmentationModel):
    def __init__(
    self,
    c_in,
    num_labels,
    depth,
    width,
    dilations=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ndim=2
):
        self.num_labels = num_labels
        # Initialize msd network.
        super().__init__(c_in, num_labels, depth, width, dilations=dilations)

        self.net = nn.Sequential(self.scale_in, self.msd)
        self.linear = nn.Linear(num_labels, num_labels, bias=False)

        # Train all parameters apart from self.scale_in.
#         self.init_optimizer(net_trained)


    # This is in place of forward
    def __call__(self, x):
        x = self.net(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.linear(x)
        return x
    
    def to(self, device):
        self.net.to(device)
        self.linear.to(device)
        return self
    
    def eval():
        self.net.eval()
        self.linear.eval()
        
    def train():
        self.net.train()
        self.linear.train()
        
    def parameters(self):
        msd_params = list(self.msd.parameters())
        linear_params = list(self.linear.parameters())
        all_params = msd_params + linear_params
        return all_params
    
    def state_dict(self):
        net_dict = self.net.state_dict()
        linear_dict = self.linear.state_dict()
        keys_net = list(net_dict.keys())
        keys_linear = list(linear_dict.keys())
        state_dict = OrderedDict()
        
        for k in keys_net:
            state_dict[k] = net_dict[k]
        for k in keys_linear:
            state_dict[k] = linear_dict[k]
            
        return state_dict
    
    
    def load_state_dict(self, state_dict):
        keys = list(state_dict.keys())
        net_dict = OrderedDict()
        linear_dict = OrderedDict()
        
        for k in keys[:-1]:
            net_dict[k] = state_dict[k]
        for k in keys[-1:]:
            linear_dict[k] = state_dict[k]
        
        self.net.load_state_dict(net_dict)
        self.linear.load_state_dict(linear_dict)