import numpy as np
import torch

class DRU:
    def __init__(self, sigma=2, hard=False):
        self.sigma = sigma # standard deviation of Gaussian noise applied by DRU 
        self.hard = hard # true if use hard discretization, soft approximation otherwise 

    def regularize(self, message): 
        m_reg = message + torch.randn(message.size()) * self.sigma # add noise to message 
        m_reg = torch.sigmoid(m_reg) 
        return m_reg

    def discretize(self, message):
        if self.hard: 
            return (message.gt(0.5).float() - 0.5).sign().float() 
        else: 
            scale = 2 * 20 
            return torch.sigmoid((message.gt(0.5).float() - 0.5) * scale) 

    def forward(self, message, mode): # mode = D for discretize / R for regularize 
        if mode=="R": 

            return self.regularize(message) # Dial used regularization during training 
        elif mode=="D": 
            return self.discretize(message) # Dial used discretization message during execution 

