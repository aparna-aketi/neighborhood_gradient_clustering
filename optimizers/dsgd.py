
from scipy.linalg import orth
import torch
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors, ForkedPdb
from collections import defaultdict

class DSGD_receiver():
    def __init__(self, model, device, rank, lr, momentum, qgm, nesterov=True, weight_decay=0):
        self.model         = model
        self.rank          = rank
        self.device        = device
        self.momentum      = momentum
        self.lr            = lr
        self.nesterov      = nesterov
        self.qgm           = qgm
        self.weight_decay  = weight_decay
        self.momentum_buff = []
        self.prev_params   = []
        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))
            self.prev_params.append(copy.deepcopy(param.data))
    

    def update_gradients(self, lr):
        """
            Returns
                applies the changes to the model
        """
        ### Applies weight_decay
        for name, p in self.model.module.named_parameters():
            if p.requires_grad:
                if self.weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)
        
        #apply momentum
        if self.momentum!=0:
            if self.qgm:
                for p, p_prev, buf in zip(self.model.module.parameters(), self.prev_params, self.momentum_buff):
                    buf.mul_(self.momentum).add_(p_prev.data-p.data, alpha=(1.0-self.momentum)/self.lr) #m_hat
                    mom_buff = copy.deepcopy(buf)
                    mom_buff.mul_(self.momentum).add_(p.grad.data) #m
                    if self.nesterov:
                        p.grad.data.add_(mom_buff, alpha=self.momentum) #nestrove momentum
                    else:
                        p.grad.data.copy_(mom_buff) 
                for p, p_prev in zip(self.model.module.parameters(), self.prev_params):
                    p_prev.data.copy_(p.data)
            else:
                for p, buf in zip(self.model.module.parameters(), self.momentum_buff):
                    buf.mul_(self.momentum).add_(p.grad.data)
                    if self.nesterov:
                        p.grad.data.add_(buf, alpha=self.momentum) #nestrove momentum
                    else:
                        p.grad.data.copy_(buf) 

        self.lr = lr
        
        
                
             
