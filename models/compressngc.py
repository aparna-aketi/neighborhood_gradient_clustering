
from scipy.linalg import orth
import torch
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors, ForkedPdb
from .evonorm import EvoNormSample2d as evonorm_s0
from collections import defaultdict

def scaled_sign(x):
    """
    :param x: torch Tensor
    :return: The sign tensor scaled by it's L1 norm and divided by the number of elements
    """
    return x.norm(p=1) / x.nelement() * torch.sign(x)

def unscaled_sign(x):
    """
    This is the standard sign compression. It has been experimented to give worse test accuracies than the scaled
    counter part.
    :param x: torch Tensor
    :return: sign(tensor)
    """
    return torch.sign(x)

class CompNGC_sender():
    def __init__(self, true_model, device):
        """
            Args
                model: the model on the sender device
                device: device on which the model is 
                include_norm: includes norm weights to gpm computation 
        """
        self.model           = copy.deepcopy(true_model)
        self.model.train()
        self.model           = self.model.to(device)
        self.gradient_buffer = {}
        self.device          = device
        self.criterion       = torch.nn.CrossEntropyLoss().to(device)
        self.param_memory    = {}
        
        

    def _update_model(self, state_dict):
        """
            Args:
                state_dict: list of device parameters 
        """
        for w, p in zip(state_dict, self.model.parameters()):
                p.data.copy_(w.data)
        return

    def _accumulate_gradients(self, x, targets, rank):
        """
            Args:
                x: inputs for which the gradients have to be accumulated
                targets: class labels for x
        """
        output = self.model(x)
        self.model.zero_grad()
        loss   = self.criterion(output, targets)
        loss.backward()
        self._clear_gradient_buffer()
        count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                #compress cross gradients
                corrected_gradient = self.param_memory[rank][count]+param.grad.data
                corrected_gradient = scaled_sign(corrected_gradient)
                self.param_memory[rank][count] = self.param_memory[rank][count]+param.grad.data - corrected_gradient
                self.gradient_buffer[name] = corrected_gradient
                count+=1
        return self.gradient_buffer

    def _clear_gradient_buffer(self):
        self.gradient_buffer = {}
        return


    def _flatten_(self, G):
        """
            Args
                G: Input to be flattened
            Returns
                flattened tensor
        """
        grad = []
        for g in G.values():
            grad.append(g)
        return flatten_tensors(grad).to(self.device)

    def __call__(self, neighbor_weight, batch_x, targets):
        """
            Args
                neighbor_weight: weights of the neighbor models
                batches: Input batches to compute variance
            Returns
                flattened gradients for each neighbor
        """

        output = {}
        if not bool(self.param_memory):
            for rank, w in neighbor_weight.items():
                self.param_memory[rank] = []
                for _, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.param_memory[rank].append(torch.zeros_like(param.data))

        for rank, w in neighbor_weight.items():
            self._update_model(w)
            g = self._accumulate_gradients(batch_x, targets, rank)
            output[rank] = self._flatten_(g)
        return output, g


class CompNGC_receiver():
    def __init__(self, model, device, rank, momentum, neighbors=2, alpha=0.5):
        self.model         = model
        self.rank          = rank
        self.device        = device
        self.proj_grads    = {}
        self.old_v         = {}
        self.pi            = 1.0/(neighbors+1)      # !!! this has to updated. right now its hard coded for bidirectional ring topology with uniform weights
        self.alpha         = alpha
        self.eps           = 1e-12
        self.margin        = 0.5
        self.momentum      = momentum
        self.momentum_buff = []
        self.param_memory  = []
        for param in self.model.module.parameters():
            if param.requires_grad:
                self.momentum_buff.append(torch.zeros_like(param.data))
                self.param_memory.append(torch.zeros_like(param.data))
    
    def average_gradients(self, grad):

        new_grad = torch.zeros_like(grad[-1])
        for g in grad:
            new_grad +=self.pi*g

        return new_grad #Ftorch.clamp(new_grad,min=-1.0,max=1.0)
 

    def _unflatten_(self, flat_tensor, ref_buf):
        """
            Args
                flat_tensor: received flat tensor to be reshaped 
                ref_buf: reference buffer for computing unflattened shape
            Returns
                unflattened tensor based on reference tensor
        """
        ref  = []
        keys = []
        for key,val in ref_buf.items():
            ref.append(val)
            keys.append(key)
        unflat_tensor =  unflatten_tensors(flat_tensor, ref)
        X = {}
        for i, key in enumerate(keys):
            X[key] = unflat_tensor[i]
        return X

    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, ref_buf, self_grads=None):
        """
            Args
                flat_tensor: received flat tensor to be reshaped 
                ref_buf: reference buffer for computing unflattened shape
            Returns
                computes orthogonal projection space and stores in self.Z
        """
        ### Unflatten the neighbor grads
        epsilon = 0
        omega   = 0
        for rank, flat_tenor  in neighbor_grads_comm.items():
            omega += torch.norm(flat_tenor.data-self_grads.data, p=1)/flat_tenor.size(0)
            neighbor_grads_comm[rank] = self._unflatten_(flat_tenor, ref_buf)
        for rank, flat_tenor  in neighbor_grads_comp.items():
            epsilon += torch.norm(flat_tenor.data-self_grads.data, p=1)/flat_tenor.size(0)
            neighbor_grads_comp[rank] = self._unflatten_(flat_tenor, ref_buf)
        
        #get the projected gradients for each parameter
        count = 0 
        for name, self_params in self.model.module.named_parameters():
            if self_params.requires_grad:
                cross_grads_comm = []
                cross_grads_comp = []
                for rank, neigh_grad in neighbor_grads_comm.items():
                    cross_grads_comm.append(neigh_grad[name])
                #compress self gradients
                corrected_gradient = self.param_memory[count]+self_params.grad.data
                corrected_gradient = scaled_sign(corrected_gradient)
                self.param_memory[count] = self.param_memory[count]+self_params.grad.data - corrected_gradient
                cross_grads_comm.append(corrected_gradient)
                for rank, neigh_grad in neighbor_grads_comp.items():
                    cross_grads_comp.append(neigh_grad[name])
                cross_grads_comp.append(corrected_gradient) # added twice so we can use self.pi/2 as weight
                p_grads_comm  = self.average_gradients(cross_grads_comm)
                p_grads_comp  = self.average_gradients(cross_grads_comp)
                p_grads       = ((1-self.alpha)*p_grads_comp)+(self.alpha*p_grads_comm)
                self.proj_grads[name] = p_grads
                count+=1
        return epsilon*self.pi, omega*self.pi
                

    def project_gradients(self):
        """
            Returns
                applies the changes to the model
        """
        ### Applies the grad projections
        for name, p in self.model.module.named_parameters():
            if p.requires_grad:
                p.grad.data = self.proj_grads[name].data 
        
        #apply momentum
        if self.momentum!=0:
            for p, buf in zip(self.model.module.parameters(), self.momentum_buff):
                if p.requires_grad:
                    buf.mul_(self.momentum).add_(p.grad.data)
                    p.grad.data.add_(buf, alpha=self.momentum) #nestrove momentum
        
        
                
             
