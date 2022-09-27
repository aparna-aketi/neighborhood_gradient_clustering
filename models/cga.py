#reference: https://github.com/yasesf93/CrossGradientAggregation/blob/main/Optimizers/CGA.py
from scipy.linalg import orth
import torch
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors, ForkedPdb
from .evonorm import EvoNormSample2d as evonorm_s0
from collections import defaultdict
import quadprog

class CGA_sender():
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
        
        

    def _update_model(self, state_dict):
        """
            Args:
                state_dict: list of device parameters 
        """
        for w, p in zip(state_dict, self.model.parameters()):
                p.data.copy_(w.data)
        return

    def _accumulate_gradients(self, x, targets):
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
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.gradient_buffer[name] = param.grad.data
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
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            g = self._accumulate_gradients(batch_x, targets)
            output[rank] = self._flatten_(g)
        return output, g


class CGA_receiver():
    def __init__(self, model, device, rank, momentum, neighbors=2):
        self.model         = model
        self.rank          = rank
        self.device        = device
        self.proj_grads    = {}
        self.old_v         = {}
        self.pi            = 1.0/float(neighbors+1)      # !!! this has to updated. right now its hard coded for bidirectional ring topology with uniform weights
        self.eps           = 1e-12
        self.margin        = 0.5
        self.momentum      = momentum
        self.momentum_buff = []
        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))

        
            

    def compute_QP_projection(self, grad, self_rank, param_name):
        '''
        Reference: https://github.com/yasesf93/CrossGradientAggregation/blob/main/Optimizers/CGA.py

        Parameters
        ----------
        grad : list of Tensors
            list of gradients of all the neighbours including self
        self_rank : scalar
            The rank of the current node
        param_name : string
            The name of the parameter for the given gradients

        Returns
        -------
        Projected gradients of the given parameter

        '''
        except_triggered = False
        
        grad_flat = torch.zeros(*grad[0].flatten().size(), len(grad)).to(self.device)
        neigh_size = len(grad)
        gradsize   = grad[0].size()
        for i in range(neigh_size):
            grad_flat[:,i] = grad[i].flatten()
        self_index = neigh_size-1
            
        grad_flat = grad_flat.cpu().double().numpy()
        grad_flat = np.transpose(grad_flat)
        for i in range(neigh_size):
            grad_flat[i,:] = self.pi*grad_flat[i,:]
        grad_np = grad_flat[self_index,:]
        grad_flat = np.delete(grad_flat, (self_index), axis=0) #memory rows
        memory_np = grad_flat[~np.all(grad_flat==0,axis=1)]    #non zerow rows
        
        t = memory_np.shape[0]
        p = np.dot(memory_np, memory_np.transpose())
        p = 0.5*(p+p.transpose()) + self.eps*np.eye(t)
        q = -1*np.dot(memory_np, grad_np)
        G = np.eye(t)
        h = np.zeros(t) + self.margin
        try:
            v = quadprog.solve_qp(p, q, G,h)[0]
            self.old_v[param_name] = v
        except ValueError:
            except_triggered = True
            print('Handling ValueError', param_name)
            v = self.old_v[param_name]					#old_v_batch
            print('v',v)
        x = np.dot(v,memory_np)+grad_np
        grad = torch.Tensor(x).view(*gradsize).to(self.device)
        if except_triggered:
            print('grad:', grad.norm(2))
        return grad
    
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
        for name, self_params in self.model.module.named_parameters():
            if self_params.requires_grad:
                cross_grads_comm = []
                cross_grads_comp = []
                for rank, neigh_grad in neighbor_grads_comm.items():
                    cross_grads_comm.append(neigh_grad[name])
                cross_grads_comm.append(self_params.grad.data)
                p_grads = self.compute_QP_projection( cross_grads_comm, self.rank, name)
                self.proj_grads[name] = p_grads
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
                buf.mul_(self.momentum).add_(p.grad.data)
                p.grad.data.add_(buf, alpha=self.momentum) #nestrove momentum
        
        
                
             
