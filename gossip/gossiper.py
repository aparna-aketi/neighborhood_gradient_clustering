
"""
Gossipers

:description: Gossiper's are designed for multi-peer communication (i.e., send
              and recv from multiple peers at each ieration)
"""

import torch
import torch.distributed as dist
import copy

from .graph_manager import GraphManager
from .mixing_manager import MixingManager
from .mixing_manager import UniformMixing
from .utils import (unsparsify_layerwise, flatten_tensors)



class dist_backend:
    UNDEFINED = -1
    TCP = 0
    MPI = 1
    GLOO = 2
    NCCL = 3


class Gossiper(object):
    """ Generic gossip averaging object for multi-peer communication """

    def __init__(self, msg, graph, device=None, mixing=None, logger=None,
                 rank=None, world_size=None, cg=False):
        """
        Initialize generic averaging class designed for multi-peer comms

        :param msg: (tensor) message used to initialize recv buffer
        :param device: (device) device on which to initialize recv buffer
        :param graph: (GraphManager) Subclass of GraphManager
        :param mixing: (MixingManager) Subclass of MixingManager
        :param logger: (python logger) module used to log results
        """

        self.logger = logger
        if rank is None or world_size is None:
            assert dist.is_initialized()
            # for now p2p communication only supported withed tcp and mpi
            assert dist._backend != dist_backend.GLOO
            assert dist._backend != dist_backend.NCCL
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        # graph topology properties
        self.rank = rank
        self.world_size = world_size
        assert isinstance(graph, GraphManager)
        self._graph_manager = graph
        self.peers_per_itr_device = torch.tensor(
            [self._graph_manager.peers_per_itr], device=device,
            dtype=msg.dtype)
        self.passive = self._graph_manager.is_passive()
        self.refresh_peers_(iteration=0, epoch=0, rotate=False)  # sets in- and out-peers attributes

        # mixing matrix
        if mixing is None:
            mixing = UniformMixing(self._graph_manager, device)
        assert isinstance(mixing, MixingManager)
        self._mixing_manager = mixing
        self.refresh_mixing_weights_()  # sets mixing-weights attribute


        # msg buffers used during send/recv
        self.device = device if device is not None else msg.device
        if not cg:
            self.out_msg_buffer = []
            self.in_msg_buffer = msg.clone().detach_().to(self.device)
            
            if self.device.type == 'cpu':
                try:
                    self.in_msg_buffer = self.in_msg_buffer.pin_memory()
                except Exception as e:
                    if self.logger is not None:
                        self.logger.error(e)
            self.placeholder   = self.in_msg_buffer.clone()
        else:
            self.out_msg_buffer = []
            self.in_msg_buffer = []

        self._pending_req = None

    @property
    def peers_per_itr(self):
        return self._graph_manager.peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v):
        self._graph_manager.peers_per_itr = v

    def refresh_peers_(self, iteration, epoch, rotate=None):
        """ Update in- and out-peers """
        # if rotate is None:
        #     rotate = True if self._graph_manager.is_dynamic_graph() else False
        # cannot cycle peers in a static graph
        #assert not (rotate and not self._graph_manager.is_dynamic_graph())
        if self._graph_manager.is_dynamic_graph():
            if iteration%1==0:
                rotate = True
        self.out_edges, self.in_edges = self._graph_manager.get_edges(rotate)


    def refresh_mixing_weights_(self):
        """ Update mixing-matrix weights """
        self.mixing_weights = self._mixing_manager.get_mixing_weights()
        

    def mix_out_msg_(self, out_msg, ps_weight):
        """ Returns a generator mixing messages on the fly """
        self.refresh_mixing_weights_()
        self.ps_weight = ps_weight

        # check whether or not we need to communicate ps_weight
        # if not self.regular:
        out_msg = torch.cat([out_msg, self.ps_weight.type(out_msg.dtype)])
        # check whether or not we need to create a buffer for each out-msg
        if self._mixing_manager.is_uniform():
            weight = self.mixing_weights['uniform']
            out_msg *= weight.type(out_msg.dtype)
            for _ in self.out_edges:
                yield out_msg
                
    def mix_self_msg_(self, out_msg, ps_weight):
        """ Returns a generator mixing messages on the fly """
        self.refresh_mixing_weights_()
        self.ps_weight = ps_weight
        out_msg = torch.cat([out_msg, self.ps_weight.type(out_msg.dtype)])
        if self._mixing_manager.is_uniform():
            weight = self.mixing_weights['uniform']-1.0
            out_msg *= weight.type(out_msg.dtype)
            yield out_msg
            
    def mix_neigh_msg_(self, params_list):
        """ Returns a generator mixing messages on the fly """
        out = {}
        self.refresh_mixing_weights_()
        if self._mixing_manager.is_uniform():
            weight = 1.0/self.mixing_weights['uniform']
            for k, v in params_list.items():
                out[k] = v.mul_(weight.type(v.dtype))
        return out

    def clean_msg_buffers_(self):
        """ Clean outgoing message buffer """
        msgs = []
        while len(self.out_msg_buffer) > 0:
            req, msg = self.out_msg_buffer.pop()
            req.wait()
            msgs.append(msg)
        while len(msgs) > 0:
            msg = msgs.pop()
            with torch.no_grad():
                msg.set_()

    def parse_in_msg_buffer(self, residual=False):
        """ Parse in-msg buffer and return msg and ps-weight separately """
        msg = self.in_msg_buffer
        if not self.regular:
            return msg.narrow(0, 0, len(msg) - 1), msg[-1]
        else:
            if residual:
                return msg, self.ps_weight * self.peers_per_itr_device
            else:
                return msg, torch.ones(1, device=self.device).type(msg.dtype)

    def mix(self):
        """ Single gossip step """
        raise NotImplementedError

class SGD_DS(Gossiper):

    def mix(self, out_msg, ref_msg, ps_weight, epoch, iteration, residual=False, uncompress=False, shapes =None):
        """ Consensus averaging step """
        # out_msg must be on the correct device
        assert out_msg.device.type == self.device.type
        if self.logger is not None:
            self.logger.debug('in/out -peers {}/{}'
                              .format(self.in_edges, self.out_edges))

        # prepare messages for gossip
        self.neigh_weights = {}
        out_copy    = copy.deepcopy(out_msg)
        ps_copy     = copy.deepcopy(ps_weight)
        placeholder = torch.zeros_like(torch.cat([out_msg,ps_weight]))
        
        if uncompress:
            values     = out_msg[:int(len(out_msg)/2)]
            indices    = out_msg[int(len(out_msg)/2):]
            mixed_out_msgs = self.mix_out_msg_(values, ps_weight)
        else:
            mixed_out_msgs = self.mix_out_msg_(out_msg, ps_weight)
       
        # non-blocking send
        # print(len(self.out_edges), len(self.in_edges))
        data_amt = 0
        for out_edge in self.out_edges:
            msg = next(mixed_out_msgs)
            if uncompress:
                msg = torch.cat([msg[0:(len(msg)-1)], indices, msg[-1].view(1)])
            assert self.rank == out_edge.src 
            req = dist.broadcast(tensor=msg, src=out_edge.src,group=out_edge.process_group, async_op=True)
            self.out_msg_buffer.append((req, msg))
            if uncompress:
                data_amt += ((msg.element_size()*(msg.nelement()/2))+msg.nelement())
            else:
                data_amt += msg.element_size()*msg.nelement()
        # blocking recv w/ some code optimization to avoid buffer prep overhead
        
        self.in_msg_buffer.zero_()
        if uncompress:
           msg_temp = unsparsify_layerwise(out_copy, shapes, ref_msg)
           msg_in   = flatten_tensors(msg_temp)
           mixed_self_msg  = self.mix_self_msg_(msg_in, ps_copy)
        else:
            mixed_self_msg = self.mix_self_msg_(out_copy, ps_copy)
                
            
        msg_mix = next(mixed_self_msg)
        self.in_msg_buffer.copy_(msg_mix) 
        for in_edge in self.in_edges:
            #print(in_edge.src, self.device)
            dist.broadcast(tensor=placeholder, src=in_edge.src, group=in_edge.process_group)
            if uncompress:
                in_weight     = placeholder[-1]
                received_temp = unsparsify_layerwise(placeholder.narrow(0, 0, len(placeholder) - 1), shapes, ref_msg)
                received_msg  = torch.cat([flatten_tensors(received_temp), in_weight.view(1)])
                self.placeholder.copy_(received_msg)
            else:
                self.placeholder.copy_(placeholder) 
                self.neigh_weights[in_edge.src] = copy.deepcopy(self.placeholder).narrow(0, 0, len(self.placeholder) - 1)
            self.in_msg_buffer.add_(self.placeholder)
                
        self.refresh_peers_(epoch=epoch, iteration=iteration)
        self.clean_msg_buffers_()
        in_msg = self.in_msg_buffer.narrow(0, 0, len(self.in_msg_buffer) - 1)
        updated_ps_weight = self.in_msg_buffer[-1]
        neigh_weights     = self.mix_neigh_msg_(self.neigh_weights)
        return in_msg, updated_ps_weight, neigh_weights, data_amt
    
class send_cg(Gossiper):

    def mix(self, out_msg):
        """ Consensus averaging step """
        # out_msg must be on the correct device
        #print(out_msg)
        
        keys = [k  for  k in  out_msg.keys()]
        assert out_msg[keys[0]].device.type == self.device.type
        if self.logger is not None:
            self.logger.debug('in/out -peers {}/{}'
                              .format(self.in_edges, self.out_edges))

        placeholder = torch.zeros_like(out_msg[keys[0]])
        
        data_amt = 0
        for out_edge in self.out_edges:
            assert self.rank == out_edge.src 
            req = dist.broadcast(tensor=out_msg[out_edge.dest], src=out_edge.src,group=out_edge.process_group, async_op=True)
            self.out_msg_buffer.append((req, out_msg[out_edge.dest]))
            data_amt += out_msg[out_edge.dest].element_size()*out_msg[out_edge.dest].nelement()
        
        self.in_msg_buffer = {}
        for i, in_edge in enumerate(self.in_edges):
            dist.broadcast(tensor=placeholder, src=in_edge.src, group=in_edge.process_group)
            self.in_msg_buffer[in_edge.src] = copy.deepcopy(placeholder)
        
        self.clean_msg_buffers_()
        return self.in_msg_buffer, data_amt

