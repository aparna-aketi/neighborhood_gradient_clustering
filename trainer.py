import argparse
import os
import shutil
import time
from tkinter import E
import numpy as np
import statistics 
import copy
import matplotlib.pyplot  as plt
#from models.cganet import cganet5

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
import torch.nn.functional as F
from math import ceil
import random


# Importing modules related to distributed processing
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.autograd import Variable
from torch.multiprocessing import spawn

###########
from gossip import GossipDataParallel
from gossip import RingGraph, GridGraph, FullGraph
from gossip import UniformMixing
from gossip import *
from models import *
from optimizers import *
from dataloader import *

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cganet', help = 'resnet or vgg or resquant' )
parser.add_argument('-depth', '--depth', default=20, type=int, help='depth of the resnet model')
parser.add_argument('--normtype',   default='evonorm', help = 'none or batchnorm or groupnorm or evonorm' )
parser.add_argument('--data-dir', dest='data_dir',    help='The directory used to save the trained models',   default='../../data', type=str)
parser.add_argument('--dataset', dest='dataset',     help='available datasets: cifar10, cifar100, imagenette', default='cifar10', type=str)
parser.add_argument('--skew', default=1.0, type=float,     help='obelongs to [0,1] where 0= completely iid and 1=completely non-iid')
parser.add_argument('--classes', default=10, type=int,     help='number of classes in the dataset')
parser.add_argument('-b', '--batch-size', default=160, type=int,  help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,     metavar='LR', help='initial learning rate')
parser.add_argument('--gamma',  default=0.1, type=float,  metavar='AR', help='averaging rate')
parser.add_argument('--alpha',  default=1.0, type=float, help='NGC mixing weight')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',     help='momentum')
parser.add_argument('--weight_decay', default=0.0, type=float,     help='weight_decay')
parser.add_argument('-world_size', '--world_size', default=10, type=int, help='total number of nodes')
parser.add_argument('--epochs', default=100, type=int, metavar='N',   help='number of total epochs to run')
parser.add_argument('--optimizer', default='ngc', type=str,  help='global optimizer = [d-psgd, cga, ngc, compcga, compngc]')
parser.add_argument('--graph', '-g',  default='ring', help = 'graph structure - [ring, torus]' )
parser.add_argument('--neighbors', default=2, type=int,     help='number of neighbors per node')
parser.add_argument('-d', '--devices', default=4, type=int, help='number of gpus/devices on the card')
parser.add_argument('-j', '--workers', default=4, type=int,  help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=321, type=int,   help='set seed')
parser.add_argument('--print-freq', '-p', default=100, type=int,  help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',    help='The directory used to save the trained models',   default='outputs', type=str)
parser.add_argument('--port', dest='port',   help='between 3000 to 65000',default='25500' , type=str)
parser.add_argument("--steplr", action="store_true", help="Uses step lr schedular for training.")
parser.add_argument('--nesterov', action='store_true', )
parser.add_argument('--qgm', action='store_true', help='quasi global momentum')
args = parser.parse_args()

# Check the save_dir exists or not
args.save_dir = os.path.join(args.save_dir, args.optimizer+"_"+args.arch+"_nodes_"+str(args.world_size)+"_"+ args.normtype+"_lr_"+ str(args.lr)+"_gamma_"+str(args.gamma)+"_alpha_"+str(args.alpha)+"_skew_"+str(args.skew)+"_"+args.graph )
if not os.path.exists(os.path.join(args.save_dir, "excel_data") ):
    os.makedirs(os.path.join(args.save_dir, "excel_data") )
torch.save(args, os.path.join(args.save_dir, "training_args.bin"))    

def run(rank, size):
    global args, best_prec1, global_steps
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:{}".format(rank%args.devices))
	##############
    best_prec1 = 0
    data_transferred = 0
    global_steps = 0
    
    
    if args.arch.lower()=='resnet':
        model = resnet(num_classes=args.classes, depth=args.depth, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'vgg11':
        model = vgg11(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'mobilenet':
        model = MobileNetV2(num_classes=args.classes, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'cganet':
        model = cganet5(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'lenet5':
        model = LeNet5()
    else:
        raise NotImplementedError
    
    if rank==0: 
        print(args)
        print('Printing model summary...')
        if args.dataset=="fmnist":
            print(summary(model, (1,28,28), batch_size=int(args.batch_size/size), device='cpu'))
        elif args.dataset=="imagenette_full":
            print(summary(model, (3, 224, 224), batch_size=int(args.batch_size/size), device='cpu'))
        elif args.dataset=="imagenet":
            print(summary(model, (3, 224, 224), batch_size=int(args.batch_size/size), device='cpu'))
        else: 
            print(summary(model, (3, 32, 32), batch_size=int(args.batch_size/size), device='cpu'))
        
    if args.optimizer.lower()=='cga':
        sender = CGA_sender(model, device)
    elif args.optimizer.lower()=="ngc":
        sender = NGC_sender(model, device)
    elif args.optimizer.lower()=='compcga':
        sender = CompCGA_sender(model, device)
    elif args.optimizer.lower()=="compngc":
        sender = CompNGC_sender(model, device)
    else:
        sender=None

    if args.graph.lower() == 'ring':
        graph = RingGraph(rank, size, args.devices, peers_per_itr=args.neighbors) #undirected ring structure => neighbors = 2 ; directed ring => neighbors=1
    elif args.graph.lower() == 'torus':   
        graph = GridGraph(rank, size, args.devices, peers_per_itr=args.neighbors) # torus graph structure
    elif args.graph.lower() == 'full':
        graph = FullGraph(rank, size, args.devices, peers_per_itr=args.world_size-1) # torus graph structure  
    elif args.graph.lower() == 'chain':   
        graph = ChainGraph(rank, size, args.devices, peers_per_itr=args.neighbors)
    else:
        raise NotImplementedError
    
    mixing = UniformMixing(graph, device)
    model = GossipDataParallel(model, 
				device_ids=[rank%args.devices],
				rank=rank,
				world_size=size,
				graph=graph, 
				mixing=mixing,
				comm_device=device, 
                level = 32,
                biased = False,
                eta = args.gamma,
                compress_ratio=0.0,
                compress_fn = 'quantize', 
                compress_op = 'top_k', 
                momentum=args.momentum,
                lr = args.lr) 
    model.to(device)
 
    train_loader, bsz_train = partition_trainDataset(args.dataset, args.data_dir, args.skew, args.seed, args.batch_size)
    val_loader, bsz_val     = test_Dataset(args.dataset, args.data_dir)
   
    if args.optimizer.lower()=='cga':
        receiver  = CGA_receiver(model, device, rank,  args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors)
    elif args.optimizer.lower()=='compcga':
        receiver = CompCGA_receiver(model, device, rank,  args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors)
    elif args.optimizer.lower()=='ngc':
        receiver  = NGC_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors, alpha = args.alpha)
    elif args.optimizer.lower()=='compngc':
        receiver = CompNGC_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors, alpha = args.alpha)
    else:
        receiver = DSGD_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay)
    

    optimizer = optim.SGD(model.parameters(), args.lr)
    
    criterion = nn.CrossEntropyLoss().to(device)
    if args.steplr:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma = 0.981, step_size=1)
    else:
        if args.dataset=='imagenet':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=[int(args.epochs*0.33), int(args.epochs*0.67), int(args.epochs*0.89)])
        else:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)])
            
    for epoch in range(0, args.epochs):  
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        model.block()
        dt, prec1, loss = train(train_loader, model, criterion, optimizer, epoch, bsz_train, optimizer.param_groups[0]['lr'], device, receiver, sender)
        data_transferred += dt
        if epoch>=0: lr_scheduler.step()
        prec1, loss = validate(val_loader, model, criterion, bsz_val,device, epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model_{}.th'.format(rank)))
      
    #############################
    average_parameters(model)
    print('Final test accuracy')
    prec1_final, _ = validate(val_loader, model, criterion, bsz_val,device, epoch)
    print("Rank : ", rank, "Data transferred(in GB) during training: ", data_transferred/1.0e9, "\n")
    #Store processed data
    torch.save((prec1, prec1_final, (data_transferred+dt)/1.0e9), os.path.join(args.save_dir, "excel_data","rank_{}.sp".format(rank)))

#def train(train_loader, model, criterion, optimizer, epoch, batch_size, writer, device):
def train(train_loader, model, criterion, optimizer, epoch, batch_size, lr, device, receiver=None, sender=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_transferred = 0

    # switch to train mode
    model.train()
    end = time.time()
    step = len(train_loader)*batch_size*epoch
    for i, (input, target) in enumerate(train_loader):
        #print(dist.get_rank(), torch.unique(target))
        data_time.update(time.time() - end)
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)
        # gossip the weights
        _, amt_data_transfer, cross_weights = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr)
        data_transferred += amt_data_transfer
        # do global update (gossip average step) in the pre forward hook, 
        # then compute output in the forward pass
        output = model(input_var)
        loss = criterion(output, target_var)
        # compute gradient 
        loss.backward()

        if 'cga' in args.optimizer.lower() or 'ngc' in args.optimizer.lower():
            #send and recieve cross gradients
            cross_grad, ref_buf                       = sender(cross_weights, input_var, target_var) 
            cross_grad_copy                           = copy.deepcopy(cross_grad)
            _, amt_data_transfer, recieved_cross_grad = model.transfer_additional(cross_grad)
            receiver(recieved_cross_grad, cross_grad_copy, ref_buf)
            data_transferred    +=amt_data_transfer
            #project the gradients
            receiver.project_gradients(lr)
        elif args.optimizer.lower() == 'd-psgd':
            receiver.update_gradients(lr)

        # do local update
        optimizer.step()
        #zero out the gradients
        optimizer.zero_grad() 
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Rank: {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      dist.get_rank(), epoch, i, len(train_loader),  batch_time=batch_time,
                      loss=losses, top1=top1))
        step += batch_size 
    return data_transferred, top1.avg, losses.avg


def validate(val_loader, model, criterion, batch_size, device, epoch=0):
#def validate(val_loader, model, criterion, batch_size, writer, device, epoch=0):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    step = len(val_loader)*batch_size*epoch
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var, target_var = Variable(input).to(device), Variable(target).to(device)
            # compute output and loss
            output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Rank: {0}\t'
                      'Test: [{1}/{2}]\t'
                      #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          dist.get_rank(),i, len(val_loader), 
                          #batch_time=batch_time, 
                          loss=losses,
                          top1=top1))
            step += batch_size
    print('Rank:{0}, Prec@1 {top1.avg:.3f}'.format(dist.get_rank(),top1=top1))
    return top1.avg, losses.avg

def average_parameters(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def flatten_tensors(tensors):
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat

def average_parameters(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

def init_process(rank, size, fn, backend='nccl'):
    """Initialize distributed enviornment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,size)

if __name__ == '__main__':
    size = args.world_size
    
    spawn(init_process, args=(size,run), nprocs=size, join=True)
    #read stored data
    excel_data = {
        'data': args.dataset,
        "graph" : args.graph,
        "nodes": size,
        'arch': args.arch,
        "norm" : args.normtype,
        'depth':args.depth,
        'optimizer' : args.optimizer,
        "learning rate": args.lr,
        "momentum":args.momentum,
        "qgm":args.qgm,
        "nesterov":args.nesterov,
        "weight_decay":args.weight_decay,
        "skew" : args.skew,
        "gamma" : args.gamma,
        "alpha" : args.alpha,
        "epochs": args.epochs,
        "avg test acc":[0.0 for _ in range(size)],
        "avg test acc final":[0.0 for _ in range(size)],
        "data transferred": [0.0 for _ in range(size)],
         "seed" :args.seed,
         }
         
    for i in range(size):
        acc, acc_final, d_tfr = torch.load(os.path.join( args.save_dir, "excel_data","rank_{}.sp".format(i) ))
        excel_data["avg test acc"][i] = acc
        excel_data["avg test acc final"][i] = acc_final
        excel_data["data transferred"][i] = d_tfr
        
    torch.save(excel_data, os.path.join(args.save_dir, "excel_data","dict"))
    #print(excel_data)
    