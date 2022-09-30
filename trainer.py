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
from gossip_ds import GossipDataParallel
from gossip_ds import RingGraph, GridGraph, FullGraph
from gossip_ds import UniformMixing
from gossip_ds import *
from models import *
import xlsxwriter

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cganet', help = 'resnet or vgg or resquant' )
parser.add_argument('-depth', '--depth', default=20, type=int, help='depth of the resnet model')
parser.add_argument('--normtype',   default='evonorm', help = 'none or batchnorm or groupnorm or evonorm' )
parser.add_argument('--data-dir', dest='data_dir',    help='The directory used to save the trained models',   default='../data', type=str)
parser.add_argument('--dataset', dest='dataset',     help='available datasets: cifar10, cifar100, imagenette', default='cifar10', type=str)
parser.add_argument('--skew', default=1.0, type=float,     help='obelongs to [0,1] where 0= completely iid and 1=completely non-iid')
parser.add_argument('--classes', default=10, type=int,     help='number of classes in the dataset')
parser.add_argument('-b', '--batch-size', default=160, type=int,  help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,     metavar='LR', help='initial learning rate')
parser.add_argument('--gamma',  default=1.0, type=float,  metavar='AR', help='averaging rate')
parser.add_argument('--alpha',  default=1.0, type=float, help='NGC mixing weight')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',     help='momentum')
parser.add_argument('-world_size', '--world_size', default=5, type=int, help='total number of nodes')
parser.add_argument('--epochs', default=100, type=int, metavar='N',   help='number of total epochs to run')
parser.add_argument('--optimizer', default='ngc', type=str,  help='global optimizer = [d-psgd, cga, ngc, compcga, compngc]')
parser.add_argument('--graph', '-g',  default='ring', help = 'graph structure - [ring, torus]' )
parser.add_argument('--neighbors', default=2, type=int,     help='number of neighbors per node')
parser.add_argument('-d', '--devices', default=4, type=int, help='number of gpus/devices on the card')
parser.add_argument('-j', '--workers', default=4, type=int,  help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=1234, type=int,   help='set seed')
parser.add_argument('--run_no', default=1, type=str, help='parallel run number, models saved as model_{rank}_{run_no}.th')
parser.add_argument('--print-freq', '-p', default=100, type=int,  help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',    help='The directory used to save the trained models',   default='outputs', type=str)
parser.add_argument('--port', dest='port',   help='between 3000 to 65000',default='25500' , type=str)
parser.add_argument('--save-every', dest='save_every',  help='Saves checkpoints at every specified number of epochs',  type=int, default=5)
parser.add_argument("--steplr", action="store_true", help="Uses step lr schedular for training.")
args = parser.parse_args()

# Check the save_dir exists or not
args.save_dir = os.path.join(args.save_dir, args.optimizer+"_"+args.arch+"_nodes_"+str(args.world_size)+"_"+ args.normtype+"_lr_"+ str(args.lr)+"_gamma_"+str(args.gamma)+"_alpha_"+str(args.alpha)+"_skew_"+str(args.skew)+"_"+args.graph )
if not os.path.exists(os.path.join(args.save_dir, "excel_data") ):
    os.makedirs(os.path.join(args.save_dir, "excel_data") )
torch.save(args, os.path.join(args.save_dir, "training_args.bin"))    

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

def skew_sort(indices, skew, classes, class_size, seed):
    # skew belongs to [0,1]
    rng = random.Random()
    rng.seed(seed)
    class_indices = {}
    for i in range(0, classes):
        class_indices[i]=indices[0:class_size[i]]
        indices = indices[class_size[i]:]
    random_indices = []
    sorted_indices = []
    for i in range(0, classes):
        sorted_size    = int(skew*class_size[i])
        sorted_indices = sorted_indices + class_indices[i][0:sorted_size]
        random_indices = random_indices + class_indices[i][sorted_size:]
    rng.shuffle(random_indices)
    return random_indices, sorted_indices
            
    
class DataPartitioner(object):
    """ Partitions a dataset into different chunks"""
    def __init__(self, data, sizes, skew, classes, class_size, seed, device):
        
        self.data = data
        self.partitions = []
        data_len = len(data)
        dataset = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False, num_workers=2)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(dataset):
              labels = labels+targets.tolist()
        #labels  = [data[i][1] for i in range(0, data_len)]
        sort_index = np.argsort(np.array(labels))
        indices = sort_index.tolist()
        indices_rand, indices = skew_sort(indices, skew=skew, classes=classes, class_size=class_size, seed=seed)
        
        for i, frac in enumerate(sizes):
            if skew==1:
                part_len = int(frac*data_len)
                self.partitions.append(indices[0:part_len])
                if len(sizes)>10 and i<10:
                    #print('here', i, len(sizes), len(indices))
                    indices = indices[2*part_len:]+indices[part_len:2*part_len]
                else:
                    indices = indices[part_len:]
            elif skew==0:
                part_len = int(frac*data_len)
                self.partitions.append(indices_rand[0:part_len])
                indices_rand = indices_rand[part_len:] 
            else:
                part_len = int(frac*data_len*skew); 
                part_len_rand = int(frac*data_len*(1-skew))
                part_ind = indices[0:part_len]+indices_rand[0:part_len_rand]
                self.partitions.append(part_ind)
                indices = indices[part_len:]
                indices_rand = indices_rand[part_len_rand:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    
def partition_trainDataset(device):
    """Partitioning dataset""" 
    if args.dataset == 'cifar10':
        normalize   = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        classes    = 10
        class_size = {x:5000 for x in range(10)}

        dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    elif args.dataset == 'cifar100':
        normalize  = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        classes    = 100
        class_size = {x:500 for x in range(100)}

        dataset = datasets.CIFAR100(root=args.data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    elif args.dataset == 'imagenette':
        normalize  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        classes    = 10
        class_size = {0: 963, 1: 955, 2: 993, 3: 858, 4: 941, 5: 956, 6: 961, 7: 931, 8: 951, 9: 960}

        data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), normalize,])

        data_dir = args.data_dir

        dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
                  
       
    size = dist.get_world_size()
    #print(size)
    bsz = int((args.batch_size) / float(size))
    
    partition_sizes = [1.0/size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes, skew=args.skew, classes=classes, class_size=class_size, seed=args.seed, device=device)

    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=2)
    return train_set, bsz

def train_analysis_Dataset():
    if args.dataset=='cifar10':
        normalize   = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        classes    = 10
        class_size = {x:5000 for x in range(10)}

        dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    train_bsz = 64
    train_set = torch.utils.data.DataLoader(dataset, batch_size=train_bsz, shuffle=False, num_workers=2)
    return train_set

def test_Dataset():
  
    if args.dataset=='cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset=='cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(root=args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset == 'imagenette':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), normalize,])

        data_dir = args.data_dir

        dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),  data_transforms)

    val_bsz = 128
    val_set = torch.utils.data.DataLoader(dataset, batch_size=val_bsz, shuffle=False, num_workers=2)
    return val_set, val_bsz


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
        model = MobileNetV2(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'cganet':
        model = cganet5(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
    else:
        raise NotImplementedError
    
    if args.optimizer.lower()=='cga':
        sender = CGA_sender(model, device)
    elif args.optimizer.lower()=="ngc":
        sender = NGC_sender(model, device)
    elif args.optimizer.lower()=='compcga':
        sender = CompCGA_sender(model, device)
    elif args.optimizer.lower()=="compngc":
        sender = CompNGC_sender(model, device)
    elif args.optimizer.lower()=="d-psgd":
        sender = NGC_sender(model, device)
    else:
        sender=None
    
    if rank==0:
        print(args)
        print('Printing model summary...')
        if 'cifar' in args.dataset: print(summary(model, (3, 32, 32), batch_size=int(args.batch_size/size), device='cpu'))
        else: print(summary(model, (3, 224, 224), batch_size=int(args.batch_size/size), device='cpu'))
        

    if args.graph.lower() == 'ring':
        graph = RingGraph(rank, size, args.devices, peers_per_itr=args.neighbors) #undirected ring structure => neighbors = 2 ; directed ring => neighbors=1
    elif args.graph.lower() == 'torus':   
        graph = GridGraph(rank, size, args.devices, peers_per_itr=args.neighbors) # torus graph structure
    elif args.graph.lower() == 'full':
        graph = FullGraph(rank, size, args.devices, peers_per_itr=args.world_size-1) # torus graph structure  
    else:
        raise NotImplementedError
    #graph = BipartiteGraph(rank, size, args.devices, peers_per_itr=int(args.world_size/2)) #undirected bipartite structure, use only for even world size
   
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
    train_loader, bsz_train = partition_trainDataset(device=device)
    val_loader, bsz_val     = test_Dataset()
   
    if args.optimizer.lower()=='cga':
        receiver  = CGA_receiver(model, device, rank, args.momentum, neighbors=args.neighbors)
    elif args.optimizer.lower()=='compcga':
        receiver = CompCGA_receiver(model, device, rank, args.momentum, neighbors=args.neighbors)
    elif args.optimizer.lower()=='ngc':
        receiver  = NGC_receiver(model, device, rank, args.momentum, neighbors=args.neighbors, alpha = args.alpha)
    elif args.optimizer.lower()=='compngc':
        receiver = CompNGC_receiver(model, device, rank, args.momentum, neighbors=args.neighbors, alpha = args.alpha)
    elif args.optimizer.lower()=='d-psgd':
        receiver = NGC_receiver(model, device, rank, args.momentum, neighbors=args.neighbors, alpha = args.alpha)
    else:
        receiver = None
    
    
    # define loss function (criterion) and nvidia-smi optimizer
    if args.optimizer.lower()=='d-psgd':
        optimizer = optim.SGD(model.parameters(), args.lr, momentum = args.momentum, nesterov=False)
    else:
        optimizer = optim.SGD(model.parameters(), args.lr)
    if rank==0: print(optimizer)
    criterion = nn.CrossEntropyLoss().to(device)
    if args.steplr:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma = 0.981, step_size=1)
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)])
    
    for i, (input, target) in enumerate(train_loader):
        input_var, _ = Variable(input).to(device), Variable(target).to(device)
        if i ==0:
            inp_batch   = copy.deepcopy(input_var)
        elif inp_batch.size(0)<128:
            inp_batch = torch.cat((inp_batch, copy.deepcopy(input_var)), dim=0)
    epsilon_list = []
    omega_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(0, args.epochs):  
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        model.block()
        dt, global_steps, epsilon, omega, acc, loss = train(train_loader, model, criterion, optimizer, epoch, bsz_train, optimizer.param_groups[0]['lr'], device, rank, inp_batch, global_steps, receiver, sender)
        data_transferred += dt
        epsilon_list.append(epsilon)
        omega_list.append(omega)
        if epoch>=0: lr_scheduler.step()
        prec1, loss = validate(val_loader, model, criterion, bsz_val,device, epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        val_loss_list.append(loss)
        val_acc_list.append(prec1)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model_{}_{}.th'.format(rank, args.run_no)))
      
    #############################
    dt = gossip_avg(train_loader, model, criterion, optimizer, epoch, bsz_train, optimizer.param_groups[0]['lr'], device, rank)
    print('Final test accuracy')
    prec1_final, _ = validate(val_loader, model, criterion, bsz_val,device, epoch, True, args.classes, return_classwise=False)
    print("Rank : ", rank, "Data transferred(in GB) during training: ", data_transferred/1.0e9, "Data transferred(in GB) in final gossip averaging rounds: ", dt/1.0e9, "\n")
    #Store processed data
    torch.save((prec1, prec1_final, (data_transferred+dt)/1.0e9, epsilon_list, omega_list, val_acc_list, val_loss_list), os.path.join(args.save_dir, "excel_data","rank_{}.sp".format(rank)))


#def train(train_loader, model, criterion, optimizer, epoch, batch_size, writer, device):
def train(train_loader, model, criterion, optimizer, epoch, batch_size, lr, device, rank, inp_batch=None, global_steps=0, receiver=None, sender=None):
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
    epsilon = 0.0
    omega = 0.0
    for i, (input, target) in enumerate(train_loader):
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
        self_gradients = []
        for _, param in model.named_parameters():
            if param.requires_grad:
                self_gradients.append(param.grad.data)
        self_grads = flatten_tensors(self_gradients).to(device)
        
        #send and recieve cross gradients
        cross_grad, ref_buf                   = sender(cross_weights, input_var, target_var) 
        cross_grad_copy = copy.deepcopy(cross_grad)
        _, amt_data_transfer, recieved_cross_grad = model.transfer_additional(cross_grad)
        e,o=receiver(recieved_cross_grad, cross_grad_copy, ref_buf, self_grads)
        data_transferred                   +=amt_data_transfer
        epsilon += e.data.item()
        omega   += o.data.item()

        if 'cga' in args.optimizer.lower() or 'ngc' in args.optimizer.lower():
            #project the gradients
            receiver.project_gradients()

        # do local update
        optimizer.step()
        global_steps+=1
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
    return data_transferred, global_steps, epsilon/float(len(train_loader)+1), omega/float(len(train_loader)+1), top1.avg, losses.avg

def gossip_avg(train_loader, model, criterion, optimizer, epoch, batch_size, lr, device, rank):
    """
       This function runs only gossip averaging for 50 iterations without local sgd updates - used to obtain the average model
    """
    data_transferred = 0 
    n = 50
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.zero_grad()
        _, amt_data_transfer, _ = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr)
        data_transferred += amt_data_transfer
        if i==n: break
    return data_transferred

def validate(val_loader, model, criterion, batch_size, device, epoch=0, class_wise=False, list_of_classes=10, return_classwise=False):
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
    acc         = [0 for c in range(list_of_classes)]
    class_count = [0 for c in range(list_of_classes)]

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
            if class_wise:
                _, preds = torch.max(output.data, 1)
                for c in range(list_of_classes):
                    acc[c] += ((preds == target_var) * (target_var == c)).sum().float() 
                    class_count[c] += (target_var == c).sum()
                

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
    if class_wise:
        for c in range(list_of_classes):
            acc[c] = (acc[c].cpu().numpy()/class_count[c].cpu().numpy())*100
        print('Class-wise accuracy for rank {} is '.format(dist.get_rank()), acc)
    if return_classwise:
        return top1.avg, acc
    return top1.avg, losses.avg


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
        'Algo': 'DS',
        'data': args.dataset,
        'arch': args.arch,
        'optimizer' : args.optimizer,
        "momentum":args.momentum,
        "learning rate": args.lr,
        "gamma" : args.gamma,
        "alpha" : args.alpha,
        "graph" : args.graph,
        "skew" : args.skew,
        "norm" : args.normtype,
        "epochs": args.epochs,
        "nodes": size,
        "avg test acc":[0.0 for _ in range(size)],
        "avg test acc final":[0.0 for _ in range(size)],
        "data transferred": [0.0 for _ in range(size)],
        "epsilon": [],
        "omega":[],
        "val_acc_list":[],
        "val_loss_list":[],
        "seed" :args.seed,
        'depth':args.depth
         }
    for i in range(size):
        acc, acc_final, d_tfr, epsilon_list, omega_list, val_acc_list, val_loss_list = torch.load(os.path.join( args.save_dir, "excel_data","rank_{}.sp".format(i) ))
        excel_data["avg test acc"][i] = acc
        excel_data["avg test acc final"][i] = acc_final
        excel_data["data transferred"][i] = d_tfr
        excel_data["epsilon"].append(epsilon_list)
        excel_data["omega"].append(omega_list)
        excel_data["val_acc_list"].append(val_acc_list)
        excel_data["val_loss_list"].append(val_loss_list)
        
    torch.save(excel_data, os.path.join(args.save_dir, "excel_data","dict"))
    #print(excel_data)
    

