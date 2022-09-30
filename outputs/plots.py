import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import argparse
### set the arguments
parser = argparse.ArgumentParser(description='Plots simulations')
parser.add_argument('--case', default='algo', type=str, help = 'takes one of the following values [iid, noniid, algo, alpha_var]' )
parser.add_argument('--plot_data', default='val_loss', type=str, help = 'takes one of the following values [omega, epsilon, val_loss, val_acc]' )
parser.add_argument('--nodes', default=5, type=int, help = 'takes one of the following values [5, 10]' )
args = parser.parse_args()


### load hyperparameters as per arguments
if args.nodes==5:
    if args.case == "iid":
        optimizers = ["ngc", "ngc"]
        networks = ["cganet", "cganet"] 
        nodes = [5, 5]
        lrs = [0.01, 0.01]
        gammas = [1.0, 1.0]
        alphas = [1.0, 1.0]
        skew = [0.0, 0.0]
        topologies = ["ring", "full"]
    elif args.case =="noniid":
        optimizers = ["ngc", "ngc"]
        networks = ["cganet", "cganet"] 
        nodes = [5, 5]
        lrs = [0.01, 0.01]
        gammas = [0.1, 1.0]
        alphas = [1.0, 1.0]
        skew = [1.0, 1.0]
        topologies = ["ring", "full"]
    elif args.case == "algo":
        optimizers = ["ngc",  "cga"] 
        networks = ["cganet", "cganet"] 
        nodes = [5, 5]
        lrs = [0.01, 0.01]
        gammas = [0.1, 0.1]
        alphas = [1.0, 1.0]
        skew = [1.0, 1.0]
        topologies = ["ring", "ring"]

elif  args.nodes==10:
    if args.case == "iid":
        optimizers = ["ngc", "ngc", "ngc"]
        networks = ["cganet", "cganet", "cganet"] 
        nodes = [10, 10, 10]
        lrs = [0.1, 0.1, 0.1]
        gammas = [1.0, 1.0, 1.0]
        alphas = [1.0, 1.0, 1.0]
        skew = [0.0, 0.0, 0.0]
        topologies = ["ring", "torus", "full"]
    elif args.case =="noniid":
        optimizers = ["ngc", "ngc", "ngc"]
        networks = ["cganet", "cganet", "cganet"] 
        nodes = [10, 10, 10]
        lrs = [0.1, 0.1, 0.1]
        gammas = [0.5, 0.5, 0.5]
        alphas = [1.0, 1.0, 1.0]
        skew = [1.0, 1.0, 1.0]
        topologies = ["ring", "torus", "full"]
    elif args.case == "algo":
        optimizers = ["ngc", "cga", "d-psgd"]
        networks = ["cganet","cganet", "cganet"]  
        nodes = [10, 10, 10]
        lrs = [0.1, 0.1, 0.1]
        gammas =  [0.5, 0.5, 1.0]
        alphas = [1.0, 1.0, 1.0]
        skew = [1.0, 1.0, 1.0]
        topologies = ["ring", "ring", "ring"]
    elif args.case == "alpha_var":
        optimizers = ["ngc", "ngc", "ngc", "ngc", "ngc"]
        networks = ["cganet", "cganet", "cganet", "cganet", "cganet"] 
        nodes = [10, 10, 10, 10, 10]
        lrs = [0.01, 0.01, 0.01, 0.01, 0.01]
        gammas = [1.0, 1.0, 0.5, 0.5, 0.25]
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        skew = [1.0, 1.0, 1.0, 1.0, 1.0]
        topologies = ["ring", "ring", "ring", "ring", "ring"]


### Read proper file and get proper data
epsilon = []
omega = []
val_acc = []
val_loss = []
labels = []
acc = []
for opt, net, n, lr, gm, a,sk, tp in zip(optimizers, networks, nodes, lrs, gammas, alphas, skew, topologies):
    if tp =="full":
        stp = "full"
        tp = "fully-connected"
    else:
        stp=tp
    try:
        file_name = f"{opt}_{net}_nodes_{n}_evonorm_lr_{lr}_gamma_{gm}_alpha_{a}_skew_{sk}_{stp}"
        try:
            excel_data= torch.load( os.path.join(file_name, "excel_data", "dict"))
            epsilon.append(np.mean(np.array(excel_data["epsilon"]), axis=0))
            omega.append(np.mean(np.array(excel_data["omega"]), axis=0))
        except:
            excel_data= torch.load( os.path.join(file_name, "excel_data", "dict"), map_location=torch.device('cpu'))
            for i,e in enumerate(excel_data["epsilon"]):
                excel_data["epsilon"][i] = torch.stack(e).cpu().numpy()
            for i,w in enumerate(excel_data["omega"]):
                excel_data["omega"][i] = torch.stack(w).cpu().numpy()
            epsilon.append(np.mean(np.array(excel_data["epsilon"]), axis=0))
            omega.append(np.mean(np.array(excel_data["omega"]), axis=0))
        val_acc.append(np.mean(np.array(excel_data["val_acc_list"]), axis=0))
        val_loss.append(np.mean(np.array(excel_data["val_loss_list"]), axis=0))
        acc.append(np.mean(np.array(excel_data["avg test acc final"])))
        if args.case=="iid" or args.case == "noniid":
            labels.append(tp)
        elif args.case == "algo" or args.case == "test":
            labels.append(opt)
        elif args.case == "alpha_var" :
            labels.append(a)
            
    except:
        continue
### plot the figure
fig,ax = plt.subplots()
fig.set_size_inches(10,8)
plt.style.use("seaborn-talk")
cs = sns.color_palette("muted")


if args.plot_data == "acc":
    ax.plot(labels, acc, color=cs[0],  marker='o', markersize=20, linestyle='--', linewidth=6)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=40)
    plt.xticks(np.arange(-0., 1.1, 0.25),fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_xlabel("Alpha", fontsize=40)
    plt.tight_layout()
    plt.savefig(args.case+"_"+args.plot_data+"_"+str(args.nodes)+".jpg")
else:
    for i, label in enumerate(labels):
        if args.plot_data == "val_loss":
            ax.plot(range(1, val_loss[i].shape[0]+1), val_loss[i], color=cs[i],  linewidth=6,label = label)
            ax.set_ylabel("Validation Loss", fontsize=40)
        elif args.plot_data == "val_acc":
            ax.plot(range(1, val_loss[i].shape[0]+1), val_acc[i], color=cs[i],  linewidth=6,label = label)
            ax.set_ylabel("Validation Accuracy", fontsize=40)
        elif args.plot_data == "omega":
            ax.plot(range(1, omega[i].shape[0]+1), omega[i], color=cs[i],  linewidth=6,label = label)
            ax.set_ylabel("Data-Variance Bias", fontsize=40)
        elif args.plot_data == "epsilon":
            ax.plot(range(1, epsilon[i].shape[0]+1), epsilon[i], color=cs[i],  linewidth=6,label = label)
            ax.set_ylabel("Model-Variance Bias", fontsize=40)

    ax.legend(fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_xlabel("Epochs", fontsize=40)
    plt.tight_layout()
    plt.savefig(args.case+"_"+args.plot_data+"_"+str(args.nodes)+".jpg")