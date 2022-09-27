# %% 
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

### set the arguments
case = "algo"#[iid, noniid, algo, alpha_var]
plot_data = "val_acc"#[omega, epsilon, train_loss, val_loss, train_acc, val_acc, acc]
node = 10#[5, 10]

### load hyperparameters as per arguments
if node==5:
    if case == "iid":
        optimizers = ["ngc", "ngc"]
        networks = ["cganet", "cganet"] 
        nodes = [5, 5]
        lrs = [0.01, 0.01]
        gammas = [1.0, 1.0]
        alphas = [1.0, 1.0]
        skew = [0.0, 0.0]
        topologies = ["ring", "full"]
    elif case =="noniid":
        optimizers = ["ngc", "ngc"]
        networks = ["cganet", "cganet"] 
        nodes = [5, 5]
        lrs = [0.01, 0.01]
        gammas = [0.1, 1.0]
        alphas = [1.0, 1.0]
        skew = [1.0, 1.0]
        topologies = ["ring", "full"]
    elif case == "algo":
        optimizers = ["ngc",  "cga",  "d-psgd"] 
        networks = ["cganet", "cganet", "cganet"] 
        nodes = [5, 5, 5]
        lrs = [0.01, 0.01, 0.01]
        gammas = [0.1, 0.1, 0.1]
        alphas = [1.0, 1.0, 1.0]
        skew = [1.0, 1.0, 1.0]
        topologies = ["ring", "ring", "ring"]

elif  node==10:
    if case == "iid":
        optimizers = ["ngc", "ngc", "ngc"]
        networks = ["cganet", "cganet", "cganet"] 
        nodes = [10, 10, 10]
        lrs = [0.1, 0.1, 0.1]
        gammas = [1.0, 1.0, 1.0]
        alphas = [1.0, 1.0, 1.0]
        skew = [0.0, 0.0, 0.0]
        topologies = ["ring", "torus", "full"]
    elif case =="noniid":
        optimizers = ["ngc", "ngc", "ngc"]
        networks = ["cganet", "cganet", "cganet"] 
        nodes = [10, 10, 10]
        lrs = [0.1, 0.1, 0.1]
        gammas = [0.5, 0.5, 0.5]
        alphas = [1.0, 1.0, 1.0]
        skew = [1.0, 1.0, 1.0]
        topologies = ["ring", "torus", "full"]
    elif case == "algo":
        optimizers = ["ngc", "cga", "d-psgd"]
        networks = ["cganet","cganet", "cganet"]  
        nodes = [10, 10, 10]
        lrs = [0.1, 0.1, 0.1]
        gammas =  [0.5, 0.5, 1.0]
        alphas = [1.0, 1.0, 1.0]
        skew = [1.0, 1.0, 1.0]
        topologies = ["ring", "ring", "ring"]
    elif case == "alpha_var":
        optimizers = ["ngc", "ngc", "ngc", "ngc", "ngc"]
        networks = ["cganet", "cganet", "cganet", "cganet", "cganet"] 
        nodes = [10, 10, 10, 10, 10]
        lrs = [0.01, 0.01, 0.01, 0.01, 0.01]
        gammas = [1.0, 1.0, 0.5, 0.5, 0.25]
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        skew = [1.0, 1.0, 1.0, 1.0, 1.0]
        topologies = ["ring", "ring", "ring", "ring", "ring"]
        
    elif case == "test":
        optimizers = ["d-psgd","cga", "ngc", "compcga", "compngc"]
        networks = ["cganet","cganet","cganet","cganet","cganet"] 
        nodes = [10, 10, 10, 10, 10]
        lrs = [0.01, 0.01, 0.01, 0.01, 0.01]
        gammas = [0.5, 0.5, 0.5, 0.5, 0.5]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0]
        skew = [1.0, 1.0, 1.0, 1.0, 1.0]
        topologies = ["ring", "ring", "ring", "ring", "ring"]


### Read proper file and get proper data
epsilon = []
omega = []
train_acc = []
train_loss = []
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
        train_acc.append(np.mean(np.array(excel_data["train_acc_list"]), axis=0))
        train_loss.append(np.mean(np.array(excel_data["train_loss_list"]), axis=0))
        val_acc.append(np.mean(np.array(excel_data["val_acc_list"]), axis=0))
        val_loss.append(np.mean(np.array(excel_data["val_loss_list"]), axis=0))
        acc.append(np.mean(np.array(excel_data["avg test acc final"])))
        if case=="iid" or case == "noniid":
            labels.append(tp)
        elif case == "algo" or case == "test":
            labels.append(opt)
        elif case == "alpha_var" :
            labels.append(a)
            
    except:
        continue
### plot the figure
fig,ax = plt.subplots()
fig.set_size_inches(10,8)
plt.style.use("seaborn-talk")
cs = sns.color_palette("muted")


if plot_data == "acc":
    ax.plot(labels, acc, color=cs[0],  marker='o', markersize=20, linestyle='--', linewidth=6)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=40)
    plt.xticks(np.arange(-0., 1.1, 0.25),fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_xlabel("Alpha", fontsize=40)
    plt.tight_layout()
    plt.savefig(case+"_"+plot_data+"_"+str(node)+".jpg")
    plt.show()
else:
    for i, label in enumerate(labels):
        if plot_data == "train_loss":
            ax.plot(range(1, val_loss[i].shape[0]+1), train_loss[i], color=cs[i],  linewidth=6,label = label)
            ax.set_ylabel("Train Loss", fontsize=40)
        elif plot_data == "val_loss":
            ax.plot(range(1, val_loss[i].shape[0]+1), val_loss[i], color=cs[i],  linewidth=6,label = label)
            ax.set_ylabel("Validation Loss", fontsize=40)
        elif plot_data == "train_acc":
            ax.plot(range(1, val_loss[i].shape[0]+1), train_acc[i], color=cs[i],  linewidth=6,label = label)
            ax.set_ylabel("Train Accuracy", fontsize=40)
        elif plot_data == "val_acc":
            ax.plot(range(1, val_loss[i].shape[0]+1), val_acc[i], color=cs[i],  linewidth=6,label = label)
            ax.set_ylabel("Validation Accuracy", fontsize=40)
        elif plot_data == "omega":
            ax.plot(range(1, omega[i].shape[0]+1), omega[i], color=cs[i],  linewidth=6,label = label)
            ax.set_ylabel("Data-Variance Bias", fontsize=40)
        elif plot_data == "epsilon":
            ax.plot(range(1, epsilon[i].shape[0]+1), epsilon[i], color=cs[i],  linewidth=6,label = label)
            ax.set_ylabel("Model-Variance Bias", fontsize=40)

    ax.legend(fontsize=30)
    plt.xticks(np.arange(0, 301, 60),fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_xlabel("Epochs", fontsize=40)
    plt.tight_layout()
    plt.savefig(case+"_"+plot_data+"_"+str(node)+".jpg")
    plt.show()
# %%