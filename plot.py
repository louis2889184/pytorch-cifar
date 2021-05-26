import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def jointly_draw_learning_curves():
    folders = [
        "checkpoints_update1_pretrain0.bin",
        "checkpoints_update1_sparsity010.bin",
        "checkpoints_update1_sparsity100.bin",
        # "checkpoints_update1_pretrain25.bin",
    ]

    fig, axs = plt.subplots(1, 1)
    # fig.set_size_inches(10.5, 14.5)

    colors = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:purple"]
    names = [
        "Sparsely updated single model (0.005)", 
        "Sparsely updated single model (0.010)", 
        "Sparsely updated single model (0.100)", 
    ]

    for folder, color, name in zip(folders, colors, names):
        accs = torch.load(folder)

        train_accs = accs["train_accs"]
        eval_accs = accs["test_accs"]

        iters = range(len(train_accs))
        
        ax1 = axs
        # ax1.plot(iters, train_accs, color=color, label=f"{name} (Train)")
        ax1.plot(iters, eval_accs, color=color, label=f"{name}", linestyle="--")
        ax1.set_ylabel('Eval Accuracy')
        ax1.grid(linewidth=2, linestyle=":")

        ax1.legend()

    plt.savefig("img/learning_curve_sparsity.png")

if __name__ == "__main__":
    jointly_draw_learning_curves()