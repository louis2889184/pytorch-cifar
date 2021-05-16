import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def jointly_draw_learning_curves():
    folders = [
        "baseline_accs.bin",
        "default_accs.bin",
        "default_accs_230.bin",
        "default_accs_lr003.bin",
        "default_accs_230_lr003.bin",
    ]

    fig, axs = plt.subplots(1, 1)
    # fig.set_size_inches(10.5, 14.5)

    colors = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:purple"]
    names = ["Densely Updated single model (lr=0.1)", 
        "Sparsely Updated single model (lr=0.1)", 
        "Sparsely Updated single model (lr=0.1)", 
        "Sparsely Updated single model (lr=0.03)",
        "Sparsely Updated single model (lr=0.03)"
    ]

    for folder, color, name in zip(folders, colors, names):
        accs = torch.load(folder)

        train_accs = accs["train_accs"]
        eval_accs = accs["test_accs"]

        iters = range(len(train_accs))
        
        ax1 = axs
        # ax1.plot(iters, train_accs, color=color, label=f"{name} (Train)")
        ax1.plot(iters, eval_accs, color=color, label=f"{name} (Evaluation)", linestyle="--")
        ax1.set_ylabel('Accuracy')
        ax1.grid(linewidth=2, linestyle=":")

        ax1.legend()

    plt.savefig("img/learning_curve.png")

if __name__ == "__main__":
    jointly_draw_learning_curves()