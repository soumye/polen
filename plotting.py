import matplotlib.pyplot as plt
import numpy as np

def plot(scores, label):
    """Plotting"""
    plt.plot(scores, 'b', label=str(label) + " lookaheads", linewidth=3.0)
    plt.legend()
    plt.xlabel('rollouts',fontsize=26)
    plt.ylabel('joint score',fontsize=26)
    plt.tight_layout()
    plt.show()

def plot_many(list_scores, args):
    """Plotting Multiple"""
    colors = ['b','c','m','r']
    for i, scores in enumerate(list_scores):
        plt.plot(scores, colors[i], label=str(i) + " lookaheads", linewidth=3.0)
    plt.legend()
    plt.xlabel('rollouts',fontsize=26)
    plt.ylabel('joint score',fontsize=26)
    plt.tight_layout()
    plt.show()