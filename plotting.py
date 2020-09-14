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

def plot_bar(p0, p1, p2):
    N = 5
    labels = ['S', 'DD', 'DC', 'CD', 'CC']

    ind = np.arange(N) 
    width = 0.23     
    plt.bar(ind, p0, width, label='Default')
    plt.bar(ind + width, p1, width, label='Agent_1')
    plt.bar(ind + 2*width, p2, width, label='Agent_2')

    plt.ylabel('P(Defect)')
    plt.title('IPD Scores')

    plt.xticks(ind + width, ('S', 'DD', 'DC', 'CD', 'CC'))
    plt.legend(loc='best')
    return plt.figure()
