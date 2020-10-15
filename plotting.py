import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
import numpy as np
import io

def plot(scores, label, logdir):
    """Plotting"""
    plt.plot(scores, 'b', label=str(label) + " lookaheads", linewidth=3.0)
    plt.legend()
    plt.xlabel('rollouts',fontsize=26)
    plt.ylabel('joint score',fontsize=26)
    plt.tight_layout()
    plt.savefig(logdir + '/curve.png')
    plt.close()

def plot_many(list_scores, args, logdir):
    """Plotting Multiple"""
    colors = ['b','c','m','r']
    for i, scores in enumerate(list_scores):
        plt.plot(scores, colors[i], label=str(i) + " lookaheads", linewidth=3.0)
    plt.legend()
    plt.xlabel('rollouts',fontsize=26)
    plt.ylabel('joint score',fontsize=26)
    plt.tight_layout()
    plt.savefig(logdir + '/curve.png')
    plt.close()

def plot_bar(p0, p1, p2, logdir=None):
    plt.figure()
    N = 5
    labels = ['S', 'DD', 'DC', 'CD', 'CC']
    ind = np.arange(N) 
    width = 0.23     
    plt.bar(ind, p0, width, label='Default')
    plt.bar(ind + width, p1, width, label='Agent_1')
    plt.bar(ind + 2*width, p2, width, label='Agent_2')

    plt.ylabel('P(Defect)')
    plt.xlabel('Previous State')
    plt.title('IPD Defect Probs')

    plt.xticks(ind + width, ('S', 'DD', 'DC', 'CD', 'CC'))
    plt.legend(loc='best')
    plt.tight_layout()
    if logdir:
        plt.savefig(logdir + '/probs.png')
        plt.close()
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image = PIL.Image.open(buf)
        return ToTensor()(image)

def plot_scatter(p0, p1, p2, logdir=None):
    plt.figure()
    p2_ = [p2[0], p2[1], p2[3], p2[2], p2[4]]
    N = 5
    labels = ['S', 'DD', 'DC', 'CD', 'CC']
    colors = ['y','r','g','b','m']
    fig, ax = plt.subplots()
    # set axes range
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    for i, label in enumerate(labels):
        ax.scatter(1-p1[i], 1-p2_[i], c=colors[i], s=100, alpha=0.61, label=label, edgecolors='none')
    ax.legend(loc='best')
    ax.grid(True)
    plt.xlabel('P(Cooperation|State) Agent 1', fontsize=20)
    plt.ylabel('P(Cooperation|State) Agent 2', fontsize=20)
    plt.title('Cooperation Prob Plot')
    if logdir:
        plt.savefig(logdir + '/scatter_plot.png')
        plt.close()
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image = PIL.Image.open(buf)
        return ToTensor()(image)

def plot_scatter_many(p1s, p2s, label=''):
    plt.figure()
    labels = ['S', 'DD', 'DC', 'CD', 'CC']
    colors = ['y','r','g','b','m']
    idx_2 = [0,1,3,2,4]
    fig, ax = plt.subplots()
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    for i, label in enumerate(labels):
            x = [1-p1[i] for p1 in p1s]
            y = [1-p2[idx_2[i]] for p2 in p2s]
            ax.scatter(x, y, c=colors[i], s=100, alpha=0.5, label=label, edgecolors='none')
    ax.legend(loc='best')
    ax.grid(True)
    plt.xlabel('P(Cooperation|State) Agent 1', fontsize=20)
    plt.ylabel('P(Cooperation|State) Agent 2', fontsize=20)
    plt.title('Cooperation Prob Plot')
    plt.savefig('scatter' + label + '.png')
    plt.close()