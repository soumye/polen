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