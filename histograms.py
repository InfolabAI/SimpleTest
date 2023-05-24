import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

HISTOGRAM_PATH = './histograms/'

def histogram(data1d):
    #plt.hist(data1d, bins=np.unique(data1d).shape[0], density=True, rwidth=0.1)
    plt.hist(data1d, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], density=True, rwidth=0.1)
    plt.savefig(HISTOGRAM_PATH + '1d.png')
    plt.close()

def histogram2d(data2d):
    plt.hist2d(data2d[:, 0], data2d[:, 1], bins=100, density=True)
    plt.savefig(HISTOGRAM_PATH + '2d.png')
    plt.close()

def test():
    data1d = np.array([0, 1, 2, 3, 4, 5, 1, 1, 2])
    data2d = np.random.randn(1000, 1000)
    histogram(data1d)
    histogram2d(data2d)

if __name__ == '__main__':
    test()  