import torch
import numpy as np
import matplotlib.pyplot as plt

from constants import *

def get_gaussian_log(x, mu, log_stddev):
    '''
    returns log probability of picking x
    from a gaussian distribution N(mu, stddev)
    '''
    # ignore constant since it will be cancelled while taking ratios
    log_prob = -log_stddev - (x - mu)**2 / (2 * torch.exp(log_stddev)**2)
    return log_prob

def plot_data(statistics):
    '''
    plots reward and loss graph for entire training
    '''
    x_axis = np.linspace(0, N_EPISODES, N_EPISODES // LOG_STEPS)
    plt.plot(x_axis, statistics["reward"])
    plt.title("Variation of mean rewards")
    plt.show()

    plt.plot(x_axis, statistics["val_loss"])
    plt.title("Variation of Critic Loss")
    plt.show()

    plt.plot(x_axis, statistics["policy_loss"])
    plt.title("Variation of Actor loss")
    plt.show()
