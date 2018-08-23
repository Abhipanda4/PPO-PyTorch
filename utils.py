import torch
import matplotlib.pyplot as plt

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
    pass
