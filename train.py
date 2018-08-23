import torch

import os
import pickle

from constants import *
from environment import Game
from model import Actor, Critic
from ppo import PPO
from utils import plot_data

env = Game()
n_input = env.state_dim

actor = Actor(n_input, N_HIDDEN)
critic = Critic(n_input, N_HIDDEN)

# retrieve previous saved model if exists
if os.path.exists(ACTOR_SAVE_PATH):
    print("Loading saved actor model...")
    actor.load_state_dict(torch.load(ACTOR_SAVE_PATH))
if os.path.exists(CRITIC_SAVE_PATH):
    print("Loading saved critic model...")
    critic.load_state_dict(torch.load(CRITIC_SAVE_PATH))

ppo_agent = PPO(env, actor, critic)

statistics = {
    'reward': [],
    'val_loss': [],
    'policy_loss': [],
}

total_R = 0
for i in range(0, N_EPISODES):
    r, policy_loss, val_loss = ppo_agent.update_params()
    total_R += r

    # log data onto stdout
    if (i + 1) % LOG_STEPS == 1:
        mean_R = total_R / (i + 1)
        print("Episode: %d, Reward: %.8f, Value loss: [%.8f], Policy Loss: [%.8f]" %(i, mean_R, val_loss, policy_loss))
        # save statistics
        statistics['reward'].append(mean_R)
        statistics['val_loss'].append(val_loss)
        statistics['policy_loss'].append(policy_loss)

    # save models and statistics
    # if (i + 1) % SAVE_STEPS == 0:
        # torch.save(ppo_agent.actor.state_dict(), ACTOR_SAVE_PATH)
        # torch.save(ppo_agent.critic.state_dict(), CRITIC_SAVE_PATH)

plot_data(statistics)
