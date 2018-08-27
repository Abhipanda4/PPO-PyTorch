import torch

import os
import pickle
import numpy as np

from constants import *
from environment import Game
from model import Actor, Critic
from ppo import PPO
from utils import plot_data
from running_state import *
from replay_memory import *

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

running_state = ZFilter((2,), clip=5)

statistics = {
    'reward': [],
    'val_loss': [],
    'policy_loss': [],
}

best_reward = 0
for i in range(0, N_EPISODES):
    memory = Memory()
    num_steps = 0
    num_ep = 0
    reward_batch = 0

    while num_steps < BATCH_SIZE:
        S = env.reset()
        S = running_state(S)
        t = 0
        reward_sum = 0

        while True:
            t += 1

            A = ppo_agent.select_best_action(S)
            S_prime, R, is_done = env.take_one_step(A.item())

            reward_sum += R
            mask = 1 - int(is_done)

            memory.push(S, np.array([A.item()]), mask, R)

            if is_done:
                break

            S = running_state(S_prime)
            # S = S_prime

        num_steps += t
        num_ep += 1
        reward_batch += reward_sum

    reward_batch /= num_ep

    # The memory is now full of rollouts. Sample from memory and optimize
    batch = memory.sample()
    policy_loss, val_loss = ppo_agent.update_params(batch)


    # log data onto stdout
    if i == 0 or i % LOG_STEPS == 0:
        print("Episode: %d, Reward: %.8f, Value loss: [%.8f], Policy Loss: [%.8f]" %(i, reward_batch, val_loss, policy_loss))
        # save statistics
        statistics['reward'].append(reward_batch)
        statistics['val_loss'].append(val_loss)
        statistics['policy_loss'].append(policy_loss)

    # save models and statistics
    if reward_batch > best_reward:
        best_reward = reward_batch
        torch.save(ppo_agent.actor.state_dict(), ACTOR_SAVE_PATH)
        torch.save(ppo_agent.critic.state_dict(), CRITIC_SAVE_PATH)

plot_data(statistics)
