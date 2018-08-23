import torch
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

from constants import *
from utils import *

class PPO:
    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=A_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=C_LEARNING_RATE)

        self.batch_S = []
        self.batch_A = []
        self.batch_R = []

    def reset(self):
        self.batch_S = []
        self.batch_A = []
        self.batch_R = []

    def select_best_action(self, S):
        S = torch.FloatTensor(S)
        mu, log_sigma = self.actor(Variable(S))
        action = torch.normal(mu, torch.exp(log_sigma))
        return action

    def do_rollout(self, render=False):
        total_R = 0

        is_done = False
        S = self.env.reset()
        while not is_done:
            A = self.select_best_action(S).item()
            S_prime, R, is_done = self.env.take_one_step(A)

            self.batch_S.append(S)
            self.batch_A.append(A)
            self.batch_R.append(R)

            S = S_prime
            total_R += R

            if render:
                self.env.env.render()

        return total_R

    def compute_advantage(self, batch_V):
        ep_len = len(self.batch_R)

        v_target = torch.FloatTensor(ep_len)
        advantages = torch.FloatTensor(ep_len)

        prev_v_target = 0
        prev_v = 0
        prev_A = 0

        for i in reversed(range(ep_len)):
            v_target[i] = self.batch_R[i] + GAMMA * prev_v_target
            delta = self.batch_R[i] + GAMMA * prev_v - batch_V.data[i]
            advantages[i] = delta + GAMMA * TAU * prev_A

            prev_v_target = v_target[i]
            prev_v = batch_V.data[i]
            prev_A = advantages[i]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        return advantages, v_target

    def update_params(self):
        total_R = self.do_rollout()

        S = torch.FloatTensor(self.batch_S)
        A = torch.FloatTensor(np.asarray(self.batch_A))

        V_S = self.critic(Variable(S))
        advantages, v_target = self.compute_advantage(V_S)

        # loss function for value net
        L_vf = torch.mean(torch.pow(V_S - Variable(v_target), 2))

        # optimize the critic net
        self.critic_optimizer.zero_grad()
        L_vf.backward()
        self.critic_optimizer.step()

        # cast into variable
        A = Variable(A)

        # new log probability of the actions
        means, log_stddevs = self.actor(Variable(S))
        new_log_prob = get_gaussian_log(A, means, log_stddevs)

        # old log probability of the actions
        old_means, old_log_stddevs = self.actor(Variable(S), old=True)
        old_log_prob = get_gaussian_log(A, old_means, old_log_stddevs)

        # save the old actor
        self.actor.backup()

        # ratio of new and old policies
        ratio = torch.exp(new_log_prob - old_log_prob)

        # find clipped loss
        advantages = Variable(advantages)
        L_cpi = ratio * advantages
        clip_factor = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages
        L_clip = -torch.mean(torch.min(L_cpi, clip_factor))
        L_entropy = 0
        actor_loss = L_clip + L_entropy

        # optimize actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
        self.actor_optimizer.step()

        # prepare for next simulation
        self.reset()

        return total_R, L_clip, L_vf
