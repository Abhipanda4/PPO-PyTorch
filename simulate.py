import os

from model import *
from environment import Game
from ppo import PPO
from constants import *

env = Game()
n_input = env.state_dim

actor = Actor(n_input, N_HIDDEN)
critic = Critic(n_input, N_HIDDEN)

# retirieve previous saved model if exists
if os.path.exists(ACTOR_SAVE_PATH):
    print("Loading saved actor model...")
    actor.load_state_dict(torch.load(ACTOR_SAVE_PATH))
if os.path.exists(CRITIC_SAVE_PATH):
    print("Loading saved critic model...")
    critic.load_state_dict(torch.load(CRITIC_SAVE_PATH))

agent = PPO(env, actor, critic)
r = agent.do_rollout(render=True)
print(r)
agent.reset()
