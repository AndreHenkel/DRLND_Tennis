import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline


from unityagents import UnityEnvironment
from drl.agent import Agent

# get environment
env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

agent = Agent(state_size=state_size, action_size=action_size, random_seed=1337)
agent.load(torch.load("checkpoints/checkpoint_actor_v2_solved.pth"),
           torch.load("checkpoints/checkpoint_critic_v2_solved.pth"));

print_every=100

scores_deque = deque(maxlen=print_every)
scores_total = []
deque_history = []
highest_score = 0.0
for i_episode in range(1, 300 + 1):
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)
    for t in range(10000):
        action = agent.act(states[0])
        actions = [action, action]
        # get new state & reward & done_information
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished

        states = next_states
        scores += rewards
        if np.any(dones):
            break

    deque_history.append(np.mean(scores_deque))
    scores_total.append(scores)
    scores_deque.append(np.max(scores))
    #scores.append(scores)
    if np.max(scores) > highest_score:
        highest_score = np.max(scores)

    print('Episode {}: Score: {:.2f}. Highest: {:.2f}. Average over {} steps is {:.2f}'.format(i_episode, np.max(scores), highest_score, print_every, np.mean(scores_deque)), end="\r")