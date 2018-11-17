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

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

print(states)

# trains only one agent
#def train_one():
# init
env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
agent = Agent(state_size=state_size, action_size=action_size, random_seed=1337)
#agent2= Agent(state_size=state_size, action_size=action_size, random_seed=42)

print("I'm fine")
def ddpg(n_episodes=2000, max_t=100, print_every=100):
    #scores_deque = deque(maxlen=print_every)
    scores_total = []
    highest_score = 0.0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)
        for t in range(max_t):
            action = agent.act(states[0])
            actions = [action, action]
            # get new state & reward & done_information
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished

            agent.step(states[0], actions[0], rewards[0], next_states[0], dones[0], 0)
            agent.step(states[0], actions[0], rewards[0], next_states[0], dones[0], 1)

            #agent2.step(states[1], actions[1], rewards[1], next_states[1], dones[1])

            states = next_states
            scores += rewards
            if np.any(dones):
                break

        scores_total.append(scores)
        #scores_deque.append(scores)
        #scores.append(scores)
        if np.max(scores) > highest_score:
            highest_score = np.max(scores)

        print('Score (max over agents) from episode {}: {:.2f}. Highest score: {:.2f}'.format(i_episode, np.max(scores), highest_score), end="\r")
        if i_episode % print_every == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoints/checkpoint_actor_v1.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoints/checkpoint_critic_v1.pth')
    return scores_total


scores_total = ddpg()
env.close()

scores_p1 = [[s[0]] for s in scores_total]
scores_p2 = [[s[1]] for s in scores_total]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores_p1) + 1), scores_p1)
plt.plot(np.arange(1, len(scores_p1) + 1), scores_p1)
plt.ylabel('Score')
plt.xlabel('Episode #')
fig.savefig('checkpoints/scores_v1.png')
plt.show()
