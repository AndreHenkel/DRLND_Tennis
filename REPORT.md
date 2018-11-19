# Implementation
Epsilon with epsilon decay to increase exploration.
Shared Replay Buffer for both agents.
L2 Weight Decay.
Shared Actor Network and different Critic heads. One idea with this was, since the space observation between both agents only differ in one field and is mirrored,
that the agent learns to adapt to that fact, since all other moves are similar.
So the critics tell each agent whether is was a good or a bad move for them, but they both learn to adapt to both situations.

# Hyperparameters

BUFFER_SIZE     = int(1e5)  # replay buffer size

BATCH_SIZE      = 128       # minibatch size

GAMMA           = 0.99      # discount factor

TAU             = 1e-3      # for soft update of target parameters

LR_ACTOR        = 1e-4       learning rate of the actor

LR_CRITIC       = 1e-3      # learning rate of the critic

WEIGHT_DECAY    = 0.00001   # L2 weight decay


n_episodes  = 2000          # maximum episodes to take 

max_t       = 10000         # maximum steps to take in one episodes, the high number was set to be sure to not end the episode without a done signal


EPSILON = 0.995             # how often the agent reacts randomly in a scale from 0 to 1

EPSILON_DECAY = 0.999       # how much decay per timestep the epsilon will have.


# Architecture
DDPG Actor Critic Architecture with a shared Actor Network for both Agents and different Critic heads for each agent.
Both, the Actor and the Critic Network have two hidden layers with 400 units in the first and 300 units in the second.

# Score
The score needs to be over +0.5 over 100 consecutive episodes to solve the environment.
You can see the solved reward plot in the "checkpoints" folder with the naming "*v2.png".


# Future
The agent so far is pretty good in a very short learning cycle.
Although additional improvements can be made, like the Generalized Advantage Estimation(GAE) to improve the prediction of the expected return.
