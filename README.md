# Overview
In this project you had to train two agents of tennis players, that try to keep the ball in the game as long as possible.



# Setup
To start, you need Python 3.6, PyTorch, Numpy, Matplotlib, and the Unity ML-Agents Toolkit.

With Anaconda or virtualenv you can create your python environment like:
conda create -n drlnd python=3.6 pytorch matplotlib

For the Unity ML-Agent you need to download the Toolkit (https://github.com/Unity-Technologies/ml-agents) go to the ml-agents/python directory and install it via:
    $pip install ml-agents/python

# Instructions

Watch the trained agent with:
$ python watch.py

and let the agent train again with:
$ python train.py
    
    
# Environment
The observation space for each agent consists of 8 values and the action space for each agent consists of 2 values, which are both continuous.
The Environment is considered solved, after taking the maximum score of each player for each episode and then taking the mean of the last 100 scores reaches 0.5 or higher for one episode.

![solved_env]('checkpoints/scores_v2.png')
'Graph showing me solving the environment'

This algorithm even succeeded 0.7 over 100 consecutive episodes once and then bounced around the 0.5 mark.


# Architecture and Algorithms
Can be viewed in the "REPORT.md" file next to this file.


# Future
The agent so far is pretty good in a very short learning cycle.
Although additional improvements can be made, like the Generalized Advantage Estimation(GAE) to improve the prediction of the expected return.


    

    

