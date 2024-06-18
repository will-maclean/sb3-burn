# Reinforcement Learning
Reinforcement learning is a family of algorithms that focus on training an agent to maximise
a reward signal coming from an environment by choosing optimal actions. Reinforcement learning
as a topic is too large to cover here in its entirety, but we can provide a quick summary and
a set of links to more comprehensive resources.

A reinforcement learning scenario will contain the following elements:
- An environment
- An agent

## Environment
The environment is what is trying to be "solved". It may be just about anything - a 
video game, a robotic control situation, a simulation of some sort. 

The key aspects of an environment are that is can take in actions, and output
signals of the form (observation, reward, done). 

## Agent
The agent is what we train to solve the environment. It takes in an observation
of the environment, and produces an action to be taken in an environment. For
example, the agent may take the pixels on the screen of a video game as input, 
and produce keyboard strokes as an output.

Typically, an agent will contain a neural network, which is trained through
backpropagation. However, it is not the case that an agent _must_ contain a
neural network - algorithms exist that can train other types of model, and other
types of algorithm do exist.

## Replay Buffer
Reinforcement learning algorithms are roughly divided into two different types - 
online and offline algorithms. All algorithms learn by interacting with the environment,
but offline algorithms can store the data they receive from the environment to use
later, whereas online algorithms cannot. Offline algorithms store data in a replay buffer.

## Algorithms
There are different types of reinforcement learning algorithm. The typical starting
point for most newcomers to reinforcement learning is the Deep Q Network or DQN. This
is a good starting point if you haven't used reinforcement learning before.

## Resources
Unfortunately, reinforcement learning as a field is far too large to fit on this page! Here
are some resources to help you get started:

- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [DeepLizard DQN Course](https://deeplizard.com/learn/video/wrBUkpiRvCA)