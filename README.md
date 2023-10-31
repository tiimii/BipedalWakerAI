# BipedalWakerAI

The BipedalWakerAI is an artificial intelligence model designed to navigate the BipedalWalker environment from the OpenAI Gym. The AI utilizes the Soft Actor-Critic (SAC) algorithm, a state-of-the-art reinforcement learning method, to learn optimal policies for the walker.

## Overview

- **Environment**: The AI is trained on the BipedalWalker-v3 environment from OpenAI Gym. This environment challenges the walker to move forward on a challenging terrain without falling.
- **Model**: The core of the AI is the SAC model, which is implemented using the stable_baselines3 library. The SAC algorithm is an off-policy actor-critic deep reinforcement learning algorithm that incorporates the entropy of the policy into the reward, promoting more exploratory policies.
- **Training**: The model is trained iteratively, with each training session consisting of a specified number of timesteps. The training logs and model checkpoints are saved in the Training directory.
- **Evaluation**: The performance of the trained model can be evaluated using the evaluate function, which calculates the mean reward over a set number of episodes.
- **Hyperparameter Tuning**: The repository includes a hyperparameter tuning function that uses the Optuna library to search for the best hyperparameters for the SAC model.

## Files

- `main.py`: The main script that initializes the model, trains it, and tests its performance.
- `sac_model.py`: Contains the implementation of the SAC model, training, evaluation, and hyperparameter tuning functions.

##Getting Started

Clone the repository.
Install the required libraries and dependencies.
Run the main.py script to train and test the model.

## Training Process

The training process involves the following steps:

- **Initialization**: The SAC model is initialized with a specified architecture and hyperparameters. The model is saved in the Training/Saved Models directory.
- **Training Loop**: The model is trained iteratively over a specified number of timesteps. After each training session, the model is saved.
- **Evaluation**: After training, the model's performance is evaluated over a set number of episodes. The mean reward is calculated and printed.
- **Hyperparameter Tuning**: If desired, the hyperparameters of the model can be tuned using the Optuna library. The best hyperparameters are saved and can be used for subsequent training sessions.
