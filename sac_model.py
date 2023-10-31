from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import os
import gymnasium as gym
import optuna


def initialize_model(model_path, params):
    # Initialize environment
    env = gym.make('BipedalWalker-v3', hardcore=False)

    # Wrap environment
    env = DummyVecEnv([lambda: env])

    # Create Logs directory
    log_path = os.path.join("Training", "Logs")

    # Neural network architecture
    policy_kwargs = dict(
        net_arch=[256, 256, 256]  # Three hidden layers with 256 neurons each
    )

    # Create model
    model = SAC("MlpPolicy", env, **params, verbose=1, tensorboard_log=log_path, policy_kwargs=policy_kwargs)

    # Save Model
    model.save(model_path)

    env.close()


def train(model_path, timesteps):
    # Initialize environment
    env = gym.make('BipedalWalker-v3', hardcore=False)

    # Wrap environment
    env = DummyVecEnv([lambda: env])

    # Load model
    model = SAC.load(model_path, env=env)

    # Train model
    model.learn(total_timesteps=timesteps, progress_bar=False)

    # Save model
    model.save(model_path)

    # Close environment
    env.close()


def evaluate(model_path):
    # Initialize environment
    env = gym.make('BipedalWalker-v3')

    # Wrap environment
    env = DummyVecEnv([lambda: env])

    # Load model
    model = SAC.load(model_path)
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(mean_reward)


def objective(trial):
    # Sample hyperparameters for the policy network architecture
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_size = trial.suggest_categorical("layer_size", [64, 128, 256, 512])
    net_arch = [layer_size for _ in range(n_layers)]
    policy_kwargs = dict(net_arch=net_arch)
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99])
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1.0, log=True)

    # Initialize environment
    env = gym.make('BipedalWalker-v3')
    env = DummyVecEnv([lambda: env])

    # Create model with hyperparameters
    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=gamma, ent_coef=ent_coef, verbose=0)

    # Train model
    model.learn(total_timesteps=50000)

    # Evaluate model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=8, render=False)

    return mean_reward


def hyperparameter_tuning():
    # Tuning process
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params

    # Print best parameters
    print("Best hyperparameters:", best_params)

    # Create Model with best parameters
    model_path = os.path.join("Training", "Saved Models", "SAC_Bipedal_Model_TUNED")
    initialize_model(model_path, best_params)


def test(model_path):
    # Load model
    model = SAC.load(model_path)
    new_env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")

    episodes = 5
    for episode in range(1, episodes + 1):
        obs, _ = new_env.reset()
        done = False
        score = 0

        while not done:
            new_env.render()
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = new_env.step(action)
            score += reward
        print(f"Episode {episode}:\tScore:{score}")
    new_env.close()
