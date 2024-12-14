# Using stable_baselines3 instead of implementing DQN manually
# referencing from https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

from environment import EmotionWorld
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

env = EmotionWorld()
# model = DQN("MlpPolicy", env, verbose=1)
model = DQN("MlpPolicy", env, learning_rate=1e-3, gamma=0.99, verbose=1)

model.learn(total_timesteps=10000)
model.save("dqn_emotionworld")

print("Model training complete and saved.")

# test model
obs, info = env.reset()

total_reward = 0
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done:
        print(f"Episode done, total reward: {total_reward}")
        obs, info = env.reset()
        total_reward = 0