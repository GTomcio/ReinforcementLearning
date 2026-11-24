import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

#global enviroment setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_ID = "CartPole-v1"

train_env = gym.make(ENV_ID)
eval_env = gym.make(ENV_ID)

class decision(nn.Module):
      def __init__(self, feature_size: int = 64):
            super().__init__()
            self.network = nn.Sequential(
                  nn.Linear(4, feature_size),
                  nn.ReLU(),
                  nn.Linear(feature_size, feature_size),
                  nn.ReLU(),
                  nn.Linear(feature_size, feature_size),
                  nn.ReLU(),
                  nn.Linear(feature_size, 2),
            )

      def forward(self, observation):
            return self.network(observation)

model = decision().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

gamma = 0.99
batches = 50
batch_episodes = 100
eval_batch = 100

for batch in range(batches):
      episode_losses = []
      episode_rewards = []

      for i in range(batch_episodes):
            observation, info = train_env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0
            log_probs = []
            episode_rewards = []

            while not terminated:
                  observation_tensor = torch.tensor(
                        observation, dtype=torch.float32, device=DEVICE
                  )

                  action_distribution = torch.distributions.Categorical(
                        logits = model(observation_tensor)
                  )

                  action = int(action_distribution.sample())
                  log_probs.append(action)

                  observation, reward, terminated, truncated, info = train_env.step(action)

                  episode_rewards.append(float(reward))
                  episode_reward += float(reward)

                  if truncated:
                        episode_reward += 20

            episode_returns = []
            cumulative_returns = 0.0

            for step_reward in reversed(episode_rewards):
                  cumulative_returns = step_reward + gamma * cumulative_returns
                  episode_returns.append(cumulative_returns)

            episode_returns.reverse()

            episode_loss = torch.stack(
                  [
                        -log_prob * episode_return
                        for log_prob, episode_return in zip(log_probs, episode_returns)
                  ]
            ).sum()

            episode_losses.append(episode_loss)
            episode_returns.append(episode_rewards)