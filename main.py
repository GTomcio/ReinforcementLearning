import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

#global enviroment setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

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

optimizer = torch.optim.Adam(model.parameters(), lr=float(input("lr: ")))

gamma = float(input("gamma: "))
num_batches = int(input("number of batches: "))
episodes_per_batch = int(input("episodes per batch: "))
evaluation_episodes = 10

# Collect a fixed number of episodes before each model update.
for batch_index in range(num_batches):
    episode_losses = []
    episode_rewards = []

    for _ in range(episodes_per_batch):
        observation, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        log_probs = []
        rewards = []

        while not terminated and not truncated:
            observation_tensor = torch.tensor(
                observation, dtype=torch.float32, device=DEVICE
            )

            action_distribution = torch.distributions.Categorical(
                logits=model(observation_tensor)
            )
            sampled_action = action_distribution.sample()
            log_probs.append(action_distribution.log_prob(sampled_action))

            action = int(sampled_action.item())
            observation, reward, terminated, truncated, info = env.step(action)

            rewards.append(float(reward))
            episode_reward += float(reward)

        # Calculate discounted returns separately for this episode.
        episode_returns = []
        cumulative_return = 0.0

        for step_reward in reversed(rewards):
            cumulative_return = step_reward + gamma * cumulative_return
            episode_returns.append(cumulative_return)
        episode_returns.reverse()

        # Calculate one policy loss for this complete episode.
        episode_loss = torch.stack(
            [
                -log_prob * episode_return
                for log_prob, episode_return in zip(log_probs, episode_returns)
            ]
        ).sum()
        episode_losses.append(episode_loss)
        episode_rewards.append(episode_reward)

    # Average the episode losses so the update represents the whole batch.
    batch_loss = torch.stack(episode_losses).mean()
    average_batch_reward = sum(episode_rewards) / len(episode_rewards)

    # Evaluate the same policy that collected this batch, without updating it.
    evaluation_rewards = []
    for _ in range(evaluation_episodes):
        observation, info = eval_env.reset()
        terminated = False
        truncated = False
        evaluation_reward = 0.0

        while not terminated and not truncated:
            observation_tensor = torch.tensor(
                observation, dtype=torch.float32, device=DEVICE
            )

            with torch.no_grad():
                action_scores = model(observation_tensor)
                action = int(torch.argmax(action_scores).item())

            observation, reward, terminated, truncated, info = eval_env.step(action)
            evaluation_reward += float(reward)

        evaluation_rewards.append(evaluation_reward)

    average_evaluation_reward = sum(evaluation_rewards) / len(evaluation_rewards)
    print(
        f"batch {batch_index + 1} reward {average_batch_reward:.2f} "
        f"eval {average_evaluation_reward:.2f}"
    )

    # Update only after both reward measurements use the same parameters.
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

env.close()
eval_env.close()
