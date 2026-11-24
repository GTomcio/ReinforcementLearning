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

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

gamma = 0.99
batches = 20
batch_episodes = 100
eval_batch = 10
position_weight = 0.01
angle_weight = 0.01

for batch in range(batches):
      episode_losses = []
      episode_rewards = []

      for i in range(batch_episodes):
            #resetting init state for each ep
            observation, info = train_env.reset()
            terminated = False
            truncated = False
            #sum of the rewards from one ep
            total_episode_reward = 0.0
            #lists of probability of an action and reward following the action respectively
            log_probs = []
            step_rewards = []

            #episode loop
            while not terminated:
                  observation_tensor = torch.tensor(
                        observation, dtype=torch.float32, device=DEVICE
                  )

                  action_distribution = torch.distributions.Categorical(
                        logits = model(observation_tensor)
                  )

                  sampled_action = action_distribution.sample()
                  log_probs.append(action_distribution.log_prob(sampled_action))
                  action = sampled_action.item()
                  #env step
                  observation, reward, terminated, truncated, info = train_env.step(action)
                  #reward shaping
                  cart_position = observation[0]
                  cart_angle = observation[2]
                  shaped_reward = (
                        reward
                        - position_weight * (cart_position ** 2)
                        - angle_weight * (cart_angle ** 2)
                  )

                  step_rewards.append(float(shaped_reward))
                  total_episode_reward += shaped_reward
                  #special case if the model completes the task
                  if truncated:
                        break

            episode_returns = []
            cumulative_returns = 0.0

            for step_reward in reversed(step_rewards):
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
            episode_rewards.append(total_episode_reward)

      batch_loss = torch.stack(episode_losses).mean()
      batch_reward = sum(episode_rewards) / len(episode_rewards)

      evaluation_rewards = []
      for i in range(eval_batch):
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
      f"batch {batch + 1} reward {batch_reward:.2f} "
      f"eval {average_evaluation_reward:.2f}"
      )
      optimizer.zero_grad()
      batch_loss.backward()
      optimizer.step()
            
train_env.close()
eval_env.close()
      