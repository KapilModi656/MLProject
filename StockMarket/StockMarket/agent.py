from environment import StockTradingEnvironment
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
class ActorCritic:
    def __init__(self, state_dim, action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Simple linear models
        self.actor = nn.Linear(state_dim, action_dim)
        self.critic = nn.Linear(state_dim, 1)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.softmax = nn.Softmax(dim=-1)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.actor(state)
        probs = self.softmax(logits)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, state, action_log_prob, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + self.gamma * next_value * (1 - done)
        advantage = target - value

        # Critic update
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -action_log_prob * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

if __name__ == "__main__":
    env = StockTradingEnvironment(stock_name="apple")
    initial_state=env.reset()
    state_dim = env.observation_space
    action_dim = env.action_space.__len__()
    agent = ActorCritic(state_dim, action_dim)

    num_episodes = 10000
    episode_length = 300
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        for t in range(episode_length):
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action_log_prob, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {episode+1}, Total Reward: {total_reward}")