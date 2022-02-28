import random
from time import sleep

import gym
from numpy import vstack
import torch
from tqdm import tqdm

torch.manual_seed(1)


class Agent:
    def __init__(
        self,
        env,
        state_value_predictor,
        action_decider,
        explore_prob=0.5,
        discount=0.95,
    ):
        self.device = torch.device("cpu")
        self.state_value_predictor = state_value_predictor.to(self.device)
        self.action_decider = action_decider.to(self.device)
        self.env = env
        # TODO implement these with static memory
        self.states = []
        self.actions = []
        self.state_scores = []
        self.discount = discount
        self.explore_prob = explore_prob
        self.optimizer = torch.optim.Adam(
            self.state_value_predictor.parameters(), lr=1e-1
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.97
        )

    def run_episode(self, render=False):
        state = self.env.reset()
        done = False
        current_rewards = []
        n = 0
        while not done:
            action = self.get_next_action(state, self.explore_prob)
            self.states.append(state)
            state, reward, done, _ = self.env.step(action)
            self.actions.append(action)
            current_rewards.append(reward)
            n += 1
            if render:
                self.env.render()
                sleep(0.01)

        # calculate cumulative rewards
        # we could speed this up with the knowledge that the rewards are always 1..
        discounted_rewards = []
        for i in range(n):
            G = 0
            for j in range(i, n):
                G += current_rewards[j] * self.discount ** (j - i)
            discounted_rewards.append(G)

        self.state_scores.extend(discounted_rewards)
        return n

    def get_next_action(self, state, eps):
        if random.uniform(0, 1) < eps:
            return self.env.action_space.sample()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        state = state.reshape(1, -1)
        action = self.action_decider(state).argmax().item()
        return action

    def train(self, epochs=20):
        states, actions, scores = self.sample_memory(size=1000)
        for _ in range(epochs):
            self.optimizer.zero_grad()
            state_values = self.state_value_predictor(states)
            loss = torch.nn.functional.mse_loss(
                state_values.take_along_dim(actions, dim=1), scores
            )
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        self.action_decider.load_state_dict(self.state_value_predictor.state_dict())
        return loss.item()

    def sample_memory(self, size):
        # this can be made better (with static mem)
        idxs = random.sample(range(len(self.states)), size)
        states = [self.states[i] for i in idxs]
        actions = [self.actions[i] for i in idxs]
        rewards = [self.state_scores[i] for i in idxs]

        states = torch.tensor(vstack(states), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)

        return states, actions.view(-1, 1), rewards.view(-1, 1)


def get_model(n_inputs, n_outputs):
    return torch.nn.Sequential(
        torch.nn.Linear(n_inputs, 16), torch.nn.ReLU(), torch.nn.Linear(16, n_outputs),
    )


def main():
    env = gym.make("CartPole-v1")
    n_inputs = env.observation_space.shape[0]
    n_outputs = env.action_space.n
    state_value_predictor = get_model(n_inputs, n_outputs)
    action_decider = get_model(n_inputs, n_outputs)
    action_decider.load_state_dict(state_value_predictor.state_dict())
    agent = Agent(env, state_value_predictor, action_decider)
    agent.explore_prob = 1
    for _ in tqdm(range(200)):
        agent.run_episode()

    agent.explore_prob = 0
    bar = tqdm(range(100))
    for _ in bar:
        nsteps = agent.run_episode()
        loss = agent.train()
        bar.set_description(f"Loss: {loss:.3f}, nsteps: {nsteps:.3f}")


if __name__ == "__main__":
    main()
