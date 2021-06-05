import torch
import numpy as np
from torch import optim
import torch.nn as nn
from memory import LearnedMemory, ReplayBuffer, WeightedMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device: {}'.format(device))


class DuelingDeepQNet(nn.Module):
    def __init__(self, n_actions, input_dim, fc1_dims, fc2_dims, lr = 0.0003, weighted = False):
        super(DuelingDeepQNet, self).__init__()

        self.fc1 = nn.Linear(*input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)
        
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()

        self.optim = optim.Adam(self.parameters(), lr = lr) 
        self.mse = nn.MSELoss()
        if weighted:
            self.crit = self.weighted_loss
        else:
            self.crit = self.mse

    def weighted_loss(self, target, pred, errors, weights):
        return torch.mean(torch.multiply(weights, errors))

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))

        V = self.V(x)
        A = self.A(x)

        Q = V + (A - torch.mean(A, dim = 1, keepdim = True))

        return Q

    def advantage(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))

        return self.A(x)

class Agent:
    def __init__(self, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_decay = 1e-8, eps_min = 0.01,
                 mem_size = 1000000, fc1_dims = 128, fc2_dims = 128, replace = 100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.eps_min = eps_min
        self.replace = replace
        self.batch_size = batch_size
        self.weighted = False
        self.learned_memory = True

        self.learn_step_counter = 0
        if self.weighted:
            self.memory  = WeightedMemory(max_size=mem_size, input_shape=input_dims)
        elif self.learned_memory:
            self.memory = LearnedMemory(max_size=mem_size, input_shape= input_dims, device=device)
        else:
            self.memory = ReplayBuffer(max_size = mem_size, input_shape = input_dims)
        self.q_eval = DuelingDeepQNet(n_actions = n_actions, input_dim = input_dims, fc1_dims = fc1_dims, fc2_dims = fc2_dims, weighted=self.weighted)
        self.q_next = DuelingDeepQNet(n_actions = n_actions, input_dim = input_dims, fc1_dims = fc1_dims, fc2_dims = fc2_dims, weighted = self.weighted)

        self.q_eval.to(device)
        self.q_next.to(device)

        

    def store_transition(self, state, action, reward, new_state, done):
        if self.learned_memory:
            self.memory.store_transition(state, action, reward, new_state, done, self.q_eval, self.q_next)
        else:
            self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            state = torch.Tensor([observation]).to(device)
            advantage = self.q_eval.advantage(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def get_errors(self,states, rewards, next_states, terminal):
        target_value_per_action = self.q_next(next_states)

        # value of next_state accorting to target_dqn
        target_value = torch.max(target_value_per_action, axis=1)
        # import pdb; pdb.set_trace()
        # terminal = terminal > 0
        bellman_target = rewards + torch.mul(target_value.values, 0.9) * (1 - torch.gt(terminal, 0).int())

        active_value_per_action = self.q_eval(states)
        # value of state according to active_dqn
        active_value = torch.max(active_value_per_action, axis=1)

        return bellman_target - active_value.values

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        if self.learn_step_counter % self.replace == 0:

            self.q_next.load_state_dict(self.q_eval.state_dict())

        if self.learned_memory:
            self.memory.learn()

        states, actions, rewards, states_, dones, weights = self.memory.sample(self.batch_size)

        states = torch.tensor(states).to(device)
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor(dones).to(device)
        actions = torch.tensor(actions).to(device)
        states_ = torch.tensor(states_).to(device)
        if weights is not None:
            weights = torch.tensor(weights).to(device)

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval(states)[indices, actions]
        q_next = self.q_next(states_)

        max_actions = torch.argmax(self.q_eval(states_), dim=1)
        # q_eval = self.q_eval(torch.Tensor(states_).to(device))[indices, actions]
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        q_next[dones] = 0.0
        self.q_eval.optim.zero_grad()
        # import pdb; pdb.set_trace()
        if self.weighted:
            errors = torch.abs(q_pred - q_target)
            # print(weights)
            loss = self.q_eval.crit(q_target, q_pred, errors, weights)
        else:
            loss = self.q_eval.crit(q_target, q_pred)
        loss.backward()

        self.q_eval.optim.step()

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.eps_min else self.eps_min
        self.learn_step_counter += 1