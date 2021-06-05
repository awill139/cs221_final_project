import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from util import SumSegmentTree, MinSegmentTree

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype = np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size

        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace = False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones, None

class WeightedMemory(ReplayBuffer):
	def __init__(self, max_size, input_shape):
		super().__init__(max_size, input_shape)

		self._alpha = 0.6
		self._beta = 0.4
		self._beta_frames = 1e5
		self._frame = 1
		self.size = 0
		self._idx = 0

		it_capacity = 1
		while it_capacity < max_size:
			it_capacity *= 2

		self._it_sum = SumSegmentTree(it_capacity)
		self._it_min = MinSegmentTree(it_capacity)
		self._max_priority = 1.0

	def beta_by_frame(self, idx):
		return min(1.0, self._beta + idx * (1.0 - self._beta) / self._beta_frames)

	def store_transition(self, state, action, reward, state_, done):
		self._it_sum[self.mem_cntr] = self._max_priority ** self._alpha
		self._it_min[self.mem_cntr] = self._max_priority ** self._alpha
		# self._idx = (self._idx + 1) % self.size
		super().store_transition(state, action, reward, state_, done)
		self.size += 1

	def _sample_proportional(self, batch_size):
		indices = []
		max_mem = min(self.mem_cntr, self.mem_size)
		while len(indices) < batch_size:
			mass = random.random() * self._it_sum.sum(0, self.size - 1)
			idx = self._it_sum.find_prefixsum_idx(mass)
			if idx < max_mem:
				indices.append(idx)
		return indices

	def sample(self, batch_size):
		indices = self._sample_proportional(batch_size)

		weights = []
		p_min = self._it_min.min() / self._it_sum.sum()
		beta = self.beta_by_frame(self._frame)
		self._frame += 1

		max_weight = (p_min * self.size) ** -beta

		for idx in indices:
			p_sample = self._it_sum[idx] / self._it_sum.sum()
			weight =  (p_sample * self.size) ** (-beta)
			weights.append(weight / max_weight)

		states = self.state_memory[indices]
		states_ = self.new_state_memory[indices]
		actions = self.action_memory[indices]
		rewards = self.reward_memory[indices]
		dones = self.terminal_memory[indices]

        
		self.last_indices = indices
		
		return states, actions, rewards, states_, dones, np.array(weights)

	def update_priorities(self, priorities):

		for idx, priority in zip(self.last_indices, priorities):
            # import pdb; pdb.set_trace()
			self._it_sum[idx] = (priority + 1e-5) ** self._alpha
			self._it_min[idx] = (priority + 1e-5) ** self._alpha

			self._max_priority = max(self._max_priority, (priority + 1e-5))


class RMNetwork(nn.Module):
	def __init__(self, input_dim, fc1_dims = 128, fc2_dims = 128, lr = 0.0003):
		super(RMNetwork, self).__init__()

		self.state = nn.Linear(*input_dim, fc1_dims)
		self.next_state = nn.Linear(*input_dim, fc1_dims)
		self.state2 = nn.Linear(fc1_dims, fc2_dims)
		self.next_state2 = nn.Linear(fc1_dims, fc2_dims)

		self.combined = nn.Linear(2* fc2_dims + 1 + 1 + 1, 1)

		# self.out = nn.Linear(fc2_dims, 1)
        
		self.relu1 = nn.LeakyReLU()
		self.relu2 = nn.LeakyReLU()
		self.relu3 = nn.LeakyReLU()
		self.relu4 = nn.LeakyReLU()

		self.optim = optim.Adam(self.parameters(), lr = lr) 
		self.crit = nn.BCELoss()

	def forward(self, state, state_, action, reward, done):
		
		x = self.relu1(self.state(state))
		x = self.relu2(self.state2(x))
		x_ = self.relu3(self.next_state(state_))
		x_ = self.relu4(self.next_state2(x_))
		try:
			combined = torch.cat([x, x_, action, reward, done], dim = 0)
		except:
			combined = torch.cat([x, x_, action, reward, done], dim = 1)
		# import pdb; pdb.set_trace()
		output = self.combined(combined.float())
		output = torch.sigmoid(output)
		return output

class LearnedMemory(ReplayBuffer):
	def __init__(self, max_size, input_shape, device, greek_letter = 0.3, batch_size = 32):
		super().__init__(max_size, input_shape,)
		self.rmnet = RMNetwork(input_shape)
		self.rmnet = self.rmnet.to(device)
		self.batch_size = batch_size
		self.greek_letter = greek_letter
		self.device = device
		

		self.label_memory = np.zeros(self.mem_size, dtype = np.float64)

	def learn(self):
		self.rmnet.train()
		for _ in range(1):
			self.rmnet.optim.zero_grad()
			states, actions, rewards, states_, dones, labels = self.self_sample(self.batch_size)

			states = torch.tensor(states).to(self.device)
			rewards = torch.tensor(rewards).to(self.device)
			dones = torch.tensor(dones).to(self.device)
			actions = torch.tensor(actions).to(self.device)
			states_ = torch.tensor(states_).to(self.device)
			labels = torch.tensor(labels).to(self.device).float()
			
			out = self.rmnet(states, states_, torch.unsqueeze(actions, 1), torch.unsqueeze(rewards, 1), torch.unsqueeze(dones, 1))

			loss = self.rmnet.crit(out.squeeze(), labels.squeeze())
			loss.backward()
			self.rmnet.optim.step()

	def label_data(self, reward):
		if reward > 0:
			return 1
		else:
			return 0

	def self_sample(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size, replace = False)

		states = self.state_memory[batch]
		states_ = self.new_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.terminal_memory[batch]
		labels = self.label_memory[batch]

		return states, actions, rewards, states_, dones, labels

	def store_transition(self, state, action, reward, state_, done, model_eval, model_target):
		# return super().store_transition(state, action, reward, state_, done)
		self.rmnet.eval()

		_state = torch.tensor(state).to(self.device)
		_reward = torch.tensor(reward).to(self.device)
		_done = torch.tensor(done).to(self.device)
		_action = torch.tensor(action).to(self.device)
		_state_ = torch.tensor(state_).to(self.device)
		
		store = self.rmnet(_state.to(self.device), _state_.to(self.device), torch.unsqueeze(_action, 0).to(self.device), torch.unsqueeze(_reward, 0).to(self.device), torch.unsqueeze(_done, 0).to(self.device))
		if store > self.greek_letter:
			idx = self.mem_cntr % self.mem_size
			q = model_eval(torch.unsqueeze(_state, 0))
			q_next = model_target(torch.unsqueeze(_state_, 0))

			self.label_memory[idx] = self.label_data(torch.max(q).item() - torch.max(q_next).item())#self.label_data(reward)

			super().store_transition(state, action, reward, state_, done)
			