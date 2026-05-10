import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(5e5)
LR = 1e-4
UPDATE_EVERY = 4
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE_HARD = 10000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True


class DQNWaypointsAgent:
    """DQN agent for waypoint-based observations (paper experiment style)."""

    def __init__(self, env, seed):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.last_loss = None

        self.state_size = int(np.prod(self.observation_space.shape))
        self.action_size = int(self.action_space.n)
        self.seed = random.seed(seed)

        print(f"State size: {self.state_size}")
        print(f"Action size: {self.action_size}")

        self.device = device
        try:
            self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed).to(self.device)
            self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed).to(self.device)
            print(f"Networks loaded on {self.device}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU out of memory, falling back to CPU")
                self.device = torch.device("cpu")
                torch.cuda.empty_cache()
                self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed).to(self.device)
                self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed).to(self.device)
                print("Networks loaded on CPU")
            else:
                raise

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device)

        self.t_step = 0
        self.total_steps = 0

    def step(self, state, action, reward, next_state, done):
        state_vector = self._extract_state(state)
        next_state_vector = self._extract_state(next_state)

        self.memory.add(state_vector, action, reward, next_state_vector, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        self.total_steps += 1
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def _extract_state(self, state):
        """Flatten observation into fixed-size float32 vector."""
        if isinstance(state, np.ndarray):
            vec = state.astype(np.float32).reshape(-1)
        elif isinstance(state, dict):
            if "observation" in state:
                vec = np.asarray(state["observation"], dtype=np.float32).reshape(-1)
            else:
                vec = np.asarray(list(state.values()), dtype=np.float32).reshape(-1)
        else:
            vec = np.asarray(state, dtype=np.float32).reshape(-1)

        if vec.size < self.state_size:
            padded = np.zeros(self.state_size, dtype=np.float32)
            padded[:vec.size] = vec
            return padded

        return vec[:self.state_size]

    def act(self, state, eps=0.0):
        state_vector = self._extract_state(state)
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        return int(random.choice(np.arange(self.action_size)))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.last_loss = float(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        if self.total_steps % TARGET_UPDATE_HARD == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def save(self, filepath):
        import os

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        checkpoint = {
            "qnetwork_local_state_dict": self.qnetwork_local.state_dict(),
            "qnetwork_target_state_dict": self.qnetwork_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "memory_size": len(self.memory),
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.qnetwork_local.load_state_dict(checkpoint["qnetwork_local_state_dict"])
        self.qnetwork_target.load_state_dict(checkpoint["qnetwork_target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {filepath}")
        print(f"Memory had {checkpoint.get('memory_size', 0)} experiences")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8))
            .float()
            .to(self.device)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    """MLP for waypoint-based DQN."""

    def __init__(self, state_size, action_size, seed=42, fc1_units=256, fc2_units=256, fc3_units=128):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# Backward-compatible name used by existing scripts.
dqn = DQNWaypointsAgent
