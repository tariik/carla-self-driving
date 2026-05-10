import numpy as np
import random
from collections import namedtuple, deque
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim



# ✅ HIPERPARÁMETROS OPTIMIZADOS PARA CONDUCIR RECTO
BUFFER_SIZE = int(5e5)  # 500k experiencias — cubrir ~5% del training
LR = 1e-4               # Learning rate conservador (loss alta → updates suaves)
UPDATE_EVERY = 4        # Actualizar cada 4 steps
BATCH_SIZE = 128
GAMMA = 0.99            # Reducido de 0.995 → Q_max teórico ~200 en vez de 400
TARGET_UPDATE_HARD = 10000  # Hard copy cada 10k steps (reemplaza soft update)
# Use GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU cache and set memory optimization if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Enable memory optimization
    torch.backends.cudnn.benchmark = True
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    print(f"GPU Memory available: ~{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**2:.2f} MB")

class dqn():
    """
    DQN Agent para lane following usando telemetría mínima.
    
    State (3 valores):
    - speed: Velocidad actual (0-300 km/h)
    - lane_distance: Distancia al centro del carril (-10 a 10)
    - lane_angle: Ángulo con el carril (-180 a 180)
    
    Action Space: 27 acciones discretas (9 steering × 3 throttle)
    """

    def __init__(self, env, seed):
        """Initialize an Agent object."""
        # Get dimensions from environment
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.last_loss = None  # Para monitorear la última pérdida de entrenamiento
        
        # State size: solo 3 valores
        self.state_size = 4  # speed, lane_distance, lane_angle
        
        action_size = env.action_space.n  # 27 actions
        self.action_size = action_size
        
        print(f"📊 State size: {self.state_size} (speed, lane_distance, lane_angle)")
        print(f"📊 Action size: {action_size}")
        
        self.seed = random.seed(seed)


        # Q-Network - Try GPU first, fallback to CPU if out of memory
        self.device = device  # Store device in instance
        try:
            self.qnetwork_local = QNetwork(self.state_size, action_size, seed).to(self.device)
            self.qnetwork_target = QNetwork(self.state_size, action_size, seed).to(self.device)
            print(f"✓ Networks loaded on {self.device}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️ GPU out of memory, falling back to CPU")
                self.device = torch.device("cpu")
                torch.cuda.empty_cache()
                self.qnetwork_local = QNetwork(self.state_size, action_size, seed).to(self.device)
                self.qnetwork_target = QNetwork(self.state_size, action_size, seed).to(self.device)
                print(f"✓ Networks loaded on CPU")
            else:
                raise e
                
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)


        # Replay memory - pass device to ReplayBuffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.total_steps = 0  # Para hard target update
    
    def step(self, state, action, reward, next_state, done):
        
        # Save experience in replay memory
        # Extract telemetry from observation dict
        state_vector = self._extract_telemetry(state)
        next_state_vector = self._extract_telemetry(next_state)

        # Save experience in replay memory
        self.memory.add(state_vector, action, reward, next_state_vector, done)

        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        self.total_steps += 1
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def _extract_telemetry(self, state):
        """
        Extract minimal telemetry vector from observation dict or array.
        
        Args:
            state: Observation from environment (dict or np.ndarray)
            
        Returns:
            np.array: [speed, lane_distance, lane_angle]
        """
        # Si ya es un array numpy plano, devolverlo directamente        
        if isinstance(state, np.ndarray):
            if state.dtype != np.object_:
                if len(state.shape) == 1 and len(state) == 4:
                    return state.astype(np.float32)
                elif len(state.shape) > 1:
                    return state.reshape(-1).astype(np.float32)[:4]
        # Si es un diccionario, extraer los valores
        # Tu environment puede retornar diferentes formatos
        if isinstance(state, dict):
            try:
                # Intentar diferentes posibles claves
                speed = None
                lane_dist = None
                lane_angle = None
                
                # Buscar 'speed'
                if 'speed' in state:
                    speed = state['speed']
                
                # Buscar 'distance_to_center' o variantes
                if 'distance_to_center' in state:
                    lane_dist = state['distance_to_center']
                elif 'lane_distance' in state:
                    lane_dist = state['lane_distance']
                
                # Buscar 'angle' o variantes
                if 'angle' in state:
                    lane_angle = state['angle']
                elif 'lane_angle' in state:
                    lane_angle = state['lane_angle']
                
                # Convertir a float si son arrays o listas
                if isinstance(speed, (list, np.ndarray)):
                    speed = float(speed[0])
                if isinstance(lane_dist, (list, np.ndarray)):
                    lane_dist = float(lane_dist[0])
                if isinstance(lane_angle, (list, np.ndarray)):
                    lane_angle = float(lane_angle[0])
                
                # Verificar que tenemos todos los valores
                if speed is None or lane_dist is None or lane_angle is None:
                    raise KeyError(f"Missing values: speed={speed}, lane_dist={lane_dist}, lane_angle={lane_angle}")
                
                return np.array([
                    float(speed),
                    float(lane_dist),
                    float(lane_angle)
                ], dtype=np.float32)
                
            except (KeyError, IndexError, TypeError) as e:
                print(f"⚠️ Error extracting telemetry from dict: {e}")
                print(f"   State keys: {state.keys() if isinstance(state, dict) else 'N/A'}")
                print(f"   State: {state}")
                raise ValueError(f"Cannot extract telemetry from state: {e}")
        
        # Si es otra cosa, intentar convertir directamente
        try:
            state_array = np.array(state, dtype=np.float32).reshape(-1)
            if len(state_array) >= 4:
                return state_array[:4]  
        except Exception:
            pass
        
        raise ValueError(f"Cannot extract telemetry from state of type {type(state)}: {state}")
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
            # Extract telemetry vector [speed, lane_distance, lane_angle]
        state_vector = self._extract_telemetry(state)
        
        # Convert to torch tensor
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(self.device)
    
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
           # return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.last_loss = float(loss.item())
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)  # ✅ FIX: previene explosión de gradientes
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # Hard target update cada TARGET_UPDATE_HARD steps (no soft update)
        if self.total_steps % TARGET_UPDATE_HARD == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save(self, filepath):
        """
        Save the agent's Q-networks and optimizer state.
        
        Params
        ======
            filepath (str): Path where to save the checkpoint
        """
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        checkpoint = {
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory_size': len(self.memory),
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath):
        """
        Load a previously saved agent checkpoint.
        
        Params
        ======
            filepath (str): Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Model loaded from {filepath}")
        print(f"   Memory had {checkpoint.get('memory_size', 0)} experiences")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device: torch device (cuda or cpu)
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    


class QNetwork(nn.Module):
    """
    Q-Network para lane keeping: 4 → 128 → 128 → 64 → 27
    Sin BatchNorm (causa inestabilidad en DQN por mismatch train/eval mode)
    Sin Dropout (exploración ya cubierta por epsilon-greedy)
    """

    def __init__(self, state_size=4, action_size=27, seed=42, fc1_units=128, fc2_units=128, fc3_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights usando Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
