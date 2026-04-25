from collections import deque
import time
from turtle import done
from typing import Dict, Optional
import gymnasium as gym

import carla
import numpy as np

from carla_app.carla_core import CarlaApi
from carla_app.collision_sensor import CollisionSensor
from carla_app.dqn import dqn
from carla_app.lane_invasion_sensor import LaneInvasionSensor

from torch.utils.tensorboard import SummaryWriter
import time

from torch.utils.tensorboard import SummaryWriter
import time
import torch
class CarlaEnv(gym.Env):
    def __init__(
        self, 
        config, 
        render_mode: Optional[str] = None,
        max_episode_steps: int = 300,
        verbose: bool = True    
    ):
        super(CarlaEnv, self).__init__()
        self.config = config
        self.carla_api = CarlaApi(config)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.verbose = verbose

        self.vehicle: Optional[carla.Vehicle] = None

        # internal state
        self.episode_step = 0
        self.terminated = False
        self.truncated = False
        self._initialized = False
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.last_invasion_count = 0
        self.initial_spawn_point = None

        # continuous control: steer, throttle, brake
        # 9 steering × 3 throttle = 27 acciones discretas
        self.action_space = gym.spaces.Discrete(27)
        # Mapeo de acciones discretas a valores continuos
        self.action_mapping = self._build_action_mapping()

        self.prev_steer = 0.0

        # Observation space: [v_norm, e_y_norm, e_psi_rad, steer_prev]
        # speed normalizado ÷120 para que todas las obs estén en rango similar
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -np.pi, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0,  np.pi,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.episodes_since_restart = 0
        self.restart_every = 10000  # Reiniciar cada 10000 episodios
        self.prev_dist_to_goal = None
        self._setup_world()
    
    def _setup_world(self):
        
        
        # connect, spawn vehicle, attach camera and components
        self.carla_api.connect_to_server()
        self.initial_spawn_point = self.carla_api.spawn_vehicle()

        # set sync mode (important for reproducibility)
        self.carla_api.set_sync_mode(render_mode=self.render_mode)
        self.vehicle = self.carla_api.vehicle

        print("[DEBUG] attaching collision sensor...")
        self.collision_sensor = CollisionSensor(self.vehicle)
        print("[DEBUG] attaching lane invasion sensor...")
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle)
        
        # Initialize CARLA world with given configuration
        if self.render_mode == 'human':
            print("[DEBUG] initialize_pygame...")
            self.carla_api.initialize_pygame()
            print("[DEBUG] setup_camera...")
            self.carla_api.setup_camera()
        
        print("[DEBUG] initialize_components...")
        self.carla_api.initialize_components(render_mode=self.render_mode)
        
        self._initialized = True
        self.episode_step = 0
        self.start_time = time.time()

    def _restart_carla_connection(self):
        """Reinicia la conexión a CARLA para limpiar memoria."""
        print("\n🔄 Restarting CARLA connection to free memory...")
        
        try:
            # 1. Destruir sensores
            if self.collision_sensor:
                try:
                    self.collision_sensor.destroy()
                except Exception:
                    pass
            
            if self.lane_invasion_sensor:
                try:
                    self.lane_invasion_sensor.destroy()
                except Exception:
                    pass
            
            # 2. Limpiar CARLA API (pero sin cerrar pygame)
            if self.carla_api:
                try:
                    # Guardar display para no cerrarlo
                    saved_display = self.carla_api.display
                    saved_clock = self.carla_api.clock
                    
                    # Limpiar solo actores de CARLA
                    if self.carla_api.camera_manager:
                        self.carla_api.camera_manager.destroy()
                    
                    for actor in self.carla_api.actors:
                        try:
                            if actor is not None and hasattr(actor, 'is_alive'):
                                if actor.is_alive:
                                    actor.destroy()
                        except Exception:
                            pass
                    
                    self.carla_api.actors.clear()
                    
                    # Desconectar del mundo sin cerrar pygame
                    self.carla_api.world = None
                    self.carla_api.client = None
                    
                    # Restaurar display
                    self.carla_api.display = saved_display
                    self.carla_api.clock = saved_clock
                    
                except Exception as e:
                    print(f"⚠ Warning during cleanup: {e}")
            
            # 3. Esperar un momento
            time.sleep(2)
            
            # 4. Reconectar
            self.carla_api.connect_to_server()
            self.initial_spawn_point = self.carla_api.spawn_vehicle()
            self.carla_api.set_sync_mode(render_mode=self.render_mode)
            self.vehicle = self.carla_api.vehicle
            
            # 5. Recrear sensores
            self.collision_sensor = CollisionSensor(self.vehicle)
            self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle)
            
            # 6. Recrear cámara
            self.carla_api.setup_camera()
            
            print("✓ CARLA connection restarted successfully")
            
        except Exception as e:
            print(f"❌ Error restarting CARLA: {e}")
            raise
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        self.prev_dist_to_goal = None
        self.episode_step = 0
        self.terminated = False
        self.truncated = False
        self.prev_steer = 0.0
        # ✅ REINICIAR CONEXIÓN SI ES NECESARIO
        self.episodes_since_restart += 1
        if self.episodes_since_restart >= self.restart_every:
            # self._restart_carla_connection()
            self.episodes_since_restart = 0

        # Respawn vehicle (una sola vez)
        if not self._initialized:
            self._setup_world()
        else:
            self.carla_api.respawn_vehicle(self.initial_spawn_point)

        if self.collision_sensor:
            self.collision_sensor.reset()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.reset()
        
        # ✅ RESETEAR contador de invasiones
        self.last_invasion_count = 0

        # Importante: hacer tick para actualizar estado
        self.carla_api.world.tick()
        
        obs = self._get_observation()
        info = {}
        telemetry = self.carla_api.vehicle_controller.get_telemetry()
        self.prev_dist_to_goal = telemetry.get("distance_to_goal", None)
        if self.verbose:    
            print("✓ Environment reset")
        
        return obs, info

    def step(self, action):
        """Apply action and return new state, reward, done, info."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if action not in self.action_mapping:
            raise ValueError(f"Invalid action {action}. Must be 0-26.")

        action_cmd = self.action_mapping[action]
        
        control = carla.VehicleControl(
            throttle=action_cmd['throttle'],
            steer=action_cmd['steering'],
            brake=action_cmd['brake'],
            reverse=action_cmd['reverse'],
            hand_brake=action_cmd['hand_brake']
        )
        
        self.vehicle.apply_control(control)
        
        if self.verbose:
            print(f"Step {self.episode_step}: Applied action {action} -> {action_cmd}")
        
        # advance simulation (synchronous)
        self.carla_api.world.tick()

        if self.render_mode == 'human':
            self.carla_api.camera_manager.update()

        # telemetry / observation
        # obs = self._get_observation()

        telemetry = self.carla_api.vehicle_controller.get_telemetry()
        speed = telemetry.get('speed', 0.0)
        vehicle_location = self.vehicle.get_location()
        distance_to_goal = telemetry.get('distance_to_goal', None)
        #goal_reached = telemetry.get('goal_reached', False)
        #current_waypoint = telemetry.get('current_waypoint', None)
        #next_waypoint = telemetry.get('next_waypoint', None)
        #waypoint_progress = telemetry.get('waypoint_progress', None)
        distance_to_center = telemetry.get('lane_info', {}).get('distance_to_center', 0.0)
        #distance_abs = telemetry.get('lane_info', {}).get('distance_abs', 0.0)
        lane_width = telemetry.get('lane_info', {}).get('lane_width', 0.0)
        #distance_percentage = telemetry.get('lane_info', {}).get('distance_percentage', 0.0)
        #is_centered = telemetry.get('lane_info', {}).get('is_centered', False)
        #side = telemetry.get('lane_info', {}).get('side', 'unknown')
        angle = telemetry.get('lane_info', {}).get('angle', 0.0)
        #angle_abs = telemetry.get('lane_info', {}).get('angle_abs', 0.0)
        #is_aligned = telemetry.get('lane_info', {}).get('is_aligned', False)

    
        

       
        # justo después de leer telemetry y lane_info
        steer = float(action_cmd['steering'])
        delta_steer = steer - self.prev_steer

        v = float(speed)  # km/h
        dt = float(distance_to_center)
        lane_w = float(lane_width)
        angle_deg = float(angle)
        e_psi = np.deg2rad(angle_deg)

        denom = max(lane_w / 2.0, 1e-3)
        e_y = np.clip(dt / denom, -1.0, 1.0)

        # pesos iniciales (rectas)
        k_v = 0.03
        k_y = 1.0
        k_psi = 0.5
        k_delta = 0.2
        k_s = 0.05

        v_capped = min(v, 80.0)  # ✅ FIX: no recompensar >80 km/h
        v_forward = v_capped * np.cos(e_psi)
        reward = (k_v * v_forward) - (k_y * e_y**2) - (k_psi * e_psi**2) - (k_delta * delta_steer**2) - (k_s * steer**2)

        # actualiza memoria de steer para el siguiente paso/obs
        self.prev_steer = steer

        
        # Update previous distance to goal

        if distance_to_goal is not None and self.prev_dist_to_goal is not None:
            progress = self.prev_dist_to_goal - distance_to_goal   # >0 si te acercas
            k_prog = 1.0                               # empieza con 1.0
            reward += k_prog * progress
        
        self.prev_dist_to_goal = distance_to_goal

        
        goal_reached = False
        if distance_to_goal is not None:
            radius = self.config.get('goal_point', {}).get('radius', 5.0)
            if distance_to_goal <= radius:
                reward += 5.0  # ✅ FIX: bonus proporcional a la escala del reward
                print("🎯 Goal reached! Bonus awarded.")
                self.terminated = True
                goal_reached = True


        # ✅ FIX: terminar si el ángulo supera 90° — no hay recuperación posible
        if np.abs(e_psi) > np.pi / 2:
            reward += -5.0
            self.terminated = True

        GRACE_PERIOD = 20  # ✅ FIX: evitar falsas terminaciones al inicio
        collision_occurred = self.collision_sensor.has_collision()
        if collision_occurred and self.episode_step >= GRACE_PERIOD:
            reward += -10.0  # ✅ FIX: aditivo y escala razonable
            self.terminated = True

        # Off-road detection (ojo lane_width == 0)
        if (lane_width is not None) and (lane_width > 1e-3):
            if np.abs(distance_to_center) > lane_width * 0.95:
                reward += -10.0  # ✅ FIX: aditivo y escala razonable
                self.terminated = True

        # Lane invasion (salida de carril)
        current_invasion_count = self.lane_invasion_sensor.get_invasion_count()
        new_invasions = current_invasion_count - self.last_invasion_count
        if new_invasions > 0:
            reward -= 5.0 * new_invasions  # ✅ FIX: penalización proporcional pero razonable
            self.last_invasion_count = current_invasion_count
            if current_invasion_count >= 5:  # ✅ FIX: más tolerante para aprender
                self.terminated = True

        self.episode_step += 1

        # Time limit
        if self.episode_step >= self.max_episode_steps:
            self.truncated = True  

        # ✅ FIX: Reward clipping — acotar TD error
        reward = float(np.clip(reward, -10.0, 10.0))
        obs = self._get_observation()
        info = {
            'speed': speed,
            'distance_to_center': distance_to_center,
            'angle': angle,
            'collision': collision_occurred,
            'lane_invasions': current_invasion_count,
            'goal_reached': goal_reached,
        }
        
        return obs, reward, self.terminated, self.truncated, info

    def _build_action_mapping(self) -> dict:
        """
        Construye el mapeo de acciones discretas (0-26) a comandos de control.
        
         Table 1 - 27 discrete actions (DQN)
        - Steering: 9 valores [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        - Throttle: 3 valores [0, 0.5, 1]
        - Total: 9 × 3 = 27 acciones
        
        Returns:
            dict: Mapeo {action_id: [throttle, steering, brake, reverse, hand_brake]}
        """
        actions = {}
        action_id = 0
        
        # Paper: throttle [0, 0.5, 1] × steering [-1, -0.75, ..., 0.75, 1]
        for throttle in [0, 0.5, 1.0]:
            for steering in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
                actions[action_id] = {
                    'throttle': throttle,
                    'steering': steering,
                    'brake': 0.0,
                    'reverse': False,
                    'hand_brake': False
                }
                action_id += 1
        
        return actions
    
    def _get_observation(self):
        telemetry = self.carla_api.vehicle_controller.get_telemetry()
        lane_info = telemetry.get('lane_info', {}) or {}

        v = float(telemetry.get('speed', 0.0))  # km/h
        dt = float(lane_info.get('distance_to_center', 0.0))  # m
        lane_w = float(lane_info.get('lane_width', 0.0))      # m
        angle_deg = float(lane_info.get('angle', 0.0))        # deg

        denom = max(lane_w / 2.0, 1e-3)
        e_y = np.clip(dt / denom, -1.0, 1.0)
        e_psi = np.deg2rad(angle_deg)

        obs = np.array([v / 120.0, e_y, e_psi, self.prev_steer], dtype=np.float32)
        return obs

    def render(self, mode='human'):
        """Render (human) or return rgb array."""
        if self.render_mode == 'human':
            # let the CarlaControl render with pygame HUD
            render =self.carla_api.render()
            return render

    def close(self):
        """Clean up resources - correct order."""
        try:
            # 1. Destroy custom sensors FIRST
            if self.collision_sensor:
                try:
                    self.collision_sensor.destroy()
                    print("  ✓ Collision sensor destroyed")
                except Exception as e:
                    print(f"  Warning: {e}")
            
            if self.lane_invasion_sensor:
                try:
                    self.lane_invasion_sensor.destroy()
                    print("  ✓ Lane invasion sensor destroyed")
                except Exception as e:
                    print(f"  Warning: {e}")
            
            # 2. Then destroy CARLA API (camera + vehicle + world cleanup)
            if self.carla_api:
                self.carla_api.clean_up()
        
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        print("✓ Environment closed")



def compute_q_values(agent, states_np: np.ndarray):
    agent.qnetwork_local.eval()
    with torch.no_grad():
        s = torch.from_numpy(states_np).float().to(agent.device)
        q = agent.qnetwork_local(s).detach().cpu().numpy()
    agent.qnetwork_local.train()
    return q


def make_epsilon_by_step(total_steps: int,
                         eps_start: float = 1.0,
                         eps_mid: float = 0.10,
                         eps_end: float = 0.01,
                         frac1: float = 0.25,   # 25% del training: start -> mid
                         frac2: float = 0.75):  # hasta 75%: mid -> end, luego fijo
    total_steps = max(1, int(total_steps))
    s1 = int(total_steps * frac1)
    s2 = int(total_steps * frac2)
    s1 = max(1, s1)
    s2 = max(s1 + 1, s2)

    def epsilon_by_step(step: int) -> float:
        step = int(step)
        if step <= 0:
            return eps_start

        # fase 1: eps_start -> eps_mid
        if step < s1:
            alpha = step / s1
            return eps_start + alpha * (eps_mid - eps_start)

        # fase 2: eps_mid -> eps_end
        if step < s2:
            alpha = (step - s1) / (s2 - s1)
            return eps_mid + alpha * (eps_end - eps_mid)

        # fase 3: fijo
        return eps_end

    return epsilon_by_step


if __name__ == '__main__':
    print("Testing CarlaWorldEnv as Gymnasium environment...")
    config = {
        'host': 'localhost',
        'port': 2000,
        'town': 'Town01',
        'weather': 'ClearNoon',
        'vehicle_model': 'vehicle.tesla.model3',
        'camera_position': [1.5, 0.0, 2.4],
        'camera_rotation': [0.0, 0.0, 0.0],
        'window_width': 800,
        'window_height': 600,
        'target_fps': 20,
        'window_width': 800,
        'window_height': 600,
        'window_title': "CARLA Gym Environment",
        'name_map': 'Town01',
        'set_weather': True,
        'weather_preset': 'ClearNoon',
        'spawn_point': {'x': 383.0, 'y': -2.0, 'z': None, 'yaw': 180.0},
        'goal_point': {'x': 11.71, 'y': -2.0, 'radius': 5.0},
        # 'goal_point': {'x': 334.0, 'y': 43.0, 'radius': 5.0},
        
        'autopilot_mode': True,
        'camera_fov': 90,
        'camera_position': [-5, 0.0, 2.8],
        'camera_rotation': [-15, 0, 0],
        'throttle_speed': 0.8,
        'brake_speed': 0.5,
        'steer_speed': 0.5,
        'hud_font_size': 20,
        'hud_bg_color': (0, 0, 0, 0),
        'hud_bg_alpha': 128,
        'hud_width': 240,
        'hud_height': 320,
        'enable_photo_capture': True,
        'photo_capture_interval': 1.0,
        'photo_capture_directory': 'run_logs',
        'photo_width': 800,
        'photo_height': 600,
        'photo_fov': 90.0,
        'collision_sensor_enabled': True,
        'lane_invasion_sensor_enabled': True,
        'carla_timeout': 20.0,
    }  # Placeholder for actual configuration

    # 🔥 PARÁMETROS DE ENTRENAMIENTO
    n_episodes = 100000  # Más episodios para aprender
    max_t = 1000      # Más steps por episodio
    eps_start = 1.0   # ✅ FIX: Empezar con exploración completa
    eps_decay = 0.9995
    eps_end = 0.01

    # 🔥 render_mode='human' para ver el entrenamiento
    env = CarlaEnv(config, render_mode=None, max_episode_steps=max_t, verbose=False)
    
    # Create agent
    agent = dqn(env, seed=42)
    
    # Training metrics
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    # ✅ AGREGAR: Tracking del mejor modelo
    best_avg_score = -np.inf
    best_episode = 0
    best_checkpoint_path = None
    global_step = 0
    
    print("\n🚀 Starting DQN Training for Lane Keeping")
    print(f"Episodes: {n_episodes}, Max steps: {max_t}")
    print(f"Epsilon: {eps_start} → {eps_end} (decay: {eps_decay})")
    print("=" * 60)

    run_name = time.strftime("carla_dqn_%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # 3 Probe states fijos para verificar convergencia de Q-values en TensorBoard
    # Estado 0: conducción recta ideal (v=60, centrado, alineado, steer=0)
    # Estado 1: desviado a la derecha con ángulo (v=40, e_y=0.5, e_psi=10°, steer=0.25)
    # Estado 2: rápido y centrado (v=80, centrado, alineado, steer=0)
    # Probe states con speed normalizado (v/120)
    probe_states = np.array([
        [60.0/120,  0.0,  0.0,              0.0],    # ideal: recto, centrado
        [40.0/120,  0.5,  np.deg2rad(10.0), 0.25],   # desviado, necesita corregir
        [80.0/120,  0.0,  0.0,              0.0],     # rápido, centrado
    ], dtype=np.float32)

    # ✅ FIX: total_steps basado en pasos REALES esperados, no max_t teórico.
    # Con speed cap 80 km/h y reward clip, episodios más largos (~100 steps).
    # 100 steps/ep × 100,000 eps = 10,000,000 pasos efectivos.
    # - Phase 1 (eps 1.0→0.10): primeros 25% = 2,500,000 steps ≈ 25,000 eps
    # - Phase 2 (eps 0.10→0.01): hasta 75%  = 7,500,000 steps ≈ 75,000 eps
    # - Phase 3 (eps fijo 0.01): resto
    EXPECTED_STEPS_PER_EP = 100  # más largo con speed cap a 80 km/h
    realistic_total_steps = n_episodes * EXPECTED_STEPS_PER_EP
    epsilon_by_step = make_epsilon_by_step(realistic_total_steps,
                                       eps_start=1.0,
                                       eps_mid=0.10,
                                       eps_end=0.01,
                                       frac1=0.25,
                                       frac2=0.75)
  
    
    try:
        for i_episode in range(1, n_episodes + 1):
            state, info = env.reset()
            score = 0
            steps_taken = 0
            
            for t in range(max_t):
                # Select action
                global_step += 1

                eps = float(epsilon_by_step(global_step))
                action = agent.act(state, eps)
                
                # Take action
                next_state, reward, done, truncated, info = env.step(action)
                
                # 🔥 RENDER para ver el coche
                #if env.render_mode == 'human':
                #    if not env.render():
                #        print("\n⚠ Window closed by user")
                #        raise KeyboardInterrupt
                
                # Store experience and learn (agent.step ya espera BATCH_SIZE muestras)
                agent.step(state, action, reward, next_state, done or truncated)
                if agent.last_loss is not None:
                    writer.add_scalar("train/loss", float(agent.last_loss), global_step)
                
                # Update state and score
                state = next_state
                score += reward
                steps_taken = t + 1

                
                
                if done or truncated:
                    break
            
            # Update metrics
            scores_window.append(score)
            scores.append(score)
            # eps = max(eps_end, eps_decay * eps)

            
            

            # ✅ VERIFICAR SI ES EL MEJOR MODELO (cada 100 episodios para estabilidad)
            if i_episode >= 100 and i_episode % 10 == 0:
                current_avg = np.mean(scores_window)
            
                if current_avg > best_avg_score:
                    best_avg_score = current_avg
                    best_episode = i_episode
                    
                    # Guardar best checkpoint
                    import os
                    os.makedirs('checkpoints_dqn', exist_ok=True)
                    best_checkpoint_path = f'checkpoints_dqn/best_checkpoint.pth'
                    agent.save(best_checkpoint_path)
                    
                    print(f'   🏆 NEW BEST! Avg: {best_avg_score:.2f} at ep {best_episode} - Saved!')
            
            # Print progress every 10 episodes
            if i_episode % 10 == 0:
                print(f'Episode {i_episode}/{n_episodes} | '
                    f'Steps: {steps_taken:3d} | '
                    f'Score: {score:7.2f} | '
                    f'Avg(last 10): {np.mean(list(scores_window)[-10:]):7.2f} | '
                    f'Eps: {eps:.3f}')

            
            writer.add_scalar("train/episode_reward", score, i_episode)
            writer.add_scalar("train/episode_steps", steps_taken, i_episode)
            writer.add_scalar("train/epsilon", eps, i_episode)

            # si estás usando average window
            if len(scores_window) > 0:
                writer.add_scalar("train/avg_reward_100", float(np.mean(scores_window)), i_episode)

            # Log Q-values cada 10 episodios (reducir I/O de TensorBoard)
            if i_episode % 10 == 0:
                q = compute_q_values(agent, probe_states)  # shape (3, 27)
                for i in range(3):
                    for a in range(q.shape[1]):
                        writer.add_scalar(f"qprobe/state{i}/action_{a:02d}", float(q[i, a]), global_step)
                    writer.add_scalar(f"qprobe/state{i}/Q_max", float(q[i].max()), global_step)
                    writer.add_scalar(f"qprobe/state{i}/Q_best_action", int(q[i].argmax()), global_step)
            
            # Save checkpoint every 50 episodes
            if i_episode % 500 == 0:
                import os
                os.makedirs('checkpoints_dqn', exist_ok=True)
                agent.save(f'checkpoints_dqn/checkpoint_ep{i_episode}.pth')
                print(f'   ✓ Checkpoint saved')
    
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    
    finally:
        env.close()
        writer.close()
        
        print("✓ Training complete!")
        if len(scores) > 0:
            print(f"✓ Total episodes: {len(scores)}")
            print(f"✓ Final average score: {np.mean(list(scores_window)[-10:]):.2f}")