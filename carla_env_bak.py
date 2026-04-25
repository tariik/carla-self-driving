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
from carla_env.sensors.lane_invasion_sensor import LaneInvasionSensor


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

        # Observation space: [velocidad, distancia_al_centro, ángulo_con_carril]
        self.observation_space = gym.spaces.Box(
            low=np.array([-50.0, -5.0, -180.0], dtype=np.float32),
            high=np.array([50.0, 5.0, 180.0], dtype=np.float32),
            dtype=np.float32
        )
        self.episodes_since_restart = 0
        self.restart_every = 10000  # Reiniciar cada 10000 episodios

        self._setup_world()
    
    def _setup_world(self):
        
        
        # connect, spawn vehicle, attach camera and components
        self.carla_api.connect_to_server()
        self.initial_spawn_point = self.carla_api.spawn_vehicle()

        # set sync mode (important for reproducibility)
        self.carla_api.set_sync_mode(render_mode=self.render_mode)
        self.vehicle = self.carla_api.vehicle

        # Planificar ruta
        # self.carla_api.plan_route()
        # self.carla_api.visualize_route(life_time=300.0)  # ← AÑADIR ESTA LÍNEA
        self.collision_sensor = CollisionSensor(self.vehicle)
        if self.verbose:
            print("✓ Collision sensor attached")
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle)
        if self.verbose:
            print("✓ Lane invasion sensor attached")
        
        # Initialize CARLA world with given configuration
        if self.render_mode == 'human':
            self.carla_api.initialize_pygame()
            self.carla_api.setup_camera()
        
        self.carla_api.initialize_components()
        
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

        self.episode_step = 0
        self.terminated = False
        self.truncated = False

        # ✅ REINICIAR CONEXIÓN SI ES NECESARIO
        self.episodes_since_restart += 1
        if self.episodes_since_restart >= self.restart_every:
            # self._restart_carla_connection()
            self.episodes_since_restart = 0
        else:
            # Reset normal
            if not self._initialized:
                self._setup_world()
            else:
                self.carla_api.respawn_vehicle(self.initial_spawn_point)
            
            
        
        # Respawn vehicle at initial spawn point
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
        #vehicle_location = self.vehicle.get_location()
        #distance_to_goal = telemetry.get('distance_to_goal', None)
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

        # Extraer variables del estado
        vt = speed      # velocidad (km/h)
        dt = distance_to_center      # distancia al centro del carril (metros)
        phi_t = angle   # ángulo con el carril (grados)
        # Convertir ángulo a radianes para cálculos trigonométricos
        phi_t_rad = np.deg2rad(phi_t)

        # Base reward: siempre positivo por sobrevivir
        reward = 0.1  # Pequeño bonus por cada step sin crash

        # Componentes del reward (Ecuación 19)
        reward_longitudinal = np.abs(vt * np.cos(phi_t_rad))  # Velocidad en dirección del carril
        reward_transversal = np.abs(vt * np.sin(phi_t_rad))   # Velocidad perpendicular (penalizar)
        reward_deviation = np.abs(vt * dt)                    # Desviación lateral ponderada
    
        # Reward total (Ecuación 19)
        reward += reward_longitudinal - reward_transversal - reward_deviation
        
        telemetry = self.carla_api.vehicle_controller.get_telemetry()
        speed = telemetry.get('speed', 0.0)
        #vehicle_location = self.vehicle.get_location()
        #distance_to_goal = telemetry.get('distance_to_goal', None)
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

        # Extraer variables del estado
        # ✅ ECUACIÓN 19 DEL PAPER
        vt = speed  # velocidad (km/h)
        dt = distance_to_center*10  # distancia al centro del carril (metros)
        phi_t = angle  # ángulo con el carril (grados)
        phi_t_rad = np.deg2rad(phi_t)
        

        # Componentes del reward (Ecuación 19)
        reward_longitudinal = np.abs(vt * np.cos(phi_t_rad))  # Velocidad en dirección del carril
        reward_transversal = np.abs(vt * np.sin(phi_t_rad))   # Velocidad perpendicular (penalizar)
        reward_deviation = np.abs(vt * dt)                    # Desviación lateral ponderada
    
        # Reward total (Ecuación 19)
        reward += reward_longitudinal - reward_transversal - reward_deviation
        
        # ✅ NORMALIZAR (el paper no lo hace, pero es necesario para DQN estable)
        SCALE_FACTOR = 50.0
        reward = reward / SCALE_FACTOR
        reward = np.clip(reward, -4.0, 2.0)
        
        # ✅ ECUACIÓN 18: Penalización colisión/salida
        GRACE_PERIOD = 5
        collision_occurred = self.collision_sensor.has_collision()
        
        if collision_occurred and self.episode_step >= GRACE_PERIOD:
            reward = -200.0 / SCALE_FACTOR  # -4.0 normalizado
            self.terminated = True
        
        # Lane invasion (salida de carril)
        current_invasion_count = self.lane_invasion_sensor.get_invasion_count()
        new_invasions = current_invasion_count - self.last_invasion_count
        
        if new_invasions > 0:
            reward -= (200.0 / SCALE_FACTOR) * new_invasions  # -4.0 por invasión
            self.last_invasion_count = current_invasion_count
            
            if current_invasion_count >= 2:  # Más estricto según paper
                self.terminated = True
        
        # Off-road detection
        if np.abs(distance_to_center) > lane_width * 0.95:
            reward = -200.0 / SCALE_FACTOR
            self.terminated = True
        
        self.episode_step += 1
        
        # ✅ ECUACIÓN 20: Bonus por llegar al objetivo
        # TODO: Implementar en Fase 3 cuando tengamos ruta con waypoints
        if self.episode_step >= self.max_episode_steps:
            reward += 100.0 / SCALE_FACTOR  # +2.0 normalizado
            self.terminated = True
        
        obs = self._get_observation()
        info = {
            'speed': speed,
            'distance_to_center': distance_to_center,
            'angle': angle,
            'collision': collision_occurred,
            'lane_invasions': current_invasion_count,
            'reward_components': {
                'longitudinal': reward_longitudinal / SCALE_FACTOR,
                'transversal': -reward_transversal / SCALE_FACTOR,
                'deviation': -reward_deviation / SCALE_FACTOR
            }
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
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Collect camera image + telemetry and return observation dict."""
        # camera
        # surface = self._app.camera_manager.get_surface()
        # if surface is not None:
        #     import pygame
        #     small = pygame.transform.scale(surface, (84, 84))
        #     arr = pygame.surfarray.array3d(small).transpose([1, 0, 2])
        #     image = arr.astype(np.uint8)
        # else:
        #     image = np.zeros((84, 84, 3), dtype=np.uint8)


        telemetry = self.carla_api.vehicle_controller.get_telemetry()
        lane_info = telemetry.get('lane_info', {}) or {}
    
        obs = np.array([
            telemetry.get('speed', 0.0),              # Velocidad en km/h
            lane_info.get('distance_to_center', 0.0), # Distancia lateral al centro (m)
            lane_info.get('angle', 0.0)               # Ángulo con el carril (grados)
        ], dtype=np.float32)
        
        return obs

        #return {
            # 'image': image,
        #    'speed': np.array([telemetry.get('speed', 0.0)], dtype=np.float32),
            # 'distance_to_goal': np.array([telemetry.get('distance_to_goal', 0.0) or 0.0], dtype=np.float32),
        #    'lane_distance': np.array([lane_info.get('distance_to_center', 0.0)], dtype=np.float32),
        #    'lane_angle': np.array([lane_info.get('angle', 0.0)], dtype=np.float32),
        #}


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
        'goal_point': {'x': 334.0, 'y': 43.0, 'radius': 5.0},
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
    n_episodes = 20000  # Más episodios para aprender
    max_t = 500      # Más steps por episodio
    eps_start = 1.0
    #eps_end = 0.01
    #eps_decay = 0.98  # ✅ Más rápido (antes 0.995)
    eps_decay = 0.9995  # Más lento
    eps_end = 0.05      # Mantener algo de exploración

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
    
    print("\n🚀 Starting DQN Training for Lane Keeping")
    print(f"Episodes: {n_episodes}, Max steps: {max_t}")
    print(f"Epsilon: {eps_start} → {eps_end} (decay: {eps_decay})")
    print("=" * 60)
    
    try:
        for i_episode in range(1, n_episodes + 1):
            state, info = env.reset()
            score = 0
            steps_taken = 0
            
            for t in range(max_t):
                # Select action
                action = agent.act(state, eps)
                
                # Take action
                next_state, reward, done, truncated, info = env.step(action)
                
                # 🔥 RENDER para ver el coche
                #if env.render_mode == 'human':
                #    if not env.render():
                #        print("\n⚠ Window closed by user")
                #        raise KeyboardInterrupt
                
                # Store experience and learn
                # ✅ Solo aprender después de 1000 experiences
                if len(agent.memory) > 1000:
                    agent.step(state, action, reward, next_state, done or truncated)
                else:
                    agent.memory.add(
                        agent._extract_telemetry(state),
                        action,
                        reward,
                        agent._extract_telemetry(next_state),
                        done or truncated
                    )
                
                # Update state and score
                state = next_state
                score += reward
                steps_taken = t + 1
                
                if done or truncated:
                    break
            
            # Update metrics
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay * eps)

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
        
        # Plot results
        if len(scores) > 0:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.plot(scores, alpha=0.6, label='Episode Score')
            if len(scores) >= 10:
                moving_avg = np.convolve(scores, np.ones(10)/10, mode='valid')
                plt.plot(range(9, len(scores)), moving_avg, 
                        label='Moving Average (10 episodes)', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.title('DQN Training Progress - Lane Keeping')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_progress.png', dpi=150)
            plt.close()
            
            print("✓ Training complete!")
            print(f"✓ Plot saved to training_progress.png")
            print(f"✓ Total episodes: {len(scores)}")
            print(f"✓ Final average score: {np.mean(list(scores_window)[-10:]):.2f}")