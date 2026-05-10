from collections import deque
import random
import time
from turtle import done
from typing import Dict, Optional
import gymnasium as gym

import carla
import numpy as np

from carla_app.carla_core import CarlaApi
from carla_app.sensores.collision_sensor import CollisionSensor
from agents.dqn import dqn
from carla_app.sensores.lane_invasion_sensor import LaneInvasionSensor
import time




class CarlaEnv(gym.Env):
    def __init__(
        self, 
        config, 
        render_mode: Optional[str] = None,
        max_episode_steps: int = 600,
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
        self.last_legal_invasions_count = 0
        self.last_illegal_invasions_count = 0
        self.last_unknown_invasions_count = 0
        self.initial_spawn_point = None
        self.route_waypoints_count = 0
        self.last_route_index = 0
        self.reward_mode = str(self.config.get('reward_mode', 'stable')).lower()
        self.random_route_enabled = bool(self.config.get('random_route_enabled', True))
        self.max_random_route_distance = float(self.config.get('max_random_route_distance', 200.0))
        self.min_random_route_distance = float(self.config.get('min_random_route_distance', 50.0))

        # continuous control: steer, throttle, brake
        # 9 steering × 4 vertical = 36 acciones discretas
        self.action_space = gym.spaces.Discrete(36)
        # Mapeo de acciones discretas a valores continuos
        self.action_mapping = self._build_action_mapping()

        self.prev_steer = 0.0

        # Paper-like waypoint state:
        # S = [d_t, phi_t, wp_x0, ..., wp_x14]
        # Total: 2 core + 15 waypoint X-coordinates = 17 dimensions
        core_dims = 2
        waypoint_dims = 15
        total_dims = core_dims + waypoint_dims
        
        low = np.zeros(total_dims, dtype=np.float32)
        high = np.zeros(total_dims, dtype=np.float32)
        
        # Core obs bounds: [d_t, phi_t]
        low[:2] = np.array([-1.0, -np.pi], dtype=np.float32)
        high[:2] = np.array([1.0, np.pi], dtype=np.float32)
        
        # Waypoint X bounds in ego frame, signed and normalized to [-1, 1]
        low[2:] = -1.0
        high[2:] = 1.0
        
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )
        self.waypoint_max_distance = 80.0  # meters, signed normalization

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

        # Plan initial route once world and vehicle are ready.
        try:
            self.carla_api.plan_route(start_location=self.vehicle.get_location())
            route_wps = self.carla_api.get_route_waypoints() or []
            self.route_waypoints_count = len(route_wps)
            if self.verbose:
                print(f"[DEBUG] initial route planned: {self.route_waypoints_count} waypoints")
        except Exception as e:
            self.route_waypoints_count = 0
            print(f"[WARN] could not plan initial route: {e}")
        
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
            if self.random_route_enabled:
                start_transform, goal_location, straight_distance = self._pick_random_start_goal(
                    max_distance=self.max_random_route_distance,
                    min_distance=self.min_random_route_distance
                )
                self.initial_spawn_point = start_transform
                self.carla_api.respawn_vehicle(start_transform)
            else:
                self.carla_api.respawn_vehicle(self.initial_spawn_point)

        # Plan route for this episode so waypoint list is always available.
        try:
            if self.random_route_enabled:
                self.carla_api.plan_route(
                    start_location=self.vehicle.get_location(),
                    end_location=goal_location,
                )
                if self.verbose:
                    print(
                        f"[DEBUG] random route: max_dist={self.max_random_route_distance:.1f}m "
                        f"straight_dist={straight_distance:.1f}m"
                    )
            else:
                self.carla_api.plan_route(start_location=self.vehicle.get_location())
            route_wps = self.carla_api.get_route_waypoints() or []
            self.route_waypoints_count = len(route_wps)
            if self.carla_api.draw_full_route_each_episode:
                self.carla_api.visualize_route(life_time=self.carla_api.route_draw_life_time)
            if self.carla_api.server_top_camera_enabled:
                self.carla_api.update_server_spectator_top_view(follow_vehicle=False)
            if self.verbose:
                print(f"[DEBUG] episode route planned: {self.route_waypoints_count} waypoints")
        except Exception as e:
            self.route_waypoints_count = 0
            print(f"[WARN] could not plan episode route: {e}")

        if self.collision_sensor:
            self.collision_sensor.reset()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.reset()
        
        # ✅ RESETEAR contadores de invasiones (legal/illegal)
        self.last_legal_invasions_count = 0
        self.last_illegal_invasions_count = 0
        self.last_unknown_invasions_count = 0
        self.last_route_index = 0

        # Importante: hacer tick para actualizar estado
        self.carla_api.world.tick()
        
        obs = self._get_observation()
        info = {
            'route_waypoints_count': self.route_waypoints_count,
        }
        if self.verbose:    
            print("✓ Environment reset")
        
        return obs, info

    def _pick_random_start_goal(self, max_distance=200.0, min_distance=50.0):
        """Select random start/goal spawn points with straight-line distance in [min_distance, max_distance]."""
        spawn_points = self.carla_api.world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            raise RuntimeError("Not enough spawn points to choose random start/goal")

        # Fast randomized search first.
        for _ in range(200):
            start_sp = random.choice(spawn_points)
            goal_sp = random.choice(spawn_points)
            if start_sp == goal_sp:
                continue
            dist = start_sp.location.distance(goal_sp.location)
            if min_distance <= dist <= max_distance:
                return start_sp, goal_sp.location, dist

        # Deterministic fallback if random attempts fail.
        valid_pairs = []
        for i, start_sp in enumerate(spawn_points):
            for j, goal_sp in enumerate(spawn_points):
                if i == j:
                    continue
                dist = start_sp.location.distance(goal_sp.location)
                if min_distance <= dist <= max_distance:
                    valid_pairs.append((start_sp, goal_sp.location, dist))

        if not valid_pairs:
            raise RuntimeError(
                f"No spawn point pairs found with distance in [{min_distance:.1f}m, {max_distance:.1f}m]"
            )

        return random.choice(valid_pairs)

    def _reward_conduccion_estable(self, v, e_y, e_psi, steer, delta_steer, speed_limit_kmh):
        """Reward base para conducción estable en rectas y curvas suaves."""
        k_v = 0.03
        k_y = 1.0
        k_psi = 0.5
        k_delta = 0.2
        k_s = 0.05

        v_capped = min(v, 80.0)
        v_forward = v_capped * np.cos(e_psi)

        reward = (
            (k_v * v_forward)
            - (k_y * e_y ** 2)
            - (k_psi * e_psi ** 2)
            - (k_delta * delta_steer ** 2)
            - (k_s * steer ** 2)
        )

        if v > speed_limit_kmh:
            speed_over = v - speed_limit_kmh
            reward -= 0.5 * (speed_over ** 2) / (speed_limit_kmh + 1e-3)

        return reward

    def _reward_conduccion_curvas(self, v, e_y, e_psi, steer, delta_steer, speed_limit_kmh, route_progress_delta):
        """Reward tolerante a giros: reduce castigo de steer en alta curvatura y ajusta velocidad objetivo."""
        curve_factor = float(np.clip(np.abs(e_psi) / (np.pi / 4.0), 0.0, 1.0))

        # En curvas permitimos más volante y más cambio de volante.
        k_v = 0.03
        k_y = 1.0
        k_psi = 0.6
        k_delta = 0.2 * (1.0 - 0.60 * curve_factor)
        k_s = 0.05 * (1.0 - 0.50 * curve_factor)

        v_capped = min(v, 80.0)
        v_forward = v_capped * np.cos(e_psi)
        reward = (
            (k_v * v_forward)
            - (k_y * e_y ** 2)
            - (k_psi * e_psi ** 2)
            - (k_delta * delta_steer ** 2)
            - (k_s * steer ** 2)
        )

        # Objetivo de velocidad dependiente de curvatura.
        target_speed = speed_limit_kmh * (1.0 - 0.45 * curve_factor)
        target_speed = max(20.0, target_speed)
        reward -= 0.8 * ((v - target_speed) / (speed_limit_kmh + 1e-3)) ** 2

        if v > speed_limit_kmh:
            speed_over = v - speed_limit_kmh
            reward -= 0.5 * (speed_over ** 2) / (speed_limit_kmh + 1e-3)

        # Recompensa explícita de progreso por waypoint, reforzada en curvas.
        # route_progress_delta > 0 significa que avanzó en la ruta.
        progress_gain = max(0.0, float(route_progress_delta))
        reward += (0.15 + 0.25 * curve_factor) * progress_gain

        return reward

    def step(self, action):
        """Apply action and return new state, reward, done, info."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if action not in self.action_mapping:
            raise ValueError(f"Invalid action {action}. Must be 0-35.")

        action_cmd = self.action_mapping[action]
        
        control = carla.VehicleControl(
            throttle=action_cmd['throttle'],
            steer=action_cmd['steering'],
            brake=action_cmd['brake'],
            reverse=action_cmd['reverse'],
            hand_brake=action_cmd['hand_brake']
        )
        
        # Apply to vehicle AND update vehicle_controller tracking
        self.vehicle.apply_control(control)
        self.carla_api.vehicle_controller.set_control(
            throttle=action_cmd['throttle'],
            steer=action_cmd['steering'],
            brake=action_cmd['brake'],
            hand_brake=action_cmd['hand_brake'],
            reverse=action_cmd['reverse']
        )
        
        if self.verbose:
            print(f"Step {self.episode_step}: Applied action {action} -> {action_cmd}")
        
        # advance simulation (synchronous)
        self.carla_api.world.tick()

        # Keep CARLA server viewport in bird's-eye mode during training.
        if self.carla_api.server_top_camera_enabled and self.carla_api.server_top_camera_follow:
            self.carla_api.update_server_spectator_top_view(
                follow_vehicle=self.carla_api.server_top_camera_follow
            )

        if self.render_mode == 'human':
            self.carla_api.camera_manager.update()

        # telemetry / observation
        # obs = self._get_observation()

        telemetry = self.carla_api.vehicle_controller.get_telemetry()
        speed = telemetry.get('speed', 0.0)
        vehicle_location = self.vehicle.get_location()
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

        lane_info = telemetry.get('lane_info', {}) or {}
        route_current_point = telemetry.get('route_current_point')
        next_5_waypoints = telemetry.get('next_5_waypoints') or []
        route_waypoints_count = telemetry.get('route_waypoints_count', 0)
        route_current_index = telemetry.get('route_current_index', 0)
        route_progress_delta = route_current_index - self.last_route_index
        self.last_route_index = route_current_index
        loc = telemetry.get('location')
        ctrl = telemetry.get('control')
        acc = telemetry.get('acceleration')
        ang_vel = telemetry.get('angular_velocity')
        loc_str = f"x={loc.x:.2f}  y={loc.y:.2f}  z={loc.z:.2f}" if loc else "None"
        acc_str = f"x={acc.x:.3f}  y={acc.y:.3f}  z={acc.z:.3f}" if acc else "None"
        ang_str = f"x={ang_vel.x:.3f}  y={ang_vel.y:.3f}  z={ang_vel.z:.3f}" if ang_vel else "None"
       
        route_point_str = "None"
        if route_current_point:
            route_point_str = (
                f"idx={route_current_point.get('index')} "
                f"xyz=({route_current_point.get('x',0):.2f}, {route_current_point.get('y',0):.2f}, {route_current_point.get('z',0):.2f}) "
                f"road={route_current_point.get('road_id')} lane={route_current_point.get('lane_id')} "
                f"s={route_current_point.get('s',0):.2f} opt={route_current_point.get('road_option')}"
            )

        next5_preview = "None"
        if next_5_waypoints:
            preview_items = []
            for p in next_5_waypoints[:5]:
                preview_items.append(
                    f"#{p.get('index')}({p.get('x',0):.1f},{p.get('y',0):.1f})"
                )
            next5_preview = " | ".join(preview_items)
        
        # Format local waypoints for display
        local_waypoints = telemetry.get('local_waypoints', []) or []
        local_wp_preview = "None"
        if local_waypoints:
            preview_items = []
            for i, (x, y) in enumerate(local_waypoints[:5]):
                preview_items.append(f"wp{i}:({x:6.2f},{y:+6.2f})")
            local_wp_preview = " | ".join(preview_items)

         #print(
            #f"\n{'='*65}\n"
            #f"  TELEMETRY  step={self.episode_step}\n"
             #f"{'='*65}\n"
             #f"  --- control (REWARD/OBS) ---\n"
             #f"  throttle         : {getattr(ctrl, 'throttle', '-')}\n"
             #f"  steer            : {getattr(ctrl, 'steer', '-')}\n"
            # f"  brake            : {getattr(ctrl, 'brake', '-')}\n"
             #f"  speed            : {telemetry.get('speed', 0.0):8.3f} km/h"
             #f"   (limit: {telemetry.get('speed_limit', '-')} km/h)\n"
             #f"  location         : {loc_str}\n"
             #f"  acceleration     : {acc_str}\n"
             #f"  angular_velocity : {ang_str}\n"
            #f"  engine_rpm       : {telemetry.get('engine_rpm', '-')}\n"
            #f"  gear             : {telemetry.get('gear', '-')}\n"
            #f"  drag             : {telemetry.get('drag', '-')}\n"
            #f"  traffic_light    : {telemetry.get('traffic_light_state', '-')}"
            #f"   at_light={telemetry.get('at_traffic_light', '-')}\n"
             #f"  waypoint_progress: {telemetry.get('waypoint_progress')}"
             #f"   ({route_current_index + 1}/{max(route_waypoints_count, 1)})\n"
             #f"  route_point      : {route_point_str}\n"
             #f"  next_5_waypoints : {next5_preview}\n"
             #f"  --- lane_info ---\n"
             #f"  lane_width       : {lane_info.get('lane_width', 0.0):8.3f} m\n"
             #f"  dist_to_center   : {lane_info.get('distance_to_center', 0.0):+8.3f} m\n"
             #f"  angle            : {lane_info.get('angle', 0.0):+8.3f} deg\n"
            #f"  side             : {lane_info.get('side', '-')}\n"
            #f"  is_centered      : {lane_info.get('is_centered', '-')}\n"
            #f"  is_aligned       : {lane_info.get('is_aligned', '-')}\n"
             #f"  --- lane_invasions (REWARD) ---\n"
             #f"  legal_invasions  : {self.lane_invasion_sensor.get_legal_invasions_count()}\n"
             #f"  illegal_invasions: {self.lane_invasion_sensor.get_illegal_invasions_count()}\n"
            #f"  unknown_invasions: {self.lane_invasion_sensor.get_unknown_invasions_count()}\n"
             #f"  --- local_waypoints (OBS) ---\n"
             #f"  next_5_local_wp  : {local_wp_preview}\n"
             #f"  all_local_wp_cnt : {len(local_waypoints)}\n"
             #f"{'='*65}"
         #)

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

        speed_limit_kmh = float(telemetry.get('speed_limit', 50.0))
        if self.reward_mode in ('curve', 'curve_aware', 'curvas'):
            reward = self._reward_conduccion_curvas(v, e_y, e_psi, steer, delta_steer, speed_limit_kmh, route_progress_delta)
        else:
            reward = self._reward_conduccion_estable(v, e_y, e_psi, steer, delta_steer, speed_limit_kmh)

        # actualiza memoria de steer para el siguiente paso/obs
        self.prev_steer = steer

        # Grace period to avoid false positives right after spawn/reset
        GRACE_PERIOD = 20

        termination_reason = None

        # ✅ FIX: terminar si el ángulo supera 90° — no hay recuperación posible
        if self.episode_step >= GRACE_PERIOD and np.abs(e_psi) > np.pi / 2:
            reward += -5.0
            self.terminated = True
            termination_reason = 'heading_error'

        collision_occurred = self.collision_sensor.has_collision()
        if collision_occurred and self.episode_step >= GRACE_PERIOD:
            reward += -10.0  # ✅ FIX: aditivo y escala razonable
            self.terminated = True
            termination_reason = 'collision'

        # Off-road detection (ojo lane_width == 0)
        if self.episode_step >= GRACE_PERIOD and (lane_width is not None) and (lane_width > 1e-3):
            if np.abs(distance_to_center) > lane_width * 0.95:
                reward += -10.0  # ✅ FIX: aditivo y escala razonable
                self.terminated = True
                termination_reason = 'offroad'

        # Lane invasion (salida de carril) - con clasificación legal/illegal
        current_legal_invasions = self.lane_invasion_sensor.get_legal_invasions_count()
        current_illegal_invasions = self.lane_invasion_sensor.get_illegal_invasions_count()
        
        new_legal_invasions = current_legal_invasions - self.last_legal_invasions_count
        new_illegal_invasions = current_illegal_invasions - self.last_illegal_invasions_count
        
        if new_illegal_invasions > 0:
            reward -= 8.0 * new_illegal_invasions  # ⚠️ Penalización mayor para ilegales
            self.last_illegal_invasions_count = current_illegal_invasions
        
        if new_legal_invasions > 0:
            reward -= 2.0 * new_legal_invasions  # ⚠️ Penalización menor para legales
            self.last_legal_invasions_count = current_legal_invasions
        
        # Terminar si demasiadas invasiones ilegales
        total_invasions = current_legal_invasions + current_illegal_invasions
        if self.episode_step >= GRACE_PERIOD and current_illegal_invasions >= 5:
            self.terminated = True
            termination_reason = 'illegal_lane_invasions'
        
        # ✅ Goal detection: si llegó al último waypoint (dentro de 5m)
        route_waypoints_count = telemetry.get('route_waypoints_count', 0)
        route_current_index = telemetry.get('route_current_index', 0)
        waypoints_remaining = route_waypoints_count - route_current_index
        
        # Si está cerca del último waypoint, lo consideramos goal
        if waypoints_remaining <= 1 and self.episode_step > 50:
            local_waypoints_x = telemetry.get('local_waypoints_x', []) or []
            if local_waypoints_x and len(local_waypoints_x) > 0:
                nearest_wp_x = local_waypoints_x[0]
                # Si el waypoint está muy cerca (dentro de 5m adelante o atrás)
                if abs(nearest_wp_x) < 5.0:
                    reward += 10.0  # ✅ Bonus por completar ruta
                    self.terminated = True
                    termination_reason = 'goal_reached'

        self.episode_step += 1

        # Time limit
        if self.episode_step >= self.max_episode_steps:
            self.truncated = True  
            if termination_reason is None:
                termination_reason = 'time_limit'

        # ✅ FIX: Reward clipping — acotar TD error
        reward = float(np.clip(reward, -10.0, 10.0))
        obs = self._get_observation()
        info = {
            'speed': speed,
            'distance_to_center': distance_to_center,
            'angle': angle,
            'collision': collision_occurred,
            'legal_invasions': self.lane_invasion_sensor.get_legal_invasions_count(),
            'illegal_invasions': self.lane_invasion_sensor.get_illegal_invasions_count(),
            'total_invasions': self.lane_invasion_sensor.get_invasion_count(),
            'termination_reason': termination_reason,
        }
        
        return obs, reward, self.terminated, self.truncated, info

    def _build_action_mapping(self) -> dict:
        """
        Construye el mapeo de acciones discretas (0-35) a comandos de control.
        
        Acciones mejoradas con brake:
        - Steering: 9 valores [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        - Control vertical: 4 opciones [full_throttle, half_throttle, coast, full_brake]
        - Total: 9 × 4 = 36 acciones (0-35)
        
        Returns:
            dict: Mapeo {action_id: [throttle, steering, brake, reverse, hand_brake]}
        """
        actions = {}
        action_id = 0
        
        # Steering: 9 valores
        steering_values = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Acciones verticales: throttle + coast + brake (4 opciones)
        vertical_actions = [
            {'throttle': 1.0, 'brake': 0.0},    # Full throttle
            {'throttle': 0.5, 'brake': 0.0},    # Half throttle
            {'throttle': 0.0, 'brake': 0.0},    # Coast
            {'throttle': 0.0, 'brake': 1.0},    # Full brake
        ]
        
        for vertical in vertical_actions:
            for steering in steering_values:
                actions[action_id] = {
                    'throttle': vertical['throttle'],
                    'steering': steering,
                    'brake': vertical['brake'],
                    'reverse': False,
                    'hand_brake': False
                }
                action_id += 1
        
        return actions
    
    def _get_observation(self):
        """Build paper-like state vector: [d_t, phi_t, wp_x0, ..., wp_x14]."""
        telemetry = self.carla_api.vehicle_controller.get_telemetry()
        lane_info = telemetry.get('lane_info', {}) or {}
        local_waypoints_x = telemetry.get('local_waypoints_x', []) or []

        v = float(telemetry.get('speed', 0.0))  # km/h
        dt = float(lane_info.get('distance_to_center', 0.0))  # m
        lane_w = float(lane_info.get('lane_width', 0.0))      # m
        angle_deg = float(lane_info.get('angle', 0.0))        # deg

        denom = max(lane_w / 2.0, 1e-3)
        e_y = np.clip(dt / denom, -1.0, 1.0)
        e_psi = np.deg2rad(angle_deg)
        
        # Normalize signed waypoint x distances to [-1, 1]
        # Pad with zeros if we don't have 15 waypoints
        wp_normalized = []
        for i in range(15):
            if i < len(local_waypoints_x):
                # Keep sign to preserve left/right (or ahead/behind) information
                wp_x = float(local_waypoints_x[i])
                wp_norm = np.clip(wp_x / self.waypoint_max_distance, -1.0, 1.0)
                wp_normalized.append(wp_norm)
            else:
                wp_normalized.append(0.0)  # Pad with zeros
        
        # State vector: [d_t, phi_t, wp_x0, ..., wp_x14]
        obs = np.array(
            [e_y, e_psi] + wp_normalized,
            dtype=np.float32
        )
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
