"""
Vehicle Controller module for CARLA Manual Control.

Handles vehicle control and telemetry data.
"""

from typing import Optional, List, Dict, Any, Tuple

import carla
import pygame
import numpy as np

from carla_app.sensores.lane_detector import LaneDetector



class VehicleController:
    """Handles vehicle control and telemetry."""
    
    def __init__(self, config):
        """
        Initialize vehicle controller.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.vehicle: Optional[carla.Vehicle] = None
        self.control = carla.VehicleControl()
        self.world: Optional[carla.World] = None
        self.route_waypoints: List[carla.Waypoint] = []
        self.route_trace: List[Tuple[carla.Waypoint, Any]] = []
        self.route_debug_points: List[Dict[str, Any]] = []
        self.current_waypoint_index: int = 0
        self.next_route_points: List[Dict[str, Any]] = []
        self.local_waypoints: List[Tuple[float, float]] = []  # ego-centric coordinates
        self.lane_detector: Optional[LaneDetector] = None
    
    def set_vehicle(self, vehicle: carla.Vehicle) -> None:
        """
        Set the vehicle to control.
        
        Args:
            vehicle: CARLA vehicle instance
        """
        self.vehicle = vehicle
    
    def set_world(self, world: carla.World) -> None:
        """
        Set the CARLA world.
        
        Args:
            world: CARLA world instance
        """
        self.world = world
        self.lane_detector = LaneDetector(world)
        
    
    def _build_route_debug_points(self, route_trace: List[Tuple[carla.Waypoint, Any]]) -> List[Dict[str, Any]]:
        """Build a fully expanded route list with XYZ and waypoint metadata."""
        debug_points: List[Dict[str, Any]] = []
        for i, (wp, road_option) in enumerate(route_trace):
            loc = wp.transform.location
            rot = wp.transform.rotation
            debug_points.append({
                'index': i,
                'x': float(loc.x),
                'y': float(loc.y),
                'z': float(loc.z),
                'road_id': int(wp.road_id),
                'section_id': int(wp.section_id),
                'lane_id': int(wp.lane_id),
                's': float(wp.s),
                'lane_width': float(wp.lane_width),
                'is_junction': bool(wp.is_junction),
                'junction_id': int(wp.junction_id),
                'yaw': float(rot.yaw),
                'pitch': float(rot.pitch),
                'roll': float(rot.roll),
                'road_option': getattr(road_option, 'name', str(road_option)),
            })
        return debug_points

    def set_route(
        self,
        waypoints: List[carla.Waypoint],
        route_trace: Optional[List[Tuple[carla.Waypoint, Any]]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Set the route waypoints.
        
        Args:
            waypoints: List of waypoints forming the route
            route_trace: Optional full route [(waypoint, RoadOption), ...]
            verbose: Print full route details when True
        """
        self.route_waypoints = waypoints
        self.route_trace = route_trace or [(wp, 'LANEFOLLOW') for wp in waypoints]
        self.route_debug_points = self._build_route_debug_points(self.route_trace)
        self.current_waypoint_index = 0
        self.next_route_points = self.route_debug_points[:15]  # 15 waypoints for lookahead

        if verbose and self.route_debug_points:
            start = self.route_debug_points[0]
            end = self.route_debug_points[-1]
            print(
                "[ROUTE] "
                f"wps={len(self.route_debug_points)} | "
                f"start=({start['x']:.1f},{start['y']:.1f}) | "
                f"end=({end['x']:.1f},{end['y']:.1f}) | "
                f"lookahead={len(self.next_route_points)}"
            )

    def get_route_debug_points(self) -> List[Dict[str, Any]]:
        """Return the fully expanded route list with xyz + metadata."""
        return self.route_debug_points

    def _get_active_route_waypoints(self) -> List[carla.Waypoint]:
        """Return a forward-looking slice of route waypoints for local tracking."""
        if not self.route_waypoints:
            return []

        start = self.current_waypoint_index
        end = min(start + 15, len(self.route_waypoints))  # 15 waypoint lookahead
        return self.route_waypoints[start:end]
    
    def _update_waypoint_progress(self) -> None:
        """Track nearest route waypoint and keep a rolling window of next 15 points."""
        if not self.route_waypoints or self.vehicle is None:
            self.next_route_points = []
            self.local_waypoints = []
            return
        
        vehicle_loc = self.vehicle.get_location()

        nearest_index = min(
            range(len(self.route_waypoints)),
            key=lambda i: vehicle_loc.distance(self.route_waypoints[i].transform.location)
        )
        self.current_waypoint_index = nearest_index

        start = nearest_index
        end = min(nearest_index + 15, len(self.route_debug_points))  # 15 waypoint window
        self.next_route_points = self.route_debug_points[start:end]
        
        # Update local waypoints (ego-centric)
        self.local_waypoints = self._waypoints_to_local(count=15)
    
    def process_input(self, keys: pygame.key.ScancodeWrapper) -> None:
        """
        Process keyboard input and update vehicle control.
        
        Args:
            keys: Pygame key state
        """
        # Reset controls
        self.control.throttle = 0.0
        self.control.brake = 0.0
        self.control.steer = 0.0
        self.control.hand_brake = False
        
        # Throttle
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.control.throttle = self.config['throttle_speed']
        
        # Brake
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.control.brake = self.config['brake_speed']
        
        # Steering
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.control.steer = -self.config['steer_speed']
        
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.control.steer = self.config['steer_speed']
        
        # Hand brake
        if keys[pygame.K_SPACE]:
            self.control.hand_brake = True
            self.control.brake = 1.0
    
    def set_control(self, throttle: float, steer: float, brake: float, hand_brake: bool = False, reverse: bool = False) -> None:
        """
        Set vehicle control values directly (from DQN agent or code).
        
        Args:
            throttle: Throttle value (0.0 to 1.0)
            steer: Steering value (-1.0 to 1.0)
            brake: Brake value (0.0 to 1.0)
            hand_brake: Hand brake flag
            reverse: Reverse flag
        """
        self.control.throttle = float(throttle)
        self.control.steer = float(steer)
        self.control.brake = float(brake)
        self.control.hand_brake = bool(hand_brake)
        self.control.reverse = bool(reverse)
    
    def _waypoints_to_local(self, count: int = 15) -> List[Tuple[float, float]]:
        """
        Convert global route waypoints to ego-centric (local) coordinates.
        
        Applies transformation (rotation + translation):
        - Translation: subtract vehicle position
        - Rotation: rotate by vehicle heading (yaw)
        
        Formula:
            x_local = (x_global - x_veh) * cos(yaw) + (y_global - y_veh) * sin(yaw)
            y_local = -(x_global - x_veh) * sin(yaw) + (y_global - y_veh) * cos(yaw)
        
        Args:
            count: Number of waypoints to convert (default 15)
        
        Returns:
            List of (x_local, y_local) tuples for next waypoints
        """
        if not self.route_debug_points or self.vehicle is None:
            return []
        
        vehicle_pos = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_yaw_rad = np.deg2rad(vehicle_transform.rotation.yaw)
        
        cos_y = np.cos(vehicle_yaw_rad)
        sin_y = np.sin(vehicle_yaw_rad)
        
        local_waypoints = []
        
        # Get next 'count' waypoints starting from current index
        for i in range(self.current_waypoint_index, min(self.current_waypoint_index + count, len(self.route_debug_points))):
            wp = self.route_debug_points[i]
            
            # Translation
            delta_x = wp['x'] - vehicle_pos.x
            delta_y = wp['y'] - vehicle_pos.y
            
            # Rotation (yaw)
            local_x = delta_x * cos_y + delta_y * sin_y
            local_y = -delta_x * sin_y + delta_y * cos_y
            
            local_waypoints.append((local_x, local_y))
        
        return local_waypoints
    
    def get_discrete_action(self) -> int:
        """
        Convierte controles continuos actuales a acción discreta (0-26) para DQN.
        
        Mapeo DQN:
        - Steering: 9 valores [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        - Throttle: 3 valores [0, 0.5, 1]
        - Total: 9 × 3 = 27 acciones (0-26)
        
        Returns:
            action_id (0-26)
        """
        steering = self.control.steer  # -steer_speed a steer_speed
        throttle = self.control.throttle  # 0 a throttle_speed
        
        # Normalizar steering a rango [-1, 1]
        steer_speed = self.config.get('steer_speed', 1.0)
        steering_norm = steering / steer_speed if steer_speed > 0 else 0.0
        steering_norm = max(-1.0, min(1.0, steering_norm))
        
        # Normalizar throttle a rango [0, 1]
        throttle_speed = self.config.get('throttle_speed', 1.0)
        throttle_norm = throttle / throttle_speed if throttle_speed > 0 else 0.0
        throttle_norm = max(0.0, min(1.0, throttle_norm))
        
        # Mapear steering normalizado (-1 a 1) a índice (0-8)
        steer_idx = max(0, min(8, int(round((steering_norm + 1.0) / 0.25))))
        
        # Mapear throttle normalizado (0 a 1) a índice (0-2)
        if throttle_norm < 0.25:
            throttle_idx = 0
        elif throttle_norm < 0.75:
            throttle_idx = 1
        else:
            throttle_idx = 2
        
        # Convertir a action_id (0-26)
        action_id = throttle_idx * 9 + steer_idx
        
        return min(26, action_id)
    
    def apply_control(self) -> None:
        """Apply current control to vehicle."""
        if self.vehicle is not None:
            self.vehicle.apply_control(self.control)
    
    def get_telemetry(self) -> dict:
        """
        Get vehicle telemetry data.
        
        Returns:
            Dictionary with speed, location, and control data
        """
        if self.vehicle is None:
            return {}
        
        # Update waypoint progress
        self._update_waypoint_progress()
        
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        
        location = self.vehicle.get_location()
        transform = self.vehicle.get_transform()
        acceleration = self.vehicle.get_acceleration()
        angular_velocity = self.vehicle.get_angular_velocity()
        
        # Engine / transmission telemetry
        td = self.vehicle.get_telemetry_data()
        engine_rpm = td.engine_rpm
        gear = td.gear
        drag = td.drag
        
        # Traffic
        speed_limit_kmh = self.vehicle.get_speed_limit()
        at_traffic_light = self.vehicle.is_at_traffic_light()
        traffic_light_state = str(self.vehicle.get_traffic_light_state())

        # Per-wheel telemetry (FL, FR, RL, RR)
        wheels_raw = td.wheels
        wheels = []
        for w in wheels_raw:
            wheels.append({
                'omega':                w.omega,
                'torque':               w.torque,
                'lat_force':            w.lat_force,
                'long_force':           w.long_force,
                'lat_slip':             w.lat_slip,
                'long_slip':            w.long_slip,
                'tire_friction':        w.tire_friction,
                'tire_load':            w.tire_load,
                'normalized_tire_load': w.normalized_tire_load,
            })
        
        # Waypoint info
        current_wp_loc = None
        next_wp_loc = None
        waypoint_progress = None
        
        if self.route_waypoints:
            if self.current_waypoint_index < len(self.route_waypoints):
                current_wp = self.route_waypoints[self.current_waypoint_index]
                current_wp_loc = current_wp.transform.location
            
            if self.current_waypoint_index + 1 < len(self.route_waypoints):
                next_wp = self.route_waypoints[self.current_waypoint_index + 1]
                next_wp_loc = next_wp.transform.location
            
            waypoint_progress = f"{self.current_waypoint_index + 1}/{len(self.route_waypoints)}"
        
        # Lane info
        lane_info = None
        if self.lane_detector is not None:
            lane_info = self.lane_detector.get_lane_info(
                self.vehicle,
                candidate_waypoints=self._get_active_route_waypoints()
            )
        
        return {
            # kinematics
            'speed': speed_kmh,
            'velocity': velocity,
            'location': location,
            'transform': transform,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            # engine / transmission
            'engine_rpm': engine_rpm,
            'gear': gear,
            'drag': drag,
            # control inputs
            'control': self.control,
            # traffic
            'speed_limit': speed_limit_kmh,
            'at_traffic_light': at_traffic_light,
            'traffic_light_state': traffic_light_state,
            # wheels
            'wheels': wheels,
            # navigation
            'current_waypoint': current_wp_loc,
            'next_waypoint': next_wp_loc,
            'waypoint_progress': waypoint_progress,
            'route_waypoints_count': len(self.route_waypoints),
            'route_current_index': self.current_waypoint_index,
            'route_current_point': self.route_debug_points[self.current_waypoint_index] if self.route_debug_points else None,
            'next_5_waypoints': self.next_route_points,  # Keep for backward compat
            'local_waypoints': self.local_waypoints,  # ego-centric (x, y) pairs
            'local_waypoints_x': [wp[0] for wp in self.local_waypoints],  # only X for DQN
            # lane
            'lane_info': lane_info,
        }
