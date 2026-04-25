"""
Vehicle Controller module for CARLA Manual Control.

Handles vehicle control and telemetry data.
"""

from typing import Optional, List

import carla
import pygame

from carla_app.lane_detector import LaneDetector



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
        self.current_waypoint_index: int = 0
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
    
    def set_route(self, waypoints: List[carla.Waypoint]) -> None:
        """
        Set the route waypoints.
        
        Args:
            waypoints: List of waypoints forming the route
        """
        self.route_waypoints = waypoints
        self.current_waypoint_index = 0
    
    def _update_waypoint_progress(self) -> None:
        """Update current waypoint based on vehicle position."""
        if not self.route_waypoints or self.vehicle is None:
            return
        
        vehicle_loc = self.vehicle.get_location()
        
        # Check if we're close to next waypoint
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            next_wp = self.route_waypoints[self.current_waypoint_index + 1]
            distance = vehicle_loc.distance(next_wp.transform.location)
            
            # Move to next waypoint if close enough (5 meters)
            if distance < 5.0:
                self.current_waypoint_index += 1
    
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
        
        # Calculate distance to goal if configured
        distance_to_goal = None
        goal_reached = False
        
        if self.config['goal_point']['x'] is not None and self.config['goal_point']['y'] is not None:
            dx = location.x - self.config['goal_point']['x']
            dy = location.y - self.config['goal_point']['y']
            distance_to_goal = (dx**2 + dy**2)**0.5
            goal_reached = distance_to_goal <= self.config['goal_point']['radius']
        
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
            lane_info = self.lane_detector.get_lane_info(self.vehicle)
        
        return {
            'speed': speed_kmh,
            'location': location,
            'control': self.control,
            'distance_to_goal': distance_to_goal,
            'goal_reached': goal_reached,
            'current_waypoint': current_wp_loc,
            'next_waypoint': next_wp_loc,
            'waypoint_progress': waypoint_progress,
            'lane_info': lane_info
        }
