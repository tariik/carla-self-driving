"""
Lane detection and position calculation.
"""

import carla
import math
from typing import Optional


class LaneDetector:
    """Detect vehicle position relative to lane center."""
    
    def __init__(self, world: carla.World):
        """
        Initialize lane detector.
        
        Args:
            world: CARLA world instance
        """
        self.world = world
        self.map = world.get_map()
    
    def get_distance_to_center(self, vehicle: carla.Vehicle) -> Optional[float]:
        """
        Get lateral distance from vehicle to lane center.
        
        Args:
            vehicle: Vehicle actor
            
        Returns:
            Distance in meters (positive = right, negative = left)
            None if not on a driveable lane
        """
        vehicle_location = vehicle.get_location()
        
        # Get nearest driveable waypoint
        waypoint = self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if waypoint is None:
            return None
        
        # Calculate 2D distance
        lane_center = waypoint.transform.location
        dx = vehicle_location.x - lane_center.x
        dy = vehicle_location.y - lane_center.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Determine side using cross product
        lane_forward = waypoint.transform.get_forward_vector()
        cross = lane_forward.x * dy - lane_forward.y * dx
        
        return distance if cross > 0 else -distance
    
    def get_lane_info(self, vehicle: carla.Vehicle) -> Optional[dict]:
        """
        Get comprehensive lane position information.
        
        Args:
            vehicle: Vehicle actor
            
        Returns:
            Dictionary with lane info or None if not on road
        """
        vehicle_location = vehicle.get_location()
        vehicle_transform = vehicle.get_transform()
        
        waypoint = self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if waypoint is None:
            return None
        
        # Calculate distance
        lane_center = waypoint.transform.location
        dx = vehicle_location.x - lane_center.x
        dy = vehicle_location.y - lane_center.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Determine side
        lane_forward = waypoint.transform.get_forward_vector()
        cross = lane_forward.x * dy - lane_forward.y * dx
        signed_distance = distance if cross > 0 else -distance
        
        # Calculate as percentage of lane width
        lane_width = waypoint.lane_width
        distance_percentage = (abs(signed_distance) / (lane_width / 2)) * 100 if lane_width > 0 else 0
        
        # Calculate angle between vehicle and lane direction
        vehicle_forward = vehicle_transform.get_forward_vector()
        
        # Dot product for angle
        dot = vehicle_forward.x * lane_forward.x + vehicle_forward.y * lane_forward.y
        dot = max(-1.0, min(1.0, dot))  # Clamp to avoid math domain errors
        angle_rad = math.acos(dot)
        angle_deg = math.degrees(angle_rad)
        
        # Determine if angle is left or right using cross product
        cross_angle = vehicle_forward.x * lane_forward.y - vehicle_forward.y * lane_forward.x
        signed_angle = angle_deg if cross_angle > 0 else -angle_deg
        
        return {
            'distance_to_center': signed_distance,
            'distance_abs': abs(signed_distance),
            'lane_width': lane_width,
            'distance_percentage': min(distance_percentage, 100),
            'is_centered': distance_percentage < 20,  # Within 20% is "centered"
            'side': 'right' if signed_distance > 0 else 'left',
            'lane_id': waypoint.lane_id,
            'road_id': waypoint.road_id,
            'waypoint': waypoint,
            'angle': signed_angle,  # Angle in degrees (+ = right, - = left)
            'angle_abs': abs(signed_angle),
            'is_aligned': abs(signed_angle) < 5.0  # Within 5 degrees is aligned
        }
    
    def is_vehicle_centered(self, vehicle: carla.Vehicle, threshold: float = 0.5) -> bool:
        """
        Check if vehicle is centered in lane.
        
        Args:
            vehicle: Vehicle actor
            threshold: Max distance from center in meters (default: 0.5m)
            
        Returns:
            True if within threshold of center
        """
        distance = self.get_distance_to_center(vehicle)
        if distance is None:
            return False
        return abs(distance) < threshold
