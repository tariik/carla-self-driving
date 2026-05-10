"""
Lane detection and position calculation.
"""

import carla
import math
from typing import Optional, Sequence


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

    def _get_reference_waypoint(
        self,
        vehicle_location: carla.Location,
        candidate_waypoints: Optional[Sequence[carla.Waypoint]] = None,
    ) -> Optional[carla.Waypoint]:
        """Return the best waypoint reference, preferring the active route slice when available."""
        if candidate_waypoints:
            return min(
                candidate_waypoints,
                key=lambda wp: vehicle_location.distance(wp.transform.location)
            )

        return self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

    @staticmethod
    def _signed_lateral_distance(
        vehicle_location: carla.Location,
        waypoint: carla.Waypoint,
    ) -> float:
        """Project vehicle offset onto the lane right axis to get a signed lateral error."""
        lane_center = waypoint.transform.location
        lane_forward = waypoint.transform.get_forward_vector()

        dx = vehicle_location.x - lane_center.x
        dy = vehicle_location.y - lane_center.y

        right_x = -lane_forward.y
        right_y = lane_forward.x
        return dx * right_x + dy * right_y

    @staticmethod
    def _signed_heading_error(
        vehicle_transform: carla.Transform,
        waypoint: carla.Waypoint,
    ) -> float:
        """Return signed heading error in degrees using atan2(cross, dot)."""
        vehicle_forward = vehicle_transform.get_forward_vector()
        lane_forward = waypoint.transform.get_forward_vector()

        dot = vehicle_forward.x * lane_forward.x + vehicle_forward.y * lane_forward.y
        cross = vehicle_forward.x * lane_forward.y - vehicle_forward.y * lane_forward.x
        return math.degrees(math.atan2(cross, dot))
    
    def get_distance_to_center(
        self,
        vehicle: carla.Vehicle,
        candidate_waypoints: Optional[Sequence[carla.Waypoint]] = None,
    ) -> Optional[float]:
        """
        Get lateral distance from vehicle to lane center.
        
        Args:
            vehicle: Vehicle actor
            
        Returns:
            Distance in meters (positive = right, negative = left)
            None if not on a driveable lane
        """
        vehicle_location = vehicle.get_location()
        waypoint = self._get_reference_waypoint(vehicle_location, candidate_waypoints)
        
        if waypoint is None:
            return None

        return self._signed_lateral_distance(vehicle_location, waypoint)
    
    def get_lane_info(
        self,
        vehicle: carla.Vehicle,
        candidate_waypoints: Optional[Sequence[carla.Waypoint]] = None,
    ) -> Optional[dict]:
        """
        Get comprehensive lane position information.
        
        Args:
            vehicle: Vehicle actor
            
        Returns:
            Dictionary with lane info or None if not on road
        """
        vehicle_location = vehicle.get_location()
        vehicle_transform = vehicle.get_transform()
        waypoint = self._get_reference_waypoint(vehicle_location, candidate_waypoints)
        
        if waypoint is None:
            return None

        signed_distance = self._signed_lateral_distance(vehicle_location, waypoint)
        
        # Calculate as percentage of lane width
        lane_width = waypoint.lane_width
        distance_percentage = (abs(signed_distance) / (lane_width / 2)) * 100 if lane_width > 0 else 0
        signed_angle = self._signed_heading_error(vehicle_transform, waypoint)
        
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
            'angle': signed_angle,  # Signed heading error in degrees
            'angle_abs': abs(signed_angle),
            'is_aligned': abs(signed_angle) < 5.0  # Within 5 degrees is aligned
        }
    
    def is_vehicle_centered(
        self,
        vehicle: carla.Vehicle,
        threshold: float = 0.5,
        candidate_waypoints: Optional[Sequence[carla.Waypoint]] = None,
    ) -> bool:
        """
        Check if vehicle is centered in lane.
        
        Args:
            vehicle: Vehicle actor
            threshold: Max distance from center in meters (default: 0.5m)
            
        Returns:
            True if within threshold of center
        """
        distance = self.get_distance_to_center(vehicle, candidate_waypoints)
        if distance is None:
            return False
        return abs(distance) < threshold
