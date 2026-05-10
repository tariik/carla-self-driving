#!/usr/bin/env python3
"""
Test script to verify legal/illegal lane invasion classification logic.
Tests the classification method with various scenarios.
"""

import sys

# Mock the LaneInvasionSensor classification method
class MockInvasionSensor:
    """Mock sensor to test classification logic."""
    
    def __init__(self):
        self.last_vehicle_heading = 0.0
    
    @staticmethod
    def _determine_crossing_direction(heading_delta):
        """Determine crossing direction based on heading change."""
        # Normalize to [-180, 180]
        while heading_delta > 180:
            heading_delta -= 360
        while heading_delta < -180:
            heading_delta += 360
        
        if heading_delta >= 0:
            return 'Left'
        else:
            return 'Right'
    
    @staticmethod
    def _classify_invasion(crossing_direction, lane_change_str):
        """Classify if invasion is legal or illegal."""
        if 'None' in lane_change_str or lane_change_str == 'None':
            return 'illegal'
        elif 'Both' in lane_change_str:
            return 'legal'
        elif 'Left' in lane_change_str and crossing_direction == 'Left':
            return 'legal'
        elif 'Right' in lane_change_str and crossing_direction == 'Right':
            return 'legal'
        else:
            return 'illegal'


def test_classification():
    """Test various scenarios."""
    sensor = MockInvasionSensor()
    
    test_cases = [
        # (crossing_direction, lane_change, expected_result, description)
        ('Left', 'Both', 'legal', 'Can change both directions, crossing left'),
        ('Right', 'Both', 'legal', 'Can change both directions, crossing right'),
        ('Left', 'Left', 'legal', 'Can change left, crossing left'),
        ('Right', 'Right', 'legal', 'Can change right, crossing right'),
        ('Left', 'Right', 'illegal', 'Can only change right, but crossing left'),
        ('Right', 'Left', 'illegal', 'Can only change left, but crossing right'),
        ('Left', 'None', 'illegal', 'Cannot change lanes, crossing left'),
        ('Right', 'None', 'illegal', 'Cannot change lanes, crossing right'),
    ]
    
    print("Testing Lane Invasion Classification Logic")
    print("=" * 70)
    
    all_passed = True
    for crossing_dir, lane_change, expected, description in test_cases:
        result = sensor._classify_invasion(crossing_dir, lane_change)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result != expected:
            all_passed = False
        
        print(
            f"{status} | Direction: {crossing_dir:<6} | LaneChange: {lane_change:<6} | "
            f"Result: {result:<8} (expected {expected:<8}) | {description}"
        )
    
    print("=" * 70)
    
    # Test heading delta calculation
    print("\nTesting Heading Delta Calculation")
    print("=" * 70)
    
    heading_tests = [
        (0, 'Left', 'Moving left (yaw increases)'),
        (45, 'Left', 'Strong left turn'),
        (90, 'Left', 'Extreme left turn'),
        (-45, 'Right', 'Right turn'),
        (-90, 'Right', 'Extreme right turn'),
        (0, 'Left', 'No change detected as left'),
        (170, 'Left', 'Near-wrap positive'),
        (-170, 'Right', 'Near-wrap negative'),
    ]
    
    for heading_delta, expected_dir, description in heading_tests:
        result_dir = sensor._determine_crossing_direction(heading_delta)
        status = "✓ PASS" if result_dir == expected_dir else "✗ FAIL"
        
        if result_dir != expected_dir:
            all_passed = False
        
        print(f"{status} | Heading Δ: {heading_delta:>4}° → {result_dir:<6} | {description}")
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(test_classification())
