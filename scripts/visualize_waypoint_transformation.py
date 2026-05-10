#!/usr/bin/env python3
"""
Visualize ego-centric waypoint transformation.

Shows how global waypoints are converted to local (vehicle-relative) coordinates,
demonstrating translation invariance for DQN training.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec


def global_to_local_waypoints(global_wps, vehicle_pos, vehicle_yaw_rad):
    """
    Convert global waypoints to ego-centric coordinates.
    
    Args:
        global_wps: List of (x, y) global positions
        vehicle_pos: (x, y) vehicle position
        vehicle_yaw_rad: Vehicle heading in radians
    
    Returns:
        List of (x, y) local positions
    """
    cos_y = np.cos(vehicle_yaw_rad)
    sin_y = np.sin(vehicle_yaw_rad)
    
    local_wps = []
    for gx, gy in global_wps:
        # Translation
        dx = gx - vehicle_pos[0]
        dy = gy - vehicle_pos[1]
        
        # Rotation
        lx = dx * cos_y + dy * sin_y
        ly = -dx * sin_y + dy * cos_y
        
        local_wps.append((lx, ly))
    
    return local_wps


def plot_scenario(ax, title, global_wps, vehicle_pos, vehicle_yaw_deg, is_global=True):
    """Plot either global or local waypoint view."""
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    vehicle_yaw_rad = np.deg2rad(vehicle_yaw_deg)
    
    if is_global:
        # Global view
        ax.scatter(*zip(*global_wps), c='blue', s=50, marker='o', label='Global waypoints', zorder=3)
        
        # Draw path
        wps_array = np.array(global_wps)
        ax.plot(wps_array[:, 0], wps_array[:, 1], 'b--', alpha=0.5, linewidth=1)
        
        # Vehicle
        vehicle_rect = patches.FancyBboxPatch(
            (vehicle_pos[0] - 1, vehicle_pos[1] - 0.5), 2, 1,
            boxstyle="round,pad=0.1",
            edgecolor='red', facecolor='lightcoral', linewidth=2, zorder=5
        )
        ax.add_patch(vehicle_rect)
        
        # Heading arrow
        arrow_len = 3
        arrow_end_x = vehicle_pos[0] + arrow_len * np.cos(vehicle_yaw_rad)
        arrow_end_y = vehicle_pos[1] + arrow_len * np.sin(vehicle_yaw_rad)
        ax.arrow(vehicle_pos[0], vehicle_pos[1], 
                arrow_end_x - vehicle_pos[0], arrow_end_y - vehicle_pos[1],
                head_width=0.5, head_length=0.4, fc='red', ec='red', linewidth=2, zorder=4)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper left')
        
    else:
        # Local view
        local_wps = global_to_local_waypoints(global_wps, vehicle_pos, vehicle_yaw_rad)
        
        ax.scatter(*zip(*local_wps), c='green', s=50, marker='o', label='Local waypoints', zorder=3)
        
        # Draw path
        wps_array = np.array(local_wps)
        ax.plot(wps_array[:, 0], wps_array[:, 1], 'g--', alpha=0.5, linewidth=1)
        
        # Vehicle at origin
        vehicle_rect = patches.FancyBboxPatch(
            (-1, -0.5), 2, 1,
            boxstyle="round,pad=0.1",
            edgecolor='red', facecolor='lightcoral', linewidth=2, zorder=5
        )
        ax.add_patch(vehicle_rect)
        
        # Heading arrow (always pointing right in local frame)
        ax.arrow(0, 0, 3, 0,
                head_width=0.5, head_length=0.4, fc='red', ec='red', linewidth=2, zorder=4)
        
        # Axes
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
        
        ax.set_xlabel('X local (m)')
        ax.set_ylabel('Y local (m)')
        ax.legend(loc='upper left')
        
        # Print local coordinates
        info_text = "Local Waypoints (X only for DQN):\n"
        for i, (x, y) in enumerate(local_wps[:5]):
            info_text += f"  #{i}: x={x:6.2f}m, y={y:6.2f}m\n"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_translation_invariance():
    """
    Demonstrate that local waypoints are invariant to vehicle position.
    Same trajectory at different locations produces identical local views.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Scenario 1: Straight road, vehicle going forward
    print("Scenario 1: Straight road (vehicle going forward)")
    print("=" * 60)
    
    # Global waypoints: straight line
    global_wps_1 = [(i, 0) for i in range(10, 40, 5)]  # 10, 15, 20, ..., 35
    
    # Vehicle position 1
    vehicle_pos_1 = (0, 0)
    vehicle_yaw_1 = 0
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_scenario(ax1, "Scenario 1a: Global View (Vehicle at origin)", 
                 global_wps_1, vehicle_pos_1, vehicle_yaw_1, is_global=True)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_scenario(ax2, "Scenario 1a: Local View (ego-centric)",
                 global_wps_1, vehicle_pos_1, vehicle_yaw_1, is_global=False)
    
    # Same trajectory, different vehicle location
    vehicle_pos_2 = (100, 50)
    vehicle_yaw_2 = 0
    
    ax3 = fig.add_subplot(gs[1, 0])
    plot_scenario(ax3, "Scenario 1b: Global View (Vehicle at (100,50))",
                 [(x + 100, y + 50) for x, y in global_wps_1], vehicle_pos_2, vehicle_yaw_2, is_global=True)
    
    ax4 = fig.add_subplot(gs[1, 1])
    plot_scenario(ax4, "Scenario 1b: Local View (IDENTICAL to 1a!)",
                 [(x + 100, y + 50) for x, y in global_wps_1], vehicle_pos_2, vehicle_yaw_2, is_global=False)
    
    # Scenario 2: Curved road with heading change
    print("\nScenario 2: Curved road (vehicle turning)")
    print("=" * 60)
    
    # Curved path
    angles = np.linspace(0, np.pi/2, 8)
    radius = 20
    global_wps_curve = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    
    vehicle_pos_3 = (0, 0)
    vehicle_yaw_3 = 45  # 45 degrees
    
    ax5 = fig.add_subplot(gs[2, 0])
    plot_scenario(ax5, "Scenario 2: Curve - Global View",
                 global_wps_curve, vehicle_pos_3, vehicle_yaw_3, is_global=True)
    
    ax6 = fig.add_subplot(gs[2, 1])
    plot_scenario(ax6, "Scenario 2: Curve - Local View",
                 global_wps_curve, vehicle_pos_3, vehicle_yaw_3, is_global=False)
    
    # Print local waypoints for both scenarios
    print("\nGlobal waypoints (Scenario 1):", global_wps_1)
    local_wps_1a = global_to_local_waypoints(global_wps_1, vehicle_pos_1, np.deg2rad(vehicle_yaw_1))
    print("Local waypoints (1a, at origin):", local_wps_1a)
    
    local_wps_1b = global_to_local_waypoints(
        [(x + 100, y + 50) for x, y in global_wps_1],
        vehicle_pos_2,
        np.deg2rad(vehicle_yaw_2)
    )
    print("Local waypoints (1b, at (100,50)):", local_wps_1b)
    print("✓ IDENTICAL! This is why local coordinates enable generalization.")
    
    fig.suptitle('Ego-Centric Waypoint Transformation: Translation Invariance Demo',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('waypoint_transformation_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: waypoint_transformation_demo.png")
    plt.show()


def plot_dqn_state_vector():
    """
    Show how local waypoint X-coordinates form the DQN state vector.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Local waypoints example
    local_wps = [(5, 0.1), (10, 0.2), (15, 0.3), (20, 0.1), (25, -0.2),
                 (30, -0.5), (35, -0.8), (40, -0.9), (45, -0.7), (50, -0.2),
                 (55, 0.3), (60, 0.8), (65, 1.0), (70, 0.9), (75, 0.5)]
    
    # Extract X coordinates only
    x_coords = [wp[0] for wp in local_wps]
    
    # Visualize local waypoints
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.scatter(*zip(*local_wps), c='green', s=50, marker='o', zorder=3)
    ax1.plot([wp[0] for wp in local_wps], [wp[1] for wp in local_wps], 
            'g--', alpha=0.5, linewidth=1)
    
    # Vehicle at origin
    vehicle_rect = patches.FancyBboxPatch(
        (-1, -0.5), 2, 1,
        boxstyle="round,pad=0.1",
        edgecolor='red', facecolor='lightcoral', linewidth=2, zorder=5
    )
    ax1.add_patch(vehicle_rect)
    ax1.arrow(0, 0, 5, 0, head_width=0.5, head_length=0.4, 
             fc='red', ec='red', linewidth=2, zorder=4)
    
    ax1.set_xlabel('X local (m)')
    ax1.set_ylabel('Y local (m)')
    ax1.set_title('15 Ego-Centric Waypoints', fontweight='bold')
    ax1.legend(['Path', 'Waypoints'], loc='upper left')
    
    # Show state vector
    state_vector_info = """
DQN STATE VECTOR (from local waypoints):
    
State = [v_norm, e_y, e_psi, steer_prev, wp_x[0], wp_x[1], ..., wp_x[14]]
         └─────────────┬──────────────┘ └──────────────┬────────────┘
            Core kinematics          Waypoint lookahead (only X)
    
Where wp_x[i] = X coordinate of i-th local waypoint
    
Example (first 8 waypoints only):
"""
    
    x_sample = x_coords[:8]
    for i, x in enumerate(x_sample):
        state_vector_info += f"  wp_x[{i:2d}] = {x:6.2f}m\n"
    
    state_vector_info += """
Benefits of using only X:
  • X encodes distance to waypoint (forward progress)
  • Y info implicit in steering angle (e_psi)  
  • Reduces state dimensionality
  • Improves training convergence
  • Generalizes to any road layout
    
FIFO Mechanism:
  Step 1: [wp_0, wp_1, wp_2, ..., wp_14]
  Step 2: [wp_1, wp_2, wp_3, ..., wp_15]  ← shift left, add new
  Step 3: [wp_2, wp_3, wp_4, ..., wp_16]  ← continues...
"""
    
    ax2.text(0.05, 0.95, state_vector_info, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax2.axis('off')
    ax2.set_title('DQN State Vector Construction', fontweight='bold')
    
    fig.suptitle('Waypoint-Based DQN State Vector: Translation Invariant Navigation',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dqn_state_vector_construction.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: dqn_state_vector_construction.png")
    plt.show()


if __name__ == '__main__':
    print("Generating ego-centric waypoint transformation visualizations...\n")
    
    plot_translation_invariance()
    print("\n" + "="*60 + "\n")
    plot_dqn_state_vector()
    
    print("\n✓ All visualizations generated!")
