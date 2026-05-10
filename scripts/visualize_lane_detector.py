"""
Visualize LaneDetector geometry using Matplotlib.

This script shows, in 2D:
- vehicle position and heading
- reference waypoint and lane center
- lane forward and right axes
- signed lateral distance used by LaneDetector
- signed heading error used by LaneDetector
- next 5 route waypoints from the global planner
"""

import argparse
import math

import carla
import matplotlib.pyplot as plt

from carla_app.planer.navigation.global_route_planner import GlobalRoutePlanner
from carla_app.sensores.lane_detector import LaneDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LaneDetector geometry")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--sampling-resolution", type=float, default=2.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--route-lookahead", type=int, default=5)
    parser.add_argument("--offset-x", type=float, default=0.0,
                        help="Artificial X offset added to vehicle position for visualization")
    parser.add_argument("--offset-y", type=float, default=0.0,
                        help="Artificial Y offset added to vehicle position for visualization")
    parser.add_argument("--yaw-offset", type=float, default=0.0,
                        help="Artificial yaw offset in degrees added to vehicle heading")
    parser.add_argument("--save", default="")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def clamp_index(index, length):
    if length == 0:
        return 0
    if index < 0:
        index = length + index
    return max(0, min(index, length - 1))


def yaw_to_vector(yaw_deg, scale=1.0):
    yaw_rad = math.radians(yaw_deg)
    return math.cos(yaw_rad) * scale, math.sin(yaw_rad) * scale


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()
    lane_detector = LaneDetector(world)

    spawn_points = carla_map.get_spawn_points()
    if len(spawn_points) < 2:
        raise RuntimeError("At least 2 spawn points are required")

    start_idx = clamp_index(args.start_index, len(spawn_points))
    end_idx = clamp_index(args.end_index, len(spawn_points))
    start_transform = spawn_points[start_idx]
    end_transform = spawn_points[end_idx]

    planner = GlobalRoutePlanner(carla_map, args.sampling_resolution)
    route = planner.trace_route(start_transform.location, end_transform.location)
    if not route:
        raise RuntimeError("GlobalRoutePlanner returned an empty route")

    route_waypoints = [wp for wp, _ in route]

    vehicle_location = carla.Location(
        x=start_transform.location.x + args.offset_x,
        y=start_transform.location.y + args.offset_y,
        z=start_transform.location.z,
    )
    vehicle_rotation = carla.Rotation(
        pitch=start_transform.rotation.pitch,
        yaw=start_transform.rotation.yaw + args.yaw_offset,
        roll=start_transform.rotation.roll,
    )
    vehicle_transform = carla.Transform(vehicle_location, vehicle_rotation)

    nearest_index = min(
        range(len(route_waypoints)),
        key=lambda i: vehicle_location.distance(route_waypoints[i].transform.location)
    )
    next_route_waypoints = route_waypoints[nearest_index: nearest_index + max(1, args.route_lookahead)]
    reference_waypoint = next_route_waypoints[0]

    signed_distance = lane_detector._signed_lateral_distance(vehicle_location, reference_waypoint)
    signed_angle = lane_detector._signed_heading_error(vehicle_transform, reference_waypoint)

    lane_center = reference_waypoint.transform.location
    lane_forward = reference_waypoint.transform.get_forward_vector()
    right_x = -lane_forward.y
    right_y = lane_forward.x

    lane_axis_scale = max(reference_waypoint.lane_width * 1.5, 4.0)
    heading_scale = max(reference_waypoint.lane_width, 3.0)

    route_x = [wp.transform.location.x for wp in route_waypoints]
    route_y = [wp.transform.location.y for wp in route_waypoints]
    next_x = [wp.transform.location.x for wp in next_route_waypoints]
    next_y = [wp.transform.location.y for wp in next_route_waypoints]

    center_x = lane_center.x
    center_y = lane_center.y
    lateral_end_x = center_x + right_x * signed_distance
    lateral_end_y = center_y + right_y * signed_distance

    lane_left_x = center_x + right_x * reference_waypoint.lane_width * 0.5
    lane_left_y = center_y + right_y * reference_waypoint.lane_width * 0.5
    lane_right_x = center_x - right_x * reference_waypoint.lane_width * 0.5
    lane_right_y = center_y - right_y * reference_waypoint.lane_width * 0.5

    fig, ax = plt.subplots(figsize=(13, 10), dpi=140)
    fig.patch.set_facecolor("#fcfcfd")
    ax.set_facecolor("#f4f7fb")

    # Global route context.
    ax.plot(route_x, route_y, color="#9aa5b1", linewidth=1.2, alpha=0.55, label="Full route")
    ax.scatter(next_x, next_y, s=65, c="#2563eb", alpha=0.95, label="Next route waypoints")

    # Reference waypoint and lane center.
    ax.scatter([center_x], [center_y], s=150, c="#111827", marker="o", label="Reference lane center")

    # Lane forward axis.
    ax.quiver(
        [center_x], [center_y],
        [lane_forward.x * lane_axis_scale], [lane_forward.y * lane_axis_scale],
        angles="xy", scale_units="xy", scale=1,
        color="#15803d", width=0.004, label="Lane forward"
    )

    # Lane right axis.
    ax.quiver(
        [center_x], [center_y],
        [right_x * lane_axis_scale], [right_y * lane_axis_scale],
        angles="xy", scale_units="xy", scale=1,
        color="#dc2626", width=0.004, label="Lane right"
    )

    # Vehicle position and heading.
    ax.scatter([vehicle_location.x], [vehicle_location.y], s=180, c="#7c3aed", marker="X", label="Vehicle")
    veh_u, veh_v = yaw_to_vector(vehicle_rotation.yaw, heading_scale)
    ax.quiver(
        [vehicle_location.x], [vehicle_location.y],
        [veh_u], [veh_v],
        angles="xy", scale_units="xy", scale=1,
        color="#7c3aed", width=0.005, label="Vehicle heading"
    )

    # Signed lateral distance segment.
    ax.plot(
        [center_x, vehicle_location.x],
        [center_y, vehicle_location.y],
        linestyle="--",
        linewidth=2.0,
        color="#f59e0b",
        label="Center-to-vehicle vector"
    )
    ax.plot(
        [center_x, lateral_end_x],
        [center_y, lateral_end_y],
        linewidth=3.0,
        color="#ef4444",
        alpha=0.95,
        label="Signed lateral projection"
    )

    # Approximate lane borders at reference waypoint.
    ax.plot([lane_left_x], [lane_left_y], marker="_", markersize=18, color="#16a34a")
    ax.plot([lane_right_x], [lane_right_y], marker="_", markersize=18, color="#dc2626")

    # Labels.
    ax.text(center_x, center_y, " lane center", fontsize=10, color="#111827")
    ax.text(vehicle_location.x, vehicle_location.y, " vehicle", fontsize=10, color="#5b21b6")
    ax.text(
        lateral_end_x,
        lateral_end_y,
        f"  e_y = {signed_distance:+.3f} m",
        fontsize=10,
        color="#b45309",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5}
    )
    ax.text(
        vehicle_location.x,
        vehicle_location.y + reference_waypoint.lane_width,
        f"e_psi = {signed_angle:+.2f} deg",
        fontsize=10,
        color="#7c3aed",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5}
    )

    for i, wp in enumerate(next_route_waypoints):
        loc = wp.transform.location
        ax.text(
            loc.x,
            loc.y,
            f"#{nearest_index + i}\nr{wp.road_id} l{wp.lane_id}\ns={wp.s:.1f}",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.2}
        )

    summary = (
        f"Map: {carla_map.name}\n"
        f"Reference road/lane: {reference_waypoint.road_id}/{reference_waypoint.lane_id}\n"
        f"Lane width: {reference_waypoint.lane_width:.3f} m\n"
        f"Signed lateral error: {signed_distance:+.3f} m\n"
        f"Signed heading error: {signed_angle:+.2f} deg\n"
        f"Route slice used: {len(next_route_waypoints)} waypoints\n"
        f"Nearest route index: {nearest_index}"
    )
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#94a3b8", "pad": 4.0}
    )

    ax.set_title("LaneDetector Visual Explanation", fontsize=16, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.35, color="#94a3b8")
    ax.legend(loc="best", fontsize=9, framealpha=0.92)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=180)
        print(f"Figure saved to: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()