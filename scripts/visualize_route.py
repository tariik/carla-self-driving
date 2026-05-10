"""
Visualize CARLA route and map context using Matplotlib.

Layers drawn:
- Drivable waypoint cloud
- Lane centerlines (grouped by road_id/lane_id)
- Junction waypoints and bounding boxes
- Global planner route
- Start and goal points
- Approximate lane borders along the route using lane_width
- Route direction arrows
- Lane-change permissions over route points
"""

import argparse
import math
from collections import defaultdict

import carla
import matplotlib.pyplot as plt

from carla_app.planer.navigation.global_route_planner import GlobalRoutePlanner


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CARLA route with map layers")
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--timeout", type=float, default=10.0, help="CARLA client timeout")

    parser.add_argument("--sampling-resolution", type=float, default=2.0,
                        help="GlobalRoutePlanner sampling resolution in meters")
    parser.add_argument("--map-distance", type=float, default=3.0,
                        help="Distance for map.generate_waypoints in meters")

    parser.add_argument("--start-index", type=int, default=0,
                        help="Start spawn point index")
    parser.add_argument("--end-index", type=int, default=-1,
                        help="End spawn point index")

    parser.add_argument("--label-step", type=int, default=30,
                        help="Write a route label every N points")
    parser.add_argument("--arrow-step", type=int, default=20,
                        help="Draw one route direction arrow every N points")
    parser.add_argument("--save", default="",
                        help="If provided, save figure to this path")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open the interactive plot window")
    return parser.parse_args()


def clamp_index(index, length):
    if length == 0:
        return 0
    if index < 0:
        index = length + index
    return max(0, min(index, length - 1))


def build_lane_groups(waypoints):
    groups = defaultdict(list)
    for wp in waypoints:
        key = (wp.road_id, wp.lane_id)
        groups[key].append(wp)

    # Keep points ordered along s to draw cleaner lines.
    for key in groups:
        groups[key].sort(key=lambda w: w.s)
    return groups


def lane_border_points(waypoint):
    loc = waypoint.transform.location
    yaw_rad = math.radians(waypoint.transform.rotation.yaw)

    # Tangent is (cos, sin). Normal is (-sin, cos).
    nx = -math.sin(yaw_rad)
    ny = math.cos(yaw_rad)

    half_w = waypoint.lane_width * 0.5
    left = (loc.x + nx * half_w, loc.y + ny * half_w)
    right = (loc.x - nx * half_w, loc.y - ny * half_w)
    return left, right


def can_change_left(waypoint):
    if not (waypoint.lane_change & carla.LaneChange.Left):
        return False
    left_wp = waypoint.get_left_lane()
    return left_wp is not None and left_wp.lane_type == carla.LaneType.Driving


def can_change_right(waypoint):
    if not (waypoint.lane_change & carla.LaneChange.Right):
        return False
    right_wp = waypoint.get_right_lane()
    return right_wp is not None and right_wp.lane_type == carla.LaneType.Driving


def lane_change_category(waypoint):
    left_ok = can_change_left(waypoint)
    right_ok = can_change_right(waypoint)
    if left_ok and right_ok:
        return "both"
    if left_ok:
        return "left"
    if right_ok:
        return "right"
    return "none"


def junction_bbox_corners(junction):
    bbox = junction.bounding_box
    cx = bbox.location.x
    cy = bbox.location.y
    ex = bbox.extent.x
    ey = bbox.extent.y
    yaw = math.radians(bbox.rotation.yaw)

    corners_local = [(-ex, -ey), (ex, -ey), (ex, ey), (-ex, ey)]
    corners_world = []
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    for lx, ly in corners_local:
        wx = cx + (lx * cos_y - ly * sin_y)
        wy = cy + (lx * sin_y + ly * cos_y)
        corners_world.append((wx, wy))
    return corners_world


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()

    print(f"Connected to CARLA at {args.host}:{args.port}")
    print(f"Map: {carla_map.name}")

    spawn_points = carla_map.get_spawn_points()
    if len(spawn_points) < 2:
        raise RuntimeError("At least 2 spawn points are required")

    start_idx = clamp_index(args.start_index, len(spawn_points))
    end_idx = clamp_index(args.end_index, len(spawn_points))

    start_location = spawn_points[start_idx].location
    end_location = spawn_points[end_idx].location

    print(f"Spawn points: {len(spawn_points)}")
    print(f"Start index: {start_idx} -> ({start_location.x:.2f}, {start_location.y:.2f})")
    print(f"End index:   {end_idx} -> ({end_location.x:.2f}, {end_location.y:.2f})")

    planner = GlobalRoutePlanner(carla_map, args.sampling_resolution)
    route = planner.trace_route(start_location, end_location)
    if not route:
        raise RuntimeError("GlobalRoutePlanner returned an empty route")

    print(f"Route waypoints: {len(route)}")

    map_waypoints = list(carla_map.generate_waypoints(args.map_distance))
    lane_groups = build_lane_groups(map_waypoints)

    print(f"Map waypoints generated: {len(map_waypoints)}")
    print(f"Lane groups (road_id, lane_id): {len(lane_groups)}")

    # Prepare route arrays.
    route_x = []
    route_y = []
    route_width = []
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    for waypoint, _ in route:
        loc = waypoint.transform.location
        route_x.append(loc.x)
        route_y.append(loc.y)
        route_width.append(waypoint.lane_width)
        left, right = lane_border_points(waypoint)
        left_x.append(left[0])
        left_y.append(left[1])
        right_x.append(right[0])
        right_y.append(right[1])

    # Junction points and bounding boxes from full map waypoint set.
    jx = []
    jy = []
    junction_boxes = []
    seen_junction_ids = set()
    for wp in map_waypoints:
        if wp.is_junction:
            loc = wp.transform.location
            jx.append(loc.x)
            jy.append(loc.y)
            junction = wp.get_junction()
            if junction is not None and junction.id not in seen_junction_ids:
                seen_junction_ids.add(junction.id)
                junction_boxes.append((junction.id, junction_bbox_corners(junction)))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 11), dpi=140)
    fig.patch.set_facecolor("#fbfcfe")
    ax.set_facecolor("#f3f6fb")

    # Layer 1: dense waypoint cloud for road footprint context.
    all_x = [w.transform.location.x for w in map_waypoints]
    all_y = [w.transform.location.y for w in map_waypoints]
    ax.scatter(all_x, all_y, s=2, c="#9fa6b2", alpha=0.25, label="Drivable waypoint cloud")

    # Layer 2: lane centerlines.
    for (road_id, lane_id), waypoints in lane_groups.items():
        xs = [w.transform.location.x for w in waypoints]
        ys = [w.transform.location.y for w in waypoints]
        if lane_id < 0:
            color = "#3d6d8a"
        else:
            color = "#8a5f2b"
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.4)

    # Layer 3: junction points and bounding boxes.
    for junction_id, corners in junction_boxes:
        xs = [p[0] for p in corners] + [corners[0][0]]
        ys = [p[1] for p in corners] + [corners[0][1]]
        ax.fill(xs, ys, color="#ffd7a8", alpha=0.28)
        ax.plot(xs, ys, color="#ff9f1c", linewidth=1.2, alpha=0.85)

    if jx:
        ax.scatter(jx, jy, s=8, c="#ff7b00", alpha=0.8, label="Junction waypoints")

    # Layer 4: route and lane borders.
    ax.plot(route_x, route_y, color="#0a2463", linewidth=3.0, label="Global route")
    ax.scatter(route_x, route_y, s=12, c="#3e8ed0", alpha=0.9, label="Route waypoints")

    # Approximate lane borders along route from lane_width.
    ax.plot(left_x, left_y, "--", linewidth=1.2, color="#1a9c5a", alpha=0.95, label="Lane border (left)")
    ax.plot(right_x, right_y, "--", linewidth=1.2, color="#e4572e", alpha=0.95, label="Lane border (right)")

    # Lane-change permissions over route points.
    route_none_x, route_none_y = [], []
    route_left_x, route_left_y = [], []
    route_right_x, route_right_y = [], []
    route_both_x, route_both_y = [], []
    for waypoint, _ in route:
        loc = waypoint.transform.location
        category = lane_change_category(waypoint)
        if category == "none":
            route_none_x.append(loc.x)
            route_none_y.append(loc.y)
        elif category == "left":
            route_left_x.append(loc.x)
            route_left_y.append(loc.y)
        elif category == "right":
            route_right_x.append(loc.x)
            route_right_y.append(loc.y)
        else:
            route_both_x.append(loc.x)
            route_both_y.append(loc.y)

    ax.scatter(route_none_x, route_none_y, s=18, c="#6b7280", alpha=0.9, marker="o", label="Lane change: none")
    ax.scatter(route_left_x, route_left_y, s=18, c="#2563eb", alpha=0.9, marker="<", label="Lane change: left")
    ax.scatter(route_right_x, route_right_y, s=18, c="#16a34a", alpha=0.9, marker=">", label="Lane change: right")
    ax.scatter(route_both_x, route_both_y, s=22, c="#7c3aed", alpha=0.9, marker="D", label="Lane change: both")

    # Route direction arrows.
    arrow_step = max(1, args.arrow_step)
    arrow_x = []
    arrow_y = []
    arrow_u = []
    arrow_v = []
    for i in range(0, len(route) - 1, arrow_step):
        p0 = route[i][0].transform.location
        p1 = route[i + 1][0].transform.location
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            continue
        arrow_x.append(p0.x)
        arrow_y.append(p0.y)
        arrow_u.append(dx / norm * 4.0)
        arrow_v.append(dy / norm * 4.0)

    if arrow_x:
        ax.quiver(
            arrow_x,
            arrow_y,
            arrow_u,
            arrow_v,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0035,
            color="#111827",
            alpha=0.9,
            label="Direction"
        )

    # Start/goal markers.
    ax.scatter([start_location.x], [start_location.y], c="#0f9d58", s=260, marker="*", edgecolors="black", linewidths=0.8, label="Start")
    ax.scatter([end_location.x], [end_location.y], c="#d7263d", s=260, marker="X", edgecolors="black", linewidths=0.8, label="Goal")

    # Optional labels every N route points to expose road/lane/s context.
    if args.label_step > 0:
        for i, (wp, road_option) in enumerate(route):
            if i % args.label_step != 0:
                continue
            loc = wp.transform.location
            label = f"r{wp.road_id} l{wp.lane_id} s{wp.s:.1f}\n{road_option.name}"
            ax.text(
                loc.x,
                loc.y,
                label,
                fontsize=7,
                alpha=0.85,
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.8}
            )

    # Lane width visual as color strip on route points.
    sc = ax.scatter(route_x, route_y, c=route_width, cmap="viridis", s=14, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Lane width (m)")

    ax.set_title(f"CARLA Route Visualization - {carla_map.name}", fontsize=15, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.35, color="#8e9aaf")
    ax.legend(loc="best", fontsize=8, framealpha=0.92)

    # Summary panel for quick analysis.
    summary = (
        f"Spawn points: {len(spawn_points)}\n"
        f"Route waypoints: {len(route)}\n"
        f"Junctions in map: {len(junction_boxes)}\n"
        f"Lane change left: {len(route_left_x)}\n"
        f"Lane change right: {len(route_right_x)}\n"
        f"Lane change both: {len(route_both_x)}\n"
        f"Lane change none: {len(route_none_x)}"
    )
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#94a3b8", "pad": 4.0}
    )

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=180)
        print(f"Figure saved to: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
