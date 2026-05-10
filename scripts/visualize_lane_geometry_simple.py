"""
Simple geometric visualization of lane detector concepts.

No CARLA map, no roads, no global planner.
Only geometry:
- lane center line
- lane width
- lane forward axis
- lane right axis
- vehicle position
- vehicle heading
- signed lateral error e_y
- signed heading error e_psi
"""

import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Arc


def parse_args():
    parser = argparse.ArgumentParser(description="Simple geometric lane detector visualization")
    parser.add_argument("--lane-x", type=float, default=0.0)
    parser.add_argument("--lane-y", type=float, default=0.0)
    parser.add_argument("--lane-yaw", type=float, default=0.0,
                        help="Lane direction in degrees")
    parser.add_argument("--lane-width", type=float, default=3.5)
    parser.add_argument("--lane-length", type=float, default=24.0)

    parser.add_argument("--vehicle-x", type=float, default=2.0)
    parser.add_argument("--vehicle-y", type=float, default=1.2)
    parser.add_argument("--vehicle-yaw", type=float, default=20.0)

    parser.add_argument("--save", default="")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def unit_from_yaw(yaw_deg):
    yaw_rad = math.radians(yaw_deg)
    return math.cos(yaw_rad), math.sin(yaw_rad)


def signed_lateral_error(vehicle_x, vehicle_y, lane_center_x, lane_center_y, lane_yaw_deg):
    lane_fx, lane_fy = unit_from_yaw(lane_yaw_deg)
    right_x = -lane_fy
    right_y = lane_fx

    dx = vehicle_x - lane_center_x
    dy = vehicle_y - lane_center_y
    return dx * right_x + dy * right_y


def signed_heading_error(vehicle_yaw_deg, lane_yaw_deg):
    vehicle_fx, vehicle_fy = unit_from_yaw(vehicle_yaw_deg)
    lane_fx, lane_fy = unit_from_yaw(lane_yaw_deg)
    dot = vehicle_fx * lane_fx + vehicle_fy * lane_fy
    cross = vehicle_fx * lane_fy - vehicle_fy * lane_fx
    return math.degrees(math.atan2(cross, dot))


def rotate_local_to_world(cx, cy, yaw_deg, local_x, local_y):
    c = math.cos(math.radians(yaw_deg))
    s = math.sin(math.radians(yaw_deg))
    wx = cx + local_x * c - local_y * s
    wy = cy + local_x * s + local_y * c
    return wx, wy


def main():
    args = parse_args()

    lane_fx, lane_fy = unit_from_yaw(args.lane_yaw)
    right_x = -lane_fy
    right_y = lane_fx

    e_y = signed_lateral_error(
        args.vehicle_x,
        args.vehicle_y,
        args.lane_x,
        args.lane_y,
        args.lane_yaw,
    )
    e_psi = signed_heading_error(args.vehicle_yaw, args.lane_yaw)

    half_length = args.lane_length * 0.5
    half_width = args.lane_width * 0.5

    distance_percentage = (abs(e_y) / half_width) * 100 if half_width > 0 else 0.0
    lane_info = {
        "distance_to_center": round(e_y, 4),
        "distance_abs": round(abs(e_y), 4),
        "lane_width": round(args.lane_width, 4),
        "distance_percentage": round(min(distance_percentage, 100.0), 2),
        "is_centered": distance_percentage < 20.0,
        "side": "right" if e_y > 0 else "left",
        "angle": round(e_psi, 4),
        "angle_abs": round(abs(e_psi), 4),
        "is_aligned": abs(e_psi) < 5.0,
    }

    # Center line endpoints.
    c0x, c0y = rotate_local_to_world(args.lane_x, args.lane_y, args.lane_yaw, -half_length, 0.0)
    c1x, c1y = rotate_local_to_world(args.lane_x, args.lane_y, args.lane_yaw, half_length, 0.0)

    # Lane borders.
    l0x, l0y = rotate_local_to_world(args.lane_x, args.lane_y, args.lane_yaw, -half_length, half_width)
    l1x, l1y = rotate_local_to_world(args.lane_x, args.lane_y, args.lane_yaw, half_length, half_width)
    r0x, r0y = rotate_local_to_world(args.lane_x, args.lane_y, args.lane_yaw, -half_length, -half_width)
    r1x, r1y = rotate_local_to_world(args.lane_x, args.lane_y, args.lane_yaw, half_length, -half_width)

    # Projection point of vehicle onto lane right axis passing through lane center.
    proj_x = args.lane_x + right_x * e_y
    proj_y = args.lane_y + right_y * e_y
    lane_left_x = args.lane_x + right_x * half_width
    lane_left_y = args.lane_y + right_y * half_width
    lane_right_x = args.lane_x - right_x * half_width
    lane_right_y = args.lane_y - right_y * half_width

    fig = plt.figure(figsize=(14, 9), dpi=140)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    info_ax = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor("#fcfcfd")
    ax.set_facecolor("#f4f7fb")
    info_ax.set_facecolor("#eef3f9")

    # Lane area.
    lane_poly_x = [l0x, l1x, r1x, r0x, l0x]
    lane_poly_y = [l0y, l1y, r1y, r0y, l0y]
    ax.fill(lane_poly_x, lane_poly_y, color="#dbeafe", alpha=0.55, label="Lane area")

    # Center and borders.
    ax.plot([c0x, c1x], [c0y, c1y], color="#0f172a", linewidth=2.5, label="Lane center")
    ax.plot([l0x, l1x], [l0y, l1y], "--", color="#16a34a", linewidth=2.0, label="Left border")
    ax.plot([r0x, r1x], [r0y, r1y], "--", color="#dc2626", linewidth=2.0, label="Right border")

    # Lane forward and right axes.
    axis_scale = max(args.lane_width * 1.3, 3.0)
    ax.quiver([args.lane_x], [args.lane_y], [lane_fx * axis_scale], [lane_fy * axis_scale],
              angles="xy", scale_units="xy", scale=1, width=0.005,
              color="#15803d", label="Lane forward")
    ax.quiver([args.lane_x], [args.lane_y], [right_x * axis_scale], [right_y * axis_scale],
              angles="xy", scale_units="xy", scale=1, width=0.005,
              color="#b91c1c", label="Lane right")

    # Vehicle point and heading.
    ax.scatter([args.vehicle_x], [args.vehicle_y], s=180, c="#7c3aed", marker="X", label="Vehicle")
    veh_fx, veh_fy = unit_from_yaw(args.vehicle_yaw)
    heading_scale = max(args.lane_width, 3.0)
    ax.quiver([args.vehicle_x], [args.vehicle_y], [veh_fx * heading_scale], [veh_fy * heading_scale],
              angles="xy", scale_units="xy", scale=1, width=0.006,
              color="#7c3aed", label="Vehicle heading")

    # Vehicle-center vector and lateral projection.
    ax.plot([args.lane_x, args.vehicle_x], [args.lane_y, args.vehicle_y],
            color="#f59e0b", linewidth=2.0, linestyle="--", label="Center-to-vehicle")
    ax.plot([args.lane_x, proj_x], [args.lane_y, proj_y],
            color="#ef4444", linewidth=3.0, label="Signed lateral error e_y")

    # Projection helper.
    ax.scatter([proj_x], [proj_y], s=100, c="#ef4444", marker="o", label="Projection on right axis")

    # Heading error arc.
    arc_radius = max(args.lane_width * 1.4, 3.5)
    start_angle = args.lane_yaw
    end_angle = args.vehicle_yaw
    theta1 = min(start_angle, end_angle)
    theta2 = max(start_angle, end_angle)
    arc = Arc((args.vehicle_x, args.vehicle_y), 2 * arc_radius, 2 * arc_radius,
              angle=0, theta1=theta1, theta2=theta2, color="#7c3aed", linewidth=2.0)
    ax.add_patch(arc)

    # Draw lane width bracket around lane center.
    ax.plot([lane_left_x, lane_right_x], [lane_left_y, lane_right_y], color="#0f172a", linewidth=2.2)
    ax.plot([lane_left_x, lane_left_x], [lane_left_y - 0.18, lane_left_y + 0.18], color="#0f172a", linewidth=2.2)
    ax.plot([lane_right_x, lane_right_x], [lane_right_y - 0.18, lane_right_y + 0.18], color="#0f172a", linewidth=2.2)
    mid_width_x = 0.5 * (lane_left_x + lane_right_x)
    mid_width_y = 0.5 * (lane_left_y + lane_right_y)
    ax.text(mid_width_x, mid_width_y + 0.35, f"lane_width = {args.lane_width:.2f} m",
            ha="center", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.4})

    # Labels.
    ax.text(args.lane_x, args.lane_y, " lane_center", fontsize=10, color="#0f172a")
    ax.text(args.vehicle_x, args.vehicle_y, " vehicle", fontsize=10, color="#5b21b6")
    ax.text(proj_x, proj_y, f"  e_y = {e_y:+.3f} m", fontsize=10, color="#b45309",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5})
    ax.text(args.vehicle_x + 0.4, args.vehicle_y + 0.8, f"e_psi = {e_psi:+.2f} deg", fontsize=10, color="#6d28d9",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5})

    summary = (
        f"Simple geometric example\n"
        f"lane_yaw = {args.lane_yaw:.2f} deg\n"
        f"vehicle = ({args.vehicle_x:.2f}, {args.vehicle_y:.2f})\n"
        f"vehicle_yaw = {args.vehicle_yaw:.2f} deg"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#94a3b8", "pad": 4.0})

    ax.set_title("LaneDetector Geometry - Simple Visual Explanation", fontsize=16, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35, color="#94a3b8")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)

    margin = max(args.lane_length * 0.12, args.lane_width * 1.8)
    xs = lane_poly_x + [args.vehicle_x]
    ys = lane_poly_y + [args.vehicle_y]
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    # Right-side panel: exact lane_info-like output.
    info_ax.set_title("lane_info returned", fontsize=14, fontweight="bold")
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.axis("off")

    info_lines = [
        "{",
        f"  'distance_to_center': {lane_info['distance_to_center']},",
        f"  'distance_abs': {lane_info['distance_abs']},",
        f"  'lane_width': {lane_info['lane_width']},",
        f"  'distance_percentage': {lane_info['distance_percentage']},",
        f"  'is_centered': {lane_info['is_centered']},",
        f"  'side': '{lane_info['side']}',",
        f"  'angle': {lane_info['angle']},",
        f"  'angle_abs': {lane_info['angle_abs']},",
        f"  'is_aligned': {lane_info['is_aligned']}",
        "}",
    ]
    info_ax.text(
        0.05,
        0.92,
        "\n".join(info_lines),
        va="top",
        family="monospace",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.94, "edgecolor": "#94a3b8", "pad": 5.0}
    )

    # Small visual meters.
    meter_left = 0.08
    meter_width = 0.84
    base_y1 = 0.28
    base_y2 = 0.14
    info_ax.text(meter_left, base_y1 + 0.06, "distance_percentage", fontsize=10, fontweight="bold")
    info_ax.add_patch(plt.Rectangle((meter_left, base_y1), meter_width, 0.04, color="#dbe4f0", ec="#94a3b8"))
    info_ax.add_patch(plt.Rectangle((meter_left, base_y1), meter_width * min(lane_info['distance_percentage'], 100.0) / 100.0, 0.04,
                                    color="#ef4444" if not lane_info['is_centered'] else "#16a34a", ec=None))
    info_ax.text(meter_left, base_y1 - 0.035, f"{lane_info['distance_percentage']:.2f}% of half-lane width", fontsize=9)

    info_ax.text(meter_left, base_y2 + 0.06, "angle_abs", fontsize=10, fontweight="bold")
    info_ax.add_patch(plt.Rectangle((meter_left, base_y2), meter_width, 0.04, color="#dbe4f0", ec="#94a3b8"))
    angle_cap = 45.0
    info_ax.add_patch(plt.Rectangle((meter_left, base_y2), meter_width * min(lane_info['angle_abs'], angle_cap) / angle_cap, 0.04,
                                    color="#8b5cf6" if not lane_info['is_aligned'] else "#16a34a", ec=None))
    info_ax.text(meter_left, base_y2 - 0.035, f"{lane_info['angle_abs']:.2f} deg (cap {angle_cap:.0f})", fontsize=9)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=180)
        print(f"Figure saved to: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
