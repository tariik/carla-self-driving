"""
Diagnostic script: runs N episodes and logs key parameters per step and per episode.
Output: logs/diagnostics_YYYYMMDD_HHMMSS.txt
"""

import os
import time
import numpy as np
from datetime import datetime
from collections import deque
from carla_env import CarlaEnv

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_EPISODES   = 10     # cuántos episodios correr
MAX_STEPS    = 300    # max steps por episodio
EPSILON      = 0.0    # 0 = greedy, 1 = random
LOAD_CHECKPOINT = "checkpoints_dqn/best_checkpoint.pth"  # None = sin modelo

ENV_CONFIG = {
    'host': 'localhost', 'port': 2000,
    'town': 'Town01', 'weather': 'ClearNoon',
    'vehicle_model': 'vehicle.tesla.model3',
    'camera_position': [-5, 0.0, 2.8], 'camera_rotation': [-15, 0, 0],
    'window_width': 800, 'window_height': 600,
    'target_fps': 20, 'window_title': "CARLA Diagnostics",
    'name_map': 'Town01', 'set_weather': True, 'weather_preset': 'ClearNoon',
    'spawn_point': {'x': 383.0, 'y': -2.0, 'z': None, 'yaw': 180.0},
    'goal_point': {'x': 11.71, 'y': -2.0, 'radius': 5.0},
    'autopilot_mode': False, 'camera_fov': 90,
    'throttle_speed': 0.8, 'brake_speed': 0.5, 'steer_speed': 0.5,
    'hud_font_size': 20, 'hud_bg_color': (0, 0, 0, 0), 'hud_bg_alpha': 128,
    'hud_width': 240, 'hud_height': 320,
    'enable_photo_capture': False,
    'collision_sensor_enabled': True, 'lane_invasion_sensor_enabled': True,
    'carla_timeout': 20.0,
}
# ─────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"logs/diagnostics_{timestamp}.txt"


def write(f, line=""):
    print(line)
    f.write(line + "\n")


def action_to_str(action, action_mapping):
    cmd = action_mapping.get(action, {})
    return (f"th={cmd.get('throttle', 0):.2f} "
            f"st={cmd.get('steering', 0):+.2f} "
            f"br={cmd.get('brake', 0):.2f}")


def run():
    env = CarlaEnv(ENV_CONFIG, render_mode=None, max_episode_steps=MAX_STEPS, verbose=False)

    # Load agent if checkpoint exists
    agent = None
    if LOAD_CHECKPOINT and os.path.exists(LOAD_CHECKPOINT):
        from agents.dqn_waypoints import dqn as DQN
        agent = DQN(env, seed=42)
        agent.load(LOAD_CHECKPOINT)
        print(f"✓ Loaded checkpoint: {LOAD_CHECKPOINT}")
    else:
        print("⚠ No checkpoint loaded — using random actions")

    episode_rewards = []
    episode_steps_list = []
    episode_collisions = []
    episode_invasions = []
    episode_goals = []

    with open(log_path, "w") as f:
        write(f, "=" * 70)
        write(f, f"CARLA DQN DIAGNOSTICS — {timestamp}")
        write(f, f"Episodes: {N_EPISODES}  MaxSteps: {MAX_STEPS}  Epsilon: {EPSILON}")
        write(f, f"Checkpoint: {LOAD_CHECKPOINT}")
        write(f, "=" * 70)

        for ep in range(1, N_EPISODES + 1):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            step_rewards = []
            speeds = []
            lane_dists = []
            lane_angles = []
            collisions = 0
            invasions = 0
            goal_reached = False

            write(f)
            write(f, f"{'─'*70}")
            write(f, f"EPISODE {ep:03d}")
            write(f, f"{'─'*70}")
            write(f, f"{'Step':>5} | {'Action':>20} | {'Reward':>8} | {'Speed':>7} | {'LaneDist':>9} | {'Angle':>7} | {'Collis':>6} | {'Invas':>5}")
            write(f, f"{'─'*70}")

            for step in range(MAX_STEPS):
                # Choose action
                if agent and EPSILON < 1.0:
                    action = agent.act(obs, eps=EPSILON)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                speed       = float(info.get('speed', 0.0))
                lane_dist   = float(info.get('distance_to_center', 0.0))
                angle       = float(info.get('angle', 0.0))
                col         = bool(info.get('collision', False))
                inv_count   = int(info.get('lane_invasions', 0))

                total_reward += reward
                step_rewards.append(reward)
                speeds.append(speed)
                lane_dists.append(lane_dist)
                lane_angles.append(angle)
                if col:
                    collisions += 1
                invasions = max(invasions, inv_count)

                action_str = action_to_str(action, env.action_mapping)

                write(f,
                    f"{step+1:>5} | {action_str:>20} | {reward:>8.3f} | "
                    f"{speed:>6.1f}k | {lane_dist:>+9.3f} | {angle:>+7.1f}° | "
                    f"{'YES' if col else 'no':>6} | {inv_count:>5}"
                )

                if terminated and info.get('distance_to_center', 999) < 5.0:
                    goal_reached = True

                if done:
                    break

            # Episode summary
            n = len(step_rewards)
            write(f, f"{'─'*70}")
            write(f, f"EPISODE {ep:03d} SUMMARY")
            write(f, f"  Steps           : {n}")
            write(f, f"  Total Reward    : {total_reward:+.3f}")
            write(f, f"  Avg Reward/Step : {(total_reward/max(n,1)):+.3f}")
            write(f, f"  Min/Max Reward  : {min(step_rewards):.2f} / {max(step_rewards):.2f}")
            write(f, f"  Avg Speed       : {np.mean(speeds):.1f} km/h  (max {np.max(speeds):.1f})")
            write(f, f"  Avg |LaneDist|  : {np.mean(np.abs(lane_dists)):.3f} m  (max {np.max(np.abs(lane_dists)):.3f})")
            write(f, f"  Avg |Angle|     : {np.mean(np.abs(lane_angles)):.1f}°  (max {np.max(np.abs(lane_angles)):.1f}°)")
            write(f, f"  Collisions      : {collisions}")
            write(f, f"  Lane Invasions  : {invasions}")
            write(f, f"  Goal Reached    : {'YES 🎯' if goal_reached else 'NO'}")
            write(f, f"  Term/Trunc      : {'TERMINATED' if terminated else 'TRUNCATED'}")

            episode_rewards.append(total_reward)
            episode_steps_list.append(n)
            episode_collisions.append(collisions)
            episode_invasions.append(invasions)
            episode_goals.append(goal_reached)

        # ── Global summary ──
        write(f)
        write(f, "=" * 70)
        write(f, "GLOBAL SUMMARY")
        write(f, "=" * 70)
        write(f, f"  Episodes run      : {N_EPISODES}")
        write(f, f"  Avg Reward        : {np.mean(episode_rewards):+.2f}  ± {np.std(episode_rewards):.2f}")
        write(f, f"  Avg Steps/Episode : {np.mean(episode_steps_list):.1f}")
        write(f, f"  Goals reached     : {sum(episode_goals)}/{N_EPISODES}  ({100*sum(episode_goals)/N_EPISODES:.0f}%)")
        write(f, f"  Total collisions  : {sum(episode_collisions)}")
        write(f, f"  Total invasions   : {sum(episode_invasions)}")
        write(f)
        write(f, "REWARD PER EPISODE:")
        for i, (r, s, c, g) in enumerate(zip(episode_rewards, episode_steps_list, episode_collisions, episode_goals), 1):
            goal_marker = " 🎯" if g else ""
            write(f, f"  Ep {i:03d}: {r:+8.2f}  steps={s:4d}  collisions={c}{goal_marker}")
        write(f)
        write(f, "DIAGNOSIS:")

        # auto diagnosis
        avg_r   = np.mean(episode_rewards)
        avg_s   = np.mean(episode_steps_list)
        goal_r  = sum(episode_goals) / N_EPISODES
        col_r   = sum(episode_collisions) / N_EPISODES

        if avg_s < 30:
            write(f, "  ⚠ Avg steps muy bajo (<30): agente choca inmediatamente.")
        elif avg_s < 100:
            write(f, "  ⚠ Avg steps moderado (<100): agente sobrevive poco.")
        else:
            write(f, "  ✓ Avg steps razonable (>100).")

        if col_r > 0.5:
            write(f, "  ⚠ >50% episodios terminan en colisión: revisar GRACE_PERIOD y reward de colisión.")
        if goal_r == 0:
            write(f, "  ⚠ Nunca llega a la meta: revisar reward de progreso y epsilon.")
        elif goal_r < 0.3:
            write(f, "  ⚠ Meta alcanzada <30%: agente sub-óptimo.")
        else:
            write(f, "  ✓ Meta alcanzada razonablemente.")

        if avg_r < -100:
            write(f, "  ⚠ Reward promedio muy negativo: recompensas desbalanceadas.")
        elif avg_r > 0:
            write(f, "  ✓ Reward promedio positivo: señal aceptable.")

        write(f, "=" * 70)
        write(f, f"Log saved to: {log_path}")

    env.close()
    print(f"\n✓ Diagnostics complete → {log_path}")


if __name__ == '__main__':
    run()
