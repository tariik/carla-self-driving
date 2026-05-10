"""
Script para probar un checkpoint DQN entrenado.
"""

import os
import sys
import importlib.util

import numpy as np
import pygame

# Deshabilitar DRI3/EGL hardware ANTES de cualquier import pesado
# para evitar crashes en algunos entornos.
# os.environ.setdefault('LIBGL_DRI3_DISABLE', '1')
# os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
# os.environ.setdefault('LIBGL_ALWAYS_INDIRECT', '1')
# os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
# os.environ.setdefault('MESA_LOADER_DRIVER_OVERRIDE', 'swrast')
# os.environ.setdefault('EGL_PLATFORM', 'x11')
# os.environ.setdefault('SDL_VIDEODRIVER', 'x11')
# os.environ.setdefault('SDL_RENDER_DRIVER', 'software')
# os.environ.setdefault('SDL_VIDEO_X11_FORCE_EGL', '0')

spec = importlib.util.spec_from_file_location("carla_env_file", "carla_env.py")
carla_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(carla_env_module)
CarlaEnv = carla_env_module.CarlaEnv

from agents.dqn_waypoints import dqn
from carla_app.vehicle_controller import VehicleController
from configs import config_test as config


DEFAULT_CHECKPOINT = 'checkpoints_dqn/best_checkpoint.pth'
LEGACY_CHECKPOINT = 'checkpoints_dqn_bkp/best_checkpoint.pth'


def test_agent(checkpoint_path, n_episodes=1, max_steps=500, render=True, manual=False):
    """Prueba un agente DQN entrenado con alternancia Agente/Manual (tecla M)."""
    use_agent = not manual

    print("DQN Agent Test")
    print("=" * 60)
    print("Controles: M alterna Agente/Manual, ESC cierra")

    render_mode = 'human' if render else None
    print(f"Creating environment (render: {render})...")
    env = CarlaEnv(config, render_mode=render_mode, max_episode_steps=max_steps, verbose=False)

    print("Creating agent...")
    agent = dqn(env, seed=42)
    controller = VehicleController(config)

    checkpoint_loaded = False
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        agent.load(checkpoint_path)
        checkpoint_loaded = True
        print("Checkpoint loaded\n")
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

    # Fallback to the current waypoint-based checkpoint if the requested one is missing/incompatible.
    if not checkpoint_loaded and checkpoint_path != DEFAULT_CHECKPOINT and os.path.exists(DEFAULT_CHECKPOINT):
        print(f"Trying fallback checkpoint: {DEFAULT_CHECKPOINT}")
        try:
            agent.load(DEFAULT_CHECKPOINT)
            checkpoint_loaded = True
            print("Fallback checkpoint loaded\n")
        except Exception as e:
            print(f"Fallback load failed: {e}")

    if not checkpoint_loaded:
        print("No compatible checkpoint could be loaded.")
        env.close()
        return

    print(f"Testing {n_episodes} episodes (max {max_steps} steps)")
    print("=" * 60)

    episode_scores = []
    episode_steps = []
    episode_outcomes = []

    try:
        for i_episode in range(1, n_episodes + 1):
            state, _ = env.reset()
            score = 0.0
            steps_taken = 0
            outcome = "Unknown"
            last_m_pressed = False

            mode = "AGENTE" if use_agent else "MANUAL"
            print(f"\nEpisode {i_episode}/{n_episodes} | Modo inicial: {mode}")

            for t in range(max_steps):
                keys = None
                if render and env.render_mode == 'human':
                    pygame.event.pump()
                    keys = pygame.key.get_pressed()

                    if keys[pygame.K_ESCAPE]:
                        print("ESC pressed")
                        raise KeyboardInterrupt

                    m_pressed = keys[pygame.K_m]
                    if m_pressed and not last_m_pressed:
                        use_agent = not use_agent
                        mode = "AGENTE" if use_agent else "MANUAL"
                        print(f"Mode -> {mode}")
                    last_m_pressed = m_pressed

                if use_agent:
                    action = agent.act(state, eps=0.0)
                else:
                    if keys is None:
                        pygame.event.pump()
                        keys = pygame.key.get_pressed()
                    controller.process_input(keys)
                    action = controller.get_discrete_action()

                next_state, reward, done, truncated, info = env.step(action)

                if render and env.render_mode == 'human':
                    if not env.carla_api.render():
                        print("Window closed")
                        raise KeyboardInterrupt

                state = next_state
                score += reward
                steps_taken = t + 1

                if t > 0 and t % 100 == 0:
                    print(
                        f"Step {t:3d} | Score: {score:7.2f} | "
                        f"Speed: {info.get('speed', 0):.1f} km/h | "
                        f"Dist: {info.get('distance_to_center', 0):.2f} m"
                    )

                if done:
                    reason = info.get('termination_reason', 'unknown')
                    if info.get('goal_reached', False):
                        outcome = "Goal reached"
                    elif info.get('collision', False):
                        outcome = "Collision"
                    elif info.get('lane_invasions', 0) >= 3:
                        outcome = "Lane invasions"
                    else:
                        outcome = f"Terminated ({reason})"
                    print(f"Termination reason: {reason}")
                    break

                if truncated:
                    outcome = "Completed (max steps)"
                    break

            episode_scores.append(score)
            episode_steps.append(steps_taken)
            episode_outcomes.append(outcome)
            print(f"{outcome} | Steps: {steps_taken} | Score: {score:.2f}")

    except KeyboardInterrupt:
        print("Interrupted")

    finally:
        print("\nClosing environment...")
        try:
            if pygame.get_init():
                pygame.quit()
                print("Pygame closed")
        except Exception as e:
            print(f"Warning pygame: {e}")

        try:
            env.close()
        except Exception as e:
            print(f"Warning env: {e}")

        if episode_scores:
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            for i, (score, steps, outcome) in enumerate(
                zip(episode_scores, episode_steps, episode_outcomes), 1
            ):
                print(f"Ep {i}: {outcome:24s} Steps: {steps:3d} | Score: {score:7.2f}")
            print("-" * 60)
            print(f"Avg Score: {np.mean(episode_scores):7.2f}")
            print(f"Avg Steps: {np.mean(episode_steps):7.1f}")
            print(f"Max Score: {np.max(episode_scores):7.2f}")
            print(f"Goals: {sum(1 for o in episode_outcomes if 'Goal' in o)}/{len(episode_outcomes)}")
            print("=" * 60)

        print("Test complete")
        sys.exit(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test DQN checkpoint')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT,
        help='Path to checkpoint',
    )
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--manual', action='store_true', help='Iniciar en modo manual')

    args = parser.parse_args()

    test_agent(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render,
        manual=args.manual,
    )
