"""
Script para probar un checkpoint DQN entrenado
"""

# Deshabilitar DRI3/EGL hardware ANTES de cualquier import
# para evitar el crash 'free(): invalid pointer' en displays sin DRI3
import os
os.environ.setdefault('LIBGL_DRI3_DISABLE', '1')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
os.environ.setdefault('LIBGL_ALWAYS_INDIRECT', '1')
os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
os.environ.setdefault('MESA_LOADER_DRIVER_OVERRIDE', 'swrast')
os.environ.setdefault('EGL_PLATFORM', 'x11')
os.environ.setdefault('SDL_VIDEODRIVER', 'x11')
os.environ.setdefault('SDL_RENDER_DRIVER', 'software')
os.environ.setdefault('SDL_VIDEO_X11_FORCE_EGL', '0')

import sys
import numpy as np
import torch

# ✅ IMPORTAR directamente del archivo .py (no de la carpeta)
import importlib.util
spec = importlib.util.spec_from_file_location("carla_env_file", "carla_env.py")
carla_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(carla_env_module)
CarlaEnv = carla_env_module.CarlaEnv

from carla_app.dqn import dqn


def test_agent(checkpoint_path, n_episodes=5, max_steps=500, render=True):
    """Prueba un agente DQN entrenado."""
    
    print("🚗 DQN Agent Test")
    print("=" * 60)
    
    config = {
        'host': 'localhost',
        'port': 2000,
        'town': 'Town01',
        'weather': 'ClearNoon',
        'vehicle_model': 'vehicle.tesla.model3',
        'camera_position': [1.5, 0.0, 2.4],
        'camera_rotation': [0.0, 0.0, 0.0],
        'window_width': 800,
        'window_height': 600,
        'target_fps': 20,
        'window_width': 800,
        'window_height': 600,
        'window_title': "CARLA Gym Environment",
        'name_map': 'Town01',
        'set_weather': True,
        'weather_preset': 'ClearNoon',
        'spawn_point': {'x': 383.0, 'y': -2.0, 'z': None, 'yaw': 180.0},
        'goal_point': {'x': 11.71, 'y': -2.0, 'radius': 5.0},
        # 'goal_point': {'x': 334.0, 'y': 43.0, 'radius': 5.0},
        'autopilot_mode': True,
        'camera_fov': 90,
        'camera_position': [-5, 0.0, 2.8],
        'camera_rotation': [-15, 0, 0],
        'throttle_speed': 0.8,
        'brake_speed': 0.5,
        'steer_speed': 0.5,
        'hud_font_size': 20,
        'hud_bg_color': (0, 0, 0, 0),
        'hud_bg_alpha': 128,
        'hud_width': 240,
        'hud_height': 320,
        'enable_photo_capture': True,
        'photo_capture_interval': 1.0,
        'photo_capture_directory': 'run_logs',
        'photo_width': 800,
        'photo_height': 600,
        'photo_fov': 90.0,
        'collision_sensor_enabled': True,
        'lane_invasion_sensor_enabled': True,
        'carla_timeout': 20.0,
    }  # Placeholder for actual configuration
    
    render_mode = 'human' if render else None
    print(f"Creating environment (render: {render})...")
    env = CarlaEnv(config, render_mode=render_mode, max_episode_steps=max_steps, verbose=False)
    
    print(f"Creating agent...")
    agent = dqn(env, seed=42)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        agent.load(checkpoint_path)
        print(f"✅ Checkpoint loaded\n")
    except FileNotFoundError:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        env.close()
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        env.close()
        return
    
    print(f"🎮 Testing {n_episodes} episodes (max {max_steps} steps)")
    print("=" * 60)
    
    episode_scores = []
    episode_steps = []
    episode_outcomes = []
    
    try:
        for i_episode in range(1, n_episodes + 1):
            state, info = env.reset()
            score = 0
            steps_taken = 0
            outcome = "Unknown"
            
            print(f"\n📍 Episode {i_episode}/{n_episodes}")
            
            for t in range(max_steps):
                action = agent.act(state, eps=0.0)
                next_state, reward, done, truncated, info = env.step(action)
                
                if render and env.render_mode == 'human':
                    if not env.carla_api.render():
                        print("\n⚠ Window closed")
                        raise KeyboardInterrupt
                
                state = next_state
                score += reward
                steps_taken = t + 1
                
                if t > 0 and t % 100 == 0:
                    print(f"   Step {t:3d} | Score: {score:6.2f} | "
                          f"Speed: {info.get('speed', 0):.1f} km/h | "
                          f"Dist: {info.get('distance_to_center', 0):.2f}m")
                
                if done:
                    if info.get('goal_reached', False):
                        outcome = "🎯 Goal reached"
                    elif info.get('collision', False):
                        outcome = "💥 Collision"
                    elif info.get('lane_invasions', 0) >= 3:
                        outcome = "🚨 Lane invasions"
                    else:
                        outcome = "⚠ Off-road"
                    break
                
                if truncated:
                    outcome = "✅ Completed (max steps)"
                    break
            
            episode_scores.append(score)
            episode_steps.append(steps_taken)
            episode_outcomes.append(outcome)
            
            print(f"   {outcome} | Steps: {steps_taken} | Score: {score:.2f}")
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    
    finally:
        print("\n🔄 Closing environment...")
        
        # ✅ CERRAR PYGAME EXPLÍCITAMENTE ANTES DE ENV.CLOSE()
        try:
            import pygame
            if pygame.get_init():
                pygame.quit()
                print("   ✓ Pygame closed")
        except Exception as e:
            print(f"   Warning pygame: {e}")
        
        # Ahora cerrar environment
        try:
            env.close()
        except Exception as e:
            print(f"   Warning env: {e}")
        
        # Estadísticas
        if len(episode_scores) > 0:
            print("\n" + "=" * 60)
            print("📊 RESULTS")
            print("=" * 60)
            
            for i, (score, steps, outcome) in enumerate(zip(episode_scores, episode_steps, episode_outcomes), 1):
                print(f"Ep {i}: {outcome:25s} Steps: {steps:3d} | Score: {score:7.2f}")
            
            print("-" * 60)
            print(f"Avg Score:  {np.mean(episode_scores):7.2f}")
            print(f"Avg Steps:  {np.mean(episode_steps):7.1f}")
            print(f"Max Score:  {np.max(episode_scores):7.2f}")
            print(f"Goals:      {sum(1 for o in episode_outcomes if 'Goal' in o)}/{len(episode_outcomes)}")
            print("=" * 60)
        
        print("\n✅ Test complete - You can close terminal now")
        
        # ✅ FORZAR EXIT
        sys.exit(0)



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DQN checkpoint')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints_dqn/checkpoint_ep100.pth',
                       help='Path to checkpoint')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--no-render', action='store_true')
    
    args = parser.parse_args()
    
    test_agent(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render
    )
