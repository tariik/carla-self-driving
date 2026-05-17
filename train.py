from collections import deque
import time
from turtle import done
import numpy as np
from agents.dqn_waypoints import dqn
from torch.utils.tensorboard import SummaryWriter
import time

from carla_env import CarlaEnv
from helpers import compute_q_values, make_epsilon_by_step

from configs import train_config

if __name__ == '__main__':
    print("Testing CarlaWorldEnv as Gymnasium environment...")
  

    # 🔥 PARÁMETROS DE ENTRENAMIENTO
    n_episodes = 50000  # Más episodios para aprender
    max_t = 800       # Steps por episodio
    eps_start = 1.0   # ✅ FIX: Empezar con exploración completa
    eps_decay = 0.9995
    eps_end = 0.01

    # 🔥 render_mode='human' para ver el entrenamiento
    env = CarlaEnv(train_config, render_mode=None, max_episode_steps=max_t, verbose=False)
    
    # Create agent
    agent = dqn(env, seed=42)
    
    # Training metrics
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    # ✅ AGREGAR: Tracking del mejor modelo
    best_avg_score = -np.inf
    best_episode = 0
    best_checkpoint_path = None
    global_step = 0
    
    print("\n🚀 Starting DQN Training for Lane Keeping")
    print(f"Episodes: {n_episodes}, Max steps: {max_t}")
    print(f"Epsilon: {eps_start} → {eps_end} (decay: {eps_decay})")
    print("=" * 60)

    run_name = time.strftime("carla_dqn_%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # Probe states con dimensionalidad completa del entorno (ej. 19D).
    # Solo fijamos núcleo [v_norm, e_y, e_psi, steer_prev] y dejamos waypoints en 0.
    obs_dim = int(np.prod(env.observation_space.shape))
    probe_states = np.zeros((3, obs_dim), dtype=np.float32)
    probe_states[0, :4] = np.array([60.0 / 120.0, 0.0, 0.0, 0.0], dtype=np.float32)
    probe_states[1, :4] = np.array([40.0 / 120.0, 0.5, np.deg2rad(10.0), 0.25], dtype=np.float32)
    probe_states[2, :4] = np.array([80.0 / 120.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # ✅ FIX: total_steps basado en pasos REALES esperados, no max_t teórico.
    # Con speed cap 80 km/h y reward clip, episodios más largos (~100 steps).
    # 100 steps/ep × 100,000 eps = 10,000,000 pasos efectivos.
    # - Phase 1 (eps 1.0→0.10): primeros 25% = 2,500,000 steps ≈ 25,000 eps
    # - Phase 2 (eps 0.10→0.01): hasta 75%  = 7,500,000 steps ≈ 75,000 eps
    # - Phase 3 (eps fijo 0.01): resto
    EXPECTED_STEPS_PER_EP = 100  # más largo con speed cap a 80 km/h
    realistic_total_steps = n_episodes * EXPECTED_STEPS_PER_EP
    epsilon_by_step = make_epsilon_by_step(realistic_total_steps,
                                       eps_start=0.50,
                                       eps_mid=0.05,
                                       eps_end=0.01,
                                       frac1=0.025,
                                       frac2=0.075)
  
    
    try:
        for i_episode in range(1, n_episodes + 1):
            state, info = env.reset()
            score = 0
            steps_taken = 0
            episode_info = None
            
            for t in range(max_t):

                # Select action
                global_step += 1

                eps = float(epsilon_by_step(global_step))
                action = agent.act(state, eps)
                
                # Take action
                next_state, reward, done, truncated, info = env.step(action)
                episode_info = info  # Store final episode info for logging
                
                # 🔥 RENDER para ver el coche
                #if env.render_mode == 'human':
                #    if not env.render():
                #        print("\n⚠ Window closed by user")
                #        raise KeyboardInterrupt
                
                # Store experience and learn (agent.step ya espera BATCH_SIZE muestras)
                agent.step(state, action, reward, next_state, done or truncated)
                if agent.last_loss is not None:
                    writer.add_scalar("train/loss", float(agent.last_loss), global_step)
                
                # Update state and score
                state = next_state
                score += reward
                steps_taken = t + 1

                
                
                if done or truncated:
                    break
            
            # Update metrics
            scores_window.append(score)
            scores.append(score)
            # eps = max(eps_end, eps_decay * eps)

            
            

            # ✅ VERIFICAR SI ES EL MEJOR MODELO (cada 100 episodios para estabilidad)
            if i_episode >= 100 and i_episode % 10 == 0:
                current_avg = np.mean(scores_window)
            
                if current_avg > best_avg_score:
                    best_avg_score = current_avg
                    best_episode = i_episode
                    
                    # Guardar best checkpoint
                    import os
                    os.makedirs('checkpoints_dqn', exist_ok=True)
                    best_checkpoint_path = f'checkpoints_dqn/best_checkpoint.pth'
                    agent.save(best_checkpoint_path)
                    
                    print(f'   🏆 NEW BEST! Avg: {best_avg_score:.2f} at ep {best_episode} - Saved!')
            
            # Print progress every 10 episodes with detailed telemetry
            if i_episode % 10 == 0:
                termination_reason = episode_info.get('termination_reason', 'unknown') if episode_info else 'unknown'
                collision = episode_info.get('collision', False) if episode_info else False
                illegal_invasions = episode_info.get('illegal_invasions', 0) if episode_info else 0
                speed = episode_info.get('speed', 0) if episode_info else 0
                
                print(f'Ep {i_episode:6d} | {termination_reason:15s} | '
                    f'Steps: {steps_taken:3d} | Score: {score:7.2f} | '
                    f'Avg(100): {np.mean(scores_window):7.2f} | Eps: {eps:.4f} | '
                    f'Collision: {collision:5} | Invasions: {illegal_invasions:2d} | '
                    f'Speed: {speed:5.1f} km/h')


            
            writer.add_scalar("train/episode_reward", score, i_episode)
            writer.add_scalar("train/episode_steps", steps_taken, i_episode)
            writer.add_scalar("train/epsilon", eps, i_episode)

            # Log telemetry data from episode
            if episode_info:
                writer.add_scalar("telemetry/collision", float(episode_info.get('collision', False)), i_episode)
                writer.add_scalar("telemetry/illegal_invasions", float(episode_info.get('illegal_invasions', 0)), i_episode)
                writer.add_scalar("telemetry/speed", float(episode_info.get('speed', 0)), i_episode)
                writer.add_scalar("telemetry/distance_to_center", float(episode_info.get('distance_to_center', 0)), i_episode)
                
                # Log termination reason as categorical (0=goal, 1=collision, 2=heading, 3=invasion, 4=time_limit, 5=other)
                termination_map = {
                    'goal_reached': 0,
                    'collision': 1,
                    'heading_error': 2,
                    'lane_invasion': 3,
                    'time_limit': 4
                }
                term_reason = episode_info.get('termination_reason', 'unknown')
                term_code = termination_map.get(term_reason, 5)
                writer.add_scalar("telemetry/termination_reason", float(term_code), i_episode)

            # si estás usando average window
            if len(scores_window) > 0:
                writer.add_scalar("train/avg_reward_100", float(np.mean(scores_window)), i_episode)

            # Log Q-values cada 10 episodios (reducir I/O de TensorBoard)
            if i_episode % 10 == 0:
                q = compute_q_values(agent, probe_states)  # shape (3, 27)
                for i in range(3):
                    for a in range(q.shape[1]):
                        writer.add_scalar(f"qprobe/state{i}/action_{a:02d}", float(q[i, a]), global_step)
                    writer.add_scalar(f"qprobe/state{i}/Q_max", float(q[i].max()), global_step)
                    writer.add_scalar(f"qprobe/state{i}/Q_best_action", int(q[i].argmax()), global_step)
            
            # Save checkpoint every 50 episodes
            if i_episode % 500 == 0:
                import os
                os.makedirs('checkpoints_dqn_bkp', exist_ok=True)
                agent.save(f'checkpoints_dqn_bkp/checkpoint_ep{i_episode}.pth')
                print(f'   ✓ Checkpoint saved')
    
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    
    finally:
        env.close()
        writer.close()
        
        print("✓ Training complete!")
        if len(scores) > 0:
            print(f"✓ Total episodes: {len(scores)}")
            print(f"✓ Final average score: {np.mean(list(scores_window)[-10:]):.2f}")