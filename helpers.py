import numpy as np
import torch


def compute_q_values(agent, states_np: np.ndarray):
    agent.qnetwork_local.eval()
    with torch.no_grad():
        s = torch.from_numpy(states_np).float().to(agent.device)
        q = agent.qnetwork_local(s).detach().cpu().numpy()
    agent.qnetwork_local.train()
    return q


def make_epsilon_by_step(total_steps: int,
                         eps_start: float = 1.0,
                         eps_mid: float = 0.10,
                         eps_end: float = 0.01,
                         frac1: float = 0.25,   # 25% del training: start -> mid
                         frac2: float = 0.75):  # hasta 75%: mid -> end, luego fijo
    total_steps = max(1, int(total_steps))
    s1 = int(total_steps * frac1)
    s2 = int(total_steps * frac2)
    s1 = max(1, s1)
    s2 = max(s1 + 1, s2)

    def epsilon_by_step(step: int) -> float:
        step = int(step)
        if step <= 0:
            return eps_start

        # fase 1: eps_start -> eps_mid
        if step < s1:
            alpha = step / s1
            return eps_start + alpha * (eps_mid - eps_start)

        # fase 2: eps_mid -> eps_end
        if step < s2:
            alpha = (step - s1) / (s2 - s1)
            return eps_mid + alpha * (eps_end - eps_mid)

        # fase 3: fijo
        return eps_end

    return epsilon_by_step

