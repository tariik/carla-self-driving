# Diagnóstico TensorBoard: Catastrophic Forgetting en DQN

## Resumen Ejecutivo

El agente muestra **Q-value overestimation → loss divergence → policy collapse cíclico**.
La loss crece de mediana 8 hasta **963** (P95: 12,000), causando que la política oscile entre "ir rápido y acertar" y "ir rápido y estrellarse".

---

## 1. Datos Clave del Entrenamiento

| Métrica | Valor |
|---------|-------|
| Episodios totales | 100,000 |
| Steps totales | 9,096,547 |
| Steps/episodio medio | ~91 |
| Pico de reward | ep 40-45k: avg=+246.7 |
| Colapso | ep 70-80k: avg=-3.1 a +37.9 |
| Reward final (avg100) | +6.36 |
| Loss final (mediana) | ~328 (debería ser <10) |

---

## 2. Curva de Reward por Fase

```
ep  0-10k:  avg=-0.5   (exploration, aprendiendo)
ep 10-20k:  avg=+47    (descubre: acelerar = reward)
ep 20-30k:  avg=+166   (optimiza: max throttle + recto)
ep 40-45k:  avg=+247   ← PICO (best_checkpoint aquí)
ep 45-55k:  avg=+142   ← primer deterioro
ep 60-65k:  avg=+177   ← recuperación parcial
ep 70-80k:  avg=+17    ← COLAPSO (51% episodios negativos!)
ep 80-100k: avg=+95    ← oscilaciones, nunca recupera pico
```

---

## 3. CAUSA RAÍZ #1: Q-Value Overestimation (FATAL)

### La Loss Explota Continuamente

| Step Range | Loss Mediana | Loss P95 | Loss Max |
|------------|-------------|----------|----------|
| 0 - 600k | 8.2 | 32.9 | 125 |
| 600k - 1.2M | 64.0 | 265 | 890 |
| 1.2M - 1.8M | 259 | 849 | 2,938 |
| 1.8M - 2.4M | 619 | 1,750 | 4,907 |
| 3.0M - 3.6M | **963** | 3,022 | **12,405** |

### ¿Por qué?

Con **GAMMA = 0.995**, el Q-value teórico para un episodio con r/step=2 es:

$$Q_{max} = \frac{r_{avg}}{1 - \gamma} = \frac{2}{0.005} = 400$$

Para episodios con r/step=5 (speed exploit): $Q = 1000+$

- El DQN estándar usa `max(Q_target)` que **sobreestima** sistemáticamente
- Los Q-targets crecen → TD error crece → loss crece → gradientes inestables
- Aunque clipeas gradientes a 1.0, la **dirección** ya está corrompida

### El Ciclo Destructivo

```
Q sobreestimados → TD error grande → loss explota
→ gradientes distorsionados → política agresiva
→ crash rápidos → reward negativo en buffer
→ Q se corrigen bruscamente → colapso
→ exploración residual (eps=0.01) rescata parcialmente
→ Q vuelven a sobreestimarse → repite
```

---

## 4. CAUSA RAÍZ #2: Speed Exploitation

### Reward/Step por Fase

| Fase | Avg r/step | Max r/step |
|------|-----------|-----------|
| 0-20k | +0.29 | +3.18 |
| 20-40k | +1.70 | +5.98 |
| 40-60k | **+1.97** | **+7.34** |
| 60-80k | +1.18 | +7.34 |
| 80-100k | +1.31 | +7.34 |

**El agente aprende que max-throttle = max-reward:**
- `k_v * v_forward = 0.03 * 150km/h * cos(0) = 4.5 reward/step` solo por velocidad
- `k_prog * progress = 1.0 * (150/3.6 * 0.05) ≈ 2.1/step` por distancia
- **Total: ~6.6 reward/step a 150 km/h** → episodio de 100 steps = +660 reward
- Pero a 150 km/h, cualquier desviación lateral → crash inmediato

### Bimodalidad en la Distribución de Rewards

**Fase de colapso (ep 50-80k):**
- 24.8% episodios con reward < -10 (crashes rápidos)
- 6.0% episodios con reward > 500 (speed runs exitosos)
- El agente alterna entre "todo bien" y "crash en 30 steps"

**Fase de pico (ep 25-45k):**
- Solo 8.6% con reward < -10
- 7.2% con reward > 500

---

## 5. CAUSA RAÍZ #3: Diseño del DQN

| Problema | Valor Actual | Impacto |
|----------|-------------|---------|
| GAMMA=0.995 | Muy alto | Q-values de 400-1000+ → overestimation |
| Sin Double DQN | max(Q_target) | Maximization bias amplifica error |
| MSE loss | (Q-Q_target)² | Outliers squared = gradientes enormes |
| TAU=5e-4 | Soft update continuo | Propaga Q corruptos al target |
| Buffer=100k | 1% de steps totales | Buffer dominado por experiencia reciente |
| Sin reward clipping | Rewards de -50 a +700 | Escala de TD error varía 100x |
| LR=5e-4 | Relativamente alto | Con loss de 1000, updates muy agresivos |

---

## 6. Episodios durante el Colapso (ep 70-80k)

### Distribución de Longitud de Episodios

| Steps | % (collapse 70-80k) | % (peak 40-45k) |
|-------|---------------------|-----------------|
| 0-20 | 6.8% | 0.4% |
| 20-50 | **43.5%** | 23.0% |
| 50-100 | 30.5% | **47.1%** |
| 100-200 | 8.7% | 22.3% |
| 200-500 | 5.0% | 6.2% |

**Mediana steps:**
- Collapse: 49 steps (crash rápido)
- Peak: 73 steps (viaja más lejos)

El agente en colapso: acelera a max throttle → se desvía → crash en <50 steps.

---

## 7. Correlación con Epsilon

| Episodio | Epsilon | Avg Reward | Observación |
|----------|---------|-----------|-------------|
| 25-30k | 0.080 | +210 | Exploración aún ayuda |
| 40-45k | 0.030 | +247 | Sweet spot |
| 50-55k | **0.010** | +129 | Epsilon min alcanzado |
| 70-80k | 0.010 | -3 a +38 | Sin exploración → collapse |

**Al llegar a eps_min=0.01, el agente depende 100% de Q-values corruptos.**
Ya no hay exploración que lo "rescate".

---

## 8. PLAN DE FIXES (Priorizado)

> **Nota**: El usuario decidió mantener **DQN estándar con MSE loss** (sin Double DQN ni Huber loss).

### ~~FIX 1: Double DQN~~ — DESCARTADO por decisión del usuario
### ~~FIX 2: Huber Loss~~ — DESCARTADO por decisión del usuario

### FIX 3: Reducir GAMMA a 0.99 ✅ APLICADO
```python
GAMMA = 0.99  # Q_max teórico: 200 en vez de 400
```
**Impacto**: Reduce Q-targets a la mitad, menos overestimation.

### FIX 4: Reward Clipping/Scaling ✅ APLICADO
```python
reward = np.clip(reward, -10.0, 10.0)  # Limitar escala de reward
```
**Impacto**: TD errors acotados, loss no explota.

### FIX 5: Target Network Hard Update periódico ✅ APLICADO
```python
# Cada 10000 steps: copiar local → target completamente
TARGET_UPDATE_HARD = 10000
if self.total_steps % TARGET_UPDATE_HARD == 0:
    self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
```
**Impacto**: Target no se "contamina" gradualmente con Q corruptos.

### FIX 6: Limitar velocidad máxima en reward ✅ APLICADO
```python
v_capped = min(v, 80.0)  # No recompensar >80 km/h
v_forward = v_capped * np.cos(e_psi)
reward = k_v * v_forward - ...
```
**Impacto**: Elimina incentivo a ir a 150 km/h.

### FIX 7: Aumentar buffer ✅ APLICADO
```python
BUFFER_SIZE = int(5e5)  # 500k (cubrir ~5% de training)
```

### FIX 8: Reducir LR ✅ APLICADO
```python
LR = 1e-4  # Más conservador con loss tan alta
```

---

## 9. Configuración Propuesta para Reentrenamiento

```python
# dqn.py
BUFFER_SIZE = int(5e5)
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
UPDATE_EVERY = 4
TARGET_UPDATE_HARD = 10000  # hard copy cada 10k steps

# En learn():
# - DQN estándar (max Q_target) — decisión del usuario
# - MSE loss — decisión del usuario
# - Grad clip = 1.0 (mantener)

# carla_env.py
# - reward = clip(reward, -10, 10)
# - v_capped = min(v, 80.0) para reward
# - max_episode_steps = 300 (mantener)
```

---

## 10. Estimación de Mejora

| Métrica | Actual | Esperado con fixes |
|---------|--------|-------------------|
| Loss (mediana estable) | 8→963 (diverge) | <50 (convergente) |
| Reward avg sostenido | +247 → colapsa | +200-300 estable |
| Goal rate | 0-10% | 30-50% |
| Crash rate (collapse) | 51% | <15% |
| Velocidad típica | 120-150 km/h | 50-80 km/h |

Las fixes #1 (Double DQN) y #2 (Huber Loss) **por sí solas** deberían eliminar el 80% del problema de collapse.



---------------------------------------------



## Resumen Completo de Cambios (v2)

---

### A. Red Neuronal — QNetwork (dqn.py)

| Componente | Antes (v1) | Ahora (v2) | Motivo |
|-----------|------------|------------|--------|
| Arquitectura | 4→128→128→64→27 | **4→128→128→64→27** (sin cambio) | |
| BatchNorm | `bn1`, `bn2`, `bn3` después de cada FC | **Eliminado** | BatchNorm causa inestabilidad en DQN: `eval()` usa running stats, `train()` usa batch stats → Q-values ruidosos e inconsistentes |
| Dropout | `Dropout(p=0.2)` en capa 1 y 2 | **Eliminado** | Redundante: la exploración ya la cubre epsilon-greedy. Dropout introduce ruido en Q-values durante `train()` |
| Xavier init | Sí | **Sí** (sin cambio) | |
| train/eval diff | **≠ 0** (por BatchNorm) | **= 0.0** exacto | Q-values deterministas → policy estable |

### B. Hiperparámetros DQN (dqn.py)

| Parámetro | Antes (v1) | Ahora (v2) | Motivo |
|-----------|------------|------------|--------|
| GAMMA | 0.995 | **0.99** | $Q_{max}$ teórico: 400→200, reduce overestimation |
| LR | 5e-4 | **1e-4** | Updates más suaves cuando loss es alta |
| BUFFER_SIZE | 100k | **500k** | Más diversidad, menos sesgo a experiencia reciente |
| BATCH_SIZE | 128 | **128** (sin cambio) | |
| UPDATE_EVERY | 4 | **4** (sin cambio) | |
| Target update | Soft (TAU=5e-4) | **Hard cada 10k steps** | Target no se contamina gradualmente con Q corruptos |
| Loss | MSE | **MSE** (sin cambio) | Decisión del usuario |
| DQN tipo | Estándar | **Estándar** (sin cambio) | Decisión del usuario (sin Double DQN) |
| Grad clip | 1.0 | **1.0** (sin cambio) | |

### C. Observaciones — Estado del Agente (carla_env.py)

| Obs | Antes (v1) | Ahora (v2) | Rango | Motivo |
|-----|------------|------------|-------|--------|
| `v` (speed) | km/h directos [0, 120] | **÷ 120** normalizado [0, 1] | [0, 1] | Antes speed dominaba los gradientes (120× más grande que e_y). Ahora todas las obs en rango comparable |
| `e_y` (lateral) | `clip(dt/denom, -1, 1)` | **sin cambio** | [-1, 1] | |
| `e_psi` (angular) | `deg2rad(angle)` | **sin cambio** | [-π, π] | |
| `prev_steer` | steer anterior | **sin cambio** | [-1, 1] | |
| **obs_space** | `Box([0,-1,-π,-1], [120,1,π,1])` | `Box([0,-1,-π,-1], [1,1,π,1])` | | Actualizado para reflejar normalización |

### D. Función de Reward (carla_env.py step)

| Componente | Fórmula | Peso | Cambio |
|-----------|---------|------|--------|
| Velocidad | `k_v × min(v, 80) × cos(e_psi)` | k_v=0.03 | **v1→v2**: cap de 80 km/h (antes sin límite → exploit a 150) |
| Lateral | `-k_y × e_y²` | k_y=1.0 | Sin cambio |
| Angular | `-k_psi × e_psi²` | k_psi=0.5 | Sin cambio |
| Δsteer | `-k_delta × (steer - prev_steer)²` | k_delta=0.2 | Sin cambio |
| Steer abs | `-k_s × steer²` | k_s=0.05 | Sin cambio |
| Progress | `+k_prog × (prev_dist - dist_to_goal)` | k_prog=1.0 | Sin cambio |
| **Reward clip** | `np.clip(reward, -10, +10)` | — | **Nuevo en v2**: acota TD error |

**Penalizaciones terminales:**

| Evento | Penalización | Termina? | Condición |
|--------|-------------|----------|-----------|
| Colisión | -10 | Sí | Después de GRACE_PERIOD=20 steps |
| Off-road | -10 | Sí | `|dist_center| > lane_width × 0.95` |
| Invasión carril | -5 × n_nuevas | Sí (≥5 total) | Acumulativo |
| Ángulo >90° | -5 | Sí | `|e_psi| > π/2` |
| Goal alcanzado | **+5** | Sí | `dist_to_goal ≤ 5m` |

### E. Entrenamiento — Training Loop (carla_env.py main)

| Parámetro | Antes (v1) | Ahora (v2) | Motivo |
|-----------|------------|------------|--------|
| n_episodes | 100k | **100k** (sin cambio) | |
| max_t | 1000 | **1000** (sin cambio) | Max steps por episodio |
| max_episode_steps | 300 | **300** (sin cambio) | Truncamiento |
| render_mode | None | **None** (sin cambio) | Sin pygame en training |
| EXPECTED_STEPS_PER_EP | 50 | **100** | Episodios más largos con speed cap 80 |
| Total steps (epsilon) | 5M | **10M** | Epsilon decae más gradualmente |
| Warmup 1000 | Bypass de `agent.step()` + manual `memory.add()` | **Eliminado**: `agent.step()` siempre | El bypass desalineaba `total_steps` con el hard target update |
| Double respawn | `reset()` hacía `respawn_vehicle()` 2 veces | **Corregido**: solo 1 vez | Bug: ahorra 1 tick/episodio × 100k eps |
| Probe Q-values log | Cada episodio (81 scalars × 100k) | **Cada 10 episodios** | 10× menos I/O TensorBoard |
| Matplotlib plots | Se generaban al final | **Eliminado** | No lanzar gráficas durante training |
| pygame en training | Se lanzaba (bug: `initialize_components()` sin arg) | **Corregido**: pasa `render_mode` | Solo se abre en `test_dqn.py` |

### F. Schedule de Epsilon (por pasos globales)

```
make_epsilon_by_step(total_steps=10,000,000)
├── Phase 1 [step 0 → 2.5M]:    eps 1.00 → 0.10  (exploración fuerte)
├── Phase 2 [step 2.5M → 7.5M]: eps 0.10 → 0.01  (explotación gradual)
└── Phase 3 [step 7.5M → ∞]:    eps 0.01 fijo     (explotación pura)
```

Con EXPECTED_STEPS_PER_EP=100:
- Phase 1 termina ~ episodio 25,000
- Phase 2 termina ~ episodio 75,000
- Phase 3: últimos 25,000 episodios

### G. Probe States para TensorBoard

Los 3 estados fijos monitorean convergencia en situaciones típicas:

| Probe | v_norm | v_real | e_y | e_psi | steer | Situación | Objetivo de acción |
|-------|--------|--------|-----|-------|-------|-----------|----------|
| **state0** | 0.50 | 60 km/h | 0.0 | 0° | 0.0 | ✅ Ideal: recto y centrado | Acelerar (throttle≥0.5), steer=0 |
| **state1** | 0.33 | 40 km/h | 0.5 | +10° | 0.25 | ⚠️ Desviado derecha: corrección urgente | Reducir veloc, girar izq (steer<0) |
| **state2** | 0.67 | 80 km/h | 0.0 | 0° | 0.0 | 🏎️ Rápido y centrado | Mantener (throttle≥0), steer=0 |

**Nota**: Probe states usan speed normalizada (÷120) como las observaciones reales.

---

### G.1 Todas las 27 Acciones Discretas

El espacio de acciones es: **9 steering valores × 3 throttle valores = 27 acciones**

#### **Mapeo Completo (action_id → (steering, throttle))**

| ID | Steering | Throttle | Descripción |
|----|----------|----------|-------------|
| **00-08** | [-1.0 a +1.0] | **0.0** (sin aceleración) |  |
| 00 | -1.00 (giro máx izq) | 0.0 | Freno + giro izquierda máximo |
| 01 | -0.75 | 0.0 | Freno + giro fuerte izquierda |
| 02 | -0.50 | 0.0 | Freno + giro moderado izquierda |
| 03 | -0.25 | 0.0 | Freno + giro leve izquierda |
| 04 | 0.00 (recto) | 0.0 | **Freno puro** (solo desacelera) |
| 05 | +0.25 | 0.0 | Freno + giro leve derecha |
| 06 | +0.50 | 0.0 | Freno + giro moderado derecha |
| 07 | +0.75 | 0.0 | Freno + giro fuerte derecha |
| 08 | +1.00 (giro máx der) | 0.0 | Freno + giro máximo derecha |
| **09-17** | [-1.0 a +1.0] | **0.5** (aceleración media) |  |
| 09 | -1.00 | 0.5 | Acelerar suave + giro izquierda máx |
| 10 | -0.75 | 0.5 | Acelerar suave + giro fuerte izq |
| 11 | -0.50 | 0.5 | Acelerar suave + giro moderado izq |
| 12 | -0.25 | 0.5 | Acelerar suave + giro leve izq |
| 13 | 0.00 | 0.5 | **Acelerar suave + recto** (ideal para cruising) |
| 14 | +0.25 | 0.5 | Acelerar suave + giro leve der |
| 15 | +0.50 | 0.5 | Acelerar suave + giro moderado der |
| 16 | +0.75 | 0.5 | Acelerar suave + giro fuerte der |
| 17 | +1.00 | 0.5 | Acelerar suave + giro máximo der |
| **18-26** | [-1.0 a +1.0] | **1.0** (aceleración máxima) |  |
| 18 | -1.00 | 1.0 | Acelerar máximo + giro izquierda máx |
| 19 | -0.75 | 1.0 | Acelerar máximo + giro fuerte izq |
| 20 | -0.50 | 1.0 | Acelerar máximo + giro moderado izq |
| 21 | -0.25 | 1.0 | Acelerar máximo + giro leve izq |
| 22 | 0.00 | 1.0 | **Acelerar máximo + recto** (fast cruise) |
| 23 | +0.25 | 1.0 | Acelerar máximo + giro leve der |
| 24 | +0.50 | 1.0 | Acelerar máximo + giro moderado der |
| 25 | +0.75 | 1.0 | Acelerar máximo + giro fuerte der |
| 26 | +1.00 | 1.0 | Acelerar máximo + giro máximo der |

---

### G.2 Acciones Óptimas Esperadas por Probe State

#### **State 0: Ideal (v=60, centered, aligned)**

```
Observación: [0.50, 0.0, 0.0, 0.0]  ← recto, centrado, sin rotación
Acción óptima: 13 o 22
- Acción 13: (steer=0, throttle=0.5)   Q ≈ +3.8 reward/step
- Acción 22: (steer=0, throttle=1.0)   Q ≈ +2.4 reward/step (más lento por velocidad)

¿Por qué 13 > 22?
- A 60 km/h ya tienes buen balance
- Ir a 80 km/h añade +0.6 por velocidad
- Pero aumenta riesgo de desvío (menos margen)
- Por eso throttle=0.5 es mejor que 1.0

El agente debería aprender: Q[13] > Q[22] en este estado
```

#### **State 1: Desviado (v=40, e_y=0.5, e_psi=10°, steer_prev=0.25)**

```
Observación: [0.33, 0.5, 0.175 rad (10°), 0.25]  ← desviado derecha, girado
Acción óptima: 12 o 11 (corrección izquierda)
- Acción 11: (steer=-0.5, throttle=0.5)   Q ≈ +1.2 reward/step
- Acción 12: (steer=-0.25, throttle=0.5)  Q ≈ +1.5 reward/step

¿Por qué 12 > 11?
- Estás desviado (e_y=0.5), necesitas girar izquierda (steer < 0)
- Pero steer=-0.5 es muy agresivo (saltas de +0.25 a -0.5 = Δsteer=0.75)
- Penalización por cambio brusco: -0.075
- Mejor: steer=-0.25 (cambio más suave, Δsteer=0.5)
- Penalización menor, corrección gradual

El agente debería aprender: Q[12] > Q[11] (corrección suave)
También: Q[12] > Q[13] (porque necesitas corregir, no mantener curso)
```

#### **State 2: Rápido y Centrado (v=80, e_y=0.0, e_psi=0°)**

```
Observación: [0.67, 0.0, 0.0, 0.0]  ← recto, centrado, va rápido
Acción óptima: 22 o 13
- Acción 22: (steer=0, throttle=1.0)   Q ≈ +2.4 reward/step
- Acción 13: (steer=0, throttle=0.5)   Q ≈ +1.8 reward/step

¿Por qué 22 > 13?
- Ya estás a 80 km/h (velocidad capped)
- No puedes ir más rápido, pero mantener throttle=1.0 mantiene velocidad
- throttle=0.5 ralentiza (aceleración menor)
- Ambas dan similar reward total por velocidad
- Pero 22 es más "confiado": mantiene dinamismo

En balance: Q[22] ≈ Q[13] (porque cap de 80 km/h)
Pero el agente podría preferir 13 por seguridad (margen extra)
```

---

### G.3 Monitoreo en TensorBoard

Cada **10 episodios**, se loguean:

| Tag | Significado |
|-----|------------|
| `qprobe/state0/action_00` ... `action_26` | Q-value de cada una de las 27 acciones en state0 |
| `qprobe/state0/Q_max` | Valor máximo Q entre todas las acciones en state0 |
| `qprobe/state0/Q_best_action` | ID de la acción con máximo Q (0-26) |
| `qprobe/state1/action_00` ... `action_26` | Q-value de cada acción en state1 |
| `qprobe/state1/Q_max` | Q máximo en state1 |
| `qprobe/state1/Q_best_action` | Mejor acción en state1 |
| `qprobe/state2/action_00` ... `action_26` | Q-value de cada acción en state2 |
| `qprobe/state2/Q_max` | Q máximo en state2 |
| `qprobe/state2/Q_best_action` | Mejor acción en state2 |

**¿Cómo interpretar convergencia?**

- **Líneas ruidosas → inestables**: El agente aún explora (epsilon > 0.1)
- **Líneas estables** (no cambian por >100 eps): Convergencia
- **Q_best_action fijo** (ej: siempre 13 en state0): Policy estabilizado

**Ejemplo de convergencia buena (ep 80k-100k):**

```
state0:
  - Q[13] ≈ 450 ± 20  (mejor acción: acelerar suave + recto)
  - Q[22] ≈ 400 ± 20  (segundo: acelerar máx + recto)
  - Q[04] ≈ 100 ± 15  (peor: freno)
  - Q_best_action = 13 (fijo)

state1:
  - Q[12] ≈ 320 ± 25  (mejor: corrección suave izquierda)
  - Q[11] ≈ 310 ± 25  (segundo: corrección fuerte)
  - Q[13] ≈ 200 ± 20  (peor: mantener curso, ignora desviación)
  - Q_best_action = 12 (fijo)

state2:
  - Q[22] ≈ 380 ± 20  (mejor: mantener velocidad)
  - Q[13] ≈ 360 ± 20  (segundo: reducir a velocidad media)
  - Q_best_action = 22 (fijo)
```

**Señales de problemas:**

- ❌ Q_max oscila entre 100 y 10000 → Q-overestimation no fixed
- ❌ Q_best_action cambia constantemente → policy inestable
- ❌ Todos los Q[i] ≈ 0 → modelo no entrenado
- ❌ Q[action_worst] ≈ Q[action_best] → no discrimina buenas acciones

### H. Reward Máximo Teórico por Step

| | v1 (original) | v2 (actual) |
|--|---------------|-------------|
| Velocidad | $0.03 \times 150 = 4.5$ | $0.03 \times 80 = 2.4$ |
| Progress | $\sim 2.1$ (a 150 km/h) | $\sim 1.1$ (a 80 km/h) |
| **Total/step** | **~6.6** | **~3.5** |
| $Q_{max}$ teórico | $\frac{6.6}{0.005} = 1320$ | $\frac{3.5}{0.01} = 350$ |

### I. Resultado Esperado

| Métrica | v1 (original) | v2 (esperado) |
|---------|---------------|---------------|
| Loss (mediana) | 8 → 963 (diverge) | **< 50** (convergente) |
| Reward avg sostenido | +247 → colapsa | **+200-300 estable** |
| Goal rate | 0-10% | **30-50%** |
| Crash rate (collapse) | 51% | **< 15%** |
| Velocidad típica | 120-150 km/h | **50-80 km/h** |
| train/eval Q consistency | Ruidoso (BatchNorm) | **Determinista (diff=0)** |

### J. Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `carla_app/dqn.py` | QNetwork sin BatchNorm/Dropout, GAMMA=0.99, LR=1e-4, BUFFER=500k, hard target update 10k |
| `carla_env.py` | Obs normalizada (v÷120), reward clip [-10,10], speed cap 80, fix double respawn, fix pygame init, warmup simplificado, probe Q-log cada 10 eps, sin matplotlib |
| `test_dqn.py` | Sin cambios (usa render_mode='human', compatible con obs normalizada vía el env) |