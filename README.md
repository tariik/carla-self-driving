# CARLA Environment - Modular Architecture

Sistema modular para control de vehículos en CARLA Simulator, diseñado para facilitar la futura integración con Gymnasium/OpenAI Gym para entrenamiento de RL.

## 📁 Estructura del Proyecto

```
carla_env/
├── envs/                    # 🎮 Gymnasium Environments (futuro)
│   └── __init__.py         # Punto de integración para RL
│
├── agents/                  # 🤖 Agentes de Control
│   ├── __init__.py
│   └── autopilot_agent.py  # PID controller con driving imperfecto
│
├── sensors/                 # 📷 Sensores
│   ├── __init__.py
│   ├── camera_manager.py   # Cámara principal (vista)
│   └── photo_capture.py    # Captura de fotos para dataset
│
├── perception/              # 👁️ Percepción
│   ├── __init__.py
│   └── lane_detector.py    # Detección de carril y posición
│
├── control/                 # 🎛️ Control del Vehículo
│   ├── __init__.py
│   └── vehicle_controller.py  # Control bajo nivel (throttle, brake, steer)
│
├── navigation/              # 🗺️ Navegación
│   ├── __init__.py
│   └── waypoint_renderer.py   # Generación de rutas y waypoints
│
├── ui/                      # 🖥️ Interfaz de Usuario
│   ├── __init__.py
│   └── hud_renderer.py     # HUD con telemetría
│
├── core/                    # ⚙️ Núcleo
│   ├── __init__.py
│   ├── config.py           # Configuración centralizada
│   └── state_machine.py    # Máquina de estados de navegación
│
└── utils/                   # 🔧 Utilidades
    └── __init__.py         # Helpers y funciones comunes
```

## 🚀 Uso Actual

```bash
# Ejecutar el sistema actual
python main_new.py
```

## 🔮 Integración Futura con Gymnasium

### Ejemplo de Uso Futuro:

```python
import gymnasium as gym
from carla_env.envs import CarlaNavigationEnv

# Crear environment
env = gym.make('CarlaNavigation-v0')

# Reset
observation, info = env.reset()

# Loop de entrenamiento
for _ in range(1000):
    action = agent.select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## 📦 Módulos

### **agents/** - Agentes de Control
- `AutopilotController`: PID controller con errores simulados (drift, ruido, oscilaciones)

### **sensors/** - Sensores
- `CameraManager`: Cámara principal para visualización
- `PhotoCapture`: Sistema de captura automática de fotos para datasets

### **perception/** - Percepción
- `LaneDetector`: Calcula distancia al centro del carril y ángulo

### **control/** - Control
- `VehicleController`: Control bajo nivel del vehículo (throttle, brake, steer)

### **navigation/** - Navegación
- `WaypointRenderer`: Genera rutas usando Global Planner de CARLA

### **ui/** - UI
- `HUDRenderer`: Muestra velocidad, posición, controles, lane info, fotos

### **core/** - Núcleo
- `Config`: Configuración centralizada
- `SimpleStateMachine`: Estados de navegación (IDLE, NAVIGATING, GOAL_REACHED)

## 🎯 Ventajas de esta Estructura

1. **Modularidad**: Cada componente es independiente y reutilizable
2. **Escalabilidad**: Fácil añadir nuevos sensores, agentes o environments
3. **Testeable**: Cada módulo se puede testear aisladamente
4. **Gymnasium-Ready**: Estructura compatible con estándar de RL
5. **Separación de Concerns**: Lógica claramente separada por responsabilidad

## 🔧 Configuración

Toda la configuración está centralizada en `carla_env/core/config.py`:

```python
# Ejemplo de configuración
START_X = 383.0
START_Y = -2.0
GOAL_X = 334.0
GOAL_Y = 43.0
AUTOPILOT_MODE = True
PHOTO_CAPTURE_INTERVAL = 1.0  # segundos
```

## 📸 Sistema de Captura de Fotos

- Captura automática cada N segundos (configurable)
- Captura manual con tecla 'P'
- Guardado asíncrono para no bloquear
- Contador visible en HUD

## 🎮 Controles

- **W / ↑**: Acelerar
- **S / ↓**: Frenar/Reversa
- **A / ←**: Girar izquierda
- **D / →**: Girar derecha
- **SPACE**: Freno de mano
- **P**: Capturar foto manual
- **ESC**: Salir

## 📊 Features

- ✅ Control manual y autopilot
- ✅ PID controller con errores realistas
- ✅ Detección de carril y ángulo
- ✅ Sistema de captura de fotos
- ✅ Máquina de estados de navegación
- ✅ HUD con telemetría completa
- ✅ Arquitectura modular y escalable
- 🔲 Gymnasium environment (próximamente)
- 🔲 Reward shaping (próximamente)
- 🔲 Multi-sensor fusion (próximamente)

## 🛠️ Dependencias

```
carla==0.9.16
numpy==2.3.4
pygame==2.6.1
```

## 📝 Próximos Pasos para RL

1. Crear `carla_env/envs/carla_navigation_env.py`
2. Implementar métodos Gymnasium: `reset()`, `step()`, `render()`
3. Definir `observation_space` y `action_space`
4. Implementar función de recompensa (reward shaping)
5. Registrar environment en Gymnasium
6. Integrar con stable-baselines3 o RLlib

## 👥 Autor

Professional Development Team - CARLA Autopilot System

## 📄 Licencia

MIT License
