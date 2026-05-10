"""
Test script para GlobalRoutePlanner
Calcula una ruta completa de punto A a punto B
"""

import carla
import matplotlib.pyplot as plt
import random
from carla_app.planer.navigation.global_route_planner import GlobalRoutePlanner
from carla_app.planer.navigation.local_planner import RoadOption


def pick_random_spawn_pair(spawn_points, max_distance=200.0):
    """Pick random start/goal spawn points with Euclidean distance <= max_distance."""
    valid_pairs = []
    for i, start_sp in enumerate(spawn_points):
        for j, goal_sp in enumerate(spawn_points):
            if i == j:
                continue
            dist = start_sp.location.distance(goal_sp.location)
            if dist <= max_distance:
                valid_pairs.append((start_sp.location, goal_sp.location, dist))

    if not valid_pairs:
        raise RuntimeError(
            f"No se encontraron pares de spawn points con distancia <= {max_distance} m"
        )

    return random.choice(valid_pairs)


def main():
    # Conectar a CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    
    print("✓ Conectado a CARLA")
    print(f"🗺️  Mapa cargado: {carla_map.name}")
    
    # Crear el GlobalRoutePlanner
    sampling_resolution = 2.0  # Distancia entre waypoints (metros)
    planner = GlobalRoutePlanner(carla_map, sampling_resolution)
    print(f"✓ GlobalRoutePlanner creado (resolución: {sampling_resolution}m)")
    
    # Obtener spawn points disponibles
    spawn_points = carla_map.get_spawn_points()
    print(f"\n📌 Spawn points disponibles: {len(spawn_points)}")
    print("   Primeros 5 spawn points:")
    for i, sp in enumerate(spawn_points[:5]):
        loc = sp.location
        print(f"     [{i}] x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}")
    
    # Usar spawn points aleatorios con distancia máxima entre inicio y fin
    if len(spawn_points) >= 2:
        start_location, goal_location, straight_distance = pick_random_spawn_pair(
            spawn_points,
            max_distance=200.0
        )
        print(f"\n✓ Usando spawn points aleatorios (distancia recta <= 200m)")
        print(f"   Distancia inicio-fin: {straight_distance:.2f} m")
    else:
        print(f"\n⚠️  Menos de 2 spawn points disponibles")
        start_location = carla.Location(x=50, y=50, z=0.5)
        goal_location = carla.Location(x=150, y=-100, z=0.5)
        print(f"   Usando coordenadas por defecto")
    
    print(f"\n📍 Punto de SALIDA: x={start_location.x:.2f}, y={start_location.y:.2f}, z={start_location.z:.2f}")
    print(f"📍 Punto de LLEGADA: x={goal_location.x:.2f}, y={goal_location.y:.2f}, z={goal_location.z:.2f}")
    
    # Calcular la ruta
    print("\n🔄 Calculando ruta...")
    try:
        route = planner.trace_route(start_location, goal_location)
        print(f"✓ Ruta calculada con {len(route)} waypoints\n")
    except Exception as e:
        print(f"✗ Error al calcular ruta: {e}")
        return
    
    # Mostrar la ruta completa
    print("=" * 80)

    # Dibujar waypoints con Matplotlib (vista superior XY)
    x_vals = []
    y_vals = []
    for waypoint, _ in route:
        loc = waypoint.transform.location
        x_vals.append(loc.x)
        y_vals.append(loc.y)

    plt.figure(figsize=(10, 8))
    plt.plot(x_vals, y_vals, '-', linewidth=1.5, label='Ruta')
    plt.scatter(x_vals, y_vals, s=8, alpha=0.8, label='Waypoints')
    plt.scatter([start_location.x], [start_location.y], c='green', s=120, marker='o', label='Inicio')
    plt.scatter([goal_location.x], [goal_location.y], c='red', s=120, marker='X', label='Destino')
    plt.title(f"Ruta Global Planner - {carla_map.name}")
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"{'#':<5} {'X (m)':<12} {'Y (m)':<12} {'Z (m)':<12} {'road_id':<10} {'lane_id':<10} {'Instrucción':<20}")
    print("=" * 80)
    
    for i, (waypoint, road_option) in enumerate(route):
        loc = waypoint.transform.location
        road_id = waypoint.road_id
        lane_id = waypoint.lane_id
        instruction = road_option.name
        
        print(f"{i:<5} {loc.x:<12.2f} {loc.y:<12.2f} {loc.z:<12.2f} {road_id:<10} {lane_id:<10} {instruction:<20}")
    
    print("=" * 80)
    
    # Estadísticas de la ruta
    total_distance = 0.0
    for i in range(len(route) - 1):
        wp1 = route[i][0]
        wp2 = route[i + 1][0]
        distance = wp1.transform.location.distance(wp2.transform.location)
        total_distance += distance
    
    print(f"\n📊 Estadísticas:")
    print(f"   • Total de waypoints: {len(route)}")
    print(f"   • Distancia total aproximada: {total_distance:.2f} metros")
    
    # Contar instrucciones
    instructions = {}
    for _, road_option in route:
        name = road_option.name
        instructions[name] = instructions.get(name, 0) + 1
    
    print(f"   • Desglose de instrucciones:")
    for instr, count in sorted(instructions.items()):
        print(f"     - {instr}: {count}")
    
    print("\n✓ Script completado")


if __name__ == "__main__":
    main()
