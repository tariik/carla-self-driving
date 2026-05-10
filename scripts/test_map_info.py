"""
Script completo para extraer TODOS los parámetros del mapa de CARLA
"""

import carla


def main():
    # Conectar
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    
    print("=" * 80)
    print("📍 INFORMACIÓN GENERAL DEL MAPA")
    print("=" * 80)
    
    # 1. Nombre del mapa
    print(f"Nombre del mapa: {carla_map.name}")
    
    # 2. Spawn points recomendados
    spawn_points = carla_map.get_spawn_points()
    print(f"\n🎯 Spawn points disponibles: {len(spawn_points)}")
    print("   Primeros 5 spawn points:")
    for i, sp in enumerate(spawn_points[:5]):
        loc = sp.location
        rot = sp.rotation
        print(f"   [{i}] Pos=(x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}) Rot=(pitch={rot.pitch:.1f}, yaw={rot.yaw:.1f}, roll={rot.roll:.1f})")
    
    print("\n" + "=" * 80)
    print("🛣️  INFORMACIÓN DE TOPOLOGÍA Y CARRETERAS")
    print("=" * 80)
    
    # 3. Topología (conexiones de carreteras)
    topology = carla_map.get_topology()
    print(f"Topología (pares de waypoints conectados): {len(topology)} conexiones")
    print("   Primeras 5 conexiones:")
    for i, (wp_start, wp_end) in enumerate(topology[:5]):
        print(f"   [{i}] road_id={wp_start.road_id} → road_id={wp_end.road_id}")
    
    # 4. Todos los waypoints del mapa
    all_waypoints = list(carla_map.generate_waypoints(distance=5.0))
    print(f"\n📌 Waypoints generados (distancia=5m): {len(all_waypoints)}")
    print("   Primeros 3 waypoints:")
    for i, wp in enumerate(all_waypoints[:3]):
        loc = wp.transform.location
        print(f"   [{i}] road_id={wp.road_id}, lane_id={wp.lane_id}, s={wp.s:.2f}")
        print(f"        Pos=(x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f})")
        print(f"        is_junction={wp.is_junction}, lane_width={wp.lane_width:.2f}m")
    
    print("\n" + "=" * 80)
    print("🔍 INFORMACIÓN DETALLADA DE UN WAYPOINT")
    print("=" * 80)
    
    # 5. Propiedades detalladas de un waypoint específico
    if all_waypoints:
        wp = all_waypoints[0]
        
        print(f"\nWaypoint: {wp}")
        print(f"  • ID único: {wp.id}")
        print(f"  • Road ID: {wp.road_id}")
        print(f"  • Section ID: {wp.section_id}")
        print(f"  • Lane ID: {wp.lane_id}")
        print(f"  • Distancia en carretera (s): {wp.s:.2f}m")
        print(f"  • Transform: {wp.transform}")
        print(f"    - Ubicación: x={wp.transform.location.x:.2f}, y={wp.transform.location.y:.2f}, z={wp.transform.location.z:.2f}")
        print(f"    - Rotación (Yaw): {wp.transform.rotation.yaw:.2f}°")
        print(f"  • Ancho del carril: {wp.lane_width:.2f}m")
        print(f"  • Es cruce (junction): {wp.is_junction}")
        print(f"  • Junction ID: {wp.junction_id if wp.is_junction else 'N/A'}")
        
        # Tipos de carril
        print(f"  • Tipo de carril: {wp.lane_type}")
        
        # Cambios de carril permitidos
        print(f"  • Cambios de carril permitidos: {wp.lane_change}")
        
        # Marcas de carril
        print(f"  • Marca de carril derecha: {wp.right_lane_marking}")
        print(f"  • Marca de carril izquierda: {wp.left_lane_marking}")
        
        # Navegación
        print(f"\n  Navegación:")
        next_wps = wp.next(10.0)  # Siguientes waypoints a 10m
        print(f"    - Siguientes waypoints (10m): {len(next_wps)}")
        if next_wps:
            print(f"      → road_id={next_wps[0].road_id}, lane_id={next_wps[0].lane_id}")
        
        prev_wps = wp.previous(10.0)
        print(f"    - Waypoints anteriores (10m): {len(prev_wps)}")
        if prev_wps:
            print(f"      ← road_id={prev_wps[0].road_id}, lane_id={prev_wps[0].lane_id}")
        
        # Carriles adyacentes
        right_lane = wp.get_right_lane()
        left_lane = wp.get_left_lane()
        print(f"    - Carril derecho: {'Sí' if right_lane else 'No'}")
        print(f"    - Carril izquierdo: {'Sí' if left_lane else 'No'}")
        
        # Junction si existe
        if wp.is_junction:
            junction = wp.get_junction()
            print(f"    - Junction: ID={junction.id}, BBox={junction.bounding_box}")
    
    print("\n" + "=" * 80)
    print("📍 INFORMACIÓN GEOGRÁFICA")
    print("=" * 80)
    
    # 6. Conversión a coordenadas geográficas
    test_location = carla.Location(x=0, y=0, z=0)
    geo_location = carla_map.transform_to_geolocation(test_location)
    print(f"Conversión de coordenadas simuladas a geográficas:")
    print(f"  Ubicación simulada: {test_location}")
    print(f"  Geo-referencia (lat/long): latitude={geo_location.latitude}, longitude={geo_location.longitude}")
    print(f"  Altura (z): {geo_location.altitude}")
    
    print("\n" + "=" * 80)
    print("🗺️  INFORMACIÓN DEL ARCHIVO OpenDRIVE")
    print("=" * 80)
    
    # 7. OpenDRIVE (raw)
    xodr = carla_map.to_opendrive()
    print(f"Contenido OpenDRIVE: {len(xodr)} caracteres")
    print(f"Primeras 500 caracteres:")
    print(xodr[:500])
    
    print("\n" + "=" * 80)
    print("✅ RESUMEN DE MÉTODOS DISPONIBLES EN carla.Map")
    print("=" * 80)
    
    methods = {
        "get_waypoint(location, project_to_road=True, lane_type=Driving)": "Obtiene waypoint más cercano a una ubicación",
        "get_waypoint_xodr(road_id, lane_id, s)": "Obtiene waypoint exacto por OpenDRIVE",
        "get_topology()": "Obtiene grafo minimal de carreteras",
        "get_spawn_points()": "Obtiene puntos recomendados de spawn",
        "generate_waypoints(distance)": "Genera waypoints espaciados uniformemente",
        "transform_to_geolocation(location)": "Convierte ubicación simulada a geo-referencias",
        "to_opendrive()": "Obtiene contenido OpenDRIVE como string",
    }
    
    for method, description in methods.items():
        print(f"\n  • {method}")
        print(f"    → {description}")
    
    print("\n" + "=" * 80)
    print("✅ PROPIEDADES DE CADA WAYPOINT")
    print("=" * 80)
    
    waypoint_properties = {
        "id": "ID único del waypoint",
        "road_id": "ID de la carretera (OpenDRIVE)",
        "section_id": "ID de la sección",
        "lane_id": "ID del carril",
        "s": "Distancia desde el inicio de la carretera",
        "transform": "Posición y rotación (Location + Rotation)",
        "lane_width": "Ancho del carril en metros",
        "is_junction": "¿Está en un cruce?",
        "junction_id": "ID del cruce (si es_junction=True)",
        "lane_type": "Tipo de carril (Driving, Shoulder, etc)",
        "lane_change": "Flags de cambios de carril permitidos",
        "right_lane_marking": "Info de marca de carril derecho",
        "left_lane_marking": "Info de marca de carril izquierdo",
    }
    
    for prop, description in waypoint_properties.items():
        print(f"\n  • {prop}")
        print(f"    → {description}")
    
    print("\n" + "=" * 80)
    print("✅ MÉTODOS DE CADA WAYPOINT")
    print("=" * 80)
    
    waypoint_methods = {
        "next(distance)": "Obtiene waypoints siguientes a distancia",
        "previous(distance)": "Obtiene waypoints anteriores a distancia",
        "get_left_lane()": "Obtiene waypoint del carril izquierdo",
        "get_right_lane()": "Obtiene waypoint del carril derecho",
        "next_until_lane_end(distance)": "Todos los waypoints hasta fin de carril",
        "previous_until_lane_start(distance)": "Todos los waypoints hasta inicio de carril",
        "get_junction()": "Obtiene objeto Junction si está en cruce",
        "get_landmarks(distance, stop_at_junction)": "Obtiene señales/landmarks cercanas",
    }
    
    for method, description in waypoint_methods.items():
        print(f"\n  • {method}")
        print(f"    → {description}")
    
    print("\n✓ Script completado\n")


if __name__ == "__main__":
    main()
