import os
# Forzar software rendering para evitar crash DRI3/EGL en displays remotos
os.environ['LIBGL_DRI3_DISABLE'] = '1'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
os.environ['SDL_RENDER_DRIVER'] = 'software'
os.environ.setdefault('SDL_VIDEODRIVER', 'x11')
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = ''

import pygame
import carla
from pyparsing import Optional

from carla_app.camera_manager import CameraManager
from carla_app.planer.navigation.global_route_planner import GlobalRoutePlanner
from carla_app.vehicle_controller import VehicleController



class CarlaApi:
    def __init__(self, config):
        self.host = config['host']
        self.port = config['port']
        self.carla_timeout = config.get('carla_timeout', 10.0)
        self.config = config
        self.client = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera_manager: Optional[CameraManager] = None
        self.world: Optional[carla.World] = None
        self.display: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.vehicle_controller: Optional[VehicleController] = None

        self.global_route_planner: Optional[GlobalRoutePlanner] = None
        self.current_route: Optional[list] = None
        self.route_waypoints: Optional[list] = None

        self.actors = []

      
    def connect_to_server(self):
        # Placeholder for actual connection logic to CARLA server
        print(f"Connecting to CARLA server at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.carla_timeout) 

        print(f"Loading map: {self.config['name_map']}...")
        self.client.load_world(self.config['name_map'])
        print(f"✓ Map loaded: {self.config['name_map']} (debug drawings cleared)")
        
        self.world = self.client.get_world()
        print(f"✓ Connected to: {self.world.get_map().name}")

        # Inicializar route planner después de cargar el mapa
        self._initialize_route_planner()

        # Set weather if configured
        if self.config.get('set_weather') is not None:
            self._set_wheather()

    def spawn_vehicle(self):
        # Placeholder for actual vehicle spawning logic
        print("Spawning vehicle...")
        blueprint_library = self.world.get_blueprint_library()

        # Get vehicle blueprint
        vehicle_bp = blueprint_library.filter(self.config['vehicle_model'])[0]


        z = self.config['spawn_point']['z'] if self.config['spawn_point']['z'] is not None else 0.5
        yaw = self.config['spawn_point']['yaw'] if self.config['spawn_point']['yaw'] is not None else 0.0

        spawn_transform = carla.Transform(
            carla.Location(x=self.config['spawn_point']['x'], y=self.config['spawn_point']['y'], z=z),
            carla.Rotation(yaw=yaw)
        )

        # Spawn vehicle
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
        print(f"✓ Vehicle spawned: {self.config['vehicle_model']}")        
        print(f"Using custom start position: X={self.config['spawn_point']['x']}, Y={self.config['spawn_point']['y']}")        
        
        self.actors.append(self.vehicle)

        
        return spawn_transform

    def respawn_vehicle(self, spawn_transform):
        if self.vehicle is not None:
            self.vehicle.set_transform(spawn_transform)
            self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            self.vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            self.world.tick()
            # if self.verbose:
            #     print("✓ Vehicle respawned at initial position")
    
    def _initialize_route_planner(self):
        """Inicializa el GlobalRoutePlanner con el mapa actual."""
        map = self.world.get_map()
        sampling_resolution = self.config.get('route_sampling_resolution', 2.0)
        self.global_route_planner = GlobalRoutePlanner(map, sampling_resolution)
        print(f"✓ Route planner initialized (sampling: {sampling_resolution}m)")

    def plan_route(self, start_location=None, end_location=None):
        """
        Genera una ruta desde start_location hasta end_location.
        Si no se especifican, usa spawn_point y goal_point del config.
        
        Returns:
            list: Lista de tuplas (waypoint, RoadOption)
        """
        if self.global_route_planner is None:
            raise RuntimeError("Route planner not initialized. Call connect_to_server first.")
        
        # Usar posiciones del config si no se especifican
        if start_location is None:
            # Si el vehículo ya existe, usar su posición actual
            if self.vehicle is not None:
                start_location = self.vehicle.get_location()
            else:
                # Si no, crear desde config
                start_location = carla.Location(
                    x=float(self.config['spawn_point']['x']),
                    y=float(self.config['spawn_point']['y']),
                    z=float(self.config['spawn_point'].get('z', 0.5))
                )
        # Si recibe un Transform, extraer la Location
        elif isinstance(start_location, carla.Transform):
            start_location = start_location.location
        
        if end_location is None:
            end_location = carla.Location(
                x=float(self.config['goal_point']['x']),
                y=float(self.config['goal_point']['y']),
                z=0.5
            )
        elif isinstance(end_location, carla.Transform):
            end_location = end_location.location
        
        # Generar ruta
        print(f"Planning route from ({start_location.x:.1f}, {start_location.y:.1f}) "
            f"to ({end_location.x:.1f}, {end_location.y:.1f})...")
        
        self.current_route = self.global_route_planner.trace_route(
            start_location, 
            end_location
        )
        
        # Extraer solo waypoints
        self.route_waypoints = [wp for wp, _ in self.current_route]
        
        print(f"✓ Route planned: {len(self.current_route)} waypoints")
        return self.current_route 
    
    def get_route(self):
            """Devuelve la ruta actual."""
            return self.current_route
    
    def get_route_waypoints(self):
        """Devuelve solo los waypoints de la ruta (sin RoadOption)."""
        return self.route_waypoints

    def visualize_route(self, life_time=120.0, color=None):
        """Visualiza la ruta con líneas entre waypoints."""
        if self.current_route is None:
            print("⚠ No route to visualize. Call plan_route first.")
            return
        
        if color is None:
            color = carla.Color(r=255, g=0, b=0)
        
        print(f"Visualizing route with {len(self.current_route)} waypoints...")
        
        # Dibujar líneas entre waypoints consecutivos
        for i in range(len(self.current_route) - 1):
            wp1 = self.current_route[i][0]
            wp2 = self.current_route[i + 1][0]
            
            self.world.debug.draw_line(
                wp1.transform.location + carla.Location(z=0.5),
                wp2.transform.location + carla.Location(z=0.5),
                thickness=0.1,
                color=color,
                life_time=life_time
            )
        
        # Marcar inicio y fin
        start_wp, _ = self.current_route[0]
        end_wp, _ = self.current_route[-1]
        
        self.world.debug.draw_point(
            start_wp.transform.location + carla.Location(z=1.0),
            size=0.3,
            color=carla.Color(g=255),
            life_time=life_time
        )
        
        self.world.debug.draw_point(
            end_wp.transform.location + carla.Location(z=1.0),
            size=0.3,
            color=carla.Color(b=255),
            life_time=life_time
        )
        
        print("✓ Route visualization completed")
        self.world.tick()
    def get_distance_to_goal(self):
        """Calcula la distancia del vehículo al punto objetivo."""
        if self.vehicle is None or self.route_waypoints is None:
            return None
        
        vehicle_location = self.vehicle.get_location()
        goal_location = self.route_waypoints[-1].transform.location
        
        distance = vehicle_location.distance(goal_location)
        return distance
    
    def setup_camera(self) -> None:
        """Setup camera system."""
        self.camera_manager = CameraManager(self.config)
        camera_sensor = self.camera_manager.setup_camera(self.world, self.vehicle)
        self.actors.append(camera_sensor)
        print("✓ Camera attached")
        
       # print("Waiting for first image...")
       # if self.camera_manager.wait_for_first_image():
       #     print("✓ Camera ready")
       # else:
       #     print("⚠ Warning: Camera timeout, continuing anyway...")
    
    def set_sync_mode(self, render_mode: str = 'human') -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True

        if render_mode == 'human':
            settings.fixed_delta_seconds = 0.05  # 20 FPS para visualización
        else:
            settings.fixed_delta_seconds = 0.1  # 10 FPS
        
        self.world.apply_settings(settings)
        
        print(f"✓ Synchronous mode set at {self.config['target_fps']} FPS")

    def initialize_components(self, render_mode: str = 'human') -> None:
        
        self.vehicle_controller = VehicleController(self.config)
        self.vehicle_controller.set_vehicle(self.vehicle)
        self.vehicle_controller.set_world(self.world)
        
        """Initialize HUD."""
        if render_mode == 'human':
            self.initialize_pygame()
            #self.hud_renderer = HudRenderer(
            #    width=self.config['hud_width'],
            #    height=self.config['hud_height'],
            #    font_size=self.config['hud_font_size'],
            #    bg_color=self.config['hud_bg_color'],
            #    bg_alpha=self.config['hud_bg_alpha']
            #)
            print("✓ Additional components initialized")


    def render(self):
        """Render frame."""

        if self.display is None:
            return True

        for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return False            
        # Render camera feed
        self.camera_manager.update()
        camera_surface = self.camera_manager.get_surface()
        if camera_surface is not None:
            # Scale and blit to display
            scaled = pygame.transform.scale(
                camera_surface,
                (self.config['window_width'], self.config['window_height'])
            )
            self.display.blit(scaled, (0, 0))
        else:
            self.display.fill((50, 50, 50))
        
        # Render HUD
        #telemetry = self.vehicle_controller.get_telemetry()
        
        # Add photo count to telemetry
        #if self.photo_capture is not None:
            #telemetry['photo_count'] = self.photo_capture.capture_count
        
        #self.hud_renderer.render(self.display, telemetry)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.config['target_fps'])
        return True
    def get_client(self):
        return self.client

    def initialize_pygame(self) -> None:
        """Initialize Pygame display."""
        import os
        # Sólo inicializar subsistema de display, no joystick/audio
        # que pueden triggear EGL/DRI3 innecesariamente
        import pygame.display
        import pygame.font
        import pygame.event
        import pygame.transform
        import pygame.time
        pygame.display.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.config['window_width'], self.config['window_height']),
            0  # no hardware flags
        )
        pygame.display.set_caption(self.config['window_title'])
        self.clock = pygame.time.Clock()
        print("✓ Pygame initialized")
    
    def _set_wheather(self) -> None:
        weather_presets = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
            'ClearSunset': carla.WeatherParameters.ClearSunset,
        }
            
        weather = weather_presets.get(
            self.config['weather_preset'], carla.WeatherParameters.ClearNoon
        )
        self.world.set_weather(weather)
        print(f"✓ Weather set: {self.config['weather_preset']}")

    def clean_up(self) -> None:
        """Clean up resources in correct order."""
        print("\n🧹 Cleaning up resources...")
        
        # 1. Destroy camera FIRST (uses world reference)
        if self.camera_manager is not None:
            try:
                self.camera_manager.destroy()
                print("  ✓ Camera destroyed")
            except Exception as e:
                print(f"  Warning: Error destroying camera: {e}")
        
        # 2. Destroy all actors (sensors + vehicle) BEFORE world
        for actor in self.actors:
            try:
                if actor is not None:
                    if hasattr(actor, 'is_alive') and actor.is_alive:
                        actor.destroy()
            except Exception as e:
                pass  # Silently ignore already destroyed actors
        self.actors.clear()
        print("  ✓ All actors destroyed")
        
        # 3. Clean up Pygame LAST
        try:
            pygame.quit()
        except Exception:
            pass
        
        print("✓ Cleanup completed")