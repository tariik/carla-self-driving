# collision_sensor.py
"""
Sensor de colisión para CARLA Simulator
Basado en: https://carla.readthedocs.io/en/latest/ref_sensors/#collision-detector
"""

import carla
import math
import weakref


class CollisionSensor:
    """Detecta y registra colisiones del vehículo en CARLA."""
    
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.collision_history = []
        self.collision_occurred = False
        
        # Crear sensor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            blueprint, 
            carla.Transform(), 
            attach_to=self._parent
        )
        
        # Configurar callback con weakref para evitar referencias circulares
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
    
    @staticmethod
    def _on_collision(weak_self, event):
        """Callback cuando ocurre una colisión"""
        self = weak_self()
        if not self:
            return
        
        self.collision_occurred = True
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        
        collision_data = {
            'frame': event.frame,
            'timestamp': event.timestamp,
            'other_actor': event.other_actor.type_id if event.other_actor else 'Unknown',
            'impulse': {'x': impulse.x, 'y': impulse.y, 'z': impulse.z},
            'intensity': intensity
        }
        
        self.collision_history.append(collision_data)
        
       # print(f"⚠️  COLLISION! Frame {event.frame} | "
       #       f"With: {collision_data['other_actor']} | "
       #       f"Intensity: {intensity:.2f}")
    
    def get_collision_history(self):
        return self.collision_history
    
    def has_collision(self):
        return self.collision_occurred
    
    def get_collision_count(self):
        return len(self.collision_history)
    
    def reset(self):
        self.collision_history = []
        self.collision_occurred = False
    
    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
