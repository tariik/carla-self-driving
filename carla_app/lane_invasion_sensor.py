# lane_invasion_sensor.py
"""
Sensor de invasión de carril para CARLA Simulator
Detecta cuando el vehículo cruza líneas de carril
Basado en: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector
"""

import carla
import weakref


class LaneInvasionSensor:
    """Detecta cuando el vehículo cruza líneas de carril."""
    
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.invasion_history = []
        
        # Crear sensor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        
        # Configurar callback
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
    
    @staticmethod
    def _on_invasion(weak_self, event):
        """Callback cuando se cruza una línea de carril."""
        self = weak_self()
        if not self:
            return
        
        invasion_data = {
            'frame': event.frame,
            'timestamp': event.timestamp,
            'crossed_lane_markings': [
                {
                    'type': str(marking.type),
                    'color': str(marking.color),
                    'lane_change': str(marking.lane_change),
                    'width': marking.width
                }
                for marking in event.crossed_lane_markings
            ]
        }
        
        self.invasion_history.append(invasion_data)
        
        marking_types = [str(m.type) for m in event.crossed_lane_markings]
        # print(f"🚧 LANE INVASION! Frame {event.frame}")
        # print(f"   Crossed markings: {', '.join(marking_types)}")
    
    def get_invasion_history(self):
        return self.invasion_history
    
    def get_invasion_count(self):
        return len(self.invasion_history)
    
    def reset(self):
        self.invasion_history = []
    
    def destroy(self):
        """Properly destroy the sensor."""
        if self.sensor is not None:
            try:
                self.sensor.stop()  # ← CRÍTICO: detener listener antes
                self.sensor.destroy()
            except Exception as e:
                pass
            finally:
                self.sensor = None
