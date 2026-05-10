# lane_invasion_sensor.py
"""
Sensor de invasión de carril para CARLA Simulator
Detecta cuando el vehículo cruza líneas de carril
Basado en: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector
"""

import carla
import weakref
import numpy as np


class LaneInvasionSensor:
    """Detecta cuando el vehículo cruza líneas de carril."""
    
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.invasion_history = []
        self.legal_invasions_count = 0
        self.illegal_invasions_count = 0
        self.unknown_invasions_count = 0
        self.last_vehicle_heading = None
        
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
        
        # Obtener dirección de cruce (basada en heading del vehículo)
        vehicle = self._parent
        current_heading = vehicle.get_transform().rotation.yaw
        crossing_direction = self._determine_crossing_direction(self, current_heading)
        self.last_vehicle_heading = current_heading
        
        # Clasificar cada marca cruzada
        for marking in event.crossed_lane_markings:
            lane_change_str = str(marking.lane_change)
            classification = self._classify_invasion(self, crossing_direction, lane_change_str)
            
            # Actualizar contadores
            if classification == 'legal':
                self.legal_invasions_count += 1
            elif classification == 'illegal':
                self.illegal_invasions_count += 1
            else:
                self.unknown_invasions_count += 1
            
            invasion_data = {
                'frame': event.frame,
                'timestamp': event.timestamp,
                'type': str(marking.type),
                'color': str(marking.color),
                'lane_change': lane_change_str,
                'width': marking.width,
                'crossing_direction': crossing_direction,
                'classification': classification,
            }
            self.invasion_history.append(invasion_data)
    
    @staticmethod
    def _determine_crossing_direction(self, current_heading):
        """
        Determina la dirección del cruce (Left/Right).
        
        En CARLA, el yaw aumenta en sentido counterclockwise:
        - 0° = +X (Este)
        - 90° = +Y (Norte)
        - 180° = -X (Oeste)
        - 270° = -Y (Sur)
        
        Para detectar un cambio a la izquierda/derecha, observamos si el yaw cambia.
        Si yaw crece o si en general estamos girando a la izquierda, es LEFT.
        Si yaw decrece, es RIGHT.
        """
        if self.last_vehicle_heading is None:
            return 'Unknown'
        
        heading_delta = current_heading - self.last_vehicle_heading
        
        # Normalizar a [-180, 180]
        while heading_delta > 180:
            heading_delta -= 360
        while heading_delta < -180:
            heading_delta += 360
        
        # Si el delta es positivo (izquierda) o muy cercano a cero, asumir LEFT
        # Si es negativo (derecha), asumir RIGHT
        if heading_delta >= 0:
            return 'Left'
        else:
            return 'Right'
    
    @staticmethod
    def _classify_invasion(self, crossing_direction, lane_change_str):
        """
        Clasifica si la invasión es legal o ilegal.
        
        Args:
            crossing_direction: 'Left' o 'Right' (dirección del cruce)
            lane_change_str: String de carla.LaneChange (ej: 'Both', 'Left', 'Right', 'None')
        
        Returns:
            'legal', 'illegal', o 'unknown'
        """
        # Parsear lane_change
        if 'None' in lane_change_str or lane_change_str == 'None':
            # No permitido cambiar de carril
            return 'illegal'
        elif 'Both' in lane_change_str:
            # Se permite cambiar en ambas direcciones
            return 'legal'
        elif 'Left' in lane_change_str and crossing_direction == 'Left':
            # Cambio a izquierda permitido y cruzamos a la izquierda
            return 'legal'
        elif 'Right' in lane_change_str and crossing_direction == 'Right':
            # Cambio a derecha permitido y cruzamos a la derecha
            return 'legal'
        else:
            # Cambio a una dirección no permitida
            return 'illegal'
    
    def get_invasion_history(self):
        return self.invasion_history
    
    def get_invasion_count(self):
        """Retorna el total de invasiones (legal + illegal + unknown)."""
        return len(self.invasion_history)
    
    def get_legal_invasions_count(self):
        """Retorna el número de invasiones legales."""
        return self.legal_invasions_count
    
    def get_illegal_invasions_count(self):
        """Retorna el número de invasiones ilegales."""
        return self.illegal_invasions_count
    
    def get_unknown_invasions_count(self):
        """Retorna el número de invasiones de clasificación desconocida."""
        return self.unknown_invasions_count
    
    def get_last_invasion_type(self):
        """Retorna el tipo de la última invasión (legal/illegal/unknown)."""
        if self.invasion_history:
            return self.invasion_history[-1].get('classification', 'unknown')
        return None
    
    def reset(self):
        self.invasion_history = []
        self.legal_invasions_count = 0
        self.illegal_invasions_count = 0
        self.unknown_invasions_count = 0
        self.last_vehicle_heading = None
    
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
