from typing import Optional
import queue

import carla
import pygame
import numpy as np



class CameraManager:
    """Manages camera sensor and image processing."""
    
    def __init__(self, config):
        self.config = config
        self.image_queue = queue.Queue()
        self.current_image: Optional[carla.Image] = None
        self.sensor: Optional[carla.Sensor] = None
    
    def setup_camera(self, world: carla.World, parent_actor: carla.Actor) -> carla.Sensor:

        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # Configure camera attributes (convert all to strings!)
        camera_bp.set_attribute('image_size_x', str(self.config['window_width']))
        camera_bp.set_attribute('image_size_y', str(self.config['window_height']))
        camera_bp.set_attribute('fov', str(self.config['camera_fov']))  # ← Aquí: str()
        
        # Set camera transform
        camera_transform = carla.Transform(
            carla.Location(*self.config['camera_position']),
            carla.Rotation(*self.config['camera_rotation'])
        )
        
        # Spawn and attach camera
        self.sensor = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=parent_actor
        )
        
        # Register callback
        self.sensor.listen(self._on_image_received)
        
        return self.sensor
    
    def _on_image_received(self, image: carla.Image) -> None:
        self.image_queue.put(image)
    
    def update(self) -> None:
        while not self.image_queue.empty():
            try:
                self.current_image = self.image_queue.get_nowait()
            except queue.Empty:
                break
    
    def wait_for_first_image(self, timeout: int = 30) -> bool:
        import time
        start_time = time.time()
        while self.image_queue.empty():
            if time.time() - start_time > timeout:
                return False
            pygame.time.wait(100)
        return True
    
    def get_surface(self) -> Optional[pygame.Surface]:
        """
        Convert current CARLA image to Pygame surface.
        
        Returns:
            Pygame surface or None if no image available
        """
        if self.current_image is None:
            return None
        
        # Convert raw data to numpy array
        array = np.frombuffer(
            self.current_image.raw_data,
            dtype=np.dtype("uint8")
        )
        array = np.reshape(
            array,
            (self.current_image.height, self.current_image.width, 4)
        )
        
        # Remove alpha channel and convert BGR to RGB
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        # Create Pygame surface
        return pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def destroy(self) -> None:
        """Clean up camera sensor."""
        if self.sensor is not None:
            try:
                self.sensor.stop()
                self.sensor.destroy()
            except Exception as e:
                print(f"Warning: Error destroying camera: {e}")
