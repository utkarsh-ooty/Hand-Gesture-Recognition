from collections import deque
import time

class SwipeDetector:
    def __init__(self, buffer_size=15, skip_frames=2, swipe_threshold_px=100, cooldown_seconds=1.5):
        # We store the wrist X,Y pixel coordinates
        self.buffer = deque(maxlen=buffer_size)
        
        # Configuration
        self.swipe_threshold_px = swipe_threshold_px
        self.cooldown_seconds = cooldown_seconds
        self.skip_frames = skip_frames
        
        self.frame_count = 0
        self.last_swipe_time = 0
        
    def add_position(self, wrist_x, wrist_y):
        """
        Adds current wrist position and checks for swipe.
        Returns a gesture string if detected, otherwise None.
        """
        self.frame_count += 1
        
        # Basic subsampling to prevent noise
        if self.frame_count % self.skip_frames != 0:
            return None
            
        self.buffer.append((wrist_x, wrist_y))
        
        # Check cooldown
        if time.time() - self.last_swipe_time < self.cooldown_seconds:
            return None
            
        return self._detect_swipe()
        
    def _detect_swipe(self):
        if len(self.buffer) < self.buffer.maxlen:
            return None
            
        # Get start and end points in our history
        start_pt = self.buffer[0]
        end_pt = self.buffer[-1]
        
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        
        # If the movement is primarily horizontal
        if abs(dx) > abs(dy) * 1.5:  
            if dx > self.swipe_threshold_px:
                # Movement from left to right -> Swipe Right
                # Note: This heavily depends on camera orientation/mirroring.
                # If frame is mirrored (selfie view), moving hand physical right 
                # goes to the right on screen.
                self.last_swipe_time = time.time()
                self.buffer.clear()
                return "Swipe Right"
            
            elif dx < -self.swipe_threshold_px:
                # Movement from right to left -> Swipe Left
                self.last_swipe_time = time.time()
                self.buffer.clear()
                return "Swipe Left"
                
        return None
        
    def reset(self):
        self.buffer.clear()
        self.frame_count = 0
