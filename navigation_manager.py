import time

class NavigationManager:
    def __init__(self, callback, stability_frames=10, cooldown=1.5):
        """
        callback: function(intent_str)
        """
        self.callback = callback
        self.stability_frames = stability_frames
        self.cooldown = cooldown
        
        self.history = []
        self.last_trigger_time = 0
        self.last_triggered_count = None
        
    def process_finger_count(self, count):
        """
        Takes raw integer 0-5. Checks for stability and debounces.
        Dispatches intents via callback.
        """
        if count is None:
            self.history.clear()
            self.last_triggered_count = None
            return
            
        now = time.time()
        
        # Block newly detected gestures during cooldown
        if now - self.last_trigger_time < self.cooldown:
            # While in cooldown, we drop everything, so they have to 'reset' 
            # or just naturally wait for the cooldown to clear
            self.history.clear()
            return

        self.history.append(count)
        if len(self.history) > self.stability_frames:
            self.history.pop(0)
            
        if len(self.history) == self.stability_frames:
            # Check if all frames in history match
            if all(x == count for x in self.history):
                # We have a stable valid count
                if count != self.last_triggered_count:
                    self.trigger_intent(count)
                    self.last_triggered_count = count
                # Apply cooldown regardless to avoid fast oscillations
                self.last_trigger_time = time.time()
                self.history.clear()
                
    def trigger_intent(self, count):
        from gesture_config import FINGER_MAPPING
        intent = FINGER_MAPPING.get(count)
        if intent:
            self.callback(intent)
