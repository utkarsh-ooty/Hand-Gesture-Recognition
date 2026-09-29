import math

class FingerDetector:
    def __init__(self):
        # Tip landmarks according to MediaPipe
        self.TIP_IDS = [4, 8, 12, 16, 20]
        self.PIP_IDS = [2, 6, 10, 14, 18] # Actually for thumb it's 2, for fingers it's 6,10,14,18 (PIP joints)
        self.MCP_IDS = [1, 5, 9, 13, 17]
        
    def _get_distance(self, p1, p2):
        return math.hypot(p1[1] - p2[1], p1[2] - p2[2])
        
    def count_fingers(self, landmarks, handedness="Right"):
        """
        Takes normalized landmarks list [[id, x, y, z], ...]
        Returns dictionary of individual fingers and total count.
        Returns None if landmarks are invalid.
        """
        if not landmarks or len(landmarks) < 21:
            return None
            
        fingers_extended = [0, 0, 0, 0, 0]
        wrist = landmarks[0]
        
        # Calculate palm height (wrist to middle finger MCP) as reference unit
        mid_mcp = landmarks[self.MCP_IDS[2]]
        palm_size = self._get_distance(wrist, mid_mcp)
        
        # Index, Middle, Ring, Pinky
        for i in range(1, 5):
            tip = landmarks[self.TIP_IDS[i]]
            pip = landmarks[self.PIP_IDS[i]]
            
            # Distance from wrist
            dist_tip_wrist = self._get_distance(tip, wrist)
            dist_pip_wrist = self._get_distance(pip, wrist)
            
            # Using a very small margin allows for palm-side perspective shortening
            if dist_tip_wrist > dist_pip_wrist + (palm_size * 0.02):
                fingers_extended[i] = 1
                
        # Thumb
        # When extended, thumb_tip is further from pinky_mcp than thumb_ip is.
        # When folded across the palm, thumb_tip is closer to pinky_mcp than thumb_ip.
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        pinky_mcp = landmarks[17]
        
        if self._get_distance(thumb_tip, pinky_mcp) > self._get_distance(thumb_ip, pinky_mcp):
            fingers_extended[0] = 1
            
        return sum(fingers_extended), {
            'thumb': fingers_extended[0],
            'index': fingers_extended[1],
            'middle': fingers_extended[2],
            'ring': fingers_extended[3],
            'pinky': fingers_extended[4]
        }
        
    def get_navigation_token(self, landmarks, handedness="Right"):
        """Returns semantic token: 1_FINGER, 2_FINGERS, THUMBS_UP, THUMBS_DOWN"""
        result = self.count_fingers(landmarks, handedness)
        if not result:
            return None
            
        count, details = result
        
        if count == 1 and details['index'] == 1:
            return "1_FINGER"
            
        if count == 2 and details['index'] == 1 and details['middle'] == 1:
            return "2_FINGERS"
            
        if count == 1 and details['thumb'] == 1:
            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            if thumb_tip[2] < wrist[2]:
                return "THUMBS_UP"
            else:
                return "THUMBS_DOWN"
                
        return None
        
    def check_gesture_match(self, gesture_name, landmarks):
        """Returns True if the landmarks somewhat resemble the requested gesture class."""
        if not landmarks or len(landmarks) < 21:
            return False
            
        result = self.count_fingers(landmarks)
        if result is None:
            return False
        count, _ = result
        
        if gesture_name == "open_palm":
            return count == 5
        elif gesture_name == "closed_fist":
            return count == 0
        elif gesture_name == "thumbs_up":
            # 1 finger (thumb) and pointing up
            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            return count == 1 and thumb_tip[2] < wrist[2]
        elif gesture_name == "thumbs_down":
            # 1 finger (thumb) and pointing down
            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            return count == 1 and thumb_tip[2] > wrist[2]
            
        return False
