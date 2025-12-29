import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class HandData:
    """Data structure for hand tracking information"""
    position: np.ndarray  # (x, y) position of hand center
    velocity: np.ndarray  # (vx, vy) velocity vector
    landmarks: Optional[np.ndarray] = None  # All hand landmarks
    confidence: float = 0.0
    is_detected: bool = False

class HandTracker:
    """Real-time hand tracking using MediaPipe"""
    
    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.7):
        """
        Initialize hand tracker
        
        Args:
            max_hands: Maximum number of hands to detect
            detection_confidence: Confidence threshold for detection
        """
        self.mp_hands, self.mp_drawing = self._resolve_mediapipe()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5
        )
        
        self.previous_positions = {}
        self.frame_width = 0
        self.frame_height = 0

    @staticmethod
    def _resolve_mediapipe():
        if hasattr(mp, "solutions"):
            return mp.solutions.hands, mp.solutions.drawing_utils
        try:
            from mediapipe.python.solutions import hands as mp_hands
            from mediapipe.python.solutions import drawing_utils as mp_drawing
        except Exception as exc:
            raise ImportError(
                "MediaPipe solutions API not available. Install the official "
                "`mediapipe` package (pip install mediapipe) in your active "
                "environment."
            ) from exc
        return mp_hands, mp_drawing
        
    def process_frame(self, frame: np.ndarray) -> List[HandData]:
        """
        Process a frame and extract hand tracking data
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            List of HandData objects for detected hands
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_data_list = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_id, (landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Extract palm center (average of wrist and middle finger landmarks)
                wrist = landmarks.landmark[0]
                middle_finger_mcp = landmarks.landmark[9]
                
                # Calculate hand center
                hand_center_x = (wrist.x + middle_finger_mcp.x) / 2
                hand_center_y = (wrist.y + middle_finger_mcp.y) / 2
                
                # Convert normalized coordinates to pixel coordinates
                x = int(hand_center_x * self.frame_width)
                y = int(hand_center_y * self.frame_height)
                
                # Calculate velocity
                velocity = np.array([0.0, 0.0])
                if hand_id in self.previous_positions:
                    prev_x, prev_y = self.previous_positions[hand_id]
                    velocity = np.array([x - prev_x, y - prev_y], dtype=np.float32)
                
                self.previous_positions[hand_id] = (x, y)
                
                # Extract all landmarks
                landmarks_array = np.array([
                    [lm.x * self.frame_width, lm.y * self.frame_height, lm.z]
                    for lm in landmarks.landmark
                ])
                
                hand_data = HandData(
                    position=np.array([x, y], dtype=np.float32),
                    velocity=velocity,
                    landmarks=landmarks_array,
                    confidence=handedness.classification[0].score,
                    is_detected=True
                )
                hand_data_list.append(hand_data)
        
        return hand_data_list
    
    def draw_hands(self, frame: np.ndarray, hand_data_list: List[HandData]) -> np.ndarray:
        """
        Draw hand tracking visualization on frame
        
        Args:
            frame: Input frame
            hand_data_list: List of detected hands
            
        Returns:
            Frame with hand visualization
        """
        output_frame = frame.copy()
        
        for hand_data in hand_data_list:
            if hand_data.is_detected:
                x, y = hand_data.position.astype(int)
                
                # Draw hand center
                cv2.circle(output_frame, (x, y), 8, (0, 255, 0), -1)
                
                # Draw velocity vector
                if np.linalg.norm(hand_data.velocity) > 0:
                    vx, vy = hand_data.velocity.astype(int)
                    cv2.arrowedLine(
                        output_frame,
                        (x, y),
                        (x + vx * 3, y + vy * 3),
                        (255, 0, 0),
                        2,
                        tipLength=0.3
                    )
        
        return output_frame
    
    def get_hand_radius(self, hand_data: HandData) -> float:
        """
        Estimate hand radius from landmarks
        
        Args:
            hand_data: Hand tracking data
            
        Returns:
            Estimated hand radius in pixels
        """
        if hand_data.landmarks is None:
            return 30.0
        
        # Calculate distance from wrist to fingertips
        wrist = hand_data.landmarks[0]
        fingertips = [hand_data.landmarks[i] for i in [4, 8, 12, 16, 20]]
        
        distances = [np.linalg.norm(fingertip[:2] - wrist[:2]) for fingertip in fingertips]
        return np.mean(distances) if distances else 30.0
