from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class StaticParams:
    BEST_POINT_CONV_SIZE = 80  # Reduced: Don't smooth the gap too much on small tracks
    EDGE_GUARD_DEG = 12.0      
    TTC_HARD_BRAKE = 0.55
    TTC_SOFT_BRAKE = 0.9       
    FWD_WEDGE_DEG = 8.0        
    STEER_RATE_LIMIT = 0.3     # Increased: Real servos need permission to move faster
    CENTER_BIAS_ALPHA = 0.5

@dataclass
class DynamicParams:
    BUBBLE_RADIUS: int 
    MAX_LIDAR_DIST: float
    STRAIGHT_SPEED: float
    CORNER_SPEED: float
    SPEED_MAX: float
    STEER_SMOOTH_ALPHA: float 
    PREPROCESS_CONV_SIZE: int

    def to_array(self) -> np.ndarray:
        "Converts objects to an array"
        return np.array([
            self.BUBBLE_RADIUS,
            self.MAX_LIDAR_DIST,
            self.STRAIGHT_SPEED,
            self.CORNER_SPEED,
            self.SPEED_MAX,
            self.STEER_SMOOTH_ALPHA,
            self.PREPROCESS_CONV_SIZE
        ])
    
    @staticmethod
    def from_array(arr:np.ndarray) -> "DynamicParams":
        "Converts from array to data class"
        # use "forward reference" bc in class using
        return DynamicParams(
            BUBBLE_RADIUS=int(arr[0]),
            MAX_LIDAR_DIST=float(arr[1]),
            STRAIGHT_SPEED=float(arr[2]),
            CORNER_SPEED=float(arr[3]),
            SPEED_MAX=float(arr[4]),
            STEER_SMOOTH_ALPHA=float(arr[5]),
            PREPROCESS_CONV_SIZE=int(arr[6])
        )
    
