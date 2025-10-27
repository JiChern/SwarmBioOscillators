#!/usr/bin/env python3
# This is the robot-side code, may not be executed on other robots.

import numpy as np

# Interface class for a hexapod robot, defining its rest pose
class RobotInterface(object):
    def __init__(self):
        # Define the rest pose for the six legs of the hexapod
        # Array shape: [6, 3], representing (x, y, z) coordinates in meters for each leg
        # Legs are ordered as: [right front, right middle, right back, left front, left middle, left back]
        self.rest_pose = np.array([
            [1.8e-01, 6.6e-02, -8.6e-02],  # Right front: x=0.18, y=0.066, z=-0.086
            [1.9e-01, 0.0, -8.6e-02],      # Right middle: x=0.19, y=0.0, z=-0.086
            [1.8e-01, -6.6e-02, -8.6e-02], # Right back: x=0.18, y=-0.066, z=-0.086
            [1.8e-01, -6.6e-02, -8.6e-02], # Left front: x=0.18, y=-0.066, z=-0.086
            [1.9e-01, 0.0, -8.6e-02],      # Left middle: x=0.19, y=0.0, z=-0.086
            [1.8e-01, 6.6e-02, -8.6e-02]   # Left back: x=0.18, y=0.066, z=-0.086
        ])