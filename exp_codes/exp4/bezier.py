import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Define a class to handle 2D Bezier curve calculations
class Bezier2(object):
    # Constructor to initialize the Bezier2 object
    def __init__(self):
        # Initialize empty lists to store x and y coordinates of control points
        self.x_pos = []
        self.y_pos = []

    # Method to set the control points for the Bezier curve
    def setPoint(self, x_vec, y_vec):
        # Store the input x-coordinates of control points
        self.x_pos = x_vec
        # Store the input y-coordinates of control points
        self.y_pos = y_vec

    # Method to calculate a point on the Bezier curve for a given parameter t
    def getPos(self, t):
        # Get the number of control points
        n_points = len(self.x_pos)
        # Create copies of control point coordinates to avoid modifying originals
        # This ensures the original control points remain unchanged during computation
        x_cal, y_cal = self.x_pos.copy(), self.y_pos.copy()
        
        # Implement the De Casteljau's algorithm to compute a point on the Bezier curve
        # Iterate through the levels of the algorithm (reduces points by one each iteration)
        for i in range(n_points-1):
            # For each level, compute new points by linear interpolation
            for j in range(n_points-i-1):
                # Linearly interpolate between consecutive points for x-coordinates
                # Formula: (1-t)*P_j + t*P_(j+1), where t is the interpolation parameter
                x_cal[j] = (1.0-t)*x_cal[j] + t*x_cal[j+1]
                # Linearly interpolate between consecutive points for y-coordinates
                y_cal[j] = (1.0-t)*y_cal[j] + t*y_cal[j+1]
        
        # After the algorithm completes, the first point contains the final x, y coordinates
        x_ret = x_cal[0]
        y_ret = y_cal[0]
   
        # Return the computed point on the Bezier curve as a tuple (x, y)
        return x_ret, y_ret