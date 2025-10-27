import numpy as np


class CPGEnv(object):
    """
    A class representing a Central Pattern Generator (CPG) environment.
    This environment simulates oscillatory dynamics for a specified number of cells
    over a given environment length, with a time step determined by the frequency (hz).
    Intended to run on a remote PC.
    """
    def __init__(self, cell_nums, env_length, hz):
        """
        Initialize the CPG environment.

        Args:
            cell_nums (int): Number of cells in the CPG system.
            env_length (float): Length of the environment (context-specific, e.g., spatial or temporal).
            hz (float): Frequency of updates (in Hertz), used to compute the time step.

        Attributes:
            dt (float): Time step size (1/hz).
            cell_nums (int): Number of cells.
            env_length (float): Environment length.
            internal_step (int): Tracks the number of steps taken in the environment.
            z_mat (numpy.ndarray): Matrix storing the state (x, y) of each cell (assumed to be initialized elsewhere).
        """
        self.dt = 1 / hz
        self.cell_nums = cell_nums
        self.env_length = env_length
        self.internal_step = 0
        # Note: z_mat is referenced later but not initialized here. It should be initialized
        # elsewhere (e.g., in a reset method) as a (cell_nums, 2) array for x, y states.

    def hopf1d(self, x, y, ax, ay):
        """
        Compute the derivatives for a single cell's state using a Hopf oscillator model.

        Args:
            x (float): Current x-coordinate of the oscillator state.
            y (float): Current y-coordinate of the oscillator state.
            ax (float): External coupling term for the x-coordinate.
            ay (float): External coupling term for the y-coordinate.

        Returns:
            tuple: (dx, dy) representing the rate of change for x and y coordinates.

        Notes:
            - Uses a Hopf oscillator model with fixed parameters (alpha, beta, omega, mu).
            - Models oscillatory behavior with a limit cycle, modified by external inputs ax, ay.
        """
        alpha = 10  # Damping parameter for x
        beta = 10   # Damping parameter for y
        omega = 2 * np.pi  # Angular frequency (2π radians per unit time)
        mu = 1      # Controls the radius of the limit cycle
        r_2 = x * x + y * y  # Squared radius of the oscillator state

        # Hopf oscillator equations
        dx = alpha * (mu - r_2) * x - omega * y + ax
        dy = beta * (mu - r_2) * y + omega * x + ay

        return dx, dy

    def step_env_simple(self, action, activated_cells):
        """
        Update the environment state for one time step based on actions and activated cells.

        Args:
            action (numpy.ndarray): Array of shape (n, 2) where n is the number of activated cells,
                                   containing (ax, ay) external inputs for each cell.
            activated_cells (list or numpy.ndarray): Indices of cells that are active (not defective).

        Returns:
            numpy.ndarray: Flattened observation array (z_mat.ravel()) containing the updated (x, y) states
                           of all cells.

        Notes:
            - Assumes z_mat is a (cell_nums, 2) array storing (x, y) states for each cell.
            - Updates only the activated cells using the Hopf oscillator dynamics.
            - Increments the internal step counter.
            - Runs on a remote PC, so ensure z_mat is properly initialized before calling.
        """
        # Convert activated_cells to a numpy array of integers
        activated_cells = np.array(activated_cells, dtype=int)

        # Reshape action to (n, 2) where n is the number of activated cells, splitting into x and y components
        action = np.reshape(action, (-1, 2))
        action_x = action[:, 0]
        action_y = action[:, 1]

        # Identify defective (inactive) cells by finding indices not in activated_cells
        all_cells = np.arange(0, 6)  # Assumes 6 cells total (hardcoded)
        defect_cells = np.setdiff1d(all_cells, np.array(activated_cells))

        # Check if there are any defective cells
        if defect_cells.size == 0:
            print('None of the cells is defected')
        else:
            pass  # No action taken for defective cells

        # Update the state of each activated cell using the Hopf oscillator
        for num, index in enumerate(activated_cells):
            x = self.z_mat[index, 0]  # Current x state
            y = self.z_mat[index, 1]  # Current y state
            dx, dy = self.hopf1d(x, y, action_x[num], action_y[num])  # Compute derivatives

            # Update states using Euler integration: new_state = old_state + derivative * time_step
            self.z_mat[index, 0] = x + dx * self.dt
            self.z_mat[index, 1] = y + dy * self.dt

        # Return flattened observation (all x, y states)
        obs = self.z_mat.ravel()

        # Increment step counter
        self.internal_step += 1

        return obs
