import numpy as np  # Import NumPy for numerical operations
from numpy.random import random  # Import random function from NumPy



# Class for a fully coupled oscillator system
class FullyCoupled1(object):
    # Constructor to initialize the oscillator system with parameters
    def __init__(self,cell_num, alpha, beta, mu, omega, gamma, hz):
        self.cell_num = cell_num  # Number of oscillator cells
        self.alpha = alpha  # Alpha parameter for Hopf oscillator
        self.beta = beta  # Beta parameter for Hopf oscillator
        self.mu = mu  # Mu parameter (amplitude squared) for Hopf oscillator
        self.omega = omega  # Angular frequency
        self.gamma = gamma  # Coupling strength (unused in this code; possibly for future extensions)
        self.dt = 1/hz  # Time step based on frequency

        self.theta = np.zeros(6)  # Phase offsets (fixed to 6, potential inconsistency with cell_num)
        self.phase = np.zeros(6)  # Phases (fixed to 6)




    # Method to reset initial states of oscillators
    def reset_ini_states(self, r):
        """Reset the environment to an initial state.
        
        Args:
            ini_state (np.ndarray, optional): Initial state of oscillators (shape: [cell_nums, 2]). 
                If None, samples random initial states. Defaults to None.
        
        Returns:
            np.ndarray: Initial observation (shape: [cell_nums * 2 + cell_nums]).
        """

        # use this before reset_goal function

        # Random initial state if not provided

        z_x = np.random.uniform(-r,r, self.cell_num)  # Random x components in [-r, r]
        radius = r*np.ones(self.cell_num)  # Fixed radius for each oscillator
        z_y = np.sqrt(radius*radius-z_x*z_x)  # Compute y for unit circle
        z_y = np.where(random(self.cell_num) > 0.5, -z_y, z_y)  # Randomly flip sign for y
        self.pos = np.array([z_x,z_y])  # Set positions as [x's, y's]

        return self.pos


    # Method to compute coupling terms using a coupling matrix
    def coupling_term(self, components, c_mat, dim):
        coup_components = np.zeros([dim,dim])  # Initialize coupling components matrix
        for i in range(dim):  # Loop through rows
            for j in range(dim):  # Loop through columns
                coup_components[i,j] = c_mat[i,j]*components[i]  
        
        
        return coup_components.sum(axis=0)  # Sum along axis 0 (columns)


    # Hopf oscillator dynamics for a single cell
    def hopf(self, x, y):
        
        r_2 = x ** 2 + y ** 2  # Squared radius
        dx = self.alpha * (self.mu - r_2) * x - self.omega * y  # x-dot
        dy = self.beta * (self.mu - r_2) * y + self.omega * x  # y-dot

        return np.array([dx,dy]).reshape(2,1)    


    # Method to set the coupling matrix
    def set_theta(self, c_mat):
        self.c_mat = c_mat  # Set coupling matrix (named theta, but it's c_mat)


    # Method to compute fully coupled updates for positions
    def fully_coupled(self, pos):
        # Given current position of all legs and step length, Compute positions at next step
        #p os = [x1, x2, x3, x4, y1, y2, y3, y4]  # (Comment for 4 cells, but general)

        x = np.array(pos[0])  # first row  # Extract x components
        y = np.array(pos[1])  # second row  # Extract y components

        dx = np.zeros(self.cell_num)  # Initialize dx
        dy = np.zeros(self.cell_num)  # Initialize dy

        # Compute coupling components for x and y
        coup_components_x = self.coupling_term(x,self.c_mat,dim=self.cell_num)
        coup_components_y = self.coupling_term(y,self.c_mat,dim=self.cell_num)

        # Clip coupling components to [-1, 1] for stability
        coup_components_x = np.clip(coup_components_x,-1,1)
        coup_components_y = np.clip(coup_components_y,-1,1)

        # Loop through each cell
        for i in range(self.cell_num):
            z_i = np.array([x[i],y[i]]).reshape(2,1)  # Current state vector
            F_zi = self.hopf(x[i],y[i])  # Compute Hopf dynamics



            # get the first derivate outputs from internal diffusive system

            dx[i] = F_zi[0,0] + coup_components_x[i]  # Update dx with Hopf + coupling
            dy[i] = F_zi[1,0] + coup_components_y[i]  # Update dy with Hopf + coupling


     
        return x + dx*self.dt, y + dy*self.dt  # Return updated x and y



    # Method to update positions softly using fully coupled dynamics
    def update_soft(self):
        x,y = self.fully_coupled(self.pos)  # Compute updates
        self.pos = np.array([x,y])  # Update positions
