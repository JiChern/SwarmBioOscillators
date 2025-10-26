import numpy as np

# Class representing a Salamander Central Pattern Generator (CPG) using coupled oscillators
class Salamander:
    # Constructor to initialize the Salamander CPG with given parameters
    def __init__(self, omega, cell_num, hz, a=20.0, R=1.0, coupling_strength=1, phase_bias=np.pi/6, desired_phase_diffs=None):
        """
        Initialize the Salamander class.
        
        Parameters:
        - omega: Oscillation frequency (rad/s), used for the intrinsic angular frequency of all oscillators.
        - cell_num: Number of oscillators (N).
        - hz: Sampling frequency (Hz), used to calculate the step size dt = 1/hz.
        - a: Convergence speed constant for amplitude dynamics (default 20.0).
        - R: Target amplitude (default 1.0, can be modulated by drive signal).
        - coupling_strength: Coupling strength (default 0.5).
        - phase_bias: Phase bias (default π/6, used for traveling wave when desired_phase_diffs is None).
        - desired_phase_diffs: Desired phase difference array (np.array, shape (N,)), if provided, uses full connection topology to generate custom phase differences; otherwise uses ring-shaped traveling wave.
          The first element should be 0, representing the phase difference relative to the first unit.
        """
        self.omega = omega  # Intrinsic angular frequency (rad/s)
        self.N = cell_num  # Number of oscillator cells
        self.dt = 1.0 / hz  # Time step based on sampling frequency
        self.a = np.ones(self.N) * a  # a_i for each oscillator (convergence speed)
        self.R = np.ones(self.N) * R  # R_i for each oscillator (target amplitude)
        
        # State variables
        self.theta = np.zeros(self.N)  # Phases of the oscillators
        self.r = np.zeros(self.N)     # Amplitudes of the oscillators
        self.dr = np.zeros(self.N)    # Derivatives of the amplitudes \dot{r}
        self.x = np.zeros(self.N)     # Output signals
        
        self.desired_phase_diffs = desired_phase_diffs  # Desired phase differences (if provided)
        
        # Set coupling matrices w_ij and desired phase phi_ij
        self.w = np.zeros((self.N, self.N))  # Coupling weights matrix
        self.phi = np.zeros((self.N, self.N))  # Phase bias matrix
        

        # Configure coupling based on whether custom phase differences are provided
        if self.desired_phase_diffs is not None:
            # Check desired phase differences
            if len(self.desired_phase_diffs) != self.N or self.desired_phase_diffs[0] != 0:
                raise ValueError("desired_phase_diffs must be of length N, with the first element as 0.")
            
            # Default ring topology with bidirectional coupling
            for i in range(self.N):
                j = (i + 1) % self.N  # Next neighbor (ring wrap-around)
                self.w[i, j] = coupling_strength  # Set coupling weight
                self.phi[i, j] = self.desired_phase_diffs[j] - self.desired_phase_diffs[i]  # Set phase bias
                # Bidirectional coupling
                self.w[j, i] = coupling_strength  # Set reverse coupling weight
                self.phi[j, i] = self.desired_phase_diffs[i] - self.desired_phase_diffs[j]  # Set reverse phase bias

    # Method to initialize the system state
    def init(self):
        """
        Initialize system state: randomly select initial phases on the unit circle, set amplitude r=1, \dot{r}=0.
        """
        self.theta = np.random.uniform(0, 2 * np.pi, self.N)  # Random initial phases
        self.r = np.ones(self.N)  # Initial amplitudes set to 1
        self.dr = np.zeros(self.N)  # Initial amplitude derivatives set to 0
        self._update_output()  # Initialize output signals

    # Method to advance the system by one time step
    def step(self):
        """
        Update the system state by one step using the Euler method to discretize the differential equations.
        """
        # Compute \dot{\theta}_i = omega + sum_j r_j w_ij sin(theta_j - theta_i - phi_ij)
        dtheta = np.ones(self.N) * self.omega  # Base angular velocity
        for i in range(self.N):
            for j in range(self.N):
                if self.w[i, j] > 0:  # Only if coupling exists
                    dtheta[i] += self.r[j] * self.w[i, j] * np.sin(self.theta[j] - self.theta[i] - self.phi[i, j])  # Add coupling term
        
        # Compute \ddot{r}_i = a_i ( (a_i / 4) (R_i - r_i) - \dot{r}_i )
        ddr = self.a * ( (self.a / 4) * (self.R - self.r) - self.dr )  # Second derivative of amplitude
        
        # Euler update for states
        self.theta += dtheta * self.dt  # Update phases
        self.dr += ddr * self.dt  # Update amplitude derivatives
        self.r += self.dr * self.dt  # Update amplitudes
        
        # Update output x_i = r_i (1 + cos(\theta_i))
        self._update_output()

    # Private method to update the output signals
    def _update_output(self):
        self.x = self.r * (1 + np.cos(self.theta))  # Compute outputs based on current phases and amplitudes
    
    # Method to get current phase differences relative to the first oscillator
    def get_phase_diffs(self):
        """
        Get the current phase differences relative to the first unit (mod 2π).
        """
        phase_diffs = (self.theta - self.theta[0]) % (2 * np.pi)  # Compute and normalize differences
        return phase_diffs

# Example usage: custom phase differences [0, π, 3π/2, π/2] in a 4-unit network
if __name__ == "__main__":
    import matplotlib.pyplot as plt  # Import Matplotlib for plotting
    
    # Define desired phase differences
    desired = np.array([0, np.pi, 3*np.pi/2, np.pi/2])
    
    # Create instance: frequency 2π*1.2 rad/s, 4 oscillators, sampling rate 100 Hz, custom phase differences
    sim = Salamander(omega=2 * np.pi * 1.2, cell_num=4, hz=100, desired_phase_diffs=desired)
    sim.init()  # Initialize state
    
    # Simulate for 2000 steps (sufficient for convergence)
    time_steps = 2000
    outputs = np.zeros((time_steps, sim.N))  # Record outputs
    phase_diffs_history = np.zeros((time_steps, sim.N))  # Record phase differences
    for t in range(time_steps):
        sim.step()  # Advance simulation
        outputs[t] = sim.x  # Store outputs
        phase_diffs_history[t] = sim.get_phase_diffs()  # Store phase differences
    
    # Plot output signals
    t_axis = np.arange(time_steps) * sim.dt  # Time axis
    plt.figure(figsize=(12, 6))  # Create figure
    for i in range(sim.N):
        plt.plot(t_axis, outputs[:, i], label=f'Osc {i+1}')  # Plot each oscillator's output
    plt.xlabel('Time (s)')  # X-axis label
    plt.ylabel('Output x_i')  # Y-axis label
    plt.title('Salamander CPG Simulation with Custom Phase Diffs')  # Title
    plt.legend()  # Show legend
    
    plt.show()  # Display plot
