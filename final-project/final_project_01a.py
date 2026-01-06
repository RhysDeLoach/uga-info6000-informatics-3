###############################################################################
# File Name: final_project_01a.py
#
# Description: This program implements a Digital Twin of the Quanser QubeServo 2 
# Furuta pendulum environment with full nonlinear dynamics, integrating a DC 
# motor-driven rotary arm and pendulum using RK4. It normalizes states, computes 
# rewards for balancing, and optionally renders the system in real time to 
# visualize arm and pendulum motion.
#
# Record of Revisions (Date | Author | Change):
# 12/04/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
import numpy as np
import math
import matplotlib.pyplot as plt

@dataclass
class furutaParams:
    jr: float = 0.000057 # Rotary arm moment of inertia (Kg*m^2)
    jp: float = 0.000033 # Pendulum moment of inertia about CoM (Kg*m^2)
    mp: float = 0.024 # Pendulum mass (kg)
    r: float = 0.085 # Rotary arm length (m)
    lp: float = 0.129 # Pendulum length (m)
    l: float = None # Pendulum center of mass (m)
    g: float = 9.81 # Acceleration due to gravity (m/s^2)
    br: float = 0.0005 # Rotary arm viscous damping torque (N*m*s/rad)
    bp: float = 0.0015 # Pendulum viscous damping torque (N*m*s/rad)
    rm: float = 8.4 # Motor resistance (ohms)
    km: float = 0.042 # Motor torque constant (N*m/A)
    thetaMin: float = -math.pi / 2 # Rotary arm min angle (rads)
    thetaMax: float = math.pi / 2 # Rotary arm max angle (rads)
    thetaDotMin: float = -50.0 # Rotary arm min angular velocity (rads/s)
    thetaDotMax: float = 50.0 # Rotary arm max angular velocity (rads/s)
    alphaMin: float = -math.pi # Pendulum min angle (rads)
    alphaMax: float = math.pi # Pendulum max angle (rads)
    alphaDotMin: float = -50.0 # Pendulum min angular velocity (rads/s)
    alphaDotMax: float = 50.0 # Pendulum max angular velocity (rads/s)
    actionMin: float = -18.0 # Motor voltage min (V)
    actionMax: float = 18.0 # Motor voltage max (V)
    maxSteps: float = 1000.0 # Max steps per iteration
    
    def __post_init__(self): # Calc l
        if self.l is None:
            self.l = self.lp / 2


class furutaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, params: furutaParams = None, dt=0.01, render_mode=None):
        super().__init__()
        self.params = params if params is not None else furutaParams()
        self.dt = dt
        self.render_mode = render_mode 
        self.currentSteps = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # Initialize env action space

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32) # Initialize env observation space

        self.state = None
        if self.render_mode == "human":
            self._init_render()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p = self.params
        self.currentSteps = 0
        self.state = np.array([ # Randomize initial conditions
            np.random.uniform(-0.8 * p.thetaMax, 0.8 * p.thetaMax), # +/- about 70 degrees b/c it can get stuck at bounds
            np.random.uniform(p.thetaDotMin / 2, p.thetaDotMax / 2), # Clamped to half of velocity bounds
            np.random.uniform(p.alphaMin, p.alphaMax),
            np.random.uniform(p.alphaDotMin / 2, p.alphaDotMax / 2) # Clamped to half of velocity bounds
        ])
        return self.normObs(), {} # Normalize output

    def step(self, action):
        p = self.params
        dt = self.dt
        self.currentSteps += 1
        
        actionVal = float(action[0]) if isinstance(action, np.ndarray) else float(action) # Convert to action to float
        actionVal = np.clip(actionVal, -1.0, 1.0) # Ensure the action is within the bounds
        voltage = actionVal * p.actionMax # Denormalize the action

        # RK4 integration
        def derivs(state):
            thetaDdot, alphaDdot = self.calcAccelRaw(state, voltage)
            return np.array([state[1], thetaDdot, state[3], alphaDdot])
        
        s = self.state
        k1 = derivs(s)
        k2 = derivs(s + 0.5*dt*k1)
        k3 = derivs(s + 0.5*dt*k2)
        k4 = derivs(s + dt*k3)
        self.state = s + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Clamp to system bounds
        s = self.state
        if s[0] <= p.thetaMin:
            s[0] = p.thetaMin
            s[1] = max(s[1], 0.0)
        elif s[0] >= p.thetaMax:
            s[0] = p.thetaMax
            s[1] = min(s[1], 0.0)
        
        s[1] = np.clip(s[1], p.thetaDotMin, p.thetaDotMax)
        s[2] = ((s[2] + math.pi) % (2 * math.pi)) - math.pi
        s[3] = np.clip(s[3], p.alphaDotMin, p.alphaDotMax)

        reward = self.calcReward(actionVal) # Calculate Reward
        terminated = False
        truncated = self.currentSteps >= p.maxSteps # End episodes once the steps max (10 seconds or 1000 steps)

        return self.normObs(), reward, terminated, truncated, {} # Normalize output


    def calcAccelRaw(self, state, voltage):
        p = self.params
        theta, thetaDot, alpha, alphaDot = state

        torque = p.km / p.rm * (voltage - p.km * thetaDot) # DC motor equation
        jpPivot = p.jp + p.mp * p.l**2 # Calculate moment of inertia about the pivot

        sin_a, cos_a = math.sin(alpha), math.cos(alpha)
        
        # Check if arm is at limit and being pushed into it
        atMin = (theta <= p.thetaMin) and (thetaDot <= 0)
        atMax = (theta >= p.thetaMax) and (thetaDot >= 0)
        
        if atMin or atMax:
            # Behaves as simple pendulum at bounds
            thetaDdot = 0.0
            alphaDdot = (-p.bp * alphaDot - p.mp * p.g * p.l * sin_a) / jpPivot
        else:
            # Full coupled dynamics
            M11 = p.jr + p.mp * p.r**2 + jpPivot * sin_a**2
            M12 = p.mp * p.l * p.r * cos_a
            M22 = jpPivot
            det_M = M11 * M22 - M12**2
            
            h = jpPivot * sin_a * cos_a
            
            rhs1 = (torque 
                    - p.br * thetaDot 
                    - h * alphaDot * thetaDot 
                    + p.mp * p.l * p.r * sin_a * alphaDot**2)
            rhs2 = (-p.bp * alphaDot 
                    + h * thetaDot**2 
                    - p.mp * p.g * p.l * sin_a)
            
            thetaDdot = (M22 * rhs1 - M12 * rhs2) / det_M
            alphaDdot = (M11 * rhs2 - M12 * rhs1) / det_M
        
        return float(thetaDdot), float(alphaDdot) # Rotary arm acceleraction, Pendulum Acceleration

    def normObs(self):
        p = self.params
        s = self.state

        # Normalize theta and thetaDot to [-1, 1]
        norm_theta = 2 * (s[0] - p.thetaMin) / (p.thetaMax - p.thetaMin) - 1
        norm_theta_dot = 2 * (s[1] - p.thetaDotMin) / (p.thetaDotMax - p.thetaDotMin) - 1
        
        # Use sin/cos for alpha
        sin_alpha = math.sin(s[2])
        cos_alpha = math.cos(s[2])
        
        # Normalize alphaDot
        norm_alpha_dot = 2 * (s[3] - p.alphaDotMin) / (p.alphaDotMax - p.alphaDotMin) - 1

        return np.array([norm_theta, norm_theta_dot, sin_alpha, cos_alpha, norm_alpha_dot], dtype=np.float32)

    def calcReward(self, action):
        p = self.params
        s = self.state
        
        uprightReward = math.cos(s[2] - math.pi) # Upright reward
        
        # Energy reward 
        if math.cos(s[2]) > 0:  # Below horizontal
            potentialEnergy = (1 - math.cos(s[2])) / 2
            kineticEnergy = 0.5 * (s[3] / p.alphaDotMax)**2
            energyReward = min(potentialEnergy + kineticEnergy, 1.0) * 0.3
        else:
            energyReward = 0.0
        
        # Smooth balance reward
        if math.cos(s[2]) < -0.8:  # Within tolerance of upright
            velocityFactor = math.exp(-0.5 * s[3]**2)  # Gaussian (1.0 at rest, decays with speed)
            angleFactor = -math.cos(s[2])  # 1.0 when perfectly upright
            balanceReward = 2.0 * velocityFactor * angleFactor
        else:
            balanceReward = 0.0
        
        # Penalties
        thetaPenalty = 0.1 * (s[0] / p.thetaMax)**2
        thetaDotPenalty = 0.01 * (s[1] / p.thetaDotMax)**2
        actionPenalty = 0.01 * action**2
        
        reward = uprightReward + energyReward + balanceReward - thetaPenalty - thetaDotPenalty - actionPenalty
        return float(reward)

    def _init_render(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-0.25, 0.25)
        self.ax.set_ylim(-0.25, 0.25)
        self.arm_line, = self.ax.plot([], [], lw=4, c='blue')
        self.pendulum_line, = self.ax.plot([], [], lw=4, c='red')
        plt.ion()
        plt.show()

    def render(self):
        if self.render_mode != "human":
            return
        theta, _, alpha, _ = self.state
        r = self.params.r
        lp = self.params.lp
        x0, y0 = 0, 0
        x_arm = r * math.cos(theta)
        y_arm = r * math.sin(theta)
        x_pend = x_arm + lp * math.sin(alpha)
        y_pend = y_arm - lp * math.cos(alpha)
        self.arm_line.set_data([x0, x_arm], [y0, y_arm])
        self.pendulum_line.set_data([x_arm, x_pend], [y_arm, y_pend])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)

