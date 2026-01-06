###############################################################################
# File Name: mini_project_03.py
#
# Description: This program creates a Digital Twin a pump system controlled by 
# a PI controller and variable frequency drive, logging and visualizing key 
# metrics such as flow rate, RPM, power, and specific energy over time. It 
# allows for setpoint sweeps and validation scenarios, providing insights 
# into pump performance under varying operating conditions.
#
# Record of Revisions (Date | Author | Change):
# 11/26/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
from dataclasses import dataclass  # For storing simulation and controller parameters
import numpy as np                 # Numerical computations
import math                        # Math functions
import pandas as pd                # Data logging in tabular format
import matplotlib.pyplot as plt    # Plotting and visualization

# Dataclass to store all pump and controller parameters
@dataclass
class twinParams:
    rpmMin: float = 800.0                  # Minimum allowable pump RPM
    rpmMax: float = 1800.0                 # Maximum allowable pump RPM
    Kp: float = 8.0                        # Proportional gain of PI controller
    Ki: float = 1.0                        # Integral gain of PI controller
    dt: float = 0.1                        # Simulation timestep [s]
    vfdRampRateRPMs: float = 2000.0        # Maximum VFD ramp rate [RPM/s]
    ratedPowerKw: float = 15.0             # Pump rated power [kW]
    h0: float = 41.0                        # Pump head coefficient
    kPump: float = 0.01                     # Pump quadratic coefficient
    rpmDesign: float = 1800.0              # Pump design RPM
    hStatic: float = 6.0                    # System static head [m]
    kSys: float = 0.012                     # System quadratic coefficient

# Class representing pump physics and operating point calculations
class pumpPhysics:
    def __init__(self, params: twinParams):
        self.hStatic = params.hStatic       # Static head in system
        self.kSys = params.kSys             # System loss coefficient
        self.h0 = params.h0                 # Pump head coefficient
        self.k = params.kPump               # Pump coefficient
        self.rpm1 = params.rpmDesign        # Design RPM
        self.ratedP = params.ratedPowerKw   # Rated power [kW]

    # Compute flow rate and ideal head for a given RPM
    def solveOperatingPoint(self, rpm):
        q = math.sqrt((self.h0 * (rpm / self.rpm1) ** 2 - self.hStatic) / (self.k + self.kSys))
        hIdeal = self.hStatic + self.kSys * q ** 2
        return q, hIdeal

    # Estimate pump power based on RPM using cubic scaling
    def estimatePowerKw(self, rpm):
        estPwr = self.ratedP * (rpm / self.rpm1) ** 3
        return estPwr

# PI controller class for regulating pump head
class piController:
    def __init__(self, params: twinParams, hSp):
        self.Kp = params.Kp                 # Proportional gain
        self.Ki = params.Ki                 # Integral gain
        self.dt = params.dt                 # Time step
        self.integral = 0                   # Initialize integral term
        self.rpmMax = params.rpmMax         # Maximum control RPM
        self.rpmMin = params.rpmMin         # Minimum control RPM
        self.hSp = hSp                      # Initial head setpoint

    # Update control RPM based on measured head
    def update(self, hMeas, rpm):
        error = self.hSp - hMeas            # Compute current error
        self.integral += error * self.dt    # Update integral term
        cmdRPM = rpm + self.Kp * error + self.Ki * self.integral  # PI control law
        cmdRPM = np.clip(cmdRPM, self.rpmMin, self.rpmMax)        # Clamp to limits
        return cmdRPM

# Variable frequency drive class to model RPM ramping
class variableFrequencyDrive:
    def __init__(self, params: twinParams):
        self.dt = params.dt
        self.rampRate = params.vfdRampRateRPMs * self.dt   # Max RPM change per timestep

    # Apply ramp-limited control command to current RPM
    def apply(self, cmdRPM, rpm):
        deltaRPM = cmdRPM - rpm                           # Desired change in RPM
        deltaRPM = max(-self.rampRate, min(self.rampRate, deltaRPM))  # Limit ramp
        rpm += deltaRPM                                   # Update RPM
        return rpm

# Simple sensor model to simulate head measurement noise
def sensorHead(hIdeal):
    hMeas = hIdeal + np.random.normal(0,0.05)           # Add small Gaussian noise
    return hMeas

# Run the pump simulation
def runSimulation(headSetpoint, simRuntime, startTimes = None):
    params = twinParams()                               # Load default parameters
    t = 0                                               # Initialize simulation time
    dt = params.dt
    ekWh = 0                                            # Accumulated energy [kWh]
    v = 0                                               # Accumulated volume [m^3]
    se = 0                                              # Specific energy [kWh/m^3]
    rpm = params.rpmMin                                 # Start at minimum RPM
    hSp = headSetpoint[0]                               # Initial head setpoint

    # Initialize pump, controller, and VFD objects
    plant = pumpPhysics(params)
    ctrl = piController(params, hSp)
    vfd = variableFrequencyDrive(params)

    # Initialize dataframe for logging
    log = pd.DataFrame(columns=['Time (s)', 'Head Setpoint (m)', 'Measured Head (m)', 
                                'Ideal Head (m)', 'Volumetric Flow Rate (m^3/s)', 
                                'Actual RPM', 'Control RPM', 'Power (kW)', 
                                'Energy (kWh)', 'Volume (m^3)', 'Specific Energy (kWh/m^3)'])

    while t < simRuntime:
        t += dt

        # Update setpoint if current time matches a start time
        for i, st in enumerate(startTimes):
            if abs(t - st) < 1e-6:
                hSp = headSetpoint[i]
                ctrl.hSp = hSp

        # Compute operating point
        q, hIdeal = plant.solveOperatingPoint(rpm)
        pwr = plant.estimatePowerKw(rpm)
        hMeas = sensorHead(hIdeal)
        cmdRPM = ctrl.update(hMeas, rpm)
        rpm = vfd.apply(cmdRPM, rpm)

        # Accumulate energy and volume
        ekWh += pwr * dt / 3600
        v += q * dt / 3600 
        se = ekWh / v

        # Log current state
        log.loc[len(log)] = [t, hSp, hMeas, hIdeal, q, rpm, cmdRPM, pwr, ekWh, v, se]

    return log

# Plot results from simulation
def plotData(log):
    t = log['Time (s)']

    plt.figure(figsize=(14, 18))

    # Head comparison
    plt.subplot(5, 1, 1)
    plt.plot(t, log['Head Setpoint (m)'], label='Head Setpoint', linewidth=2)
    plt.plot(t, log['Measured Head (m)'], label='Measured Head', linewidth=2)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Head (m)')
    plt.title('Head: Setpoint vs Measured')
    plt.legend()

    # RPM comparison
    plt.subplot(5, 1, 2)
    plt.plot(t, log['Control RPM'], label='Control RPM', linewidth=2)
    plt.plot(t, log['Actual RPM'], label='Actual RPM', linewidth=2)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('RPM')
    plt.title('Pump Speed')
    plt.legend()

    # Volumetric flow rate
    plt.subplot(5, 1, 3)
    plt.plot(t, log['Volumetric Flow Rate (m^3/s)'], linewidth=2)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Flow (m³/s)')
    plt.title('Volumetric Flow Rate Q')

    # Pump power
    plt.subplot(5, 1, 4)
    plt.plot(t, log['Power (kW)'], linewidth=2)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Power (kW)')
    plt.title('Pump Power')

    # Specific energy
    plt.subplot(5, 1, 5)
    plt.plot(t, log['Specific Energy (kWh/m^3)'], label='Specific Energy (kWh/m³)', linewidth=2)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.title('Energy and Specific Energy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function to run baseline, sweep, and validation scenarios
def main():
    # Baseline simulation
    headSetpoint = [25]
    startTimes = [0]
    simRuntime = 60
    log = runSimulation(headSetpoint=headSetpoint, simRuntime=simRuntime, startTimes=startTimes)
    plotData(log)
    print(f"RPM: {log.iloc[-1]['Actual RPM']}\n"
          f"Volumetric Flow Rate (m^3/s): {log.iloc[-1]['Volumetric Flow Rate (m^3/s)']}\n"
          f"Power (kW): {log.iloc[-1]['Power (kW)']}\n"
          f"Specific Energy (kWh/m^3): {log.iloc[-1]['Specific Energy (kWh/m^3)']}\n")

    # Setpoint sweep simulation
    headSetpoint = [18, 20, 22, 25, 28]
    startTimes = [0, 40, 80, 120, 160]
    simRuntime = 200
    log = runSimulation(headSetpoint=headSetpoint, simRuntime=simRuntime, startTimes=startTimes)
    plotData(log)
    print(f"RPM: {log.iloc[-1]['Actual RPM']}\n"
          f"Volumetric Flow Rate (m^3/s): {log.iloc[-1]['Volumetric Flow Rate (m^3/s)']}\n"
          f"Power (kW): {log.iloc[-1]['Power (kW)']}\n"
          f"Specific Energy (kWh/m^3): {log.iloc[-1]['Specific Energy (kWh/m^3)']}\n")

    # Validation simulation with quick setpoint change
    headSetpoint = [25, 15]
    startTimes = [0, 1]
    simRuntime = 200
    log = runSimulation(headSetpoint=headSetpoint, simRuntime=simRuntime, startTimes=startTimes)
    plotData(log)
    print(f"RPM: {log.iloc[-1]['Actual RPM']}\n"
          f"Volumetric Flow Rate (m^3/s): {log.iloc[-1]['Volumetric Flow Rate (m^3/s)']}\n"
          f"Power (kW): {log.iloc[-1]['Power (kW)']}\n"
          f"Specific Energy (kWh/m^3): {log.iloc[-1]['Specific Energy (kWh/m^3)']}\n")

if __name__ == "__main__":
    main()