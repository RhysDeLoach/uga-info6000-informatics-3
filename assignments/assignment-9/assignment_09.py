###############################################################################
# File Name: assignment_09.py
#
# Description: This script simulates a free-fall system with quadratic drag 
# using a classical RK4 integrator, then optionally trains a Physics-Informed 
# Neural Network (PINN) to approximate the position and velocity over time. It 
# logs the true plant states and compares them to the PINN predictions, plotting 
# both for visualization.
#
# Record of Revisions (Date | Author | Change):
# 11/06/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

@dataclass
class fallParams:
    m: float = 80 # Mass (kg)
    g: float = 9.81 # Acceleration due to gravity (m/s^2)
    cD: float = 0.31 # Linear drag coefficient (kg/m)

class freeFallPlant:
    def __init__(self, params: fallParams):
        # Initialize the free-fall system with given parameters
        self.p = params
        # State vector: [position, velocity]
        self.state = np.array([0.0, 0.0], dtype=float)

    def reset(self, pos: float = 0.0, pos_dot: float = 0.0):
        # Reset plant to initial state
        self.state = np.array([pos, pos_dot], dtype=float)
        return self.state.copy()
    
    def f(self, x: np.ndarray) -> np.ndarray:
        # Compute derivatives [dx/dt, dv/dt] given current state
        _, pos_dot = x
        m, g, cD = self.p.m, self.p.g, self.p.cD
        # Quadratic drag: dv/dt = g - (cD/m)*v^2
        pos_ddot = g - (cD * pos_dot ** 2) / m
        return np.array([pos_dot, pos_ddot])
    
    def step(self, dt: float) -> np.ndarray:
        # Advance one time step using RK4 integration
        x = self.state
        k1 = self.f(x)
        k2 = self.f(x + 0.5 * dt * k1)
        k3 = self.f(x + 0.5 * dt * k2)
        k4 = self.f(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.state = x_next
        return x_next.copy()

    
class DigitalTwin:
    def __init__(self, plant: freeFallPlant, dt: float = 0.01):
        self.plant = plant
        self.dt = dt
        self.log: Dict[str, List[float]] = {k: [] for k in ['t', 'pos', 'pos_dot']}

    def run(self, T: float = 6.0, pos0: float = 0.0, pos_dot0: float = 0.0):
        self.plant.reset(pos0, pos_dot0)
        N = int(T / self.dt)

        for k in range(N):
            t = k * self.dt
            pos, pos_dot = self.plant.state

            x_next = self.plant.step(self.dt)

            # Log Data
            self.log['t'].append(t)
            self.log['pos'].append(pos)
            self.log['pos_dot'].append(pos_dot)

        return self.get_log()
    
    def get_log(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype = float) for k, v in self.log.items()}

from dataclasses import dataclass as _dataclass

@_dataclass
class PINNArtifact:
    hidden: int
    state_dict: dict
    t_min: float
    t_max: float

    def build_model(self):
        m = PINNStateOfTime(hidden = self.hidden)
        m.load_state_dict(self.state_dict)
        m.eval()
        return m
    
    def predict_pos(self, t_array: np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE:
            raise RuntimeError('Pytorch not available for PINN prediction')
        with torch.no_grad():
            t = torch.tensor(t_array, dtype = torch.float32).unsqueeze(1)
            t_norm = (2.0 * (t - self.t_min) / (self.t_max - self.t_min + 1e-9)) - 1.0
            model = self.build_model()
            y = model(t_norm)
            pos_hat = y[:, 0].cpu().numpy()
        return pos_hat
    
class PINNStateOfTime(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )
        
    def forward(self, t: torch.Tensor) -> torch.tensor:
        return self.net(t)
    
def train_pinn(log: Dict[str, np.ndarray], m: float, g: float, cD: float,
                epochs: int = 2500, lr: float = 1e-3, data_weight: float = 5e-2,
                warmup_epochs: int = 300, device: str = 'cpu'):
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available — skipping PINN training.")
        return None
    
    t = torch.tensor(log['t'], dtype=torch.float32, device=device).unsqueeze(1)
    pos_meas = torch.tensor(log['pos'], dtype=torch.float32, device=device).unsqueeze(1)
    pos_dot_meas = torch.tensor(log['pos_dot'], dtype=torch.float32, device=device).unsqueeze(1)

    # Normalize time
    t_min, t_max = t.min(), t.max()
    t_norm = (2.0 * (t - t_min) / (t_max - t_min + 1e-9)) - 1.0
    t_norm = t_norm.clone().detach().requires_grad_(True)

    model = PINNStateOfTime(hidden=64).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        opt.zero_grad()
        y = model(t_norm)  # [pos, pos_dot]
        pos = y[:, :1]
        pos_dot = y[:, 1:2]
        dpos_dt = torch.autograd.grad(pos, t_norm, grad_outputs=torch.ones_like(pos),
                                        create_graph=True, retain_graph=True)[0]
        dpos_dot_dt = torch.autograd.grad(pos_dot, t_norm, grad_outputs=torch.ones_like(pos_dot),
                                        create_graph=True, retain_graph=True)[0]
        dt_norm_dt = 2.0 / (t_max - t_min + 1e-9)
        pos_t = dpos_dt * dt_norm_dt
        pos_dot_t = dpos_dot_dt * dt_norm_dt

        r1 = pos_t - pos_dot
        r2 = pos_dot_t - (g - (cD * pos_dot ** 2) / m)
        loss_resid = (r1 ** 2).mean() + (r2 ** 2).mean()

        loss_data = ((pos - pos_meas) ** 2).mean()
        bc_pos = (pos[0] - pos_meas[0])**2
        bc_pos_dot = (pos_dot[0] - 0)**2  # Initial velocity


        if ep < warmup_epochs:
            loss = data_weight * loss_data + 1e-2 * bc_pos + 1e-3 * bc_pos_dot
        else:
            loss = loss_resid + data_weight * loss_data + 1e-2 * bc_pos + 1e-3 * bc_pos_dot

        loss.backward()
        opt.step()

        if (ep + 1) % 200 == 0 or ep == 0:
            print(f"[PINN] epoch {ep+1:4d}  loss={loss.item():.4e}  resid={loss_resid.item():.4e}  data={loss_data.item():.4e}")

    with torch.no_grad():
        y_hat = model(t_norm)
        pos_hat = y_hat[:, 0].cpu().numpy()
        pos_dot_hat = y_hat[:, 1].cpu().numpy()

    artifact = PINNArtifact(hidden=64,
                           state_dict=model.state_dict(),
                           t_min=float(t_min.cpu().item() if hasattr(t_min, 'cpu') else t_min),
                           t_max=float(t_max.cpu().item() if hasattr(t_max, 'cpu') else t_max))
    return {
        't': log['t'],
        'pos_hat': pos_hat,
        'pos_dot_hat': pos_dot_hat,
        'loss_final': float(loss.item()),
        'artifact': artifact
    }

def plot_timeseries(log: Dict[str, np.ndarray], pinn_out: Dict[str, np.ndarray] | None = None):
    t = log['t']
    pos = log['pos']
    pos_dot = log['pos_dot']

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(t, pos, label='Position (plant)')  # Plot plant angle
    if pinn_out is not None:
        axs[0].plot(t, pinn_out['pos_hat'], '--', label='Position (PINN)')
        axs[0].set_title(f"Position — PINN")
    else:
        axs[0].set_title("Position")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, pos_dot, label='v')
    if pinn_out is not None:
        axs[1].plot(t, pinn_out['pos_dot_hat'], '--', label='Velocity (PINN)')
        axs[1].set_title(f"Velocity — PINN")
    else:
        axs[1].set_title("Velocity")
    axs[1].set_title("Velocity")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # True plant params (you can change these for demos)
    params = fallParams(m=80, g=9.81, cD=0.31)

    plant = freeFallPlant(params)

    dt = 0.01
    twin = DigitalTwin(plant, dt=dt)

    log = twin.run(T=20.0, pos0=0.0, pos_dot0=0.0)

    # Visualize time series
    pinn_out = None

    # Train PINN (optional — set to False to skip in class)
    DO_PINN = True and TORCH_AVAILABLE
    if DO_PINN:
        pinn_out = train_pinn(log, m=params.m, g=params.g, cD=params.cD,
                              epochs=17000, lr=1e-3, data_weight=2e-2,
                              device='mps')
        
    plot_timeseries(log, pinn_out)

if __name__ == "__main__":
    main()