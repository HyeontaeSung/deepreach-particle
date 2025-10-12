from dataclasses import dataclass
import torch
from torch import Tensor
import math

def to_tensor(x, device=None, dtype=torch.float32) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def angle_wrap(theta: Tensor) -> Tensor:
    # wrap to [-pi, pi] MKEHDM
    return (theta + math.pi) % (2 * math.pi) - math.pi
# this is for discrete time system

@dataclass
class Dubins3D2D:
    v: float = 2.0
    dt: float = 0.01
    omega_min: float = -2.0
    omega_max: float =  2.0
    v_max: float = 2.0
    v_min: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    u_dim: int = 2 # 2D control [heading_angle, velocity]

    def step(self, x: Tensor, u: Tensor) -> Tensor:
        """
        x: (..., 3)  [px, py, theta]
        u: (...)     yaw-rate, velocity
        returns x_next: (..., 3)
        """
        x = to_tensor(x, self.device, self.dtype)
        omega = torch.clamp(to_tensor(u[...,0], self.device, self.dtype), self.omega_min, self.omega_max)
        v = torch.clamp(to_tensor(u[...,1], self.device, self.dtype), self.v_min, self.v_max)

        px, py, th = x[..., 0], x[..., 1], x[..., 2]
        dt = self.dt
        # this is for discrete time system
        pxn = px + v * torch.cos(th+0.5*omega*dt) * dt
        pyn = py + v * torch.sin(th+0.5*omega*dt) * dt
        thn = angle_wrap(th + omega * dt)
        return torch.stack([pxn, pyn, thn], dim=-1)

    def f1_f2(self, x: Tensor):
        # not a control affine system 
        v = to_tensor(self.v, device = x.device, dtype = x.dtype)
        px, py, th = x[...,0], x[...,1], x[...,2]
        f1 = torch.stack([v*torch.cos(th), v*torch.sin(th), torch.zeros_like(th)], dim=-1)  # (..., 3)
        z = torch.zeros_like(th)
        o = torch.ones_like(th)
        f2 = torch.stack([z, z, o], dim = -1)  # (..., 3)
        # f2 = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=x.dtype)  # shape (3,)
        return f1, f2
    
    def f(self, x: Tensor, u: Tensor):
        # not a control affine system 
        # dsdt in dynamics
        omega, v  = u[...,0], u[...,1]
        th = x[...,2]
        f = torch.stack([v*torch.cos(th), v*torch.sin(th), omega], dim=-1)  # (..., 3)
        return f

@dataclass
class Dubins4D:
    v: float = 2.0
    dt: float = 0.01
    omega_min: float = -2.0
    omega_max: float =  2.0
    v_max: float = 2.0
    v_min: float = 0.1
    a_max: float = 2.0
    a_min: float = -2.0
    c: float = -2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    def step(self, x: Tensor, u: Tensor) -> Tensor:
        """
        x: (..., 3)  [px, py, theta]
        u: (...)     yaw-rate, velocity
        returns x_next: (..., 3)
        """
        x = to_tensor(x, self.device, self.dtype)
        omega = torch.clamp(to_tensor(u[...,0], self.device, self.dtype), self.omega_min, self.omega_max)
        acc = torch.clamp(to_tensor(u[...,1], self.device, self.dtype), self.a_min, self.a_max)

        px, py, th, v = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        dt = self.dt

        pxn = px + v * torch.cos(th) * dt
        pyn = py + v * torch.sin(th) * dt
        thn = angle_wrap(th + omega * dt)
        v = v + acc * dt
        v = torch.clamp(v, self.v_min, self.v_max)
        return torch.stack([pxn, pyn, thn, v], dim=-1)

    # def f1_f2(self, x: Tensor):
    #     # not a control affine system 
    #     v = to_tensor(self.v, device = x.device, dtype = x.dtype)
    #     px, py, th = x[...,0], x[...,1], x[...,2]
    #     f1 = torch.stack([v*torch.cos(th), v*torch.sin(th), torch.zeros_like(th)], dim=-1)  # (..., 3)
    #     z = torch.zeros_like(th)
    #     o = torch.ones_like(th)
    #     f2 = torch.stack([z, z, o], dim = -1)  # (..., 3)
    #     # f2 = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=x.dtype)  # shape (3,)
    #     return f1, f2
    
    def f(self, x: Tensor, u: Tensor):
        # not a control affine system 
        # dsdt in dynamics
        omega, acc  = u[...,0], u[...,1]
        th, v = x[...,2], x[...,3]
        f = torch.stack([v*torch.cos(th), v*torch.sin(th), omega, acc], dim=-1)  # (..., 3)
        return f


@dataclass
class Pedestrian:
    dt: float = 0.01
    v_max: float = 1.0
    v_min: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    def step(self, x: Tensor, u: Tensor) -> Tensor:
        """
        x: (..., 2)  [px, py]
        u: (..., 2)     heading, velocity
        returns x_next: (..., 2)
        """
        px, py = x[..., 0], x[..., 1]
        heading, velocity = u[..., 0], u[..., 1]

        pxn = px + velocity * torch.cos(heading) * self.dt
        pyn = py + velocity * torch.sin(heading) * self.dt
        return torch.stack([pxn, pyn], dim=-1)

    def f(self, x: Tensor, u: Tensor):
        # dsdt in dynamics
        heading, velocity = u[..., 0], u[..., 1]
        f = torch.stack([velocity * torch.cos(heading), velocity * torch.sin(heading)], dim=-1)  # (..., 2)
        return f


def heading_to_goal_u_nom_3d(x: Tensor, goal: Tensor, k: float = 2.5, 
                         omega_min: float = -2.0, omega_max: float = 2.0,
                         v_min: float = 0.1, v_max: float = 2.0) -> Tensor:
    """
    Nominal controller: 2D control [angular_velocity, linear_velocity]
    Returns control to reduce angle error to goal
    """
    x = to_tensor(x); goal = to_tensor(goal, device=x.device, dtype=x.dtype)
    px, py, th = x[...,0], x[...,1], x[...,2]
    gx, gy = goal[...,0], goal[...,1]
    
    # Compute desired heading angle
    desired_heading = torch.atan2(gy - py, gx - px)
    
    err = angle_wrap(desired_heading - th)
    
    # Angular velocity control (same as before)
    omega_nom = k * err
    omega_nom = torch.clamp(omega_nom, omega_min, omega_max)
    
    # Linear velocity control - could be constant or distance-based
    # Option 1: Constant velocity
    v_nom = torch.full_like(omega_nom, v_max)  # Maximum velocity
    
    # Option 2: Distance-based velocity (uncomment if you prefer this)
    # dist_to_goal = torch.sqrt((gx - px)**2 + (gy - py)**2)
    # v_nom = torch.clamp(dist_to_goal * 0.5, v_min, v_max)  # Scale by distance
    
    # Return 2D control
    return torch.stack([omega_nom, v_nom], dim=-1)  # (..., 2)


def heading_to_goal_u_nom(x: Tensor, goal: Tensor, k_omega: float = 5.0, k_v: float = 5.0,
                             omega_min: float = -2.0, omega_max: float = 2.0,
                             a_min: float = -2.0, a_max: float = 2.0,
                             v_min: float = 0.1, v_max: float = 2.0) -> Tensor:
    """
    Nominal controller for Dubins4D: 2D control [angular_velocity, acceleration]
    State: [px, py, theta, v]
    Control: [omega, acceleration]
    Returns control to reduce angle error to goal and regulate velocity
    """
    x = to_tensor(x); goal = to_tensor(goal, device=x.device, dtype=x.dtype)
    px, py, th, v = x[...,0], x[...,1], x[...,2], x[...,3]
    gx, gy = goal[...,0], goal[...,1]
    
    # Compute desired heading angle
    desired_heading = torch.atan2(gy - py, gx - px)
    
    # Angular velocity control to reduce heading error
    heading_err = angle_wrap(desired_heading - th)
    omega_nom = k_omega * heading_err
    omega_nom = torch.clamp(omega_nom, omega_min, omega_max)
    
    # Distance to goal
    dist_to_goal = torch.sqrt((gx - px)**2 + (gy - py)**2)
    
    # Velocity control - accelerate towards desired velocity based on distance
    # Option 1: Distance-based desired velocity
    v_desired = torch.clamp(dist_to_goal * k_v, v_min, v_max)
    
    # Option 2: Constant desired velocity (uncomment if you prefer this)
    # v_desired = torch.full_like(v, v_max)
    
    # Acceleration control to reach desired velocity
    v_err = v_desired - v
    acc_nom = k_v * v_err
    acc_nom = torch.clamp(acc_nom, a_min, a_max)
    
    # Return 2D control [omega, acceleration]
    return torch.stack([omega_nom, acc_nom], dim=-1)  # (..., 2)


def heading_to_goal_u_nom_pedestrian(x: Tensor, goal: Tensor, k: float = 50.0, 
                                          v_max: float = 1.0) -> Tensor:
    """
    Nominal controller for Pedestrian: 2D control [heading_angle, velocity]
    Returns control to move toward goal
    
    Args:
        x: (..., 2) [px, py] current position
        goal: (..., 2) [gx, gy] goal position
        k: velocity scaling factor
        v_max: maximum velocity magnitude
    """
    x = to_tensor(x); goal = to_tensor(goal, device=x.device, dtype=x.dtype)
    px, py = x[...,0], x[...,1]
    gx, gy = goal[...,0], goal[...,1]
    
    # Compute desired heading angle
    desired_heading = torch.atan2(gy - py, gx - px)
    
    # Compute distance to goal for velocity scaling
    dist_to_goal = torch.sqrt((gx - px)**2 + (gy - py)**2)
    
    # Compute desired velocity magnitude (distance-based scaling)
    v_magnitude = torch.clamp(dist_to_goal * k, 0.0, v_max)
    # v_magnitude = torch.full_like(desired_heading, v_max) # use max velocity
    
    # Return 2D control [heading_angle, velocity]
    return torch.stack([desired_heading, v_magnitude], dim=-1)  # (..., 2)










    