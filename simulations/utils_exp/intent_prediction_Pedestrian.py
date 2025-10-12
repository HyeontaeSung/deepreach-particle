import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math

# ====== CONFIGURATION ======
@dataclass
class IntentConfig:
    """Configuration for intent inference system"""
    # Goals (configurable)
    goals: List[List[float]] = None  # [[x1,y1], [x2,y2], [x3,y3]]
    
    # Rationality levels (beta values)
    beta_grid: List[float] = None  # [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    
    # Prediction parameters
    prediction_horizon: int = 4  # T = 4 time steps
    dt: float = 0.25  # time step for prediction
    
    # Control parameters
    u_min: float = -1.2
    u_max: float = 1.2
    num_control_samples: int = 64  # number of control sequences to sample
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        if self.goals is None:
            self.goals = [[8.0, 0.0], [-8.0, 0.0], [0.0, 8.0]]  # 3 default goals
        if self.beta_grid is None:
            self.beta_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# ====== UTILITIES ======
def to_tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def angle_wrap(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]"""
    return (theta + math.pi) % (2 * math.pi) - math.pi

def stable_softmax(logits, dim=-1):
    """Numerically stable softmax"""
    z = logits - logits.max(dim=dim, keepdim=True).values
    probs = torch.exp(z)
    return probs / probs.sum(dim=dim, keepdim=True)

# ====== DUBINS DYNAMICS ======
class Dubins3D:
    """Dubins 3D dynamics for obstacle vehicle"""
    def __init__(self, v: float = 1.0, dt: float = 0.01, 
                 u_min: float = -1.2, u_max: float = 1.2,
                 device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.v = v
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.device = device
        self.dtype = dtype

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 3)  [px, py, theta]
        u: (...)     yaw-rate
        returns x_next: (..., 3)
        """
        x = to_tensor(x, self.device, self.dtype)
        u = torch.clamp(to_tensor(u, self.device, self.dtype), self.u_min, self.u_max)

        px, py, th = x[..., 0], x[..., 1], x[..., 2]
        dt = self.dt
        v = self.v

        pxn = px + v * torch.cos(th) * dt
        pyn = py + v * torch.sin(th) * dt
        thn = angle_wrap(th + u * dt)
        return torch.stack([pxn, pyn, thn], dim=-1)

# ====== Pedestrian DYNAMICS ======
class Pedestrian:
    """Pedestrian dynamics"""
    def __init__(self, v: float = 1.0, dt: float = 0.01,
                 device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.v = v
        self.dt = dt
        self.device = device
        self.dtype = dtype

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 2)  [px, py]
        u: (...)     heading, velocity
    
        returns x_next: (..., 2)
        """
        x = to_tensor(x, self.device, self.dtype)
        heading, velocity = u[..., 0], u[..., 1]
        px, py = x[..., 0], x[..., 1]
        dt = self.dt
        v = self.v
        pxn = px + velocity * torch.cos(heading) * dt
        pyn = py + velocity * torch.sin(heading) * dt
        return torch.stack([pxn, pyn], dim=-1)


# ====== Q-FUNCTION (from paper) ======
def Q_example(x: torch.Tensor, U: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    Q-function from the paper: Q(x,u; g) = -||u||^2 - ||x + u - g||^2
    x: (2,), U: (K,2), g: (2,) --> returns (K,)
    """
    return -(U**2).sum(dim=1) - ((x.unsqueeze(0) + U - g.unsqueeze(0))**2).sum(dim=1)

# ====== BOLTZMANN POLICY ======
def P_u_given_x_beta_g(x: torch.Tensor, U: torch.Tensor, beta: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    Boltzmann policy: P(u | x; beta, g) over discrete action set
    beta: scalar tensor
    """
    Qs = Q_example(x, U, g)  # (K,)
    if beta.item() == 0.0:
        return torch.full((U.size(0),), 1.0/U.size(0), device=U.device, dtype=U.dtype)
    logits = beta * Qs
    return stable_softmax(logits, dim=0)  # (K,)

def P_x_given_x_beta_g(x: torch.Tensor, U: torch.Tensor, beta: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Equation (6) from Fisac et al. paper: P(x' | x; β, g) = Σ_u P(x' | x, u; β, g) · P(u | x; β, g)
    
    Where P(x' | x, u; β, g) = 1{x' = f̃_H(x, u)} (indicator function)
    So: P(x' | x; β, g) = Σ_u 1{x' = f̃_H(x, u)} · P(u | x; β, g)
    
    Args:
        x: (2,) current state [px, py]
        U: (K, 2) discrete action set [vx, vy]
        beta: scalar rationality parameter
        g: (2,) goal position [gx, gy]
    
    Returns:
        next_states: (K, 2) possible next states x'
        next_state_probs: (K,) probabilities for each next state
    """
    # Get action probabilities P(u | x; β, g)
    action_probs = P_u_given_x_beta_g(x, U, beta, g)  # (K,)
    
    # Compute next states for each action using pedestrian dynamics f̃_H(x, u)
    dt = 0.25  # prediction timestep
    next_states = []
    for action in U:
        # Convert [vx, vy] to [heading, velocity] for pedestrian dynamics
        vx, vy = action[0], action[1]
        heading = torch.atan2(vy, vx)
        velocity = torch.sqrt(vx**2 + vy**2 + 1e-8)
        
        # Compute next state: x' = f̃_H(x, u)
        px, py = x[0], x[1]
        px_next = px + velocity * torch.cos(heading) * dt
        py_next = py + velocity * torch.sin(heading) * dt
        next_state = torch.stack([px_next, py_next], dim=0)
        next_states.append(next_state)
    
    next_states = torch.stack(next_states, dim=0)  # (K, 2)
    
    # Since f̃_H(x,u) is deterministic, each action leads to exactly one next state
    # The indicator function 1{x' = f̃_H(x, u)} = 1 for the action that leads to x', 0 otherwise
    # So P(x' | x; β, g) = P(u | x; β, g) where x' = f̃_H(x, u)
    
    # The probability of each next state equals the probability of the action that leads to it
    return next_states, action_probs

# ====== BAYESIAN INTENT UPDATER ======
class BayesianIntentUpdater:
    """
    Bayesian intent inference system following Fisac et al. paper
    """
    
    def __init__(self, config: IntentConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        
        # Convert goals to tensors
        self.goals = [to_tensor(g, self.device, self.dtype) for g in config.goals]
        self.G = len(self.goals)  # number of goals
        
        # Beta grid
        self.beta_grid = to_tensor(config.beta_grid, self.device, self.dtype)
        self.B = self.beta_grid.numel()
        
        # Initialize joint belief b(beta, g) - uniform
        self.b_BG = self._init_joint_belief()
        
        # Pedestrian dynamics for obstacle
        self.dynamics = Pedestrian(dt = self.config.dt, device=self.device, dtype=self.dtype)
        
        # Control set (discrete actions)
        self._setup_control_set()
        
    def _init_joint_belief(self) -> torch.Tensor:
        """Initialize uniform joint belief over (beta, goal)"""
        b = torch.ones((self.B, self.G), device=self.device, dtype=self.dtype)
        return b / b.sum()  # (B,G)
    
    def _setup_control_set(self):
        """Setup discrete control set for obstacle vehicle"""
        # Simple control set: stop + 16 directions at max speed
        speeds = torch.tensor([0.0, 1.0], device=self.device, dtype=self.dtype)
        angles = torch.linspace(0, 2*torch.pi, steps=8+1, device=self.device)[:-1]
        
        self.U = torch.vstack([
            torch.zeros(1, 2, device=self.device, dtype=self.dtype),
            torch.stack([speeds[1]*torch.cos(angles), speeds[1]*torch.sin(angles)], dim=1)
        ])  # (K,2)
        self.K = self.U.size(0)
    
    @torch.no_grad()
    def update_belief(self, x: torch.Tensor, u_obs: torch.Tensor) -> torch.Tensor:
        """
        Update joint belief b(beta, g) given observed action u_obs at state x
        Following the paper's Bayesian update procedure
        
        Args:
            x: (3,) current obstacle state [px, py, theta]  # ← PREVIOUS state!
            u_obs: (2,) observed action [vx, vy]           # ← OBSERVED action!
        
        Returns:
            updated b_BG: (B,G) joint belief
        """
        # Map u_obs to nearest action in discrete set
        dists = torch.norm(self.U - u_obs.unsqueeze(0), dim=1)
        k_obs = torch.argmin(dists).item()
        
        # Compute likelihoods P(u_obs | x; beta, g) for all (beta, g) combinations
        g_stacked = torch.stack(self.goals, dim=0)  # (G,2)
        Qg = torch.stack([Q_example(x[:2], self.U, g_stacked[g]) for g in range(self.G)], dim=0)  # (G,K)
        
        # Vectorized likelihood computation
        logits_BGK = self.beta_grid.view(self.B,1,1) * Qg.view(1,self.G,self.K)  # (B,G,K)
        z = logits_BGK - logits_BGK.max(dim=2, keepdim=True).values
        probs_BGK = torch.exp(z) / torch.exp(z).sum(dim=2, keepdim=True)
        like_BG = probs_BGK[:,:,k_obs]  # (B,G) likelihoods of observed action
        
        # Bayesian update
        post_BG = like_BG * self.b_BG  # unnormalized posterior
        s = post_BG.sum()
        if s <= 1e-12:
            # Degenerate case - fall back to uniform
            post_BG = torch.ones_like(post_BG) / post_BG.numel()
        else:
            post_BG = post_BG / s
        
        self.b_BG = post_BG
        return post_BG
    
    def get_goal_marginals(self) -> torch.Tensor:
        """Get marginal probabilities over goals"""
        return self.b_BG.sum(dim=0)  # (G,)
    
    def get_beta_marginal(self) -> torch.Tensor:
        """Get marginal probabilities over beta values"""
        return self.b_BG.sum(dim=1)  # (B,)
    
    def get_beta_expectation(self) -> float:
        """Get expected beta value"""
        p_beta = self.get_beta_marginal()
        return (p_beta * self.beta_grid).sum().item()
    
    def get_most_likely_goal(self) -> int:
        """Get index of most likely goal"""
        p_g = self.get_goal_marginals()
        return p_g.argmax().item()
    
    @torch.no_grad()
    def predict_obstacle_actions(self, x: torch.Tensor, num_samples: int = None) -> torch.Tensor:
        """
        Predict obstacle actions using current belief
        
        Args:
            x: (2,) current obstacle state [px, py] for pedestrian
            num_samples: number of control sequences to sample (default from config)
        
        Returns:
            control_sequences: (num_samples, T) predicted control sequences (averaged control values)
        """
        if num_samples is None:
            num_samples = self.config.num_control_samples
        
        T = self.config.prediction_horizon
        control_sequences = torch.zeros((num_samples, T, 2), device=self.device, dtype=self.dtype)  # Store [heading, velocity]
        
        # Sample (beta, goal) pairs according to current belief
        flat_belief = self.b_BG.flatten()  # (B*G,)
        indices = torch.multinomial(flat_belief, num_samples, replacement=True)
        
        for i, idx in enumerate(indices):
            beta_idx = idx // self.G
            goal_idx = idx % self.G
            
            beta = self.beta_grid[beta_idx]
            goal = self.goals[goal_idx]
            
            # Generate control sequence for this (beta, goal) pair
            x_current = x.clone()  # (2,) [px, py]
            for t in range(T):
                # Get action distribution at current state
                action_probs = P_u_given_x_beta_g(x_current, self.U, beta, goal)
                
                # Sample action
                action_idx = torch.multinomial(action_probs, 1).item()
                action = self.U[action_idx]  # (2,) [vx, vy]
                
                # Convert [vx, vy] to [heading, velocity] for pedestrian dynamics
                vx, vy = action[0], action[1]
                heading = torch.atan2(vy, vx)
                velocity = torch.sqrt(vx**2 + vy**2 + 1e-8)  # avoid division by zero
                pedestrian_action = torch.stack([heading, velocity], dim=0)
                
                # Store both heading and velocity
                control_sequences[i, t] = pedestrian_action  # [heading, velocity]
                
                # Update state for next timestep using pedestrian dynamics
                x_current = self.dynamics.step(x_current, pedestrian_action)
        
        return control_sequences
    
    def average_control_sequences(self, control_sequences: torch.Tensor) -> torch.Tensor:
        """
        Average control sequences over horizon T for NN input
        
        Args:
            control_sequences: (num_samples, T, 2) control sequences [heading, velocity]
        
        Returns:
            averaged_controls: (num_samples, 2) averaged control values [heading, velocity]
        """
        return control_sequences.mean(dim=1)  # (num_samples, 2)
    
    def get_trajectories(self, x: torch.Tensor, control_sequences: torch.Tensor) -> torch.Tensor:
        """
        Get trajectories from control sequences
        
        Args:
            control_sequences: (num_samples, T, 2) control sequences [heading, velocity]
        Returns:
            trajectories: (num_samples, T, 2) trajectories [px, py]
        """
        trajectories = torch.zeros((control_sequences.shape[0], control_sequences.shape[1]+1, 2), device=self.device, dtype=self.dtype)
        trajectories[:, 0] = x.unsqueeze(0)
        for i in range(control_sequences.shape[0]):
            for t in range(control_sequences.shape[1]):
                trajectories[i, t+1] = self.dynamics.step(trajectories[i, t], control_sequences[i, t])
        return trajectories
    
    def get_prediction_particles_avg(self, x: torch.Tensor, num_samples: int = None) -> torch.Tensor:
        """
        Get averaged control particles for MPPI planner
        
        Args:
            x: (2,) current obstacle state [px, py] for pedestrian
            num_samples: number of particles to generate
        
        Returns:
            particles: (num_samples, 2) averaged control values [heading, velocity] for pedestrian dynamics
        """
        control_sequences = self.predict_obstacle_actions(x, num_samples)
        trajectories = self.get_trajectories(x, control_sequences)
        return self.average_control_sequences(control_sequences), trajectories

    def get_prediction_particles(self, x: torch.Tensor, num_samples: int = None) -> torch.Tensor:
        """
        Get control particles for MPPI planner
        
        Args:
            x: (2,) current obstacle state [px, py] for pedestrian
            num_samples: number of particles to generate
        
        Returns:
            particles: (num_samples, 2) averaged control values [heading, velocity] for pedestrian dynamics
        """
        control_sequences = self.predict_obstacle_actions(x, num_samples)
        trajectories = self.get_trajectories(x, control_sequences)
        return control_sequences, trajectories
    
    @torch.no_grad()
    def get_most_likely_particles(self, x: torch.Tensor, num_samples: int = None) -> tuple:
        """
        Generate P particles by sampling actions from the most likely (beta, goal) pair
        
        Args:
            x: (2,) current obstacle state [px, py] for pedestrian
            num_samples: number of particles to generate (default from config)
        
        Returns:
            control_sequences: (num_samples, T, 2) control sequences [heading, velocity]
            trajectories: (num_samples, T+1, 2) trajectories [px, py]
        """
        if num_samples is None:
            num_samples = self.config.num_control_samples
        
        # Find (beta, goal) pair with maximum probability
        max_idx = torch.argmax(self.b_BG.flatten())
        beta_idx = max_idx // self.G
        goal_idx = max_idx % self.G
        
        beta = self.beta_grid[beta_idx]
        goal = self.goals[goal_idx]
        
        T = self.config.prediction_horizon
        control_sequences = torch.zeros((num_samples, T, 2), device=self.device, dtype=self.dtype)
        trajectories = torch.zeros((num_samples, T + 1, 2), device=self.device, dtype=self.dtype)
        
        # Generate num_samples particles using the most likely (beta, goal)
        for i in range(num_samples):
            trajectories[i, 0] = x
            x_current = x.clone()
            
            for t in range(T):
                # Get action distribution at current state
                action_probs = P_u_given_x_beta_g(x_current, self.U, beta, goal)
                
                # Sample action from the distribution
                action_idx = torch.multinomial(action_probs, 1).item()
                action = self.U[action_idx]  # (2,) [vx, vy]
                
                # Convert [vx, vy] to [heading, velocity] for pedestrian dynamics
                vx, vy = action[0], action[1]
                heading = torch.atan2(vy, vx)
                velocity = torch.sqrt(vx**2 + vy**2 + 1e-8)
                pedestrian_action = torch.stack([heading, velocity], dim=0)
                
                # Store control
                control_sequences[i, t] = pedestrian_action
                
                # Update state for next timestep
                x_current = self.dynamics.step(x_current, pedestrian_action)
                trajectories[i, t + 1] = x_current
        
        return control_sequences, trajectories

# ====== CONVENIENCE FUNCTIONS ======
def create_intent_updater(goals: List[List[float]], 
                         beta_grid: List[float] = None,
                         prediction_horizon: int = 4,
                         dt: float = 0.25,
                         device: str = None) -> BayesianIntentUpdater:
    """
    Create and configure intent updater
    
    Args:
        goals: List of 3 goals [[x1,y1], [x2,y2], [x3,y3]]
        beta_grid: List of beta values (default: [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        prediction_horizon: T = 4 time steps
        dt: time step = 0.25
        device: device to use
    
    Returns:
        configured BayesianIntentUpdater
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = IntentConfig(
        goals=goals,
        beta_grid=beta_grid,
        prediction_horizon=prediction_horizon,
        dt=dt,
        device=device
    )
    
    return BayesianIntentUpdater(config)

# ====== EXAMPLE USAGE ======
if __name__ == "__main__":
    # Example: 3 goals for obstacle vehicle
    goals = [[8.0, 0.0], [-8.0, 0.0], [0.0, 8.0]]
    
    # Create intent updater
    intent_updater = create_intent_updater(goals)
    
    # Simulate some observations
    x_obs = torch.tensor([0.0, -6.0, 0.0], device=intent_updater.device)
    
    print("Initial belief:")
    print(f"Goal marginals: {intent_updater.get_goal_marginals()}")
    print(f"Beta expectation: {intent_updater.get_beta_expectation():.3f}")
    
    # Simulate observations toward goal 0
    for t in range(5):
        # Generate synthetic observation (toward first goal)
        u_obs = (goals[0] - x_obs[:2].cpu().numpy())
        u_obs = u_obs / (np.linalg.norm(u_obs) + 1e-9)  # normalize
        u_obs = torch.tensor(u_obs, device=intent_updater.device)
        
        # Update belief
        intent_updater.update_belief(x_obs, u_obs)
        
        # Get predictions
        particles = intent_updater.get_prediction_particles(x_obs, num_samples=10)
        
        print(f"t={t}: Goal marginals: {intent_updater.get_goal_marginals()}")
        print(f"      Beta expectation: {intent_updater.get_beta_expectation():.3f}")
        print(f"      Prediction particles: {particles[:5].cpu().numpy()}")
        
        # Update obstacle state (simplified)
        x_obs = intent_updater.dynamics.step(x_obs, 0.0)  # no control for simplicity