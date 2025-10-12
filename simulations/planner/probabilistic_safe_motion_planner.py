from dataclasses import dataclass
from typing import Callable, Optional, Union
import torch
from torch import Tensor
from utils_exp.dynamics import Dubins4D, Pedestrian, heading_to_goal_u_nom
from utils_exp.intent_prediction_Pedestrian import BayesianIntentUpdater

# Special return value for infeasible cases
INFEASIBLE = "INFEASIBLE"

def compute_collision_probability(x_next: Tensor, obs_particles: Tensor, obs_R: float) -> Tensor:
    """
    Compute collision probability for each trajectory over particles P
    
    Args:
        x_next: (K, 2) next positions for K trajectories
        obs_particles: (P, 2) obstacle particle positions at timestep t
        obs_R: safe distance from obstacle
    
    Returns:
        collision_probs: (K,) collision probability for each trajectory
    """
    P, _ = obs_particles.shape  # obs_particles is already (P, 2)
    K, _ = x_next.shape
    
    # Expand for broadcasting: x_next (K, 1, 2), obs_particles (1, P, 2)
    x_next_expanded = x_next[:, None, :]  # (K, 1, 2)
    obs_particles_expanded = obs_particles[None, :, :]  # (1, P, 2)
    
    # Compute distances: (K, P)
    distances_squared = ((x_next_expanded - obs_particles_expanded) ** 2).sum(dim=2)
    
    # Check collision: (K, P) - True if particle is within collision radius
    collision_mask = distances_squared <= obs_R ** 2
    
    # Compute collision probability for each trajectory: (K,)
    collision_probs = collision_mask.float().mean(dim=1)
    
    return collision_probs


@dataclass  
class Prob_Safe_Motion_Planner:
    """
        Probabilistic safe motion planner in Probabilistically Safe Robot Planning with Confidence-Based Human Predictions
        Solve it by sampling-based MPC.
    """
    sys_ego: Dubins4D
    sys_obs: Pedestrian
    intent_updater: BayesianIntentUpdater  # Intent prediction system
    horizon: int = 25
    dt: float = 0.25
    num_samples: int = 256
    lambda_temp: float = 4.0    # sample weight
    u_sigma: float = 0.4
    particle_count: int = 64
    alpha: float = 0.1          # CVaR level for lower tail
    collision_threshold: float = 0.1  # Collision probability threshold for rejection
    control_weights: torch.Tensor = None  # Control weight vector [w_omega, w_velocity]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    var_grid_history = None # to avoid error when calling this

    def __post_init__(self):
        self.U = torch.zeros(self.horizon, 2, device=self.device, dtype=self.dtype)
        self.U[:,1] = (self.sys_ego.a_max - self.sys_ego.a_min) * 0.5
        
        # Initialize control weights if not provided
        if self.control_weights is None:
            self.control_weights = torch.tensor([1.0, 1.0], device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def rollout_cost_batch(self, x0: Tensor, U_batch: Tensor,
                        goal_xy: Tensor, obs_center_nom: Tensor, obs_R: float) -> Tensor:
        """
        Simulate K rollouts in parallel.
        x0: (3,)
        U_batch: (K,H)  candidate control sequences
        goal_xy: (2,)
        obs_center_nom: (2,)
        obs_R: float # safe distance from obstacle
        returns: (K,)  cost for each rollout
        """
        K, H, _ = U_batch.shape 
        sys_ego, P = self.sys_ego, self.particle_count
        sys_obs = self.sys_obs

        # Get intent-based obstacle control predictions
        u_obs_particles, self.obs_particles_trajectories = self.intent_updater.get_prediction_particles(
            obs_center_nom, num_samples=P
        )  # (P,) averaged control values
        # Convert to proper format for CBVF
        # u_obs_samples = u_obs_particles.unsqueeze(-1)  # (P, 1)
        self.u_obs_samples = u_obs_particles
        # initial states expanded: (K,3)
        x = x0.expand(K, -1).clone()
        total = torch.zeros(K, device=self.device, dtype=self.dtype)
        
        # Track which trajectories are still active (not rejected)
        active_mask = torch.ones(K, dtype=torch.bool, device=self.device)

        for t in range(H):
            u = U_batch[:, t]   # (K, 2) - [yaw_rate, velocity] controls

            # Only process active trajectories
            if active_mask.any():
                # stage cost: goal distance + control effort
                x_current = x  # (K,3) - full state for Dubins3D
                x_next = sys_ego.step(x_current, u)  # (K,3)


                if t == 0:
                    # distance-basd cost for states 
                    u_nom = heading_to_goal_u_nom(x_current, goal_xy)
                    # Weighted control cost: (u - u_nom)^T * W * (u - u_nom)
                    diff = u - u_nom  # (K, 2)
                    cost_u = ((diff ** 2) * self.control_weights).sum(dim=-1)  # (K,)
                    total += cost_u
                    # total += ((x_next[:, :2] - goal_xy) ** 2).sum(-1)*10.0  # + 0.1 * (u * u)

                # reject trajectories that prob. collide with obstacle geq to threshold
                # Compute collision probability ONLY for active trajectories
                obs_particles = self.obs_particles_trajectories[:,t+1,:2] # (P, T, d) -> (P, 2)
                
                # Only compute collision probabilities for active trajectories
                active_x_next = x_next[active_mask, :2]  # (active_K, 2) - only position coordinates
                active_collision_probs = compute_collision_probability(
                    active_x_next, obs_particles, obs_R
                )  # (active_K,)

                # Reject trajectories with collision probability >= threshold (strictly)
                active_reject_mask = active_collision_probs >= self.collision_threshold
                
                # Map back to full trajectory indices
                reject_mask = torch.zeros(K, dtype=torch.bool, device=self.device)
                reject_mask[active_mask] = active_reject_mask
                
                # Set rejected trajectories to a very high cost
                # This ensures they get very low weight in MPPI update
                total[reject_mask] = 1e6
                
                # Update active mask: remove rejected trajectories
                active_mask = active_mask & ~reject_mask
                
                # If all trajectories are rejected, break early
                if not active_mask.any():
                    break
                
            x = x_next  # update state (K,3)

        return total  # (K,)


    @torch.no_grad()
    def optimize(self, x0: Tensor, goal_xy: Tensor, obs_center_nom: Tensor, obs_R: float, iters: int = 1) -> Union[Tensor, str]:
        """
        Batch-parallel MPPI optimization with hard constraint: returns Infeasible if no feasible trajectories found.
        """
        H, K = self.horizon, self.num_samples
        lam = max(1e-6, self.lambda_temp)
        
        # Track if we've found any feasible trajectories
        feasible_found = False

        for iter_idx in range(iters):
            # sample perturbations: (K,H,2) for 2D controls [yaw_rate, velocity]
            eps = torch.randn(K, H, 2, device=self.device, dtype=self.dtype) * self.u_sigma
            # candidate controls: (K,H,2)
            U_batch = torch.clamp(self.U[None, :] + eps, 
                torch.tensor([self.sys_ego.omega_min, self.sys_ego.a_min], device=self.device, dtype=self.dtype),  # CUDA
                torch.tensor([self.sys_ego.omega_max, self.sys_ego.a_max], device=self.device, dtype=self.dtype))  # CUDA

            # rollout all candidates in batch
            costs = self.rollout_cost_batch(x0, U_batch, goal_xy, obs_center_nom, obs_R)  # (K,)

            # Check if any trajectories are feasible (cost < 1e6 means not rejected)
            feasible_mask = costs < 1e6
            if feasible_mask.any():
                feasible_found = True
                
                # Only use feasible trajectories for MPPI update
                feasible_costs = costs[feasible_mask]
                feasible_eps = eps[feasible_mask]
                
                # importance weights for feasible trajectories only
                c_min = feasible_costs.min()
                w_feasible = torch.exp(-(feasible_costs - c_min) / lam)
                W_feasible = w_feasible.sum() + 1e-12
                
                # weighted update using only feasible trajectories
                # dU = (w_feasible[:, None] * feasible_eps).sum(dim=0) / W_feasible  # (H,)
                dU = (w_feasible[:, None, None] * feasible_eps).sum(dim=0) / W_feasible  # (H, 2)
                self.U = torch.clamp(self.U + dU, 
                    torch.tensor([self.sys_ego.omega_min, self.sys_ego.a_min], device=self.device, dtype=self.dtype),  # CUDA
                    torch.tensor([self.sys_ego.omega_max, self.sys_ego.a_max], device=self.device, dtype=self.dtype)) 
            else:
                # No feasible trajectories found in this iteration
                # Continue to next iteration to try again
                continue

        # # Hard constraint: if no feasible trajectories found after all iterations
        if not feasible_found:
            print("WARNING: No feasible trajectories found within collision probability threshold!")
        #     return INFEASIBLE

        return self.U.clone()

    def get_obstacle_trajectory(self) -> Tensor:
        """
        Get obstacle trajectory from control sequences
        """
        try:
            return self.obs_particles_trajectories
        except:
            print("Have not generated trajectories yet")
            return None

    def update_intent(self, obs_state: Tensor, obs_action: Tensor):
        """
        Update intent beliefs based on observed obstacle action
        
        Args:
            obs_state: (3,) obstacle state [px, py, theta]
            obs_action: (2,) observed obstacle action [vx, vy]
        """
        self.intent_updater.update_belief(obs_state, obs_action)
        
    def get_intent_info(self):
        """
        Get current intent information for monitoring
        """
        goal_marginals = self.intent_updater.get_goal_marginals()
        beta_expectation = self.intent_updater.get_beta_expectation()
        most_likely_goal = self.intent_updater.get_most_likely_goal()
        
        return {
            'goal_marginals': goal_marginals.cpu().numpy(),
            'beta_expectation': beta_expectation,
            'most_likely_goal': most_likely_goal
        }
    def get_control_info(self, x: Tensor):
        """
        Get control information for the intent-aware MPPI planner
        """
        return self.intent_updater.get_prob_control_inputs_all(x)
