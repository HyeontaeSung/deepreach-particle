from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List
from torch import Tensor
from utils_exp.intent_prediction_Pedestrian import BayesianIntentUpdater, create_intent_updater, IntentConfig
from utils_exp.dynamics import Dubins4D
from utils_exp.dynamics import Pedestrian
import torch
from utils_exp.util import var_lower_tail
from utils_exp.util import to_tensor, angle_wrap
from utils_exp.dynamics import Dubins4D, Pedestrian, heading_to_goal_u_nom

@dataclass
class MPPI_CVaR_CBVF_IntentAware:
    """
    MPPI planner with intent-aware obstacle predictions
    Integrates Bayesian intent inference with CVaR-CBVF safety constraints
    """
    sys_ego: Dubins4D
    sys_obs: Pedestrian
    intent_updater: BayesianIntentUpdater  # Intent prediction system
    horizon: int = 25
    num_samples: int = 256
    lambda_temp: float = 4.0    # sample weight
    u_sigma: float = 0.4
    particle_count: int = 64
    alpha: float = 0.1          # CVaR level for lower tail
    control_weights: torch.Tensor = None  # Control weight vector [w_omega, w_velocity]

    gamma: float = 0.2          # dCBF progress gain
    k_nom: float = 2.5         # P gain for nominal control
    cvar_weight: float = 30.0   # penalty weight
    cvar_softplus_beta: float = 30.0
    obs_noise_std: float = 0.001

    collect_data: bool = True
    most_likely: bool = False # use most likely particles or particles

    cbvf_dvdt_dvds: Optional[
        Callable[[Tensor, Tensor, Tensor, float], Tuple[Tensor, Tensor, Tensor]]
    ] = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.U = torch.zeros(self.horizon, 2, device=self.device, dtype=self.dtype)
        self.U[:,1] = (self.sys_ego.a_max - self.sys_ego.a_min) * 0.5

        # Initialize control weights if not provided
        # TODO: dimension should be same as the control input
        if self.control_weights is None:
            self.control_weights = torch.tensor([1.0, 1.0], device=self.device, dtype=self.dtype)
        
        # Initialize data collection for CVaR plotting
        self.var_history = []
        self.control_history = []
        self.var_grid_history = []

    @torch.no_grad()
    def rollout_cost_batch(self, x0: Tensor, U_batch: Tensor,
                        goal_xy: Tensor, obs_center_nom: Tensor, obs_R: float) -> Tensor:
        """
        Simulate K rollouts in parallel with intent-aware obstacle predictions
        """
        K, H, _ = U_batch.shape 
        sys_ego, P = self.sys_ego, self.particle_count
        sys_obs = self.sys_obs

        # Get intent-based obstacle control predictions
        if self.most_likely == False:
            u_obs_particles, self.obs_particles_trajectories = self.intent_updater.get_prediction_particles(
                obs_center_nom, num_samples=P
        )
        else: 
            u_obs_particles, self.obs_particles_trajectories = self.intent_updater.get_most_likely_particles(
                obs_center_nom, num_samples=P
            )  
        # Convert to proper format for CBVF
        # u_obs_samples = u_obs_particles.unsqueeze(-1)  # (P, 1)
        self.u_obs_samples = self.get_particle_first_last_control_inputs(u_obs_particles)

        # initial states expanded: (K,3)
        x = x0.expand(K, -1).clone()
        total = torch.zeros(K, device=self.device, dtype=self.dtype)
        for t in range(H):
            u = U_batch[:, t]
            u_nom = heading_to_goal_u_nom(x, goal_xy)
            # Weighted control cost: (u - u_nom)^T * W * (u - u_nom)
            diff = u - u_nom  # (K, 2)
            cost_u = ((diff ** 2) * self.control_weights).sum(dim=-1)  # (K,)
            total += cost_u
            # stage cost: control distance
            if P > 1: # particle method
                # Use intent-based obstacle control particles
                centers = obs_center_nom.to(self.device, self.dtype).view(1, 1, -1).expand(K, P, -1)
                # barrier values, dvdt, dvds: (K,P)
                value, dvdt, dvds = self.cbvf_dvdt_dvds(
                    x[:, None, :].expand(-1, P, -1), 
                    centers, 
                    self.u_obs_samples.unsqueeze(0).expand(K, -1, -1), 
                    obs_R
                )

                # f of Dubins
                f_ego = sys_ego.f(x[:, None, :], u[:, None, :])    # (K, 1, 3)
                # u_b = u.view(-1,1,1)                                    # (K, 1, 1)

                f_ego = f_ego.repeat(1, P, 1)                     # (K,P,3) - repeat to match P particles
                # f of Pedestrian
                f_obs = sys_obs.f(centers[:,:1,:], self.u_obs_samples.unsqueeze(0).expand(K, -1, -1))        # (K, P, 2)
                flow = torch.cat([f_ego, f_obs], dim=-1)          # (K,P,6)

                adv = (dvds * flow).sum(dim=-1)                         # (K,P)  == ∇V·(f)
                m = dvdt + adv + self.gamma * value                      # (K,P)
                
                # CVaR over lower tail (vectorized)

                cvar_lo = var_lower_tail(m, alpha=self.alpha)  # (K,)
                # total += self.cvar_weight * softplus(-cvar_lo, beta=self.cvar_softplus_beta)
                total += self.cvar_weight * torch.relu(-cvar_lo)
                # total += self.cvar_weight * (-cvar_lo)                
            else: # worst-case
                # Use intent-based obstacle control particles
                centers = obs_center_nom.to(self.device, self.dtype).view(1, 1, -1).expand(K, P, -1)
                
                # barrier values, dvdt, dvds: (K,P)
                value, dvdt, dvds = self.cbvf_dvdt_dvds(
                    x[:, None, :].expand(-1, P, -1), 
                    centers, 
                    self.u_obs_samples.unsqueeze(0).expand(K, -1, -1), 
                    obs_R
                )
                
                # f of Dubins
                f_ego = sys_ego.f(x[:, None, :], u[:, None, :])    # (K, 1, 3)
                # u_b = u.view(-1,1,1)                                    # (K, 1, 1)

                f_ego = f_ego.repeat(1, P, 1)                     # (K,P,3) - repeat to match P particles
                # f of Pedestrian
                worst_heading_human = torch.atan2(-dvds[...,5], -dvds[...,4])
                worst_v_human = torch.full_like(worst_heading_human, self.sys_obs.v_max)
                worst_u_human = torch.stack([worst_heading_human, worst_v_human], dim=-1)
                f_obs = sys_obs.f(centers[:,:1,:], worst_u_human.expand(K, -1, -1))        # (K, P, 2)
                flow = torch.cat([f_ego, f_obs], dim=-1)          # (K,P,6)

                adv = (dvds * flow).sum(dim=-1)                         # (K,P)  == ∇V·(f)
                m = dvdt + adv + self.gamma * value                      # (K,P)
                
                # CVaR over lower tail (vectorized)

                cvar_lo = var_lower_tail(m, alpha=self.alpha)  # (K,)
                # total += self.cvar_weight * softplus(-cvar_lo, beta=self.cvar_softplus_beta)
                total += self.cvar_weight * torch.relu(-cvar_lo)
                # total += self.cvar_weight * (-cvar_lo)

        return total, u_nom  # (K,)

    @torch.no_grad()
    def optimize(self, x0: Tensor, goal_xy: Tensor, obs_center_nom: Tensor, obs_R: float, iters: int = 1) -> Tensor:
        """
        Batch-parallel MPPI optimization with intent-aware predictions
        """
        H, K = self.horizon, self.num_samples
        lam = max(1e-6, self.lambda_temp)
        for _ in range(iters):
            # sample perturbations: (K,H)
            eps = torch.randn(K, H, 2, device=self.device, dtype=self.dtype) * self.u_sigma
            # candidate controls: (K,H)
            U_batch = torch.clamp(self.U[None, :] + eps, 
                torch.tensor([self.sys_ego.omega_min, self.sys_ego.a_min], device=self.device, dtype=self.dtype),  # CUDA
                torch.tensor([self.sys_ego.omega_max, self.sys_ego.a_max], device=self.device, dtype=self.dtype))  # CUDA

            # rollout all candidates in batch
            costs, u_nom = self.rollout_cost_batch(x0, U_batch, goal_xy, obs_center_nom, obs_R)  # (K,)
            # importance weights
            c_min = costs.min()

            w = torch.exp(-(costs - c_min) / lam)  # (K,)
            W = w.sum() + 1e-12
           
            # weighted update
            dU = (w[:, None, None] * eps).sum(dim=0) / W  # (H,)

            self.U = torch.clamp(self.U + dU, 
                torch.tensor([self.sys_ego.omega_min, self.sys_ego.a_min], device=self.device, dtype=self.dtype),  # CUDA
                torch.tensor([self.sys_ego.omega_max, self.sys_ego.a_max], device=self.device, dtype=self.dtype)) 
            cvar_lo = self.get_cvar_lo(x0, obs_center_nom, self.U[0], self.u_obs_samples, obs_R)
            
        # Collect data for CVaR plotting if enabled
        if self.collect_data:
            # self.var_history.append(cvar_lo.cpu().item())
            # self.control_history.append(self.U[0].cpu().clone())
            cvar_grid_data = self.compute_cvar_over_control_range(x0, obs_center_nom, obs_R, self.U[0], u_nom[0])
            self.var_grid_history.append(cvar_grid_data)

                
        return self.U.clone(), cvar_lo

    @torch.no_grad()
    def optimize_linspace(self, x0: Tensor, goal_xy: Tensor, obs_center_nom: Tensor, obs_R: float, iters: int = 1) -> Tensor:
        """
        Batch-parallel MPPI optimization with intent-aware predictions
        sample control space uniformly (linspace)
        """
        H, K = self.horizon, self.num_samples
        lam = max(1e-6, self.lambda_temp)

        # sample perturbations: (K,H)

        # Create a grid of control perturbations
        omega_samples = torch.linspace(self.sys_ego.omega_min, self.sys_ego.omega_max, 
                                    int(K**0.5), device=self.device, dtype=self.dtype)
        accel_samples = torch.linspace(self.sys_ego.a_min, self.sys_ego.a_max, 
                                    int(K**0.5), device=self.device, dtype=self.dtype)

        # Create meshgrid for all combinations
        omega_mesh, accel_mesh = torch.meshgrid(omega_samples, accel_samples, indexing='ij')

        # Flatten to get K samples
        omega_flat = omega_mesh.flatten()[:K]  # (K,)
        accel_flat = accel_mesh.flatten()[:K]  # (K,)

        # Create control sequences - repeat the same control over horizon
        U_batch = torch.zeros(K, H, 2, device=self.device, dtype=self.dtype)
        U_batch[:, :, 0] = omega_flat.unsqueeze(1)  # (K, H)
        U_batch[:, :, 1] = accel_flat.unsqueeze(1)  # (K, H)
        # rollout all candidates in batch
        costs, u_nom = self.rollout_cost_batch(x0, U_batch, goal_xy, obs_center_nom, obs_R)  # (K,)
        # === SIMPLE ARGMIN - Pick the best control ===
        best_idx = costs.argmin()
        self.U = U_batch[best_idx].clone()  # Simply pick the best control sequence
        
        # Get CVaR for the best control
        cvar_lo = self.get_cvar_lo(x0, obs_center_nom, self.U[0], self.u_obs_samples, obs_R)
            
                
        # Collect data for CVaR plotting if enabled
        if self.collect_data:
            # self.var_history.append(cvar_lo.cpu().item())
            # self.control_history.append(self.U[0].cpu().clone())
            cvar_grid_data = self.compute_cvar_over_control_range(x0, obs_center_nom, obs_R, self.U[0], u_nom[0])
            self.var_grid_history.append(cvar_grid_data)

                
        return self.U.clone(), cvar_lo

    @torch.no_grad()
    def get_cvar_lo(self, x0: Tensor, obs_center_nom: Tensor, u_opt: Tensor, particles: Tensor, obs_R: float) -> Tensor:
        """
        get final m: dvdt + adv + self.gamma * value
        """
        # barrier values, dvdt, dvds: (K,P)
        value, dvdt, dvds = self.cbvf_dvdt_dvds(
            x0, 
            obs_center_nom, 
            particles.unsqueeze(0).expand(1, -1, -1),
            obs_R
        )
        
        # f of Dubins
        f_ego = self.sys_ego.f(x0, u_opt)    # (K, 1, 3)
        # u_b = u.view(-1,1,1)                                    # (K, 1, 1)

        f_ego = f_ego.repeat(1, particles.shape[0], 1)                     # (K,P,3) - repeat to match P particles
        # f of Pedestrian
        f_obs = self.sys_obs.f(obs_center_nom, particles.unsqueeze(0).expand(1, -1, -1))        # (K, P, 2)
        flow = torch.cat([f_ego, f_obs], dim=-1)          # (K,P,6)

        adv = (dvds * flow).sum(dim=-1)                         # (K,P)  == ∇V·(f)
        m = dvdt + adv + self.gamma * value                      # (K,P)
        cvar_lo = var_lower_tail(m, alpha=self.alpha)

        return cvar_lo

    def get_obstacle_trajectory(self) -> Tensor:
        """
        Get obstacle trajectory from control sequences
        """
        try:
            return self.obs_particles_trajectories
        except:
            print("Have not generated trajectories yet")
            return None
    
    def get_particle_first_last_control_inputs(self, u_obs_particles: Tensor):
        """
        Extract the first and last time control inputs for each particle and concatenate them
        
        Args:
            u_obs_particles: (P, T, 2) where P is particles, T is time steps, 2 is control dim
            
        Returns:
            Tensor: (P, 1, 4) where P is particles, 1 is time dimension, 4 is concatenated controls
        """
        first_controls = u_obs_particles[:, 0, :]  # (P, 2) - first time step for each particle
        last_controls = u_obs_particles[:, -1, :]   # (P, 2) - last time step for each particle
        
        # Concatenate first and last controls along the last dimension
        concatenated = torch.cat([first_controls, last_controls], dim=-1)  # (P, 4)
        
        return concatenated


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
    
    def enable_cvar_data_collection(self):
        """Enable collection of CVaR data during optimization"""
        self.collect_data = True
        self.var_history = []
        self.control_history = []
        self.var_grid_history = []
    
    def disable_cvar_data_collection(self):
        """Disable collection of CVaR data during optimization"""
        self.collect_data = False
    
    def get_cvar_data(self):
        """Get collected CVaR data"""
        return {
            'cvar_values': self.var_history.copy(),
            'control_inputs': [ctrl.clone() for ctrl in self.control_history]
        }
    
    def clear_cvar_data(self):
        """Clear collected CVaR data"""
        self.var_history = []
        self.control_history = []
        self.var_grid_history = []
    
    def get_var_grid_history(self):
        """Get collected CVaR grid history"""
        return self.var_grid_history.copy()
    
    def extract_cvar_grids(self):
        """
        Extract just the CVaR grid values from var_grid_history
        
        Returns:
            list of numpy arrays: CVaR grids from each iteration
        """
        return [grid_data['cvar_grid'] for grid_data in self.var_grid_history]
    
    def get_cvar_grid_at_iteration(self, iteration):
        """
        Get CVaR grid data from a specific iteration
        
        Args:
            iteration: Iteration index (0-based)
            
        Returns:
            dict: Contains 'omega_grid', 'accel_grid', 'cvar_grid', etc.
        """
        if iteration < 0 or iteration >= len(self.var_grid_history):
            raise IndexError(f"Iteration {iteration} out of range [0, {len(self.var_grid_history)})")
        return self.var_grid_history[iteration]
    
    @torch.no_grad()
    def compute_cvar_over_control_range(self, x0: Tensor, obs_center_nom: Tensor, obs_R: float, u_opt: Tensor, u_nom: Tensor,
                                      omega_range=None, accel_range=None, num_points=25):
        """
        Compute CVaR values over a range of control inputs for plotting
        
        Args:
            x0: Current ego state
            obs_center_nom: Obstacle center position
            obs_R: Obstacle radius
            omega_range: (min, max) angular velocity range, if None uses system limits
            accel_range: (min, max) acceleration range, if None uses system limits  
            num_points: Number of points to sample in each dimension
            
        Returns:
            dict: Contains control grids and CVaR values
        """
        if omega_range is None:
            omega_range = (self.sys_ego.omega_min, self.sys_ego.omega_max)
        if accel_range is None:
            accel_range = (self.sys_ego.a_min, self.sys_ego.a_max)
            
        # Create control grids
        omega_grid = torch.linspace(omega_range[0], omega_range[1], num_points, device=self.device)
        accel_grid = torch.linspace(accel_range[0], accel_range[1], num_points, device=self.device)
        omega_mesh, accel_mesh = torch.meshgrid(omega_grid, accel_grid, indexing='ij')
        
        # Flatten for batch processing
        omega_flat = omega_mesh.flatten()
        accel_flat = accel_mesh.flatten()
        controls = torch.stack([omega_flat, accel_flat], dim=1)  # (num_points^2, 2)
        
        # TRUE PARALLEL COMPUTATION - Process controls in large batches
        P = self.u_obs_samples.shape[0]  # Number of particles
        total_controls = len(controls)
        
        # Process in batches to handle memory constraints
        batch_size = min(500, total_controls)  # Process up to 500 controls at once
        all_cvar_values = []
        
        # Pre-compute obstacle dynamics once (same for all ego controls)
        f_obs_base = self.sys_obs.f(
            obs_center_nom.unsqueeze(0), 
            self.u_obs_samples.unsqueeze(0)
        )  # (1, P, 2)
        
        for batch_start in range(0, total_controls, batch_size):
            batch_end = min(batch_start + batch_size, total_controls)
            batch_controls = controls[batch_start:batch_end]
            B = len(batch_controls)
            
            # Expand state and obstacle for batch with particle dimension
            # We need shape (B, P, ...) for proper broadcasting
            x0_batch = x0.unsqueeze(0).unsqueeze(0).expand(B, P, -1)  # (B, P, 4)
            obs_batch = obs_center_nom.unsqueeze(0).unsqueeze(0).expand(B, P, -1)  # (B, P, 2)
            particles_batch = self.u_obs_samples.unsqueeze(0).expand(B, -1, -1)  # (B, P, 4)
            
            # Compute CBVF for batch in parallel
            value, dvdt, dvds = self.cbvf_dvdt_dvds(
                x0_batch, 
                obs_batch, 
                particles_batch,
                obs_R
            )  # value, dvdt: (B, P), dvds: (B, P, 6)
            
            # Compute ego dynamics for batch in parallel
            f_ego_batch = torch.stack([self.sys_ego.f(x0, ctrl) for ctrl in batch_controls], dim=0)  # (B, 4)
            f_ego_batch = f_ego_batch.unsqueeze(1).expand(-1, P, -1)  # (B, P, 4)
            
            # Expand obstacle dynamics for batch
            f_obs_batch = f_obs_base.expand(B, -1, -1)  # (B, P, 2)
            
            # Concatenate flows for batch: (B, P, 6)
            flow = torch.cat([f_ego_batch, f_obs_batch], dim=-1)
            
            # Compute Lie derivative for batch in parallel: (B, P)
            adv = (dvds * flow).sum(dim=-1)
            
            # Compute m for batch: (B, P)
            m = dvdt + adv + self.gamma * value
            
            # Compute CVaR for batch (this is the only sequential part)
            for i in range(B):
                cvar_val = var_lower_tail(m[i:i+1], alpha=self.alpha)
                all_cvar_values.append(cvar_val.item())
        
        # Convert to tensor and reshape to grid
        cvar_grid = torch.tensor(all_cvar_values, device=self.device).reshape(num_points, num_points)
        
        return {
            'omega_grid': omega_mesh.cpu().numpy(),
            'accel_grid': accel_mesh.cpu().numpy(), 
            'cvar_grid': cvar_grid.cpu().numpy(),
            'omega_range': omega_range,
            'accel_range': accel_range,
            'u_opt': u_opt.cpu().numpy(),
            'u_nom': u_nom.cpu().numpy()
        }
    
    @torch.no_grad()
    def compute_cvar_over_ego_obstacle_controls(self, x0: Tensor, obs_center_nom: Tensor, obs_R: float,
                                              ego_omega_range=None, ego_accel_range=None, 
                                              obs_omega_range=None, obs_accel_range=None,
                                              num_points=20):
        """
        Compute CVaR values over both ego and obstacle control ranges
        
        Args:
            x0: Current ego state
            obs_center_nom: Obstacle center position  
            obs_R: Obstacle radius
            ego_omega_range: Ego angular velocity range
            ego_accel_range: Ego acceleration range
            obs_omega_range: Obstacle angular velocity range
            obs_accel_range: Obstacle acceleration range
            num_points: Number of points per dimension
            
        Returns:
            dict: Contains control grids and CVaR values
        """
        if ego_omega_range is None:
            ego_omega_range = (self.sys_ego.omega_min, self.sys_ego.omega_max)
        if ego_accel_range is None:
            ego_accel_range = (self.sys_ego.a_min, self.sys_ego.a_max)
        if obs_omega_range is None:
            obs_omega_range = (-1.0, 1.0)  # Obstacle heading range
        if obs_accel_range is None:
            obs_accel_range = (0.0, 1.0)   # Obstacle velocity range
            
        # Create ego control grids
        ego_omega_grid = torch.linspace(ego_omega_range[0], ego_omega_range[1], num_points, device=self.device)
        ego_accel_grid = torch.linspace(ego_accel_range[0], ego_accel_range[1], num_points, device=self.device)
        ego_omega_mesh, ego_accel_mesh = torch.meshgrid(ego_omega_grid, ego_accel_grid, indexing='ij')
        
        # Create obstacle control grids  
        obs_omega_grid = torch.linspace(obs_omega_range[0], obs_omega_range[1], num_points, device=self.device)
        obs_accel_grid = torch.linspace(obs_accel_range[0], obs_accel_range[1], num_points, device=self.device)
        obs_omega_mesh, obs_accel_mesh = torch.meshgrid(obs_omega_grid, obs_accel_grid, indexing='ij')
        
        # Flatten for processing
        ego_omega_flat = ego_omega_mesh.flatten()
        ego_accel_flat = ego_accel_mesh.flatten()
        obs_omega_flat = obs_omega_mesh.flatten()
        obs_accel_flat = obs_accel_mesh.flatten()
        
        # Create control pairs
        ego_controls = torch.stack([ego_omega_flat, ego_accel_flat], dim=1)  # (num_points^2, 2)
        obs_controls = torch.stack([obs_omega_flat, obs_accel_flat], dim=1)  # (num_points^2, 2)
        
        # Compute CVaR for each ego-obstacle control pair
        cvar_values = []
        
        for i in range(len(ego_controls)):
            ego_control = ego_controls[i]
            obs_control = obs_controls[i]
            
            # Create obstacle control samples (single sample for this specific obstacle control)
            obs_control_samples = obs_control.unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
            
            cvar_val = self.get_cvar_lo(x0, obs_center_nom, ego_control, obs_control_samples, obs_R)
            cvar_values.append(cvar_val.cpu().item())
        
        cvar_grid = torch.tensor(cvar_values, device=self.device).reshape(num_points, num_points)
        
        return {
            'ego_omega_grid': ego_omega_mesh.cpu().numpy(),
            'ego_accel_grid': ego_accel_mesh.cpu().numpy(),
            'obs_omega_grid': obs_omega_mesh.cpu().numpy(), 
            'obs_accel_grid': obs_accel_mesh.cpu().numpy(),
            'cvar_grid': cvar_grid.cpu().numpy(),
            'ego_omega_range': ego_omega_range,
            'ego_accel_range': ego_accel_range,
            'obs_omega_range': obs_omega_range,
            'obs_accel_range': obs_accel_range
        }

print("Intent-Aware MPPI Planner defined successfully!")



