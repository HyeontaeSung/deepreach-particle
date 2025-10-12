"""
Configuration file for intent-aware planning experiments
"""
from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class ExperimentConfig:
    """Configuration for intent-aware planning experiments"""
    
    # Environment setup
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # === PLANNER SELECTION ===
    planner_type: str = "MPPI"  # Options: "MPPI", "Probabilistic"
    
    # Dynamics model selection (ONLY used for MPPI planner)
    dynamics_type: Optional[str] = "Predict_TV"  # Options: "Predict_TV", "Worst" (MPPI only)
    experiment_dir: Optional[str] = "../runs/Dubins4D_Human_Predict_TV_MPC12/"
    
    # Obstacle goals
    obstacle_goals: List[List[float]] = None  # [[x1,y1], [x2,y2], ...]
    target_goal: List[float] = None  # Which goal obstacle actually moves toward
    
    # Initial conditions
    ego_init: List[float] = None  # [x, y, theta, v]
    obs_init: List[float] = None  # [x, y]
    ego_goal: List[float] = None  # [x, y]
    obs_radius: float = 0.5
    
    # Time parameters
    T_max: int = 50  # Maximum simulation steps
    dt_plan: float = 0.1  # Planning dt
    dt_exec: float = 0.1   # Execution dt
    
    # Intent prediction config
    beta_grid: List[float] = None
    prediction_horizon: int = 15

    # animation parameters
    fps: int = 10 # frames per second
    
    # === PLANNER PARAMETERS ===
    # Common parameters (both planners)
    horizon: int = 1
    num_samples: int = 512
    lambda_temp: float = 4.0
    u_sigma: float = 3.0
    particle_count: int = 256
    alpha: float = 0.05
    iters: int = 1
    
    # MPPI-specific parameters
    gamma: float = 3.0
    cvar_weight: float = 1000.0
    
    # Probabilistic planner-specific parameters
    collision_threshold: float = 0.05
    
    # Data collection
    collect_data: bool = True
    save_plots: bool = True
    plot_interval: int = 20  # Print progress every N steps
    
    def __post_init__(self):
        """Set defaults if not provided"""
        if self.obstacle_goals is None:
            self.obstacle_goals = [[-2.0, 0.0], [0.0, -2.0], [0.0, 2.0]]
            # self.obstacle_goals = [[-1.0, 2.0], [1.0, 2.0]]
            # self.obstacle_goals = [ [0.0, -2.0], [0.0, 2.0]]

        
        if self.target_goal is None:
            # self.target_goal = self.obstacle_goals[0]
            self.target_goal = [-2.0, 0.0]
        
        if self.ego_init is None:
            self.ego_init = [-2.0, 0.0, 0.0, 2.0]
        
        if self.obs_init is None:
            self.obs_init = [2.0, 0.0]
            # self.obs_init = [0.0, -1.0]

        if self.ego_goal is None:
            self.ego_goal = [2.0, 0.0]
        
        if self.beta_grid is None:
            self.beta_grid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        
        # Map dynamics type to directory (only for MPPI)
        if self.planner_type == "MPPI":
            if self.dynamics_type == "Worst":
                self.experiment_dir = "../runs/Dubins4D_Human_Worst/"
            elif self.dynamics_type == "Predict_TV":
                self.experiment_dir = "../runs/Dubins4D_Human_Predict_TV_MPC12/"
            else:
                raise ValueError(f"MPPI planner requires dynamics_type 'Predict_TV' or 'Worst', got: {self.dynamics_type}")
        else:
            # Probabilistic planner doesn't use dynamics model
            self.dynamics_type = None
            self.experiment_dir = None


# === PRESET CONFIGURATIONS ===
def get_config_mppi_particle():
    """MPPI planner with Predict_TV (particle) dynamics"""
    return ExperimentConfig(
        planner_type="MPPI",
        dynamics_type="Predict_TV",
        # target_goal=[-2.0, 0.0],
        T_max=40,
        cvar_weight=1000.0,
        particle_count=256
    )

def get_config_mppi_worst():
    """MPPI planner with worst-case dynamics"""
    return ExperimentConfig(
        planner_type="MPPI",
        dynamics_type="Worst",
        # target_goal=[-2.0, 0.0],
        particle_count=1,
        T_max=40,
        cvar_weight=1000.0
    )

def get_config_probabilistic():
    """Probabilistic safe motion planner (no dynamics model)"""
    return ExperimentConfig(
        planner_type="Probabilistic",
        dynamics_type=None,  # Not used
        # target_goal=[-2.0, 0.0],
        T_max=40,
        collision_threshold=0.05,
        u_sigma=4.0,  # Probabilistic planner typically uses higher sigma
        dt_plan=0.1,
        dt_exec=0.1,
    )

def get_config_custom(planner_type="MPPI", dynamics_type="Predict_TV", 
                     target_goal=None, T_max=50, **kwargs):
    """
    Create custom configuration
    
    Args:
        planner_type: "MPPI" or "Probabilistic"
        dynamics_type: "Predict_TV" or "Worst" (ignored if planner_type="Probabilistic")
        target_goal: Goal for obstacle
        T_max: Maximum simulation steps
        **kwargs: Additional parameters
    """
    config = ExperimentConfig(
        planner_type=planner_type,
        dynamics_type=dynamics_type if planner_type == "MPPI" else None,
        target_goal=target_goal,
        T_max=T_max,
        **kwargs
    )
    return config


# === QUICK ACCESS ALIASES ===
# Keep backward compatibility with old names
get_config_predict_tv = get_config_mppi_particle
get_config_worst_case = get_config_mppi_worst