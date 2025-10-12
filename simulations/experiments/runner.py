"""
Main experiment runner for intent-aware planning
"""
import time
import numpy as np
import torch
import pickle
import os
from typing import Optional, Tuple, List

from utils_exp.dynamics import Dubins4D, Pedestrian, heading_to_goal_u_nom_pedestrian
from planner.cbf import make_cbvf_fn_from_trained_model_continous_dubinsPedestrian
from planner.planner import MPPI_CVaR_CBVF_IntentAware
from planner.probabilistic_safe_motion_planner import Prob_Safe_Motion_Planner
from utils_exp.intent_prediction_Pedestrian import create_intent_updater
from utils_exp.util import to_tensor, metrics_min_distance
from utils_exp.plot import plot_intent_trajectories, animate_two_dubins_with_predictions, animate_min_value, animate_control_inputs, animate_cvar_landscape_evolution

from experiments.config import ExperimentConfig
from IPython.display import Image, display

import sys
sys.path.append(os.path.abspath(".."))
from dynamics import dynamics
from utils import modules

class IntentAwarePlanner:
    """Main class for running intent-aware planning experiments"""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize planner with configuration
        
        Args:
            config: ExperimentConfig object with all settings
        """
        self.config = config
        self.device = config.device
        torch.manual_seed(config.seed)
        
        # === ONLY LOAD DYNAMICS MODEL FOR MPPI ===
        if config.planner_type == "MPPI":
            self._load_dynamics_model()
        else:
            # Probabilistic planner doesn't need CBVF model
            self.cbvf_dvdt_dvds = None
        
        # Create intent updater (both planners need this)
        self._create_intent_updater()
        
        # Create planner
        self._create_planner()
        
        print(f"   IntentAwarePlanner initialized")
        print(f"   Planner: {config.planner_type}")
        if config.planner_type == "MPPI":
            print(f"   Dynamics: {config.dynamics_type}")
        print(f"   Target goal: {config.target_goal}")
        print(f"   Device: {self.device}")
    
    def _load_dynamics_model(self):
        """Load the CBVF dynamics model (ONLY for MPPI planner)"""
        config = self.config
        
        # === ADD VALIDATION ===
        if config.experiment_dir is None:
            raise ValueError(
                f"experiment_dir cannot be None for {config.planner_type} planner with dynamics {config.dynamics_type}"
            )
        
        # Load configuration
        config_path = os.path.join(config.experiment_dir, 'orig_opt.pickle')
        
        # === ADD FILE CHECK ===
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Make sure experiment_dir is correct: {config.experiment_dir}"
            )
        
        with open(config_path, 'rb') as f:
            orig_opt = pickle.load(f)
        
        # Create dynamics instance based on type
        if config.dynamics_type == "Predict_TV":
            self.dynamics_instance = dynamics.Dubins4D_Human_Predict_TV(
                set_mode=orig_opt.set_mode
            )
        elif config.dynamics_type == "Worst":
            self.dynamics_instance = dynamics.Dubins4D_Human_Worst(
                set_mode=orig_opt.set_mode
            )
        else:
            raise ValueError(f"Unknown dynamics type: {config.dynamics_type}")
        
        # Load model
        self.model = modules.SingleBVPNet(
            in_features=self.dynamics_instance.input_dim,
            out_features=1,
            type=orig_opt.model,
            mode=orig_opt.model_mode,
            final_layer_factor=1.,
            hidden_features=orig_opt.num_nl,
            num_hidden_layers=orig_opt.num_hl,
            periodic_transform_fn=self.dynamics_instance.periodic_transform_fn
        )
        self.model.to(self.device)
        
        # Load weights
        checkpoint_path = os.path.join(
            config.experiment_dir, 'training', 'checkpoints', 'model_final.pth'
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        # Create CBVF function
        self.cbvf_dvdt_dvds = make_cbvf_fn_from_trained_model_continous_dubinsPedestrian(
            self.model, self.dynamics_instance, time_embed=1.0, device=self.device
        )
        
        print(f"   Loaded CBVF model: {config.dynamics_type}")
    
    def _create_intent_updater(self):
        """Create intent prediction system"""
        config = self.config
        self.intent_updater = create_intent_updater(
            goals=config.obstacle_goals,
            beta_grid=config.beta_grid,
            prediction_horizon=config.prediction_horizon,
            dt=config.dt_plan,
            device=self.device
        )
    
    def _create_planner(self):
        """Create planner based on configuration"""
        config = self.config
        
        # Create dynamics systems
        self.sys_ego = Dubins4D(dt=config.dt_plan, device=self.device)
        self.sys_obs = Pedestrian(dt=config.dt_plan, device=self.device)
        
        # Execution dynamics (finer timestep)
        self.sys_ego_exec = Dubins4D(dt=config.dt_exec, device=self.device)
        self.sys_obs_exec = Pedestrian(dt=config.dt_exec, device=self.device)
        
        # === CREATE PLANNER BASED ON TYPE ===
        if config.planner_type == "MPPI":
            # === VALIDATE CBVF IS LOADED ===
            if self.cbvf_dvdt_dvds is None:
                raise ValueError(
                    "MPPI planner requires CBVF model, but it wasn't loaded. "
                    "Check that dynamics_type is set correctly."
                )
            
            self.planner = MPPI_CVaR_CBVF_IntentAware(
                sys_ego=self.sys_ego,
                sys_obs=self.sys_obs,
                intent_updater=self.intent_updater,
                horizon=config.horizon,
                num_samples=config.num_samples,
                lambda_temp=config.lambda_temp,
                u_sigma=config.u_sigma,
                particle_count=config.particle_count,
                alpha=config.alpha,
                gamma=config.gamma,
                cvar_weight=config.cvar_weight,
                cbvf_dvdt_dvds=self.cbvf_dvdt_dvds,
                collect_data=config.collect_data,
                device=self.device
            )
            self.planner_has_cvar = True
            
        elif config.planner_type == "Probabilistic":
            self.planner = Prob_Safe_Motion_Planner(
                sys_ego=self.sys_ego,
                sys_obs=self.sys_obs,
                intent_updater=self.intent_updater,
                horizon=config.prediction_horizon,
                dt=config.dt_plan,
                num_samples=config.num_samples,
                lambda_temp=config.lambda_temp,
                u_sigma=config.u_sigma,
                particle_count=config.particle_count,
                alpha=config.alpha,
                collision_threshold=config.collision_threshold,
                device=self.device
            )
            self.planner_has_cvar = False
            
        else:
            raise ValueError(f"Unknown planner type: {config.planner_type}")
    
    @torch.no_grad()
    def run(self) -> dict:
        """
        Run the planning experiment
        
        Returns:
            dict: Results containing trajectories, intent history, time-to-goal, etc.
        """
        config = self.config
        
        # Initialize states
        x = to_tensor(config.ego_init, self.device)
        goal = to_tensor(config.ego_goal, self.device)
        obs_center_nom = to_tensor(config.obs_init, self.device)
        
        # Storage
        traj = [x.cpu().numpy()]
        obs_traj = [obs_center_nom.cpu().numpy()]
        controls = []
        cvar_values = []
        intent_history = []
        update_times = []
        planning_times = []
        obs_traj_particles = []
        
        # === ADD TIME TRACKING ===
        goal_reached = False
        time_to_goal = None  # Will store the time when goal is reached
        steps_to_goal = None  # Will store the number of steps
        
        print(f"\n Starting simulation: T_max={config.T_max}")
        print(f"   Obstacle target: {config.target_goal}")
        
        # Main simulation loop
        for t in range(config.T_max):
            start_update = time.time()
            
            # === INTENT UPDATE ===
            if t > 0:
                obs_velocity = obs_center_nom[:2].cpu().numpy() - obs_center_nom_prev[:2].cpu().numpy()
                obs_velocity = obs_velocity / (np.linalg.norm(obs_velocity) + 1e-9)
                obs_velocity = torch.tensor(obs_velocity, device=self.device)
                self.planner.update_intent(obs_center_nom_prev, obs_velocity)
            
            update_times.append(time.time() - start_update)
            
            # === PLANNING ===
            start_planning = time.time()
            if config.planner_type == "MPPI":
                U, cvar_lo = self.planner.optimize(x, goal, obs_center_nom, config.obs_radius, iters=config.iters)
                cvar_values.append(cvar_lo.cpu().numpy())
            else:
                U = self.planner.optimize(x, goal, obs_center_nom, config.obs_radius, iters=config.iters)
            u0 = U[0]
            planning_times.append(time.time() - start_planning)
            
            # === EXECUTION ===
            x = self.sys_ego_exec.step(x, u0)
            traj.append(x.cpu().numpy())
            controls.append(u0.cpu().numpy())

            
            # Update obstacle
            obs_control = heading_to_goal_u_nom_pedestrian(obs_center_nom, config.target_goal)
            obs_center_nom_prev = obs_center_nom.clone()
            obs_center_nom = self.sys_obs_exec.step(obs_center_nom, obs_control)
            obs_traj.append(obs_center_nom.cpu().numpy())
            
            # Store data
            obs_particles = self.planner.get_obstacle_trajectory()
            if obs_particles is not None:
                obs_traj_particles.append(obs_particles.cpu().numpy())
            
            intent_info = self.planner.get_intent_info()
            intent_history.append(intent_info)
            
            # Shift horizon
            self.planner.U[:-1] = self.planner.U[1:].clone()
            self.planner.U[-1] = 0.0
            
            # === CHECK TERMINATION AND RECORD TIME ===
            dist_to_goal = torch.linalg.vector_norm(x[:2] - goal)
            if dist_to_goal < 0.4 and not goal_reached:
                goal_reached = True
                steps_to_goal = t + 1
                time_to_goal = (t + 1) * config.dt_exec
                print(f" Goal reached at step {steps_to_goal} (time: {time_to_goal:.2f}s)")
                break
            
            # # Print progress
            # if t % config.plot_interval == 0:
            #     print(f"t={t:3d}: Dist to goal: {dist_to_goal:.3f}, "
            #         f"Goal marginals: {intent_info['goal_marginals']}")
            #     print(f"        Beta: {intent_info['beta_expectation']:.3f}, "
            #         f"Likely goal: {intent_info['most_likely_goal']}")
        
        # === HANDLE CASE WHERE GOAL NOT REACHED ===
        if not goal_reached:
            final_dist = torch.linalg.vector_norm(x[:2] - goal).item()
            print(f"  Goal not reached within {config.T_max} steps")
            print(f"   Final distance to goal: {final_dist:.3f}m")
            steps_to_goal = config.T_max
            time_to_goal = config.T_max * config.dt_exec
        
        # Compute results
        tr = np.array(traj)
        obs_tr = np.array(obs_traj)
        min_dist = metrics_min_distance(tr, obs_tr)
        
        print(f"\n Simulation Complete!")
        print(f"   Steps: {t+1}")
        print(f"   Time to goal: {time_to_goal:.2f}s")  # ← NEW
        print(f"   Steps to goal: {steps_to_goal}")     # ← NEW
        print(f"   Min distance to obstacle: {min_dist:.3f}m")
        print(f"   Avg update time: {np.mean(update_times):.4f}s")
        print(f"   Avg planning time: {np.mean(planning_times):.4f}s")
        
        # Create results dictionary
        results = {
            'trajectory': tr,
            'obs_trajectory': obs_tr,
            'controls': np.array(controls),
            'cvar_values': cvar_values,
            'intent_history': intent_history,
            'obs_particles': obs_traj_particles,
            'var_grid_history': self.planner.var_grid_history,
            'min_distance': min_dist,
            'update_times': update_times,
            'planning_times': planning_times,
            'time_to_goal': time_to_goal,        # ← NEW
            'steps_to_goal': steps_to_goal,      # ← NEW
            'goal_reached': goal_reached,        # ← NEW
            'config': config
        }
        
        # Save plots if requested
        if config.save_plots:
            self.plot_results(results)
        
        return results
    
    def plot_results(self, results: dict):
        """Plot and save results"""
        config = self.config
        import os
        os.makedirs('figs', exist_ok=True)
        # Main trajectory plot
        plot_intent_trajectories(
            results['trajectory'],
            results['obs_trajectory'],
            results['intent_history'],
            config.obstacle_goals,
            ego_start=config.ego_init[:2],
            obs_start=config.obs_init[:2],
            ego_goal=config.ego_goal,
            title=f"Intent-Aware Planning ({config.dynamics_type})",
            save_path=f"figs/results_{config.dynamics_type}_target_goal{config.target_goal}.png"
        )
        ami = animate_two_dubins_with_predictions(
            results['trajectory'],
            results['obs_trajectory'],
            results['obs_particles'],
            goal_xy=config.ego_goal,
            goals=config.obstacle_goals,
            obs_R=config.obs_radius/2,
            filename=f"figs/results_{config.dynamics_type}_target_goal{config.target_goal}.gif",
            fps=config.fps,
            show_footprints=True,
            show_arrows=False,
            show_predictions=True,
            prediction_alpha=0.3,
            prediction_color="gray"
        )
        display(Image(filename=ami)) 
        anim_controls = animate_control_inputs(
            results['controls'], 
            filename=f"figs/results_{config.dynamics_type}_target_goal{config.target_goal}_controls.gif", 
            fps=config.fps, 
            show_inline=False) 
        display(Image(filename=anim_controls))  
        try:
            anim_val = animate_min_value(
                results['cvar_values'],
                filename=f"figs/results_{config.dynamics_type}_target_goal{config.target_goal}_val.gif",
                fps=config.fps,
                show_inline=False
            )
            display(Image(filename=anim_val))  
            anim_cvar = animate_cvar_landscape_evolution(
                results['var_grid_history'],
                filename=f"figs/results_{config.dynamics_type}_target_goal{config.target_goal}_var.gif",
                fps=config.fps,
                show_inline=False
            )
            display(Image(filename=anim_cvar))  
        except:
            print("No cvar values to plot")


        print(f"📊 Plots saved!")


def run_experiment(config: Optional[ExperimentConfig] = None) -> dict:
    """
    Convenience function to run experiment
    
    Args:
        config: ExperimentConfig object, or None to use defaults
        
    Returns:
        dict: Results dictionary
    """
    if config is None:
        config = ExperimentConfig()
    
    planner = IntentAwarePlanner(config)
    results = planner.run()
    
    return results