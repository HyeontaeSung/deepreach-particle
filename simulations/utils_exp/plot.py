import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display

def _triangle_verts(x, y, theta, L, W):
    """Return 3 vertices of a triangle centered/rotated at (x,y,theta)."""
    # local triangle (tip points along +x in the local frame)
    local = np.array([[+L/2, 0.0], [-L/2, -W/2], [-L/2, +W/2]])  # (3,2)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s,  c]])
    return local @ R.T + np.array([x, y])

def animate_two_dubins(
    traj_ego,          # (T, 3+)  [x, y, theta, ...]
    traj_obs,          # (T, 2+)  [x, y, (theta, ...)]
    goal_xy=None,      # (2,) optional
    obs_R=0.25,         # circle radius for BOTH ego and obstacle
    filename="two_dubins.gif",
    fps=20,
    xlim=None, ylim=None,
    show_footprints=True,   # draw oriented triangles
    show_arrows=False,      # optional: also draw arrows
    # If veh_L/W are None, they’ll be set relative to obs_R to keep triangles inside circles
    veh_L=None, veh_W=None,           # ego triangle size
    veh_L_obs=None, veh_W_obs=None,   # obstacle triangle size
    arrow_len=0.3
):
    traj_ego = np.asarray(traj_ego)
    traj_obs = np.asarray(traj_obs)
    T = min(len(traj_ego), len(traj_obs))

    # Default triangle sizes so they fit nicely inside the circle of radius obs_R
    if veh_L is None:     veh_L     = 0.9 * obs_R
    if veh_W is None:     veh_W     = 0.6 * obs_R
    if veh_L_obs is None: veh_L_obs = 0.9 * obs_R
    if veh_W_obs is None: veh_W_obs = 0.6 * obs_R

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # Auto limits
    if xlim is None or ylim is None:
        xs = np.concatenate([traj_ego[:T, 0], traj_obs[:T, 0]])
        ys = np.concatenate([traj_ego[:T, 1], traj_obs[:T, 1]])
        pad = 0.5
        xlim = (xs.min() - pad, xs.max() + pad) if xlim is None else xlim
        ylim = (ys.min() - pad, ys.max() + pad) if ylim is None else ylim
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    # Goal marker
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], s=120, label="goal", zorder=5)

    # Trajectory lines + current points
    ego_line, = ax.plot([], [], "-o", ms=2, alpha=0.3, label="ego traj", color="C0", zorder=2)
    obs_line, = ax.plot([], [], "-",  alpha=0.2, label="obs traj", color="C3", zorder=2)
    ego_pt,   = ax.plot([], [], "o",  color="C0", label="ego", zorder=4)
    obs_pt,   = ax.plot([], [], "o",  color="C3", label="obs", zorder=4)

    # Circles for ego and obstacle (both use obs_R so they look consistent)
    ego_circle = patches.Circle((traj_ego[0,0], traj_ego[0,1]), radius=obs_R,
                                fill=False, linestyle="-", alpha=0.8, color="C0", lw=2, zorder=3)
    obs_circle = patches.Circle((traj_obs[0,0], traj_obs[0,1]), radius=obs_R,
                                fill=False, linestyle="--", alpha=0.8, color="C3", lw=2, zorder=3)
    ax.add_patch(ego_circle)
    ax.add_patch(obs_circle)

    # Oriented triangles (footprints) centered on the same circle centers
    thE0 = traj_ego[0, 2] if traj_ego.shape[1] >= 3 else 0.0
    thO0 = traj_obs[0, 2] if traj_obs.shape[1] >= 3 else 0.0
    ego_tri = patches.Polygon(
        _triangle_verts(traj_ego[0,0], traj_ego[0,1], thE0, L=veh_L, W=veh_W),
        closed=True, color="C0", alpha=0.9, zorder=4
    )
    obs_tri = patches.Polygon(
        _triangle_verts(traj_obs[0,0], traj_obs[0,1], thO0, L=veh_L_obs, W=veh_W_obs),
        closed=True, color="C3", alpha=0.9, zorder=4
    )
    if show_footprints:
        ax.add_patch(ego_tri); ax.add_patch(obs_tri)

    # Optional arrows
    ego_arrow = None
    obs_arrow = None
    def _make_arrow(x, y, theta, color):
        return patches.FancyArrow(
            x, y, arrow_len*np.cos(theta), arrow_len*np.sin(theta),
            width=0.02, length_includes_head=True, color=color, zorder=5
        )

    ax.legend()
    ax.set_title("Two Dubins Animation")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    def update(i):
        xE, yE = traj_ego[i, 0], traj_ego[i, 1]
        xO, yO = traj_obs[i, 0], traj_obs[i, 1]
        thE = traj_ego[i, 2] if traj_ego.shape[1] >= 3 else 0.0
        thO = traj_obs[i, 2] if traj_obs.shape[1] >= 3 else 0.0

        # Lines and points
        ego_line.set_data(traj_ego[:i+1, 0], traj_ego[:i+1, 1])
        obs_line.set_data(traj_obs[:i+1, 0], traj_obs[:i+1, 1])
        ego_pt.set_data([xE], [yE])
        obs_pt.set_data([xO], [yO])

        # Circles stay centered on agents
        ego_circle.center = (xE, yE)
        obs_circle.center = (xO, yO)

        # Triangles stay centered on the same positions (so they “stick” to circles)
        if show_footprints:
            ego_tri.set_xy(_triangle_verts(xE, yE, thE, L=veh_L, W=veh_W))
            obs_tri.set_xy(_triangle_verts(xO, yO, thO, L=veh_L_obs, W=veh_W_obs))

        # Optional arrows (recreate per frame so they track positions)
        artists = [ego_line, obs_line, ego_pt, obs_pt, ego_circle, obs_circle]
        if show_footprints:
            artists += [ego_tri, obs_tri]

        nonlocal ego_arrow, obs_arrow
        if show_arrows:
            if ego_arrow is not None: ego_arrow.remove()
            if obs_arrow is not None: obs_arrow.remove()
            ego_arrow = _make_arrow(xE, yE, thE, "C0")
            # obs_arrow = _make_arrow(xO, yO, thO, "C3")
            ax.add_patch(ego_arrow); ax.add_patch(obs_arrow)
            artists += [ego_arrow, obs_arrow]

        return artists

    ani = FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=False)
    ani.save(filename, writer=PillowWriter(fps=fps))  # use .mp4 with FFMpegWriter if preferred
    plt.close(fig)
    return filename


def animate_two_dubins_with_predictions(
    traj_ego,          # (T, 3+)  [x, y, theta, ...]
    traj_obs,          # (T, 2+)  [x, y, (theta, ...)]
    obs_traj_particles_save,  # List of predicted trajectories at each time step
    goal_xy=None,      # (2,) optional ego goal
    goals=None,        # List of obstacle goals [[x1, y1], [x2, y2], ...]

    obs_R=0.25,         # circle radius for BOTH ego and obstacle
    filename="two_dubins_with_predictions.gif",
    fps=20,
    xlim=None, ylim=None,
    show_footprints=True,   # draw oriented triangles
    show_arrows=False,      # optional: also draw arrows
    show_predictions=True,  # show predicted trajectories
    prediction_alpha=0.3,   # transparency for predictions
    prediction_color="gray", # single color for all predictions
    # If veh_L/W are None, they'll be set relative to obs_R to keep triangles inside circles
    veh_L=None, veh_W=None,           # ego triangle size
    veh_L_obs=None, veh_W_obs=None,   # obstacle triangle size
    arrow_len=0.3
):
    traj_ego = np.asarray(traj_ego)
    traj_obs = np.asarray(traj_obs)
    T = min(len(traj_ego), len(traj_obs))

    # Default triangle sizes so they fit nicely inside the circle of radius obs_R
    if veh_L is None:     veh_L     = 0.9 * obs_R
    if veh_W is None:     veh_W     = 0.6 * obs_R
    if veh_L_obs is None: veh_L_obs = 0.9 * obs_R
    if veh_W_obs is None: veh_W_obs = 0.6 * obs_R

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # Auto limits - include goals in calculation
    if xlim is None or ylim is None:
        xs = np.concatenate([traj_ego[:T, 0], traj_obs[:T, 0]])
        ys = np.concatenate([traj_ego[:T, 1], traj_obs[:T, 1]])
        
        # Add goal positions to limits calculation
        if goal_xy is not None:
            xs = np.concatenate([xs, [goal_xy[0]]])
            ys = np.concatenate([ys, [goal_xy[1]]])
        
        if goals is not None:
            goal_xs = [goal[0] for goal in goals]
            goal_ys = [goal[1] for goal in goals]
            xs = np.concatenate([xs, goal_xs])
            ys = np.concatenate([ys, goal_ys])
        
        pad = 0.5
        xlim = (xs.min() - pad, xs.max() + pad) if xlim is None else xlim
        ylim = (ys.min() - pad, ys.max() + pad) if ylim is None else ylim
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    # Goal markers
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], s=120, label="ego goal", zorder=5, marker='*', color='gold')
    
    # Plot obstacle goals
    if goals is not None:
        for i, goal in enumerate(goals):
            ax.scatter(goal[0], goal[1], c=f'C{i}', s=80, label=f'obstacle goal {i+1}', 
                       marker='s', alpha=0.8, zorder=5)
    


    # Trajectory lines + current points
    ego_line, = ax.plot([], [], "-o", ms=5, alpha=0.1, label="ego traj", color="C0", zorder=2)
    obs_line, = ax.plot([], [], "-",  alpha=0.3, label="obs traj", color="C3", zorder=2)
    ego_pt,   = ax.plot([], [], "o",  color="C0", zorder=4)
    obs_pt,   = ax.plot([], [], "o",  color="C3", zorder=4)

    # Circles for ego and obstacle (both use obs_R so they look consistent)
    ego_circle = patches.Circle((traj_ego[0,0], traj_ego[0,1]), radius=obs_R,
                                fill=False, linestyle="-", alpha=0.8, color="C0", lw=2, zorder=3)
    obs_circle = patches.Circle((traj_obs[0,0], traj_obs[0,1]), radius=obs_R,
                                fill=False, linestyle="--", alpha=0.8, color="C3", lw=2, zorder=3)
    ax.add_patch(ego_circle)
    ax.add_patch(obs_circle)

    # Oriented triangles (footprints) centered on the same circle centers
    thE0 = traj_ego[0, 2] if traj_ego.shape[1] >= 3 else 0.0
    thO0 = traj_obs[0, 2] if traj_obs.shape[1] >= 3 else None
    ego_tri = patches.Polygon(
        _triangle_verts(traj_ego[0,0], traj_ego[0,1], thE0, L=veh_L, W=veh_W),
        closed=True, color="C0", alpha=0.9, zorder=4
    )
    if show_footprints:
        ax.add_patch(ego_tri)
        # Only create and add obstacle triangle if it has heading information
        if thO0 is not None:
            obs_tri = patches.Polygon(
                _triangle_verts(traj_obs[0,0], traj_obs[0,1], thO0, L=veh_L_obs, W=veh_W_obs),
                closed=True, color="C3", alpha=0.9, zorder=4
            )
            ax.add_patch(obs_tri)

    # Prediction trajectories (will be updated each frame)
    prediction_lines = []
    prediction_points = []
    
    # Initialize prediction lines and points
    if show_predictions and obs_traj_particles_save:
        num_predictions = len(obs_traj_particles_save[0]) if obs_traj_particles_save else 0
        
        for i in range(num_predictions):
            line, = ax.plot([], [], "-", alpha=prediction_alpha, color=prediction_color, 
                           linewidth=1, zorder=1)
            point, = ax.plot([], [], "o", alpha=prediction_alpha, color=prediction_color, 
                           ms=2, zorder=1)
            prediction_lines.append(line)
            prediction_points.append(point)

    # Optional arrows
    ego_arrow = None
    obs_arrow = None
    def _make_arrow(x, y, theta, color):
        return patches.FancyArrow(
            x, y, arrow_len*np.cos(theta), arrow_len*np.sin(theta),
            width=0.02, length_includes_head=True, color=color, zorder=5
        )

    ax.legend()
    ax.set_title("Intent-Aware Animation with Predictions and Goals")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    def update(i):
        xE, yE = traj_ego[i, 0], traj_ego[i, 1]
        xO, yO = traj_obs[i, 0], traj_obs[i, 1]
        thE = traj_ego[i, 2] if traj_ego.shape[1] >= 3 else 0.0
        thO = traj_obs[i, 2] if traj_obs.shape[1] >= 3 else None

        # Lines and points
        ego_line.set_data(traj_ego[:i+1, 0], traj_ego[:i+1, 1])
        obs_line.set_data(traj_obs[:i+1, 0], traj_obs[:i+1, 1])
        ego_pt.set_data([xE], [yE])
        obs_pt.set_data([xO], [yO])

        # Circles stay centered on agents
        ego_circle.center = (xE, yE)
        obs_circle.center = (xO, yO)

        # Triangles stay centered on the same positions (so they "stick" to circles)
        if show_footprints:
            ego_tri.set_xy(_triangle_verts(xE, yE, thE, L=veh_L, W=veh_W))
            # Only update obstacle triangle if it exists (has heading information)
            if thO is not None and 'obs_tri' in locals():
                obs_tri.set_xy(_triangle_verts(xO, yO, thO, L=veh_L_obs, W=veh_W_obs))

        # Update prediction trajectories
        if show_predictions and obs_traj_particles_save and i < len(obs_traj_particles_save):
            current_predictions = obs_traj_particles_save[i]  # List of predicted trajectories
            for j, (line, point) in enumerate(zip(prediction_lines, prediction_points)):
                if j < len(current_predictions):
                    pred_traj = np.array(current_predictions[j])
                    if len(pred_traj) > 0:
                        # Plot the full predicted trajectory
                        line.set_data(pred_traj[:, 0], pred_traj[:, 1])
                        # Mark current position in prediction
                        point.set_data([pred_traj[0, 0]], [pred_traj[0, 1]])
                    else:
                        # Clear if no prediction
                        line.set_data([], [])
                        point.set_data([], [])
                else:
                    # Clear if no prediction for this particle
                    line.set_data([], [])
                    point.set_data([], [])

        # Optional arrows (recreate per frame so they track positions)
        artists = [ego_line, obs_line, ego_pt, obs_pt, ego_circle, obs_circle]
        if show_footprints:
            artists += [ego_tri]
            # Only include obstacle triangle if it exists (has heading information)
            if thO is not None and 'obs_tri' in locals():
                artists += [obs_tri]
        if show_predictions:
            artists += prediction_lines + prediction_points

        nonlocal ego_arrow, obs_arrow
        if show_arrows:
            if ego_arrow is not None: ego_arrow.remove()
            if obs_arrow is not None: obs_arrow.remove()
            ego_arrow = _make_arrow(xE, yE, thE, "C0")
            ax.add_patch(ego_arrow)
            artists += [ego_arrow]
            # Only draw obstacle arrow if it has heading information
            if thO is not None:
                obs_arrow = _make_arrow(xO, yO, thO, "C3")
                ax.add_patch(obs_arrow)
                artists += [obs_arrow]

        return artists

    ani = FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=False)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return filename


def animate_min_value(vals_min_np, filename="min_value.gif", fps=20, show_inline=True):
    vals_min_np = np.asarray(vals_min_np).reshape(-1)
    T = len(vals_min_np)
    t = np.arange(T)

    # reasonable y-limits with margin
    y_min = float(vals_min_np.min())
    y_max = float(vals_min_np.max())
    pad = 0.1 * max(1e-6, (y_max - y_min))
    ylo, yhi = y_min - pad, y_max + pad

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, T-1 if T > 0 else 1)
    ax.set_ylim(ylo, yhi)
    ax.grid(True)
    ax.set_xlabel("time step")
    ax.set_ylabel("value (safe ≥ 0)")
    ax.set_title("VaR(cbf) value over time")

    (line,) = ax.plot([], [], lw=2, label="var(cbf) value")
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.6, label="safety boundary")
    ax.legend()

    viol_poly = None

    def update(i):
        nonlocal viol_poly
        x = t[: i + 1]
        y = vals_min_np[: i + 1]
        line.set_data(x, y)

        # re-draw violation shading each frame
        if viol_poly is not None:
            viol_poly.remove()
            viol_poly = None
        neg = y < 0.0
        if neg.any():
            viol_poly = ax.fill_between(
                x, y, 0.0, where=neg, interpolate=True, color="red", alpha=0.15
            )

        artists = [line]
        if viol_poly is not None:
            artists.append(viol_poly)
        return artists

    ani = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)

    if show_inline:
        from IPython.display import Image, display
        display(Image(filename=filename))
    return filename


def plot_cvar_values(cvar_lo, title="m along trajectory", save_path=None):
    """
    Plot CVaR values over time with safety boundary and violation shading
    
    Args:
        cvar_lo: List or array of CVaR values
        title: Plot title
        save_path: Optional path to save the figure
    """
    # Convert to numpy array and flatten if needed
    if isinstance(cvar_lo, list):
        # If it's a list of tensors, convert each to numpy and flatten
        cvar_lo = np.array([item.cpu().numpy() if hasattr(item, 'cpu') else item for item in cvar_lo])
    
    # Ensure it's 1D
    cvar_lo = np.asarray(cvar_lo).flatten()
    
    t_axis = np.arange(len(cvar_lo))
    plt.figure(figsize=(6, 6))
    plt.plot(t_axis, cvar_lo, lw=2, label='m')
    plt.axhline(0.0, color='k', linestyle='--', alpha=0.6, label='safety boundary')
    
    # shade violations (value < 0)
    neg = cvar_lo < 0
    if neg.any():
        plt.fill_between(t_axis, cvar_lo, 0.0, where=neg,
                        color='red', alpha=0.15, interpolate=True, label='violation')
    
    plt.xlabel("time step")
    plt.ylabel("value (safe ≥ 0)")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
# Usage examples:
# plot_cvar_values(cvar_lo)
# plot_cvar_values(cvar_lo, title="CVaR Values Over Time")
# plot_cvar_values(cvar_lo, save_path="particle/fig/TwoDubins/particlesRS92_ep10_mv")


def plot_intent_trajectories(tr, obs_tr, intent_history, goals, 
                           ego_start=None, obs_start=None, ego_goal=None,
                           title="Intent-Aware Robot Planning", save_path=None):
    """
    Plot trajectories from intent update model with intent evolution
    
    Args:
        tr: Ego vehicle trajectory (T, 3) [x, y, theta]
        obs_tr: Obstacle trajectory (T, 2) [x, y]
        intent_history: List of intent info dictionaries
        goals: List of obstacle goals [[x1, y1], [x2, y2], ...]
        ego_start: Ego start position [x, y] (optional)
        obs_start: Obstacle start position [x, y] (optional)
        ego_goal: Ego goal position [x, y] (optional)
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Trajectories
    ax1.plot(tr[:, 0], tr[:, 1], '-o', ms=2, label='ego trajectory', color='blue', alpha=0.7)
    ax1.plot(obs_tr[:, 0], obs_tr[:, 1], '-o', ms=2, label='obstacle trajectory', color='red', alpha=0.7)
    
    # Plot goals
    for i, goal in enumerate(goals):
        ax1.scatter(goal[0], goal[1], c=f'C{i}', s=80, label=f'obstacle goal {i+1}', 
                   marker='s', alpha=0.8)
    
    # Plot start/end points
    if ego_start is not None:
        ax1.scatter(ego_start[0], ego_start[1], c='blue', s=50, marker='o')
    if obs_start is not None:
        ax1.scatter(obs_start[0], obs_start[1], c='red', s=50, marker='o')
    if ego_goal is not None:
        ax1.scatter(ego_goal[0], ego_goal[1], c='purple', s=100, label='ego goal', marker='*')
    
    # Add final positions
    ax1.scatter(tr[-1, 0], tr[-1, 1], c='blue', s=100, label='ego end', marker='^')
    ax1.scatter(obs_tr[-1, 0], obs_tr[-1, 1], c='red', s=100, label='obstacle end', marker='^')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Trajectories')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    # Plot 2: Intent Evolution
    if intent_history:
        intent_array = np.array([h['goal_marginals'] for h in intent_history])
        beta_expectations = np.array([h['beta_expectation'] for h in intent_history])
        t_axis = np.arange(len(intent_array))
        
        # Create subplots for goal probabilities and beta expectation
        ax2_goals = ax2
        ax2_beta = ax2.twinx()  # Create a second y-axis
        
        # Plot goal probabilities
        for i in range(len(goals)):
            ax2_goals.plot(t_axis, intent_array[:, i], label=f'Goal {i+1}', linewidth=2)
        
        ax2_goals.axhline(1.0/len(goals), color='k', linestyle='--', alpha=0.5, label='uniform')
        ax2_goals.set_xlabel('time step')
        ax2_goals.set_ylabel('goal probability', color='blue')
        ax2_goals.tick_params(axis='y', labelcolor='blue')
        ax2_goals.grid(True)
        
        # Plot beta expectation on the right y-axis
        ax2_beta.plot(t_axis, beta_expectations, color='red', linewidth=2, linestyle='-', label='Beta expectation')
        ax2_beta.set_ylabel('beta expectation', color='red')
        ax2_beta.tick_params(axis='y', labelcolor='red')
        
        # Combine legends
        lines1, labels1 = ax2_goals.get_legend_handles_labels()
        lines2, labels2 = ax2_beta.get_legend_handles_labels()
        ax2_goals.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2_goals.set_title('Intent Evolution and Beta Expectation')

    
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Usage example:
# plot_intent_trajectories(tr, obs_tr, intent_history, OBSTACLE_GOALS,
#                         ego_start=[-1.9, 0], obs_start=[1.9, 0], ego_goal=[2.0, 0],
#                         title="Intent-Aware Planning Results")


def plot_cvar_over_control_range(cvar_data, title="CVaR over Control Input Range", save_path=None):
    """
    Plot CVaR values as a heatmap over control input ranges
    
    Args:
        cvar_data: Dictionary from compute_cvar_over_control_range containing:
            - omega_grid: (N, N) angular velocity grid
            - accel_grid: (N, N) acceleration grid  
            - cvar_grid: (N, N) CVaR values
            - omega_range: (min, max) angular velocity range
            - accel_range: (min, max) acceleration range
            - u_opt: (2,) optimal control [omega, accel] (optional)
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.contourf(cvar_data['omega_grid'], cvar_data['accel_grid'], 
                     cvar_data['cvar_grid'], levels=50, cmap='RdYlBu_r')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CVaR Value', rotation=270, labelpad=20)
    
    # Add safety boundary (CVaR = 0)
    ax.contour(cvar_data['omega_grid'], cvar_data['accel_grid'], 
               cvar_data['cvar_grid'], levels=[0], colors='black', linewidths=2, linestyles='--')
    
    # Plot optimal control if available
    if 'u_opt' in cvar_data:
        u_opt = cvar_data['u_opt']
        ax.plot(u_opt[0], u_opt[1], 'r*', markersize=20, markeredgewidth=2, 
                markeredgecolor='white', label='Optimal Control', zorder=10)
        # Add text annotation
        ax.annotate(f'u*=({u_opt[0]:.2f}, {u_opt[1]:.2f})', 
                   xy=(u_opt[0], u_opt[1]), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                 color='red', lw=2),
                   fontsize=10, fontweight='bold')
        ax.legend(loc='upper right')
    if 'u_nom' in cvar_data:
        u_nom = cvar_data['u_nom']
        ax.plot(u_nom[0], u_nom[1], 'g*', markersize=20, markeredgewidth=2, 
                markeredgecolor='white', label='Nominal Control', zorder=10)
        # ax.annotate(f'u_nom=({u_nom[0]:.2f}, {u_nom[1]:.2f})', 
        #            xy=(u_nom[0], u_nom[1]), 
        #            xytext=(10, 10), 
        #            textcoords='offset points',
        #            bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.7),
        #            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
        #                          color='green', lw=2),
        #            fontsize=10, fontweight='bold')
        ax.legend(loc='upper right')
    
    ax.set_xlabel('Angular Velocity (ω)')
    ax.set_ylabel('Acceleration (a)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_cvar_evolution(cvar_data, control_data=None, title="CVaR Evolution During Optimization", save_path=None):
    """
    Plot CVaR values over time during optimization
    
    Args:
        cvar_data: List of CVaR values over time
        control_data: List of control inputs over time (optional)
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2 if control_data else 1, 1, figsize=(12, 6 if control_data else 4))
    
    if control_data:
        ax1, ax2 = axes
    else:
        ax1 = axes
        ax2 = None
    
    # Plot CVaR values
    t_axis = np.arange(len(cvar_data))
    ax1.plot(t_axis, cvar_data, 'b-', linewidth=2, label='CVaR')
    ax1.axhline(0.0, color='k', linestyle='--', alpha=0.6, label='Safety boundary')
    ax1.set_xlabel('Optimization Iteration')
    ax1.set_ylabel('CVaR Value')
    ax1.set_title('CVaR Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Shade violations
    neg = np.array(cvar_data) < 0
    if neg.any():
        ax1.fill_between(t_axis, cvar_data, 0.0, where=neg, 
                        color='red', alpha=0.15, interpolate=True, label='Violations')
    
    # Plot control evolution if provided
    if control_data and ax2 is not None:
        controls_array = np.array([ctrl.numpy() for ctrl in control_data])
        ax2.plot(t_axis, controls_array[:, 0], 'r-', linewidth=2, label='Angular Velocity (ω)')
        ax2.plot(t_axis, controls_array[:, 1], 'g-', linewidth=2, label='Acceleration (a)')
        ax2.set_xlabel('Optimization Iteration')
        ax2.set_ylabel('Control Input')
        ax2.set_title('Control Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_cvar_3d_surface(cvar_data, title="CVaR Surface over Control Space", save_path=None):
    """
    Plot CVaR values as a 3D surface over control input ranges
    
    Args:
        cvar_data: Dictionary from compute_cvar_over_control_range
        title: Plot title
        save_path: Optional path to save the figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D surface
    surf = ax.plot_surface(cvar_data['omega_grid'], cvar_data['accel_grid'], 
                          cvar_data['cvar_grid'], cmap='RdYlBu_r', alpha=0.8)
    
    # Add safety plane (CVaR = 0)
    ax.plot_surface(cvar_data['omega_grid'], cvar_data['accel_grid'], 
                   np.zeros_like(cvar_data['cvar_grid']), alpha=0.3, color='gray')
    
    ax.set_xlabel('Angular Velocity (ω)')
    ax.set_ylabel('Acceleration (a)')
    ax.set_zlabel('CVaR Value')
    ax.set_title(title)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def animate_control_inputs(controls, dt=0.1, 
                          control_names=['Angular Velocity (ω)', 'Acceleration (a)'],
                          control_limits=None,
                          filename="control_inputs.gif", 
                          fps=20, show_inline=True):
    """
    Create animation showing control inputs evolution over time
    
    Args:
        controls: Array of control inputs (T, 2) where T is time steps
        dt: Time step size
        control_names: List of names for each control dimension
        control_limits: Optional list of (min, max) tuples for each control
        filename: Output filename
        fps: Frames per second
        show_inline: Whether to display inline in Jupyter
        
    Returns:
        str: Path to saved animation file
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    controls = np.asarray(controls)
    T = len(controls)
    t_axis = np.arange(T) * dt
    
    # Get control limits if not provided
    if control_limits is None:
        control_limits = [
            (controls[:, 0].min() - 0.1, controls[:, 0].max() + 0.1),
            (controls[:, 1].min() - 0.1, controls[:, 1].max() + 0.1)
        ]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Initialize lines
    lines = []
    current_points = []
    
    for i, (ax, name, limits) in enumerate(zip(axes, control_names, control_limits)):
        line, = ax.plot([], [], 'b-', linewidth=2, label=name)
        point, = ax.plot([], [], 'ro', markersize=10, label='Current')
        lines.append(line)
        current_points.append(point)
        
        ax.set_xlim(0, t_axis[-1])
        ax.set_ylim(limits[0], limits[1])
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add zero line if it's within range
        if limits[0] <= 0 <= limits[1]:
            ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    
    # Time counter
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle('Control Input Evolution', fontsize=16, fontweight='bold', y=0.98)
    
    def update(frame):
        """Update function for animation"""
        # Update each control dimension
        for i, (line, point) in enumerate(zip(lines, current_points)):
            line.set_data(t_axis[:frame+1], controls[:frame+1, i])
            if frame < T:
                point.set_data([t_axis[frame]], [controls[frame, i]])
        
        # Update time counter
        # time_text.set_text(f'Time: {t_axis[frame]:.2f}s / {t_axis[-1]:.2f}s')
        
        return lines + current_points + [time_text]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=T, 
                        interval=1000//fps, blit=False, repeat=True)
    
    # Save
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close(fig)
    
    print(f"✅ Control animation saved to: {filename}")
    
    if show_inline:
        from IPython.display import Image, display
        display(Image(filename=filename))
    
    return filename


def animate_cvar_landscape_evolution(var_grid_history, title="VaR(cbf) over control space", 
                                     filename="var_landscape_evolution.gif", 
                                     fps=2, show_inline=True):
    """
    Create animation showing CVaR landscape evolution over optimization iterations
    
    Args:
        var_grid_history: List of CVaR grid dictionaries from planner.var_grid_history
        title: Animation title
        filename: Output filename (gif or mp4)
        fps: Frames per second
        show_inline: Whether to display inline in Jupyter
        
    Returns:
        str: Path to saved animation file
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    if len(var_grid_history) == 0:
        print("❌ No CVaR grid history to animate!")
        return None
    
    T = len(var_grid_history)
    
    # Get global min/max for consistent colorbar
    all_cvar_grids = [data['cvar_grid'] for data in var_grid_history]
    vmin = min(grid.min() for grid in all_cvar_grids)
    vmax = max(grid.max() for grid in all_cvar_grids)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initialize with first frame
    first_data = var_grid_history[0]
    
    # Create initial contour plot
    im = ax.contourf(first_data['omega_grid'], first_data['accel_grid'], 
                     first_data['cvar_grid'], levels=50, cmap='RdYlBu_r',
                     vmin=vmin, vmax=vmax)
    
    # Add colorbar (stays fixed)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('VaR(cbf) Value', rotation=270, labelpad=20)
    
    # Create placeholders for markers
    u_opt_marker, = ax.plot([], [], 'r*', markersize=20, markeredgewidth=2, 
                            markeredgecolor='white', label='Optimal Control', zorder=10)
    u_nom_marker, = ax.plot([], [], 'g*', markersize=15, markeredgewidth=2, 
                            markeredgecolor='white', label='Nominal Control', zorder=9)
    
    # Text annotation for u_opt
    annotation = ax.annotate('', xy=(0, 0), xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                         color='red', lw=2),
                           fontsize=10, fontweight='bold', visible=False)
    
    # Iteration counter
    iter_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=14, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Stats text
    stats_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                        fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.set_xlabel('Angular Velocity (ω)', fontsize=12)
    ax.set_ylabel('Acceleration (a)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    def update(frame):
        """Update function for each frame"""
        # Clear previous contours
        for coll in ax.collections:
            coll.remove()
        
        data = var_grid_history[frame]
        
        # Update contour plot
        im = ax.contourf(data['omega_grid'], data['accel_grid'], 
                        data['cvar_grid'], levels=50, cmap='RdYlBu_r',
                        vmin=vmin, vmax=vmax)
        
        # Update safety boundary (CVaR = 0)
        ax.contour(data['omega_grid'], data['accel_grid'], 
                  data['cvar_grid'], levels=[0], 
                  colors='black', linewidths=2, linestyles='--')
        
        # Update optimal control marker
        if 'u_opt' in data:
            u_opt = data['u_opt']
            u_opt_marker.set_data([u_opt[0]], [u_opt[1]])
            
            # Update annotation
            annotation.xy = (u_opt[0], u_opt[1])
            annotation.set_text(f'u*=({u_opt[0]:.2f}, {u_opt[1]:.2f})')
            annotation.set_visible(True)
        else:
            u_opt_marker.set_data([], [])
            annotation.set_visible(False)
        
        # Update nominal control marker
        if 'u_nom' in data:
            u_nom = data['u_nom']
            u_nom_marker.set_data([u_nom[0]], [u_nom[1]])
        else:
            u_nom_marker.set_data([], [])
        
        # Update iteration counter
        iter_text.set_text(f'Iteration: {frame}/{T}')
        
        # Update stats
        cvar_min = data['cvar_grid'].min()
        cvar_max = data['cvar_grid'].max()
        safe_pct = (data['cvar_grid'] >= 0).sum() / data['cvar_grid'].size * 100
        stats_text.set_text(
            f"CVaR: [{cvar_min:.3f}, {cvar_max:.3f}]\n"
            # f"Safe region: {safe_pct:.1f}%"
        )
        
        return ax.collections + [u_opt_marker, u_nom_marker, annotation, iter_text, stats_text]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=T, 
                        interval=1000//fps, blit=False, repeat=True)
    
    # Save animation
    if filename.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
    elif filename.endswith('.mp4'):
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Intent-Aware Planner'))
        anim.save(filename, writer=writer)
    else:
        # Default to gif
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
    
    plt.close(fig)
    print(f"✅ Animation saved to: {filename}")
    
    # Display inline if requested
    if show_inline:
        from IPython.display import Image, display
        display(Image(filename=filename))
    
    return filename