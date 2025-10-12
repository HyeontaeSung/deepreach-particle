import torch
from torch import Tensor
from utils_exp.util import to_tensor, angle_wrap

def make_cbvf_fn_from_trained_model_continous_dubinsPedestrian(
    model,
    dynamics_instance,
    time_embed: float = 1.0,
    u_obs: float = 0.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    def wrap_theta(theta: Tensor) -> Tensor:
        return (theta + torch.pi) % (2 * torch.pi) - torch.pi

    @torch.no_grad()
    def cbvf_dvdt_dvds(x: Tensor, obs: Tensor, u_obs: Tensor, radius: float = 0.1) -> Tensor:
        with torch.enable_grad():
            x      = x.to(device=device, dtype=dtype)
            obs    = obs.to(device=device, dtype=dtype)
            u_obs  = torch.as_tensor(u_obs, device=device, dtype=dtype)  # accepts scalar/array

            # target broadcast shape over batch dims
            target_shape = torch.broadcast_shapes(
                x.shape[:-1], obs.shape[:-1], u_obs.shape[:-1]
            )

            # expand to common batch shape
            x_exp     = x.expand(*target_shape, 4)                         # (...,4)
            obs_exp   = obs.expand(*target_shape, obs.size(-1))            # (...,2 or 3)
            u_obs_exp = u_obs.expand(*target_shape, u_obs.size(-1))         # (...,4)

            # headings
            theta_x = angle_wrap(x_exp[..., 2])
            theta_obs_0 = angle_wrap(u_obs_exp[..., 0])                    # First heading from concatenated controls
            theta_obs_1 = angle_wrap(u_obs_exp[..., 1])                    # Second heading from concatenated controls

            # build [x, y, theta_x, obs_x, obs_y, theta_obs, theta_obs, u_obs, u_obs]
            state = torch.stack(
                [x_exp[..., 0], x_exp[..., 1], theta_x, x_exp[..., 3],
                obs_exp[..., 0], obs_exp[..., 1], theta_obs_0, theta_obs_1,
                u_obs_exp[..., 2], u_obs_exp[..., 3]],  # Last two elements from concatenated controls
                dim=-1
            )  # (..., 10)

            # coords = [t, state]
            t = torch.full((*target_shape, 1), float(time_embed), device=device, dtype=dtype)
            coord = torch.cat([t, state], dim=-1)                          # (..., 11)
            try:
                coords_org = dynamics_instance.coord_to_input(coord)            # (..., 11)
            except:        
                coords_org = dynamics_instance.coord_to_input(coord[...,:7])            # (..., 7)

            out = model({'coords': coords_org})
            model_in  = out['model_in']                 # (..., 10)
            model_out = out['model_out'].squeeze(-1)    # (...)

            h   = dynamics_instance.io_to_value(model_in.detach(), model_out.detach())
            dvs = dynamics_instance.io_to_dv(model_in, model_out)
            dvdt, dvds = dvs[..., 0].detach(), dvs[..., 1:7].detach()       # (...), (...,5)
        return h, dvdt, dvds


    return cbvf_dvdt_dvds