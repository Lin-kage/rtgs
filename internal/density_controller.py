import torch
import torch.nn as nn 
from .utils import build_rotation

class Util:
    """
        a static class
        helper to modify the optimizer
    """
    @staticmethod
    def cat_tensors_to_optimizers(new_properties: dict[str, torch.Tensor], optimizers: list[torch.optim.Optimizer]) -> dict[str, torch.Tensor]:
        new_parameters = {}
        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                assert group["name"] not in new_parameters, "parameter `{}` appears in multiple optimizers".format(group["name"])

                extension_tensor = new_properties[group["name"]]
                
                # # test
                # if group["name"] == 'means':
                #     print("density_controller 1:\n", group['params'][0].grad)
                #     param_copy = group['params'][0].clone()
                #     print("density_controller 2:\n", param_copy.grad)
                #     param_copy = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0), requires_grad=True)
                #     print("density_controller 3:\n", param_copy.grad)
                #     assert False
                    

                # get current sates
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    # append states for new properties
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    # delete old state key by old params from optimizer
                    del opt.state[group['params'][0]]
                    # append new parameters to optimizer
                    group["params"][0] = nn.Parameter(torch.cat(
                        (group["params"][0], extension_tensor),
                        dim=0,
                    ).requires_grad_(True))
                    # update optimizer states
                    opt.state[group['params'][0]] = stored_state
                else:
                    # append new parameters to optimizer
                    group["params"][0] = nn.Parameter(torch.cat(
                        (group["params"][0], extension_tensor),
                        dim=0,
                    ).requires_grad_(True))

                # add new `nn.Parameter` from optimizers to the dict returned later
                new_parameters[group["name"]] = group["params"][0]
        return new_parameters
    
    
    @staticmethod
    def prune_optimizers(mask, optimizers):
        """
        :param mask: The `False` indicating the ones to be pruned
        :param optimizers:
        :return: a new dict
        """

        new_parameters = {}
        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                assert group["name"] not in new_parameters, "parameter `{}` appears in multiple optimizers".format(group["name"])

                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))

                new_parameters[group["name"]] = group["params"][0]

        return new_parameters
    

class DensityController():
    def __init__(self, 
            densify_epoch_from_until=[0, 1000], 
            densify_grad_threshold=0.0002,
            cull_opacity_threshold=[0.005, 0.03], 
            cull_scale_threshold=[0.005, 0.5],
            clone_split_threshold=0.05):
        self.densify_epoch_from_until = densify_epoch_from_until
        self.densify_grad_threshold = densify_grad_threshold
        self.cull_opacity_threshold = cull_opacity_threshold
        self.cull_scale_threshold = cull_scale_threshold
        self.clone_split_threshold = clone_split_threshold

        
    def before_step(self, means_grad: torch.Tensor, model, global_step: int):
        if global_step < self.densify_epoch_from_until[0] or global_step >= self.densify_epoch_from_until[1]:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
        with torch.no_grad():
            return self._densify_and_prune_prework(means_grad, model)
    
    
    def after_step(self, new_properties, prune_mask, model, optimizers: list, global_step: int):
        if global_step < self.densify_epoch_from_until[0] or global_step >= self.densify_epoch_from_until[1]:
            return
    
        with torch.no_grad():
            return self._densify_and_prune(new_properties, prune_mask, model, optimizers)   
    
    
    def _densify_and_prune_prework(self, means_grad: torch.Tensor, model):
        
        grad_norm = torch.norm(means_grad, dim=-1)
        
        # densify
        clone_new_propertoes, clone_mask = self._clone(grad_norm, model)
        split_new_properties, split_prune_mask = self._split(grad_norm, model)
        
        new_properties = {}
        for key, value in clone_new_propertoes.items():
            new_properties[key] = torch.cat((clone_new_propertoes[key], split_new_properties[key]), dim=0)
        
        
        # prune
        scale_opacity_prune_mask = torch.logical_or(model.opacities / torch.mean(model.scales, dim=-1) < self.cull_opacity_threshold[0], model.opacities > self.cull_opacity_threshold[1])
        scale_prune_mask = torch.logical_or(torch.min(model.scales, dim=-1).values < self.cull_scale_threshold[0], torch.max(model.scales, dim=-1).values > self.cull_scale_threshold[1])
        
        prune_mask = torch.logical_or(split_prune_mask, torch.logical_or(scale_opacity_prune_mask, scale_prune_mask)).squeeze()
        
        return new_properties, prune_mask, clone_mask
    
    
    def _densify_and_prune(self, new_properties, prune_mask, model, optimizers: list):
        self._prune(prune_mask, model, optimizers)
        new_parameters = self._cat_tensor_to_optimizer(new_properties, optimizers)
        model.set_properties(new_parameters)
    
    
    def _clone(self, grad_norm: torch.Tensor, model):
        
        selected_mask = (grad_norm >= self.densify_grad_threshold)
        selected_mask = torch.logical_and(selected_mask, torch.max(model.scales, dim=-1).values < self.clone_split_threshold)
        
        # copy selected gaussians
        new_properties = {}
        for key, value in model.properties.items():
            if key == 'opacities':
                new_properties[key] = model.opacity_inverse_activation(model.opacity_activation(value[selected_mask]) / 2.)
            else:
                new_properties[key] = value[selected_mask]
            
        return new_properties, selected_mask
    
    
    def _split(self, grad_norm: torch.Tensor, model, N: int = 2):
        
        device = model.device
        
        padded_grad_norm = torch.zeros((model.n_gaussians,), device=device)
        padded_grad_norm[:grad_norm.shape[0]] = grad_norm
        
        selected_mask = (padded_grad_norm >= self.densify_grad_threshold)
        selected_mask = torch.logical_and(selected_mask, torch.max(model.scales, dim=-1).values >= self.clone_split_threshold)
        
        new_properties = self._split_properties(model, selected_mask, N)
        
        return new_properties, selected_mask
        

    def _prune(self, prune_mask: torch.Tensor, model, optimizers: list):
        new_parameters = self._prune_optimizer(prune_mask, optimizers)
        model.set_properties(new_parameters)
    
    
    def _split_properties(self, model, selected_mask: torch.Tensor, N):
        
        stds = model.scales[selected_mask].repeat(N, 1)
        means = torch.zeros((stds.shape[0], 3), device=model.device)  # [M*N, 3]
        samples = torch.normal(means, stds)            # [M*N, 3]
        rots = build_rotation(model.rotations[selected_mask]).repeat(N, 1, 1) # [M*N, 3, 3]
        # split: means are radomly sampled from original gaussian
        new_means = (rots @ samples.unsqueeze(-1)).squeeze(-1) + model.means[selected_mask].repeat(N, 1) # [M*N, 3]
        new_scales = model.scale_inverse_activation(model.scales[selected_mask].repeat(N, 1) / (.8 * N))
        new_opacities = model.opacity_inverse_activation(model.opacities[selected_mask].repeat(N) / N)
        
        new_properties = {
            'means': new_means,
            'scales': new_scales,
            'opacities' : new_opacities,
        }
        
        for key, value in model.properties.items():
            if key in new_properties.keys():
                continue
            new_properties[key] = value[selected_mask].repeat(N, *[1 for _ in range(value[selected_mask].dim() - 1)])
        
        return new_properties
    
    
    @staticmethod
    def _cat_tensor_to_optimizer(new_properties: dict, optimizer: list):
        return Util.cat_tensors_to_optimizers(new_properties, optimizer)
    
    @staticmethod
    def _prune_optimizer(mask, optimizer: list):
        return Util.prune_optimizers(~mask, optimizer)
        
        
