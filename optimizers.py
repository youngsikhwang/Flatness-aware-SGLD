import torch
import math
from torch.optim import Optimizer


class fSGLD(Optimizer):
    def __init__(self, params, lr, sigma, n_pert=1,
                 momentum=0.9, weight_decay=5e-4, beta_inv=1e-14, 
                 pert_type='normal', antithetic=False):
        defaults = dict(lr=lr, sigma=sigma, n_pert=n_pert,
                        momentum=momentum, weight_decay=weight_decay,
                        beta_inv=beta_inv, pert_type=pert_type, antithetic=antithetic)
        super().__init__(params, defaults)
        self.base_opt = torch.optim.SGD(self.param_groups, lr=lr,
                                        momentum=momentum,
                                        weight_decay=weight_decay,
                                        dampening=(1-lr),
                                        nesterov=False)

    def _generate_perturbation(self, tensor, sigma, pert_type='normal'):
        if pert_type == 'rademacher': # Rademacher 
            rademacher = 2 * torch.randint(0, 2, tensor.shape, device=tensor.device, dtype=tensor.dtype) - 1
            return sigma * rademacher
        else:  # normal
            return sigma * torch.randn_like(tensor)

    def step(self, closure):
        assert closure is not None
        group = self.param_groups[0]
        sigma, M = group['sigma'], group['n_pert']
        lr, betainv = group['lr'], group['beta_inv']
        momentum = group['momentum']
        pert_type = group['pert_type']
        antithetic = group['antithetic']
        params = group['params']

        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        if M > 0:
            if antithetic and M == 2:
                eps_cache = []
                with torch.no_grad():
                    for p in params:
                        eps = self._generate_perturbation(p, sigma, pert_type)
                        eps_cache.append(eps)

                # Forward perturbation
                with torch.no_grad():
                    for p, eps in zip(params, eps_cache):
                        p.add_(eps)
                with torch.enable_grad():
                    loss1 = closure()

                # Backward perturbation (antithetic)
                with torch.no_grad():
                    for p, eps in zip(params, eps_cache):
                        p.sub_(2 * eps)
                with torch.enable_grad():
                    loss2 = closure()

                # Reset to original position
                with torch.no_grad():
                    for p, eps in zip(params, eps_cache):
                        p.add_(eps)

                # Average gradients
                for p in params:
                    if p.grad is not None:
                        p.grad.div_(2.0)

                loss = 0.5 * (loss1 + loss2)
            
            else:
                # perturbation with chosen distribution
                for _ in range(M):
                    eps_cache = []
                    with torch.no_grad():
                        for p in params:
                            eps = self._generate_perturbation(p, sigma, pert_type)
                            p.add_(eps)
                            eps_cache.append(eps)

                    with torch.enable_grad():
                        loss = closure()

                    with torch.no_grad():
                        for p, eps in zip(params, eps_cache):
                            p.sub_(eps)

                for p in params:
                    if p.grad is not None:
                        p.grad.div_(float(M))
        else:
            with torch.enable_grad():
                loss = closure()

        # Noise injection
        if momentum > 0:
            noise_std = math.sqrt(2 * (1 - momentum) * betainv)
            with torch.no_grad():
                for p in params:
                    if p.grad is None:
                        continue
                    
                    param_state = self.base_opt.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(p)
                    
                    momentum_buffer = param_state['momentum_buffer']
                    noise = torch.randn_like(momentum_buffer) * noise_std
                    momentum_buffer.add_(noise)
        else:
            noise_std = math.sqrt(2 * lr * betainv)
            with torch.no_grad():
                for p in params:
                    p.add_(torch.randn_like(p) * noise_std)
        
        self.base_opt.step()
        self.zero_grad()

        return loss




class SGLD(Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=5e-4, beta_inv=1e-14):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        beta_inv=beta_inv)
        super().__init__(params, defaults)
        self.base_opt = torch.optim.SGD(self.param_groups, lr=lr,
                                        momentum=momentum,
                                        weight_decay=weight_decay,
                                        nesterov=False)

    def step(self, closure):
        assert closure is not None
        group = self.param_groups[0]
        lr, betainv = group['lr'], group['beta_inv']
        momentum = group['momentum']
        params = group['params']

        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        with torch.enable_grad():
            loss = closure()

        if momentum > 0:
            noise_std = math.sqrt(2 * (1 - momentum) * betainv)
            with torch.no_grad():
                for p in params:
                    if p.grad is None:
                        continue
                    
                    param_state = self.base_opt.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(p)
                    
                    momentum_buffer = param_state['momentum_buffer']
                    noise = torch.randn_like(momentum_buffer) * noise_std
                    momentum_buffer.add_(noise)
        else:
            noise_std = math.sqrt(2 * lr * betainv)
            with torch.no_grad():
                for p in params:
                    p.add_(torch.randn_like(p) * noise_std)
        
        self.base_opt.step()
        self.zero_grad()

        return loss



class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if not isinstance(base_optimizer, Optimizer):
            raise ValueError("base_optimizer must be an instance of torch.optim.Optimizer")
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        self.adaptive = adaptive
        super().__init__(params, {})

    def step(self, closure):
        assert closure is not None, "SAM requires closure for gradient computation"

        with torch.enable_grad():
            loss = closure()
        grad_norm = self._grad_norm()
        if grad_norm == 0:
            self.base_optimizer.step()
            return loss
        with torch.no_grad():
            for group in self.param_groups:
                scale = self.rho / grad_norm
                for p in group["params"]:
                    if p.grad is None: 
                        continue
                    e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)
                    self.state[p]["e_w"] = e_w
        
        self.zero_grad()
        
        with torch.enable_grad():
            loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: 
                        continue
                    p.sub_(self.state[p]["e_w"]) 
        self.base_optimizer.step()
        self.zero_grad()
        
        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]), p=2)
        return norm
