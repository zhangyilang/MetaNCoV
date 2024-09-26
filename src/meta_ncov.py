import torch
import torch.nn as nn
from typing import Iterable, Optional
from collections import OrderedDict

from src.meta_alg_base import MetaLearningAlgBase
from src.container import MetaParameterDict
from src.meta_curvature import TensorModeProd
from src.meta_sgd import _log_lr
from torchmeta.modules import MetaModule


class Reparameterize(MetaModule):
    def __init__(self, named_params: Iterable[tuple[str, nn.Parameter]]) -> None:
        super().__init__()
        self.mean = MetaParameterDict()
        self.log_std = MetaParameterDict()

        for name, param in named_params:
            name_sub = name.replace('.', '_')
            self.mean[name_sub] = nn.Parameter(torch.zeros_like(param))
            self.log_std[name_sub] = nn.Parameter(torch.zeros_like(param))

    def forward(self, input_tensors: OrderedDict[str, nn.Parameter],
                params: Optional[OrderedDict[str, nn.Parameter]] = None) -> OrderedDict:
        output_tensors = OrderedDict()
        
        if params is None:
            params = OrderedDict(self.named_parameters())

        for name, input_tensor in input_tensors.items():
            name_sub = name.replace('.', '_')
            output_tensors[name] = input_tensor * params['log_std.' + name_sub].exp() \
                                   + params['mean.' + name_sub]

        return output_tensors


class TriuPositiveDiag(nn.Module):
    def forward(self, input_mat: torch.Tensor) -> torch.Tensor:
        output_mat = input_mat.triu()
        output_diag = output_mat.diagonal()
        output_diag *= output_diag.sign()

        return output_mat


class HouseholderOrthogonal(nn.Module):
    def forward(self, input_mat: torch.Tensor) -> torch.Tensor:
        A, tau = torch.geqrf(input_mat)
        output_mat = torch.linalg.householder_product(A, tau)
        output_mat = output_mat * A.diagonal().sign().unsqueeze(0)

        return output_mat


class Orthogonal(nn.Module):
    def forward(self, input_mat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.qr(input_mat)[0]


class InjectiveSylvesterFlow(MetaModule):     # bijective
    def __init__(self, model_params: Iterable[nn.Parameter], hidden_dim: int) -> None:
        super().__init__()
        params_vec = nn.utils.parameters_to_vector(model_params).detach().clone()
        orth = self.init_orth(params_vec, hidden_dim)
        orth_norm = torch.linalg.vector_norm(orth, dim=0)
        self.orth_weight = nn.Parameter(orth)
        self.triu_weight_1 = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.triu_weight_2 = nn.Parameter(torch.diag(orth_norm))
        self.orth_transform = Orthogonal()
        self.triu_transform = TriuPositiveDiag()

        self.bias = nn.Parameter(torch.randn(hidden_dim))
        self.non_linear = nn.Sigmoid()

    @torch.no_grad()
    def init_orth(self, params_vec, dim) -> torch.Tensor:
        assert params_vec.numel() >= dim

        orth = torch.zeros(params_vec.numel(), dim)
        pointer = 0
        for dim_idx, chunk in enumerate(torch.tensor_split(params_vec, dim)):
            chunk_size = chunk.numel()
            orth[pointer:pointer+chunk_size, dim_idx] = chunk
            pointer += chunk_size

        return orth

    def forward(self, input_tensors: OrderedDict[str, nn.Parameter], params=None) -> OrderedDict[str, nn.Parameter]:
        input_vec = nn.utils.parameters_to_vector(input_tensors.values())
        orth = self.orth_transform(self.orth_weight)
        triu_1 = self.triu_transform(self.triu_weight_1)
        triu_2 = self.triu_transform(self.triu_weight_2)

        output_vec = triu_1 @ orth.t() @ input_vec + self.bias
        output_vec = input_vec + orth @ triu_2 @ self.non_linear(output_vec)

        output_tensors = OrderedDict()
        pointer = 0
        for name, param in input_tensors.items():
            num_param = param.numel()
            output_tensors[name] = output_vec[pointer:pointer+num_param].view_as(param)
            pointer += num_param

        return output_tensors


def InjectivePlanarFlow(params: Iterable[nn.Parameter]) -> MetaModule:
    return InjectiveSylvesterFlow(params, hidden_dim=1)


class NoninjectiveSylvesterFlow(MetaModule):    # non-injective
    def __init__(self, model_params: Iterable[nn.Parameter], hidden_dim: int) -> None:
        super().__init__()
        params_vec = nn.utils.parameters_to_vector(model_params).detach().clone()
        self.weight_inner = nn.Parameter(torch.zeros(hidden_dim, params_vec.numel()))
        self.bias = nn.Parameter(torch.randn(hidden_dim))
        self.non_linear = nn.Sigmoid()
        self.weight_dot = nn.Parameter(params_vec.unsqueeze(1).repeat(1, hidden_dim) / hidden_dim)

    def forward(self, input_tensors: OrderedDict[str, nn.Parameter],
                params: Optional[OrderedDict[str, nn.Parameter]] = None) -> OrderedDict[str, nn.Parameter]:
        input_vec = nn.utils.parameters_to_vector(input_tensors.values())

        if params is None:
            params = OrderedDict(self.named_parameters())

        output_vec = params['weight_inner'] @ input_vec + params['bias']
        output_vec = input_vec + params['weight_dot'] @ self.non_linear(output_vec)

        output_tensors = OrderedDict()
        pointer = 0
        for name, param in input_tensors.items():
            num_param = param.numel()
            output_tensors[name] = output_vec[pointer:pointer+num_param].view_as(param)
            pointer += num_param

        return output_tensors


def NoninjectivePlanarFlow(params: Iterable[nn.Parameter]) -> MetaModule:
    return NoninjectiveSylvesterFlow(params, hidden_dim=1)


class MetaNCoVMAML(MetaLearningAlgBase):     # MAML backbone
    def __init__(self, args) -> None:
        super().__init__(args)

    def _get_meta_model(self) -> dict[str, MetaModule]:
        return {'NCoV': NoninjectiveSylvesterFlow(self._base_model.parameters(), hidden_dim=self.args.syl_dim)}

    def _get_meta_optimizer(self) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
        # diverge with Adam in a few setups; use SGD instead for stability
        meta_optimizer = torch.optim.SGD([{'params': module.meta_parameters()}
                                          for module in self._meta_model.values()],
                                         lr=self.args.meta_lr,
                                         momentum=0.9,
                                         weight_decay=1e-4,
                                         nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, step_size=20000)

        return meta_optimizer, lr_scheduler

    def adapt(self, trn_input: torch.Tensor, trn_target: torch.Tensor,
              first_order: bool = False) -> OrderedDict[str, nn.Parameter]:
        latent_params = OrderedDict()
        for name, model_param in self._base_model.named_parameters():
            latent_params[name] = torch.zeros(model_param.size(), requires_grad=True, device=self.args.device)
        model_params = self._meta_model['NCoV'](latent_params)

        for _ in range(self.args.task_iter):
            trn_logit = self._base_model(trn_input, params=model_params)
            task_nll = self._nll(trn_logit, trn_target)
            nll_grads = torch.autograd.grad(task_nll,
                                            latent_params.values(),
                                            create_graph=not first_order)

            for (name, latent_param), nll_grad in zip(latent_params.items(), nll_grads):
                # Note: the grad of nlp induced by Gaussian prior is latent_param itself
                latent_params[name] = (1 - self.args.task_lr * self.args.relative_weight) * latent_param \
                                      - self.args.task_lr * nll_grad
            model_params = self._meta_model['NCoV'](latent_params)

        return model_params


class MetaNCoVSGD(MetaLearningAlgBase):   # MetaSGD backbone
    def __init__(self, args):
        super().__init__(args)

    def _get_meta_model(self) -> dict[str, MetaModule]:
        return {'NCoV': NoninjectiveSylvesterFlow(self._base_model.parameters(), hidden_dim=self.args.syl_dim),
                'log_lr_scale': _log_lr(self._base_model.named_parameters())}

    def _get_meta_optimizer(self) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
        meta_optimizer = torch.optim.SGD([{'params': module.meta_parameters()}
                                          for module in self._meta_model.values()],
                                         lr=self.args.meta_lr,
                                         momentum=0.9,
                                         weight_decay=1e-4,
                                         nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, step_size=20000)

        return meta_optimizer, lr_scheduler

    def adapt(self, trn_input: torch.Tensor, trn_target: torch.Tensor,
              first_order: bool = False) -> OrderedDict[str, nn.Parameter]:
        latent_params = OrderedDict()
        for name, model_param in self._base_model.named_parameters():
            latent_params[name] = torch.zeros(model_param.size(), requires_grad=True, device=self.args.device)
        model_params = self._meta_model['NCoV'](latent_params)
        lr = OrderedDict({name.replace('_', '.'): log_lr_scale.exp() * self.args.task_lr
                          for name, log_lr_scale in self._meta_model['log_lr_scale'].items()})

        for _ in range(self.args.task_iter):
            trn_logit = self._base_model(trn_input, params=model_params)
            task_nll = self._nll(trn_logit, trn_target)
            nll_grads = torch.autograd.grad(task_nll,
                                            latent_params.values(),
                                            create_graph=not first_order)

            for (name, latent_param), nll_grad in zip(latent_params.items(), nll_grads):
                # Gaussian N(0,I) prior
                latent_params[name] = (1 - lr[name] * self.args.relative_weight) * latent_param - lr[name] * nll_grad
                # Uniform U(0,1) prior
                # latent_params[name] = torch.clamp(latent_param - lr[name] * nll_grad, min=0, max=1)
            model_params = self._meta_model['NCoV'](latent_params)

        return model_params


class MetaNCoVMC(MetaLearningAlgBase):   # MetaCurvature backbone
    def __init__(self, args):
        super().__init__(args)

    def _get_meta_model(self) -> dict[str, MetaModule]:
        return {'NCoV': NoninjectiveSylvesterFlow(self._base_model.parameters(), hidden_dim=self.args.syl_dim),
                'MC': TensorModeProd(self._base_model.parameters())}

    def _get_meta_optimizer(self) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
        meta_optimizer = torch.optim.SGD([{'params': module.meta_parameters()}
                                          for module in self._meta_model.values()],
                                         lr=self.args.meta_lr,
                                         momentum=0.9,
                                         weight_decay=1e-4,
                                         nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, step_size=20000)

        return meta_optimizer, lr_scheduler

    def adapt(self, trn_input: torch.Tensor, trn_target: torch.Tensor,
              first_order: bool = False) -> OrderedDict[str, nn.Parameter]:
        latent_params = OrderedDict()
        for name, model_param in self._base_model.named_parameters():
            latent_params[name] = torch.zeros(model_param.size(), requires_grad=True, device=self.args.device)
        model_params = self._meta_model['NCoV'](latent_params)

        for _ in range(self.args.task_iter):
            trn_logit = self._base_model(trn_input, params=model_params)
            task_nll = self._nll(trn_logit, trn_target)
            nll_grads = torch.autograd.grad(task_nll,
                                            latent_params.values(),
                                            create_graph=not first_order)
            nll_grads = self._meta_model['MC'](nll_grads)

            for (name, latent_param), nll_grad in zip(latent_params.items(), nll_grads):
                # Gaussian N(0,I) prior
                latent_params[name] = (1 - self.args.task_lr * self.args.relative_weight) * latent_param \
                                      - self.args.task_lr * nll_grad
            model_params = self._meta_model['NCoV'](latent_params)

        return model_params
