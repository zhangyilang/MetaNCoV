import torch
import torch.nn as nn
from typing import Optional, Iterable
from collections import OrderedDict

from src.container import MetaParameterList
from src.meta_alg_base import MetaLearningAlgBase
from torchmeta.modules import MetaModule


class TensorModeProd(MetaModule):
    def __init__(self, params: Iterable[torch.Tensor]) -> None:
        super().__init__()
        self.mc_weight = MetaParameterList()

        for param in params:
            param_size = param.size()
            param_dim = param.dim()

            if param_dim == 1:      # Conv2d().bias / Linear().bias
                self.mc_weight.append(nn.Parameter(torch.ones_like(param)))
            else:                   # Linear().weight / Conv2d().weight
                self.mc_weight.append(nn.Parameter(torch.eye(param_size[0])))
                self.mc_weight.append(nn.Parameter(torch.eye(param_size[1])))
                if param_dim == 4:  # Conv2d().weight
                    self.mc_weight.append(nn.Parameter(torch.eye(param_size[2] * param_size[3])))

    def forward(self, input_grads: Iterable[torch.Tensor],
                params: Optional[OrderedDict[str, nn.Parameter]] = None) -> Iterable[torch.Tensor]:
        output_grads = list()
        pointer = 0

        if params is None:
            params = list(self.meta_parameters())

        for input_grad in input_grads:
            param_dim = input_grad.dim()

            if param_dim == 1:  # Conv2d().bias / Linear().bias
                output_grad = params[pointer] * input_grad
                pointer += 1
            elif param_dim == 2:  # Linear().weight
                output_grad = params[pointer] @ input_grad @ params[pointer+1]
                pointer += 2
            elif param_dim == 4:  # Conv2d().weight
                output_grad = torch.einsum('ijk,il->ljk',
                                           input_grad.flatten(start_dim=2),
                                           params[pointer])
                output_grad = params[pointer+1] @ output_grad @ params[pointer+2]
                pointer += 3
            else:
                raise NotImplementedError

            output_grads.append(output_grad.view_as(input_grad))

        return output_grads


class MetaCurvature(MetaLearningAlgBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def _get_meta_model(self) -> dict[str, MetaModule]:
        return {'init': self._get_base_model(),
                'MC': TensorModeProd(self._base_model.parameters())}

    def adapt(self, trn_input: torch.Tensor, trn_target: torch.Tensor,
              first_order: bool = False) -> OrderedDict[str, nn.Parameter]:
        params = OrderedDict(self._meta_model['init'].named_parameters())

        for _ in range(self.args.task_iter):
            trn_logit = self._base_model(trn_input, params=params)
            task_nll = self._nll(trn_logit, trn_target)
            grads = torch.autograd.grad(task_nll,
                                        params.values(),
                                        create_graph=not first_order)
            grads = self._meta_model['MC'](grads)

            for (name, param), grad in zip(params.items(), grads):
                params[name] = param - self.args.task_lr * grad

        return params
