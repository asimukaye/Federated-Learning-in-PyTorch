import torch
from .fedavg import FedavgOptimizer


class CgsvOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(CgsvOptimizer, self).__init__(params=params, **kwargs)
        self.gamma = kwargs.get('gamma')
        self.lr = kwargs.get('lr')
        
        
    def step(self, closure=None):
        # single step in a round of cgsv
        loss = None
        # TODO: What is closure?
        if closure is not None:
            loss = closure()

        # print(self.param_groups)
        for group in self.param_groups:
            # print("group type: ", group.keys())

            # print("Step keys: ", group.keys())
            # beta = group['momentum']

            for param in group['params']:
                # print("Param shape: ", param.shape)
                if param.grad is None:
                    continue
                
                # gradient
                delta = param.grad.data
                
                # w = w - ∆w 
                param.data.sub_(delta)
        return loss


    def accumulate(self, importance_coefficient, local_param_iterator):
        # THis function is called per client. i.e. n clients means n calls
        # TODO: Rewrite this function to match gradient aggregate step
        # NOTE: Note that accumulate is called before step

        for group in self.param_groups:
            for server_param, (name, local_param) in zip(group['params'], local_param_iterator):
                # u_t,i = w_t,i / ||w_t,i|| * 𝜞 

                local_grad_norm = local_param.grad.div(local_param.grad.norm()).mul(self.gamma)

                if server_param.grad is None:
                    server_param.grad = local_grad_norm.mul(importance_coefficient)
                else:
                    server_param.grad.add_(local_grad_norm.mul(importance_coefficient))
