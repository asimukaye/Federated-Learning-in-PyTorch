from .fedavg import FedavgOptimizer
from torch.nn import CosineSimilarity
from torch.nn.utils import parameters_to_vector

class CgsvOptimizer(FedavgOptimizer):
    def __init__(self, params, client_ids, **kwargs):
        super(CgsvOptimizer, self).__init__(params=params, **kwargs)
        self.gamma = kwargs.get('gamma')
        self.lr = kwargs.get('lr')
        self.alpha = kwargs.get('alpha')
        self.local_grad_norm = None
        self.server_grad_norm = None

        self._cos_sim = CosineSimilarity(dim=0)
        self._importance_coefficients = dict.fromkeys(client_ids, 0.0)
        
    
    def _compute_cgsv(self,server_param, local_param):
        # self.local_grad_norm = self.server_param
        server_param_vec = parameters_to_vector(server_param)
        local_param_vec = parameters_to_vector(local_param)
        print("Server param shape: ", server_param_vec.shape)
        return self._cos_sim(server_param_vec, local_param_vec)

    def _update_coefficients(self, client_id, cgsv):
        self._importance_coefficients[client_id] = self.alpha * self._importance_coefficients[client_id] + (1 - self.alpha)* cgsv
        
    def _sparsify_gradients(self, client_ids):
        # Implement gradient sparsification for reward 
        pass

    def step(self, closure=None):
        # single step in a round of cgsv
        loss = None
        if closure is not None:
            loss = closure()

        # print(self.param_groups)
        # TODO: what to do if param groups are multiple
        
        for group in self.param_groups:
            # beta = group['momentum']
            for param in group['params']:
                # print("Param shape: ", param.shape)
                if param.grad is None:
                    continue
                # gradient
                delta = param.grad.data
                
                # FIXME: switch to an additive gradient with LR?
                # w = w - ∆w 
                param.data.sub_(delta)
        return loss


    def accumulate(self, local_params_itr, client_id):
        # THis function is called per client. i.e. n clients means n calls
        # TODO: Rewrite this function to match gradient aggregate step
        # NOTE: Note that accumulate is called before step

        # NOTE: Currently supporting only one param group
        self._server_params = self.param_groups[0]['params']

        # print(type(local_params))
        # print(type(self.param_groups))
        # print(type(self.param_groups[0]))
        # print(type(self._server_params))
        local_params = [param.data.float() for param in local_params_itr]
        print(len(self._server_params))
        print(len(local_params))

        # print(self._server_params)


        cgsv = self._compute_cgsv(self._server_params, local_params)
        print(cgsv)

        self._update_coefficients(client_id, cgsv)

        i = 0
        for server_param, local_param in zip(self._server_params, local_params_itr):
                i += 1
                print(i)
                # u_t,i = w_t,i / ||w_t,i|| * 𝜞 
                local_grad_norm = local_param.grad.div(local_param.grad.norm()).mul(self.gamma)

                if server_param.grad is None:
                    print("Grad is none")
                    server_param.grad = local_grad_norm.mul(self._importance_coefficients[client_id])
                else:
                    print("Grad is not none")
                    server_param.grad.add_(local_grad_norm.mul(self._importance_coefficients[client_id]))


        # for group in self.param_groups:
        #     for server_param, (name, local_param) in zip(group['params'], local_param_iterator):

        #         # u_t,i = w_t,i / ||w_t,i|| * 𝜞 
        #         local_grad_norm = local_param.grad.div(local_param.grad.norm()).mul(self.gamma)

        #         if server_param.grad is None:
        #             server_param.grad = local_grad_norm.mul(importance_coefficient)
        #         else:
        #             server_param.grad.add_(local_grad_norm.mul(importance_coefficient))