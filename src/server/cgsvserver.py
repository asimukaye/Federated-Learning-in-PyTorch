import logging
import torch

# from .fedavgserver import FedavgServer
from .baseserver import BaseServer

logger = logging.getLogger(__name__)

class CgsvServer(BaseServer):
    def __init__(self, **kwargs):
        super(CgsvServer, self).__init__(**kwargs)
        
        #  
        self.server_optimizer = self._get_algorithm(self.model, lr=self.args.lr, gamma=self.args.gamma)

        # lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.server_optimizer, gamma=self.args.lr_decay, step_size=self.args.lr_decay_step)

        self.importance_coefficients = dict()

    def _sparsify_gradients(self, client_ids):
        # 
        pass
    
    def _init_coefficients(self):
        pass

    def _update_coefficients(self):
        pass

    def _aggregate(self, ids, updated_sizes):
        # Calls client upload and server accumulate
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')

        # calculate importance coefficients according to sample sizes
        # coefficients = {identifier: float(coefficient / sum(updated_sizes.values())) for identifier, coefficient in updated_sizes.items()}

        coefficients = self._update_coefficients()
        
        # accumulate weights
        for identifier in ids:
            locally_updated_weights_iterator = self.clients[identifier].upload()
            # Accumulate weights
            self.server_optimizer.accumulate(coefficients[identifier], locally_updated_weights_iterator)

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')

    def update(self):
        """Update the global model through federated learning.
        """
        # randomly select clients
        selected_ids = self._sample_clients()

        #TODO: Sparsify gradients here
        self._sparsify_gradients(selected_ids)

        # broadcast the current model at the server to selected clients
        self._broadcast_models(selected_ids)
        
        # request update to selected clients
        updated_sizes = self._request(selected_ids, eval=False)
        
        print("Update sizes type: ", type(updated_sizes))
        print("Update sizes: ", updated_sizes)

        # request evaluation to selected clients
        self._request(selected_ids, eval=True, participated=True)

        # receive updates and aggregate into a new weights
        self.server_optimizer.zero_grad() # empty out buffer

        self._aggregate(selected_ids, updated_sizes) # aggregate local updates
        
        self.server_optimizer.step() # update global model with the aggregated update
        self.lr_scheduler.step() # update learning rate

        # remove model copy in clients
        self._cleanup(selected_ids)

        return selected_ids

    
    # def _request(self, ids, eval=False, participated=False):
    #     # TODO: maybe this can be split into two functions
    #     def __update_clients(client):
    #         # getter function for client update
    #         client.args.lr = self.lr_scheduler.get_last_lr()[-1]
    #         update_result = client.update()
    #         return {client.id: len(client.training_set)}, {client.id: update_result}

    #     def __evaluate_clients(client):
    #         # getter function for client evaluate
    #         eval_result = client.evaluate() 
    #         return {client.id: len(client.test_set)}, {client.id: eval_result}

    #     logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Request {"updates" if not eval else "evaluation"} to {"all" if ids is None else len(ids)} clients!')
    #     if eval:
    #         if self.args._train_only: return
    #         results = []
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
    #             for idx in TqdmToLogger(
    #                 ids, 
    #                 logger=logger, 
    #                 desc=f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...evaluate clients... ',
    #                 total=len(ids)
    #                 ):
    #                 results.append(workhorse.submit(__evaluate_clients, self.clients[idx]).result()) 
    #         eval_sizes, eval_results = list(map(list, zip(*results)))
    #         eval_sizes, eval_results = dict(ChainMap(*eval_sizes)), dict(ChainMap(*eval_results))
    #         self.results[self._round][f'clients_evaluated_{"in" if participated else "out"}'] = self._log_results(
    #             eval_sizes, 
    #             eval_results, 
    #             eval=True, 
    #             participated=participated
    #         )
    #         logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...completed evaluation of {"all" if ids is None else len(ids)} clients!')
    #     else:
    #         results = []
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
    #             for idx in TqdmToLogger(
    #                 ids, 
    #                 logger=logger, 
    #                 desc=f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...update clients... ',
    #                 total=len(ids)
    #                 ):
    #                 results.append(workhorse.submit(__update_clients, self.clients[idx]).result()) 
    #         update_sizes, update_results = list(map(list, zip(*results)))
    #         update_sizes, update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*update_results))
    #         self.results[self._round]['clients_updated'] = self._log_results(
    #             update_sizes, 
    #             update_results, 
    #             eval=False, 
    #             participated=True
    #         )
    #         logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...completed updates of {"all" if ids is None else len(ids)} clients!')
    #         return update_sizes
    