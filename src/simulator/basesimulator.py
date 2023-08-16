from abc import ABCMeta, abstractmethod

# TODO: develop this into an eventual simulator class
'''
 Features this class needs:
 client creation
 server creation
 dataset distribution
 client availability
 client 
'''
class BaseSimulator(metaclass=ABCMeta):
    """Simulator orchestrating the whole process of federated learning.
    """
    def __init__(self, **kwargs):
        self._round = 0
        self._model = None
        self._clients = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
    
    @property
    def round(self):
        return self._round

    @round.setter
    def round(self, round):
        self._round = round

    @property
    def clients(self):
        return self._clients

    @clients.setter
    def clients(self, clients):
        self._clients = clients

    @abstractmethod
    def _init_server(self, model):
        pass

    @abstractmethod
    def _init_model(self, model):
        pass

    def _get_algorithm(self, model, **kwargs):
        # Imports the algorithm from algorithms modules
        ALGORITHM_CLASS = import_module(f'..algorithm.{self.args.algorithm}', package=__package__).__dict__[f'{self.args.algorithm.title()}Optimizer']
        # 
        return ALGORITHM_CLASS(params=model.parameters(), **kwargs)


    def _create_clients(self, client_datasets):
        # NOTE: This can be moved out of the server function eventually
        # Acess the client class
        CLIENT_CLASS = import_module(f'..client.{self.args.algorithm}client', package=__package__).__dict__[f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, datasets):
            client = CLIENT_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[-1])
            client.id = identifier
            return client

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Create clients!')
        
        # NOTE: Using concurrency for managing clients
        clients = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.K), os.cpu_count() - 1)) as workhorse:
            for identifier, datasets in TqdmToLogger(
                enumerate(client_datasets), logger=logger, 
                desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...creating clients... ',
                total=len(client_datasets)):
                clients.append(workhorse.submit(__create_client, identifier, datasets).result())            
        
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully created {self.args.K} clients!')
        return clients

    @abstractmethod
    def _broadcast_models(self, indices):
        raise NotImplementedError

    @abstractmethod
    def _sample_clients(self):
        raise NotImplementedError

    @abstractmethod
    def _request(self, indices, eval=False):
        raise NotImplementedError
    
    @abstractmethod
    def _aggregate(self, indices, update_sizes):
        raise NotImplementedError

    @abstractmethod
    def _cleanup(self, indices):
        raise NotImplementedError

    @abstractmethod
    def _central_evaluate(self):
        raise NotImplementedError
        
    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError
