import copy

from .fedavgclient import FedavgClient
from src import MetricManager


class CgsvClient(FedavgClient):
    def __init__(self, **kwargs):
        super(CgsvClient, self).__init__(**kwargs)
    