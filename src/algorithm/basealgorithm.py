from abc import ABCMeta, abstractmethod

class BaseOptimizer(metaclass=ABCMeta):
    """Federated optimization algorithm base class.

    Args:
        metaclass (_type_, optional): _description_. Defaults to ABCMeta.

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
    """
    # FIXME: LR is a required parameter as per current implementation
    @abstractmethod
    def step(self, closure=None):
        raise NotImplementedError
     
    @abstractmethod
    def accumulate(self, **kwargs):
        raise NotImplementedError

      