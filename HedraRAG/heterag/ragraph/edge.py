from abc import ABC, abstractmethod

# nodes of RAGraph
class BaseEdge(ABC):
    def __init__(self, typein: str):
        if (typein == 'D' or typein == 'S'):
            self.type = typein
        elif (typein == 'I' or typein == 'II')
        else:
            raise ValueError("Wrong RAGEdge type!")
        
        self.schedule_list = []

    #@abstractmethod
    def add_task(self):
        pass

    #@abstractmethod
    def schedule(self):
        pass
    
    @abstractmethod
    def take_this_edge(self, input_str: str, **params):
        pass

class DEdge(BaseEdge):
    def __init__(self, ):
        super().__init__('D')
    
    def take_this_edge(self, input_str: str):
        return True

class IterEdge(BaseEdge):
    def __init__(self, iter_num):
        self.iter_num = iter_num
        super().__init__('I')
    
    def take_this_edge(self, input_str: str, iter_num: int):
        iter_num <= self.iter_num
        return True

class IterInverseEdge(BaseEdge):
    def __init__(self, iter_num):
        self.iter_num = iter_num
        super().__init__('II')
    
    def take_this_edge(self, input_str: str, iter_num: int):
        iter_num > self.iter_num
        return True