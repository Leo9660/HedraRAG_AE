from abc import ABC, abstractmethod
from heterag.prompt.template import *

# information structure of the next stage
class NextInfo:
    def __init__(self, next_id: int, next_input, **params):
        self.next_id = next_id
        self.next_input = next_input

# nodes of RAGraph
class BaseNode(ABC):
    def __init__(self, typein: str):
        if typein == 'R' or typein == 'G':
            self.type = typein
        elif typein == 'Iter':
            self.type = typein
        elif typein == 'END':
            self.type = typein
        else:
            raise ValueError("Wrong RAGNode type!")

        self.succeed_list = []
    
    def add_edge(self, node_id: int):
        self.succeed_list.append(node_id)
    
    @abstractmethod
    def get_next(self, **params):
        pass

class RNode(BaseNode):
    def __init__(self, config, prompt: BasePrompt = None):
        super().__init__('R')
        if prompt == None:
            prompt = IterativeGenPrompt(config)
        self.prompt = prompt

    def get_prompt(self, **params):
        return self.prompt.get_string(question=params["current_query"], reference=params["documents"], initial_query=params["initial_query"])

    def get_next(self, **params):
        next_tasks = []
        for succeed_node in self.succeed_list:
            next_tasks.append(NextInfo(succeed_node, self.get_prompt(**params)))
        return next_tasks

class GNode(BaseNode):
    def __init__(self, config, prompt: BasePrompt = None):
        super().__init__('G')

    def get_next(self, **params):
        next_tasks = []
        for succeed_node in self.succeed_list:
            next_tasks.append(NextInfo(succeed_node, params["current_query"]))
        return next_tasks

class IterNode(BaseNode):
    def __init__(self, config, prompt: BasePrompt = None):
        super().__init__('Iter')

        if (config["iter_num"] != None):
            self.iter_num = config["iter_num"]
        else:
            self.iter_num = 3

    def get_next(self, **params):
        next_tasks = []
        
        if (params["iter_num"] > self.iter_num):
            next_tasks = [NextInfo(self.succeed_list[0], params["current_query"])]
        else:
            next_tasks = [NextInfo(self.succeed_list[1], params["current_query"])]

        return next_tasks
    
    def add_iter_edge(self, node_id2: int, node_id3: int):
        self.succeed_list = [node_id2, node_id3]


class EndNode(BaseNode):
    def __init__(self):
        super().__init__('END')

    def get_next(self, **params):
        return [NextInfo(-1, "")]