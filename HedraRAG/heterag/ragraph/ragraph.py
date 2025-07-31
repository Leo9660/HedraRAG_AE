# Construct RAGraph for a workflow
# Use Node 0 for query input

from heterag.ragraph.node import RNode, GNode, IterNode, EndNode
from heterag.prompt.template import *
from heterag.executor.request import *
from abc import ABC, abstractmethod
import networkx as nx

class BaseRAGraph(ABC):
    def __init__(self, name, config):
        self.name = name
        self.request_queue = []
        self.entry_point = -1
        self.config = config

        self.node_list = []

    def add_node(self, node, prompt: BasePrompt = None):
        self.node_list.append(node)

    def get_node(self, node_id: int):
        return self.node_list[node_id]
    
    def add_edge(self, node_id1: int, node_id2: int):
        self.node_list[node_id1].add_edge(node_id2)

    def add_iter_edge(self, node_id1: int, node_id2: int, node_id3: int):
        self.node_list[node_id1].add_iter_edge(node_id2, node_id3)
    
    def get_next(self, node_id: int, **params):
        return self.get_node(node_id).get_next(**params)
        

class SequentialRAGraph(BaseRAGraph):
    def __init__(self, config):
        super().__init__("Sequential", config)

        self.add_node(RNode(self.config))
        self.add_node(GNode(self.config))
        self.add_node(EndNode())

        self.add_edge(0, 1)
        self.add_edge(1, 2)

        self.entry_point = 0

class IRGRAGraph(BaseRAGraph):
    def __init__(self, config):
        super().__init__("IRG", config)

        if (config["iter_num"] != None):
            iter_num = config["iter_num"]
        else:
            iter_num = 3

        self.add_node(RNode(self.config), prompt = IterativeGenPrompt(self.config))
        self.add_node(IterNode(self.config))
        self.add_node(EndNode())

        self.add_edge(0, 1)
        self.add_iter_edge(1, 2, 0)

        self.entry_point = 0
