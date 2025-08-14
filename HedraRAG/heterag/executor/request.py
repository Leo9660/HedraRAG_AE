from abc import ABC, abstractmethod
from heterag.prompt.template import *
from heterag.retriever.utils import EmbeddingInfo
from heterag.utils import *

# an information structure for each request
class Request(ABC):

    request_num = 0

    def __init__(self, query: str, workflow: str = "Sequential"):
        self.initial_query = query
        self.current_query = query
        self.documents = None
        self.workflow = workflow

        self.id = self.request_num
        Request.request_num += 1

        self.graph = None
        self.exec_stage = -1
        self.spec_stage = -1
        # whether the execution stage and speculative stage are waiting for results
        self.exec_waiting = False
        self.spec_waiting = False

        self.answer = ""

        self.query_emb = []
        self.retrieval_score = []
    
    # bind the request with ragraph
    def enter_graph(self, graph):
        self.graph = graph
        self.exec_stage = self.graph.entry_point
    
    def get_rtasks(self, input_stage = None, input_str = None):
        if (input_stage == None):
            input_stage = self.exec_stage
        if (input_str == None):
            input_str = self.current_query
        return TaskID(self.id, input_stage, "R", input_str)
    
    def get_gtasks(self, input_stage = None, input_str = None, is_spec: bool = False):
        if (input_stage == None):
            input_stage = self.exec_stage
        if (input_str == None):
            input_str = self.current_query
        return TaskID(self.id, input_stage, "G", input_str)
    
    def add_retrieval_emb(self, retrieval_emb):
        self.retrieval_emb.append(retrieval_emb)
    
    def update_retrieval_emb(self, emb_info: EmbeddingInfo, idx: int = 0):
        self.query_emb.append(emb_info.query_emb[idx])
        self.retrieval_score.append(emb_info.retrieval_score[idx])
    
    def transfer_to_stageid(self, stage_id: int, is_spec = False):
        if is_spec:
            self.spec_stage = stage_id
        else:
            self.exec_stage = stage_id

    def update_documents(self, documents, is_spec = False):
        if not is_spec:
            self.documents = documents

    @abstractmethod
    def update_stage(self, stage_input, stage_id: int = -1, emb = None, is_spec: bool = False):
        pass

    def get_result(self):
        pass

class SequentialRequest(Request):
    # stage 0: retrieval stage
    # stage 1: generation stage
    def __init__(self, query: str, config):
        super().__init__(query, workflow = "Sequential")
        self.exec_stage = -1
        self.prompt = SequentialGenPrompt(config)
    
    def update_stage(self, stage_input: str = "", stage_id: int = -1, is_spec: bool = False):

        if stage_id == self.exec_stage or stage_id == -1:
            if self.exec_stage == -1:
                # self.exec_stage = 0
                self.transfer_to_stageid(0, is_spec)
                return [self.get_rtasks()]
            elif self.exec_stage == 0:
                # self.exec_stage = 1
                self.transfer_to_stageid(1, is_spec)
                # self.documents = stage_input
                self.update_documents(stage_input, is_spec)
                generation_query = self.prompt.get_string(reference = stage_input, question = self.initial_query)
                return [self.get_gtasks(input_str = generation_query, is_spec = is_spec)]
            elif self.exec_stage == 1:
                self.current_query = stage_input
                self.answer = stage_input
                # self.exec_stage = 2
                self.transfer_to_stageid(2, is_spec)
                return None
            else:
                raise ValueError("Wrong stage number!")

class IterRequest(Request):
    # stage 0: retrieval stage
    # stage 1: generation stage
    def __init__(self, query: str, config, iter_num: int = 3):
        super().__init__(query, workflow = "IRG")
        self.iter_num = iter_num
        self.current_num = 0
        self.exec_stage = -1

        self.prompt = IterativeGenPrompt(config)
    
    def update_stage(self, stage_input: str = "", stage_id: int = -1, is_spec: bool = False):

        #     "iter num",self.iter_num)

        if stage_id == self.exec_stage or stage_id == -1:
            if self.exec_stage == -1:
                self.transfer_to_stageid(0, is_spec)
                return [self.get_rtasks()]
            elif self.exec_stage == 0:
                self.transfer_to_stageid(1, is_spec)
                self.update_documents(stage_input, is_spec)
                generation_query = self.prompt.get_string(initial_query = stage_input, reference = self.documents, question = self.answer)
                return [self.get_gtasks(input_str = generation_query, is_spec = is_spec)]
            elif self.exec_stage == 1:
                self.current_query = stage_input
                self.answer += stage_input
                
                if self.current_num >= self.iter_num:
                    self.transfer_to_stageid(2, is_spec)
                    return None
                else:
                    self.transfer_to_stageid(0, is_spec)
                    if not is_spec:
                        self.current_num += 1
                    return [self.get_rtasks(input_str = self.current_query)]
            else:
                raise ValueError("Wrong stage number!")

class MultistepRequest(Request):
    # stage 0: reasoning stage
    # stage 1: retrieval stage
    # stage 2: answering stage
    def __init__(self, query: str, config, iter_num: int = 3, is_spec: bool = False):
        super().__init__(query, workflow = "Multistep")
        self.iter_num = iter_num
        self.current_num = 0
        self.exec_stage = -1

        self.prev_reasoning = ""

        self.reasoning_prompt = MultistepGenPrompt(config)
        self.answering_prompt = SequentialFullGenPrompt(config)
    
    def update_stage(self, stage_input: str = "", stage_id: int = -1, is_spec: bool = False):

        if stage_id == self.exec_stage or stage_id == -1:

            if self.exec_stage == -1:
                # self.exec_stage = 0
                self.transfer_to_stageid(0, is_spec)
                generation_query = self.reasoning_prompt.get_string(initial_query = self.initial_query, reference = "", question = self.initial_query, prev_reasoning = "None")
                return [self.get_gtasks(input_str = generation_query, is_spec = is_spec)]
            if self.exec_stage == 0:
                if "None" in stage_input or self.current_num >= self.iter_num:
                    return None
                else:
                    self.transfer_to_stageid(1, is_spec)
                    self.current_query = stage_input
                    if not is_spec:
                        self.current_num += 1
                    return [self.get_rtasks()]
            elif self.exec_stage == 1:
                self.transfer_to_stageid(2, is_spec)
                self.update_documents(stage_input, is_spec)
                generation_query = self.answering_prompt.get_string(reference = stage_input, question = self.current_query)
                return [self.get_gtasks(input_str = generation_query, is_spec = is_spec)]
            elif self.exec_stage == 2:
                self.transfer_to_stageid(0, is_spec)
                self.answer = stage_input
                self.prev_reasoning += "- " + self.current_query + "\n"
                self.prev_reasoning += "- " + self.answer + "\n"
                generation_query = self.reasoning_prompt.get_string(initial_query = self.initial_query, reference = self.documents, question = self.initial_query, prev_reasoning = self.prev_reasoning)
                return [self.get_gtasks(input_str = generation_query, is_spec = is_spec)]

class HyDERequest(Request):
    # stage 0: retrieval stage
    # stage 1: generation stage
    def __init__(self, query: str, config):
        super().__init__(query, workflow = "HyDE")
        self.exec_stage = -1

        self.hydeprompt = HyDEPrompt(config)
        self.genprompt = SequentialGenPrompt(config)


    def update_stage(self, stage_input: str = "", stage_id: int = -1, is_spec: bool = False):

        if stage_id == self.exec_stage or stage_id == -1:
            if self.exec_stage == -1:
                self.transfer_to_stageid(0, is_spec)
                generation_query = self.hydeprompt.get_string(question = self.initial_query)
                return [self.get_gtasks(input_str = generation_query, is_spec = is_spec)]
            elif self.exec_stage == 0:
                self.transfer_to_stageid(1, is_spec)
                self.current_query = stage_input
                return [self.get_rtasks()]
            elif self.exec_stage == 1:
                self.transfer_to_stageid(2, is_spec)
                self.update_documents(stage_input, is_spec)
                generation_query = self.genprompt.get_string(reference = stage_input, question = self.initial_query)
                return [self.get_gtasks(input_str = generation_query, is_spec = is_spec)]
            elif self.exec_stage == 2:
                self.answer = stage_input
                self.transfer_to_stageid(3, is_spec)
                return None
            else:
                raise ValueError("Wrong stage number!")


class RecompRequest(Request):
    def __init__(self, query: str, config):
        super().__init__(query, workflow = "Recomp")
        self.exec_stage = -1
        self.compress_prompt = RecompPrompt(config)
        self.answering_prompt = SequentialGenPrompt(config)

    def update_stage(self, stage_input: str = "", stage_id: int = -1, is_spec: bool = False):

        if stage_id == self.exec_stage or stage_id == -1:
            if self.exec_stage == -1:
                self.transfer_to_stageid(0, is_spec)
                return [self.get_rtasks()]
            elif self.exec_stage == 0:
                self.transfer_to_stageid(1, is_spec)
                self.update_documents(stage_input, is_spec)
                generation_query = self.compress_prompt.get_string(reference = stage_input, question = self.initial_query)
                return [self.get_gtasks(input_str = generation_query, is_spec = is_spec)]
            elif self.exec_stage == 1:
                self.transfer_to_stageid(2, is_spec)
                self.update_documents(stage_input, is_spec)
                generation_query = self.answering_prompt.get_string(reference = stage_input, question = self.initial_query)
                return [self.get_gtasks(input_str = generation_query, is_spec = is_spec)]
            elif self.exec_stage == 2:
                self.answer = stage_input
                self.transfer_to_stageid(3, is_spec)
                return None
            else:
                raise ValueError("Wrong stage number!")