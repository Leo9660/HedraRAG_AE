# An structure for generation/retrieval task
class TaskID():
    def __init__(self, id_in: int, stage_in: int, type_in: str, input_str: str, is_spec: bool = False):
        self.id = id_in
        self.stage = stage_in
        self.type = type_in
        self.input = input_str
        self.is_spec = is_spec
        self.begin_spec = False
        self.need_regeneration = True

    def __str__(self):
        return f"TaskID(id={self.id}, stage={self.stage}, type={self.type}, input={self.input})"
