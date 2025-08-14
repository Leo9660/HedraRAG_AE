from transformers import AutoTokenizer, AutoConfig
import tiktoken
import sys
import os
from abc import ABC, abstractmethod
from heterag.prompt.prompt_example import *

def get_content(s):
    if ('contents' in s):
        return s["contents"]
    elif ('title' in s and 'text' in s):
        return s['title'] + "\n" + s['text']
    elif isinstance(s, str):
        return s
    else:
        raise ValueError("Get data content failed!")

class BasePrompt(ABC):
    _shared_config = None
    _shared_tokenizer = None

    def __init__(self, name, config):

        self.config = config
        self.is_openai = config["framework"] == "openai"
        self.max_input_len = config['generator_max_input_len']
        if not self.is_openai:
            if BasePrompt._shared_config is None or BasePrompt._shared_tokenizer is None:
                print("Loading AutoConfig and AutoTokenizer...")
                generator_path = config["generator_model_path"]
                BasePrompt._shared_config = AutoConfig.from_pretrained(generator_path, trust_remote_code=True)
                BasePrompt._shared_tokenizer = AutoTokenizer.from_pretrained(generator_path, trust_remote_code=True)

            self.tokenizer = BasePrompt._shared_tokenizer
            model_name = BasePrompt._shared_config._name_or_path.lower()

            self.is_chat = "chat" in model_name or "instruct" in model_name
            self.enable_chat = self.is_chat
        else:
            self.is_chat = True
            self.enable_chat = True
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

        self.name = name
        self.system_prompt = ""
        self.user_prompt = ""
        self.placeholders = []

    def check_placeholder(self):
        # check placeholder in prompt
        for holder in self.placeholders:
            flag = False
            for prompt in [self.system_prompt, self.user_prompt]:
                if f"{holder}" in prompt:
                    print(f"Find `{holder}` in template")
                    flag = True
                    break
            if not flag and holder != "reference":
                assert False

    def truncate_prompt(self, prompt):
        if self.is_openai:
            truncated_messages = []
            total_tokens = 0
            assert isinstance(prompt, list)
            for message in prompt:
                role_content = message['content']
                encoded_message = self.tokenizer.encode(role_content)

                if total_tokens + len(encoded_message) <= self.max_input_len:
                    truncated_messages.append(message)
                    total_tokens += len(encoded_message)
                else:
                    print(f"The input text length is greater than the maximum length ({total_tokens + len(encoded_message)} > {self.max_input_len}) and has been truncated!")
                    remaining_tokens = self.max_input_len - total_tokens
                    truncated_message = self.encoding.decode(encoded_message[:remaining_tokens])
                    message['content'] = truncated_message
                    truncated_messages.append(message)
                    break

            return truncated_messages

        else:
            assert isinstance(prompt, str)
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

            if len(tokenized_prompt) > self.max_input_len:
                print(f"The input text length is greater than the maximum length ({len(tokenized_prompt)} > {self.max_input_len}) and has been truncated!")
                half = int(self.max_input_len / 2)
                prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                        self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            return prompt

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = get_content(doc_item)
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            if self.reference_template is not None:
                format_reference += self.reference_template.format(idx=idx, title=title, text=text)
            else:
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def get_string(self, question=None, reference=None, messages=None, formatted_reference=None, previous_gen=None, **params):

        if messages is not None:
            if isinstance(messages, str):
                return self.truncate_prompt(messages)
            if self.is_chat and self.enable_chat:
                if self.is_openai:
                    for item in input:
                        if item["role"] == "system":
                            item["role"] = "assistant"
                    self.truncate_prompt(messages)
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    return self.truncate_prompt(prompt)
            else:
                prompt = "\n\n".join(
                    [message['content'] for message in messages if message['content']]
                )
                return self.truncate_prompt(prompt)

        if formatted_reference is None:
            if reference is not None:
                formatted_reference = self.format_reference(reference)
            else:
                formatted_reference = ""

        input_params = {"question": question, "reference": formatted_reference}
        input_params.update(**params)

        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        if self.is_chat and self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            if self.is_openai:
                for item in input:
                    if item["role"] == "system":
                        item["role"] = "assistant"
            else:
                input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])

        if previous_gen is not None and previous_gen not in ["", " "] and self.is_openai is False:
            input += previous_gen

        return self.truncate_prompt(input)

class SequentialGenPrompt(BasePrompt):
    def __init__(self, config, reference_template=None):
        super().__init__("Sequential", config)
        self.system_prompt = (
        "Answer the question based on the given document."
        "Only give me the answer and do not output any other words."
        "\nThe following are given documents.\n\n{reference}")
        self.user_prompt = "Question: {question}"
        self.placeholders = ["reference", "question"]
        self.reference_template = reference_template

class SequentialFullGenPrompt(BasePrompt):
    def __init__(self, config, reference_template=None):
        super().__init__("Sequential", config)
        self.system_prompt = (
        "Answer the question based on the given document."
        "Please give a complete sentence."
        "\nThe following are given documents.\n\n{reference}")
        self.user_prompt = "Question: {question}"
        self.placeholders = ["reference", "question"]
        self.reference_template = reference_template

class IterativeGenPrompt(BasePrompt):
    def __init__(self, config, reference_template=None):
        super().__init__("Iterative", config)
        self.system_prompt = (
        # "Please generate a sentence for the following paragraph, based on the information of given documents, do not output other words."
        "Please generate a sentence for the following paragraph, based on the information of given documents, do not output other words."
        "\nThe following are given documents.\n\n{reference}")
        self.user_prompt = "Question: {initial_query}, Paragraph: {question}"
        self.placeholders = ["reference", "initial_query", "question"]
        self.reference_template = reference_template

class MultistepGenPrompt(BasePrompt):
    def __init__(self, config, reference_template=None):
        super().__init__("Multistep", config)
        self.system_prompt = DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_SYS_TMPL
        self.user_prompt = DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_USR_TMPL
        self.placeholders = ["reference", "initial_query", "question", "prev_reasoning"]
        self.reference_template = reference_template

class RecompPrompt(BasePrompt):
    def __init__(self, config, reference_template=None):
        super().__init__("Recomp", config)
        self.system_prompt = (
        # "Please generate a sentence for the following paragraph, based on the information of given documents, do not output other words."
        "Please created a summarized paragraph based on the information of given documents and the question, do not output other words."
        "\nThe following are given documents.\n\n{reference}")
        self.user_prompt = "Question: {question}"
        self.placeholders = ["reference", "question"]
        self.reference_template = reference_template

class HyDEPrompt(BasePrompt):
    def __init__(self, config, reference_template=None):
        super().__init__("HyDE", config)
        self.system_prompt = (
        # "Please generate a sentence for the following paragraph, based on the information of given documents, do not output other words."
        "Please created a hypothsis scentence based on the the question, do not output other words.")
        self.user_prompt = "Question: {question}"
        self.placeholders = ["reference", "question"]
        self.reference_template = reference_template