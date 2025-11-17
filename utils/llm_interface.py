import time
from typing import Optional
import time
import logging
import concurrent
from random import random
from openai import OpenAI
from typing import Any, Dict, Iterable, List, Sequence, Union
import numpy as np
logger = logging.getLogger(__name__)

class BaseClient(object):
    def __init__(self, model, temperature=1.0):
        self.model = model
        self.temperature = temperature
    
    def _chat_completion_api(self, messages, temperature, n=1):
        raise NotImplemented
    
    def chat_completion(self, n, messages, temperature=None):
        """
        Generate n responses using OpenAI Chat Completions API
        """
        temperature = temperature or self.temperature
        time.sleep(random())
        for attempt in range(1000):
            try:
                response_cur = self._chat_completion_api(messages, temperature, n)
            except Exception as e:
                logger.exception(e)
                logger.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
            else:
                break
        if response_cur is None:
            logger.info("Code terminated due to too many failed attempts!")
            exit()
            
        return response_cur
    
    def multi_chat_completion(self, messages_list, n=1, temperature=None):
        """
        An example of messages_list:
        
        messages_list = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            [
                {"role": "system", "content": "You are a knowledgeable guide."},
                {"role": "user", "content": "How are you?"},
            ],
            [
                {"role": "system", "content": "You are a witty comedian."},
                {"role": "user", "content": "Tell me a joke."},
            ]
        ]
        param: n: number of responses to generate for each message in messages_list
        """
        # If messages_list is not a list of list (i.e., only one conversation), convert it to a list of list
        assert isinstance(messages_list, list), "messages_list should be a list."
        if not isinstance(messages_list[0], list):
            messages_list = [messages_list]
        
        if len(messages_list) > 1:
            assert n == 1, "Currently, only n=1 is supported for multi-chat completion."
        
        if "gpt" not in self.model:
            # Transform messages if n > 1
            messages_list *= n
            n = 1

        with concurrent.futures.ThreadPoolExecutor() as executor:
            args = [dict(n=n, messages=messages, temperature=temperature) for messages in messages_list]
            choices = executor.map(lambda p: self.chat_completion(**p), args)

        contents: list[str] = []
        for choice in choices:
            for c in choice:
                contents.append(c.message.content)
        return contents

class OpenAIClient(BaseClient):

    ClientClass = OpenAI

    def __init__(self, model, temperature=1.0, base_url=None, api_key=None):
        super().__init__(model, temperature)
        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)
        
        self.client = self.ClientClass(api_key=api_key, base_url=base_url)
    
    def _chat_completion_api(self, messages, temperature):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, stream=False, timeout=120
        )
        return response.choices[0].message.content


class OpenAIEmbedding:
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs: Any):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.embedding_kwargs: Dict[str, Any] = dict(kwargs)

    def encode(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """
        生成输入文本的向量嵌入。

        参数:
            texts (Union[str, Sequence[str]]): 单条或多条文本。

        返回:
            numpy.ndarray: 当输入为字符串时返回形状为 (d,) 的数组；序列时返回 (n, d) 数组。
        """
        single_input = isinstance(texts, str)
        inputs: Iterable[str]
        if single_input:
            inputs = [texts]
        else:
            inputs = list(texts)

        response = self.client.embeddings.create(
            model=self.model,
            input=list(inputs),
            **self.embedding_kwargs,
        )
        vectors: List[List[float]] = [item.embedding for item in response.data]
        embeddings = np.array(vectors, dtype=np.float32)
        return embeddings[0] if single_input else embeddings