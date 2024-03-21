#!/usr/bin/python3

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM

class ChatGLM3(LLM):
  tokenizer: AutoTokenizer = None
  model: AutoModelForCausalLM = None
  use_history: bool = None
  past_key_values: List = None
  def __init__(self, device = 'cuda', use_history = True):
    assert device in {'cpu', 'cuda'}
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
    self.model = AutoModelForCausalLM.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
    self.model = self.model.to(torch.device(device))
    self.model.eval()
    self.use_history = use_history
    self.history = list()
    self.past_key_values = None
  def _call(self, prompt, stop = None, run_manager = None, **kwargs):
    if not self.use_history:
      self.history = list()
    response, self.history, self.past_key_values = self.model.stream_chat(self.tokenizer, prompt, history = self.history, past_key_values = self.past_key_values, use_cache = True, return_past_key_values = True)
    if len(self.history) > 10:
      self.history.pop(0)
      self.past_key_values = None
    return response
  @property
  def _llm_type(self):
    return "ChatGLM3-6B"
