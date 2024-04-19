#!/usr/bin/python3

from typing import List
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper, RepetitionPenaltyLogitsProcessor
from langchain.llms.base import LLM

class ChatGLM3(LLM):
  tokenizer: AutoTokenizer = None
  model: AutoModelForCausalLM = None
  use_history: bool = None
  history: List = None
  def __init__(self, device = 'cuda', use_history = True):
    assert device in {'cpu', 'cuda'}
    super().__init__()
    login(token = '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')
    self.tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
    self.model = AutoModelForCausalLM.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
    self.model = self.model.to(torch.device(device))
    self.model.eval()
    self.use_history = use_history
    self.history = list()
  def _call(self, prompt, stop = None, run_manager = None, **kwargs):
    if not self.use_history:
      self.history = list()
    response, self.history = self.model.chat(self.tokenizer, prompt, history = self.history, use_cache = True)
    if len(self.history) > 10:
      self.history.pop(0)
    return response
  @property
  def _llm_type(self):
    return "ChatGLM3-6B"

class Llama2(LLM):
  tokenizer: AutoTokenizer = None
  model: AutoModelForCausalLM = None
  def __init__(self, device = 'cuda'):
    assert device in {'cpu', 'cuda'}
    super().__init__()
    login(token = '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')
    self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    self.model = self.model.to(torch.device(device))
    self.model.eval()
  def _call(self, prompt, stop = None, run_manager = None, **kwargs):
    logits_processor = LogitsProcessorList()
    logits_processor.append(TemperatureLogitsWarper(0.8))
    logits_processor.append(TopPLogitsWarper(0.8))
    inputs = self.tokenizer(prompt, return_tensors = 'pt')
    inputs = inputs.to(torch.device(self.model.device))
    outputs = self.model.generate(**inputs, logits_processor = logits_processor, do_sample = True, use_cache = True, return_dict_in_generate = True)
    input_ids = outputs.sequences
    outputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens = True)
    response = outputs[0][len(prompt):]
    return response
  @property
  def _llm_type(self):
    return "Llama-2-7b-chat-hf"

class ChemDFM(LLM):
  tokenizer: AutoTokenizer = None
  model: AutoModelForCausalLM = None
  def __init__(self, device = 'cuda'):
    assert device in {'cpu', 'cuda'}
    super().__init__()
    login(token = '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')
    self.tokenizer = AutoTokenizer.from_pretrained('OpenDFM/ChemDFM-13B-v1.0')
    self.model = AutoModelForCausalLM.from_pretrained('OpenDFM/ChemDFM-13B-v1.0')
    self.model = self.model.to(torch.device(device))
    self.model.eval()
  def _call(self, prompt, stop = None, run_manager = None, **kwargs):
    logits_processor = LogitsProcessorList()
    logits_processor.append(TemperatureLogitsWarper(0.9))
    logits_processor.append(TopKLogitsWarper(20))
    logits_processor.append(TopPLogitsWarper(0.9))
    logits_processor.append(RepetitionPenaltyLogitsProcessor(1.05))
    s = '[Round 0]\nHuman: %s\nAssistant: ' % prompt
    inputs = self.tokenizer(s, return_tensors = 'pt')
    inputs = inputs.to(torch.device(self.model.device))
    outputs = self.model.generate(**inputs, logits_processor = logits_processor, do_sample = False, use_cache = True, return_dict_in_generate = True, max_new_tokens = 2048)
    input_ids = outputs.sequences
    outputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens = True)
    response = outputs[0][len(s):]
    return response
  @property
  def _llm_type(self):
    return "ChemDFM-13b"

