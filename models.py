#!/usr/bin/python3

from typing import List
from copy import deepcopy
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper, RepetitionPenaltyLogitsProcessor
from langchain.llms.base import LLM

class ChatGLM3(LLM):
  tokenizer: AutoTokenizer = None
  model: AutoModelForCausalLM = None
  def __init__(self, device = 'cuda', use_history = True):
    assert device in {'cpu', 'cuda'}
    super().__init__()
    login(token = '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')
    self.tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
    self.model = AutoModelForCausalLM.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
    self.model = self.model.to(torch.device(device))
    self.model.eval()
  def process_response(self, output, history):
    content = ""
    history = deepcopy(history)
    for response in output.split("<|assistant|>"):
      if "\n" in response:
        metadata, content = response.split("\n", maxsplit=1)
      else:
        metadata, content = "", response
      if not metadata.strip():
        content = content.strip()
        history.append({"role": "assistant", "metadata": metadata, "content": content})
        content = content.replace("[[训练时间]]", "2023年")
      else:
        history.append({"role": "assistant", "metadata": metadata, "content": content})
        if history[0]["role"] == "system" and "tools" in history[0]:
          content = "\n".join(content.split("\n")[1:-1])
          def tool_call(**kwargs):
            return kwargs
          parameters = eval(content)
          content = {"name": metadata.strip(), "parameters": parameters}
        else:
          content = {"name": metadata.strip(), "content": content}
    return content, history
  def _call(self, prompt, stop = None, run_manager = None, **kwargs):
    logits_processor = LogitsProcessorList()
    logits_processor.append(TemperatureLogitsWarper(0.8))
    logits_processor.append(TopPLogitsWarper(0.8))
    inputs = self.tokenizer(prompt, return_tensors = 'pt')
    outputs = self.model.generate(**inputs, logits_processor = logits_processor, do_sample = True, use_cache = True)
    outputs = outputs.tolist()[0][len(inputs['input_ids'][0]):-1]
    response = self.tokenizer.decode(outputs)
    history = list()
    response, history = self.process_response(response, history)
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
    self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', attn_implementation = 'flash_attention_2')
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

class Llama3(LLM):
  tokenizer: AutoTokenizer = None
  model: AutoModelForCausalLM = None
  def __init__(self, device = 'cuda'):
    assert device in {'cpu', 'cuda'}
    super().__init__()
    login(token = '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')
    self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', attn_implementation = 'flash_attention_2')
    self.model = self.model.to(torch.device(device))
    self.model.eval()
  def _call(self, prompt, stop = None, run_manager = None, **kwargs):
    logits_processor = LogitsProcessorList()
    logits_processor.append(TemperatureLogitsWarper(0.6))
    logits_processor.append(TopPLogitsWarper(0.9))
    inputs = self.tokenizer(prompt, return_tensors = 'pt')
    inputs = inputs.to(torch.device(self.model.device))
    outputs = self.model.generate(**inputs, logits_processor = logits_processor, do_sample = True, use_cache = True, return_dict_in_generate = True, eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')], max_new_tokens = 4096)
    input_ids = outputs.sequences
    response = self.tokenizer.decode(input_ids[0][inputs.shape[-1]:], skip_special_tokens = True)
    return response
  @property
  def _llm_type(self):
    return "Llama-3-8b-hf"

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

