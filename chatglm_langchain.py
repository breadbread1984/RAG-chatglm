#!/usr/bin/python3

from langchain.llms.base import LLM

class ChatGLM3(LLM):
  tokenizer: AutoTokenizer = None
  model: AutoModelForCausalLM = None
  def __init__(self, dev = 'cuda', use_history = True):
    assert dev in {'cpu', 'cuda'}
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
    self.model = AutoModelForCausalLM.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
    self.model = self.model.to(device(dev))
    self.model.eval()
    self.use_history = use_history
    self.history = list()
  def _call(self, prompt, stop = None, run_manager = None, **kwargs):
    if not self.use_history:
      self.history = list()
    response, self.history = self.model.chat(self.tokenizer, prompt, history = self.history)
    if len(self.history) > 10: self.history.pop(0)
    return response
  @property
  def _llm_type(self):
    return "ChatGLM3-6B"
