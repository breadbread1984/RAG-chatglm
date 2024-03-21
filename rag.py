#!/usr/bin/python3

from shutil import rmtree
from os.path import exists
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from models import ChatGLM3, Llama2
from db import DocDatabase

class RAG(object):
  def __init__(self, device = 'cuda', model = 'chatglm', db_dir = None, doc_dir = None):
    if model == 'chatglm':
      prompt = PromptTemplate.from_template("使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。尽量使答>案简明扼要。总是在回答的最后说“谢谢你的提问！”。\n{context}\n问题: {question}\n有用的回答:")
      llm = ChatGLM3(device = device)
    elif model == 'llama2':
      prmopt = PromptTemplate.from_template("[INST] {context} {question} [/INST]")
      llm = Llama2(device = device)
    else:
      raise Exception('unknown model!')
    if db_dir is not None and doc_dir is not None:
      if exists(db_dir): rmtree(db_dir)
      db = DocDatabase.load_doc(doc_dir, db_dir)
    elif db_dir is not None:
      db = DocDatabase.load_db(db_dir)
    else:
      raise Exception('at least db_dir should be given')
    self.chain = RetrievalQA.from_chain_type(llm, retriever = db.as_retriever(), return_source_documents = True, chain_type_kwargs = {"prompt": prompt})
  def query(self, question):
    return self.chain({'query': question})

