#!/usr/bin/python3

from shutil import rmtree
from os.path import exists
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from models import ChatGLM3, Llama2, Llama3, ChemDFM
from db import DocDatabase
from prompts import rag_template

class RAG(object):
  def __init__(self, device = 'cuda', model = 'chatglm', db_dir = None, doc_dir = None):
    if model == 'chatglm':
      llm = ChatGLM3(device = device)
    elif model == 'llama2':
      llm = Llama2(device = device)
    elif model == 'llama3':
      llm = Llama3(device = device)
    elif model == 'chemdfm':
      llm = ChemDFM(device = device)
    else:
      raise Exception('unknown model!')
    prompt = rag_template(llm.tokenizer, model == 'chatglm')
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

