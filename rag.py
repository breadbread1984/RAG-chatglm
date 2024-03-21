#!/usr/bin/python3

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from models import ChatGLM3
from db import DocDatabase

class RAG(object):
  def __init__(self, device = 'cuda', db_dir = None, doc_dir = None):
    prompt = PromptTemplate.from_template("使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。\n{context}\n问题: {question}\n有用的回答:")
    llm = ChatGLM3(device = device)
    if db_dir is not None:
      db = DocDatabase.load_db(db_dir)
    elif db_dir is not None:
      db = docDatabase.load_doc(doc_dir)
    else:
      raise Exception('either db_dir or doc_dir is given')
    self.chain = RetrievalQA.from_chain_type(llm, retriever = db.as_retriever(), return_source_documents = True, chain_type_kwargs = {"prompt": prompt})
  def query(self, question):
    return self.chain({'query': question})

