#!/usr/bin/python3

from langchain_core.prompts.prompt import PromptTemplate

def rag_template(tokenizer, is_chatglm = False):
  if is_chatglm:
    messages = [
      {'role': 'system', 'content': '使用给定的上下文来回答问题。如果上下文中找不到答案的线索，就说你不知道，不要试图编造答案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！'},
      {'role': 'user', 'content': '上下文：{context}\n问题：{question}\n回答：'}
    ]
  else:
    messages = [
      {'role': 'system', 'content': "Please answer the question based on the given context. If the context gives no clue to the question, just say you don't know and don't try to make up a answer. Try to keep your answers as concise as possible. Always end your answer by saying \"Thank you for asking!\"."},
      {'role': 'user', 'content': "context: {context}\nquestion: {question}\nanswer:"}
    ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context', 'question'])
  return template

