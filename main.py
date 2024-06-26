#!/usr/bin/python3

from absl import flags, app
from os.path import exists
import gradio as gr
from rag import RAG

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('doc_dir', default = None, help = 'path to document directory')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'host address')
  flags.DEFINE_integer('port', default = 8880, help = 'port number')
  flags.DEFINE_enum('model', default = 'chatglm', enum_values = {'chatglm', 'llama2', 'llama3', 'chemdfm'}, help = 'model to use')

class Warper(object):
  def __init__(self):
    if FLAGS.doc_dir is not None:
      self.rag = RAG(device = FLAGS.device, model = FLAGS.model, doc_dir = FLAGS.doc_dir, db_dir = 'vector_database')
    elif exists('vector_database'):
      self.rag = RAG(device = FLAGS.device, model = FLAGS.model, db_dir = 'vector_database')
    else:
      raise Exception('vector database has not been initialized, please specify directory containing documents!')
  def query(self, question, history):
    try:
      answer = self.rag.query(question)
      history.append((question, answer['result']))
      return "", history
    except Exception as e:
      return e, history

def main(unused_argv):
  warper = Warper()
  block = gr.Blocks()
  with block as demo:
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>文献问答系统</center></h1>")
    with gr.Row():
      with gr.Column(scale = 4):
        chatbot = gr.Chatbot(height = 450, show_copy_button = True)
        msg = gr.Textbox(label = "需要问什么？")
        with gr.Row():
          submit_btn = gr.Button("发送")
          clear_btn = gr.ClearButton(components = [chatbot], value = "清空问题")
      submit_btn.click(warper.query, inputs = [msg, chatbot], outputs = [msg, chatbot])
  gr.close_all()
  demo.launch(server_name = FLAGS.host, server_port = FLAGS.port)

if __name__ == "__main__":
  add_options()
  app.run(main)

