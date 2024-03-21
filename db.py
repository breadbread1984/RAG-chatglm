#!/usr/bin/python3

from os import listdir
from os.path import splitext, join, exists
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain,document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader

class DocDatabase(object):
  def __init__(self,):
    pass
  def load_doc(self, doc_dir)
    docs = list()
    for f in listdir(doc_dir):
      stem, ext = splitext(f)
      loader_types = {'.md': UnstructuredMarkdownL,
                      '.txt': UnstructuredFileLoader,
                      '.pdf': UnstructuredPDFLoader}
      loader = loader_types[ext](join(doc_dir, f), mode = "single", strategy = "fast")
      # load pages of a document to a list
      docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
    split_docs = text_splitter.split_documents(docs)

