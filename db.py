#!/usr/bin/python3

from os import listdir
from os.path import splitext, join, exists
from langchain,document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

class DocDatabase(object):
  def __init__(self):
    pass
  @staticmethod
  def load_db(db_dir):
    vectordb = Chroma(persist_directory = db_dir)
    return vectordb
  @staticmethod
  def load_doc(doc_dir, db_dir)
    # 1) load pages of documents to list docs
    docs = list()
    for f in listdir(doc_dir):
      stem, ext = splitext(f)
      loader_types = {'.md': UnstructuredMarkdownL,
                      '.txt': UnstructuredFileLoader,
                      '.pdf': UnstructuredPDFLoader}
      loader = loader_types[ext](join(doc_dir, f), mode = "single", strategy = "fast")
      # load pages of a document to a list
      docs.extend(loader.load())
    # 2) split pages into chunks and save to split_docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
    split_docs = text_splitter.split_documents(docs)
    # 3) encode strings to feature vectors
    # NOTE: alternative model "distilbert/distilbert-base-uncased"
    embeddings = HuggingFaceEmbeddings(model_name = "autodl-tmp/sentence-transformer")
    vectordb = Chroma.from_documents(
        documents = split_docs,
        embedding = embeddings,
        persis_directory = db_dir)
    vectordb.persist()
