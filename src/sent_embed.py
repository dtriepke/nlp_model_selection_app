import nlu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 40)
pd.options.display.float_format = "{:.2f}".format


# Document
data_doc = pd.read_csv("data/doc_embed_demo.csv", sep=",", header=[0], encoding="utf-8", dtype = "unicode")
model_pipe_doc = nlu.load("elmo")
predictions_doc = model_pipe_doc.predict(data_doc[["doc"]], output_level='document', positions=True)

predictions_doc.elmo_embeddings.shape # 20 docs
predictions_doc.elmo_embeddings[0].__len__() # 121  # token
predictions_doc.elmo_embeddings[0][0].__len__() # 512 embedding dim
# Document Embedding dim
# (#Docs, #Token, #Embed)
predictions_doc.columns

# Sentence
data_sent = pd.read_csv("data/sent_embed_demo.csv", sep =",", header = [0], encoding = "utf-8", dtype = "unicode")
model_pipe_sent = nlu.load("bert")
predictions_sent = model_pipe_sent.predict(data_sent[["doc"]], output_level='sentence', positions=True)

predictions_sent.bert_embeddings.shape # 9 sentences
predictions_sent.bert_embeddings[0].__len__() # 10 word
data_sent.doc[0].split().__len__() # 10
predictions_sent.bert_embeddings[0][0].shape # 128 embedding dim
# Sentence Embedding dim
# (9, #Token, 128)

predictions_sent.columns





