import os
import pandas as pd, matplotlib.pyplot as plt, numpy as np, json, csv
import mne

# sklearn
import sklearn
import sklearn.model_selection
from sklearn.metrics import ndcg_score

from pypdf import PdfReader
import glob

import kagglehub

import chromadb

# embedding models and chroma vector store
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
from huggingface_hub import model_info

from langchain_classic.chains import RetrievalQA

from langchain_groq import ChatGroq

# split text in chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

from functools import partial



def open_jsonl(path):
    if os.path.isdir(path):
        path = os.path.join(path, os.listdir(path)[0])
        data = pd.read_json(path, lines=True)
        return data
    with open(path,"r") as file:
        data=pd.read_json(path, lines=True)
    return data




def split_and_chunk(dataframe, text_splitter):
    dataframe['chunks'] = dataframe['text'].apply(lambda t: text_splitter.split_text(t))

    texts = []
    metadatas = []
    
    for _, row in dataframe.iterrows():
        for chunk in row["chunks"]:
            texts.append(chunk)
            metadatas.append({
                        "doc_id": row["_id"],
                        "title": row["title"]
            })

    return dataframe, texts, metadatas




def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)




def call_llm(prompt_value, llm_model, client, **kwargs):
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt_value.to_string()}],
    )
    return response.choices[0].message.content




def compute_ndcg_at_k(pred_df, gt_df, k=10):
    """
    pred_df: colonne query_id, corpus_id_pred, rank
    gt_df: colonne query_id, corpus_id (i qrels, uno o più doc rilevanti per query)
    """
    scores = []

    for query_id in gt_df['query_id'].unique():
        relevant_ids = set(gt_df[gt_df['query_id'] == query_id]['corpus_id'])

        query_preds = pred_df[pred_df['query_id'] == query_id].sort_values('rank')
        retrieved_ids = query_preds['corpus_id_pred'].tolist()[:k]

        if len(retrieved_ids) == 0:
            scores.append(0.0)
            continue

        relevance = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids]
        ideal_relevance = sorted(relevance, reverse=True)

        if sum(ideal_relevance) == 0:  # nessun documento rilevante trovabile
            scores.append(0.0)
            continue

        score = ndcg_score([ideal_relevance], [relevance], k=k)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0




def load_datasets(DATA_PATH, DATASETS_TO_USE):
    data = {}
    for name in DATASETS_TO_USE:
        p = lambda suffix: os.path.join(DATA_PATH, f'{name.lower()}_{suffix}')
        data[name] = {
            "corpus":  open_jsonl(p('corpus.jsonl'))  if os.path.exists(p('corpus.jsonl'))  else None,
            "queries": open_jsonl(p('queries.jsonl')) if os.path.exists(p('queries.jsonl')) else None,
            "qrels":   pd.read_csv(p('qrels.tsv'), sep='\t', engine='python', quoting=csv.QUOTE_NONE)
                       if os.path.exists(p('qrels.tsv')) else None,
        }
    return data




def build_embedding_model(model_name):  
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )




def get_vector_store(ds_name, corpus, embedding_model, text_splitter, CHROMA_PATH):
    collection_name = f"{ds_name.lower()}_store"
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = [c.name for c in chroma_client.list_collections()]

    if collection_name in existing:
        print(f"{collection_name}: already existing. Loading it...")
        return Chroma(collection_name=collection_name,
                      embedding_function=embedding_model,
                      persist_directory=CHROMA_PATH)

    print(f"{collection_name}: not found, computing embeddings...")
    _, texts, metadatas = split_and_chunk(corpus, text_splitter)
    return Chroma.from_texts(texts=texts, collection_name=collection_name,
                             embedding=embedding_model, metadatas=metadatas,
                             persist_directory=CHROMA_PATH)




def build_text_splitter(kind):
    if kind.lower() != "recursive":
        print("Only recursive available for now..")
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=0,
    )