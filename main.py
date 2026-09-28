# ================= 0. SETUP =================
# import, constants (paths, DATASETS_TO_USE), parse_args()
# load_datasets()        -> corpus, queries, qrels of each dataset
# build_embedding_model()-> HuggingFaceEmbeddings
# get_vector_store(...)  -> load or create crhoma vector store

# ================= TASK 1: RETRIEVAL =================
# run_task1(...)  -> for each query: retriever.invoke -> top-10 doc_id
#                    salva RESULTS/<ds>_predictions.csv
#                    computes NDCG@10 -> RESULTS/metrics.txt

# ================= TASK 2: GENERATION =================
# build_llm()     -> ChatGroq
# run_task2(...)  -> for each query: retrieval + prompt + LLM
#                    saves RESULTS/<ds>_answers.csv

# ================= MAIN =================
# main(): parse_args -> setup -> run_task1 and/or run_task2



# BLOCK 0: SETUP

from utils.functions import *
import argparse
from functools import partial
from dotenv import load_dotenv

load_dotenv()


BASE_PATH = '/workspaces/llm_finetuning'
DATA_PATH = '/workspaces/llm_finetuning/FinanceRAG'
DATASETS_TO_USE = ['FinDER', 'FinQABench', 'FinanceBench']
RESULTS_PATH = "/workspaces/llm_finetuning/RESULTS"
CHROMA_PATH = os.path.join(BASE_PATH, 'vector_store_chroma')

os.makedirs(RESULTS_PATH, exist_ok=True) 

RESULTS_TXT = os.path.join(RESULTS_PATH, "metrics.txt")


custom_rag_prompt = PromptTemplate.from_template("""Use the context provided to answer
the user's question below. If you do not know the answer based on the context provided, say so.

context: {context}

question: {question}

answer: """)


DEFAULT_LLM = {
    "groq": "meta-llama/llama-prompt-guard-2-22m",                       # ID da verificare su Groq
    "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1", "2", "both"], default="1")
    parser.add_argument("--splitter", default="recursive")
    parser.add_argument("--embedding_model", default='all-MiniLM-L6-v2')
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--api_provider", choices=["groq", "huggingface"], default="groq")
    parser.add_argument("--llm", default=None)
    parser.add_argument("--max_queries", type=int, default=None)  
    args = parser.parse_args()

    return args



# TASK 1
def run_task1(ds_name, d, vector_store, top_k):
    pred_csv_path = os.path.join(RESULTS_PATH, f"{ds_name.lower()}_predictions.csv")

    if os.path.exists(pred_csv_path):
        print(f"{ds_name}: predictions already present.")
        pred_df = pd.read_csv(pred_csv_path)
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        rows = []
        for query_id in d["qrels"]["query_id"].unique():
            query_text = d["queries"][d["queries"]["_id"] == query_id].iloc[0]["text"]
            docs = retriever.invoke(query_text)
            for rank, doc in enumerate(docs, start=1):
                rows.append([query_id, doc.metadata.get("doc_id"), rank])

        pred_df = pd.DataFrame(rows, columns=["query_id", "corpus_id_pred", "rank"])
        pred_df.to_csv(pred_csv_path, index=False)

    ndcg = compute_ndcg_at_k(pred_df, d["qrels"], k=top_k)
    print(f"{ds_name} — NDCG@{top_k}: {ndcg:.4f}")
    return ndcg



def build_llm(provider, model_name):
    """Given a prompt, returns a string"""
    if provider == "groq":
        return ChatGroq(model=model_name, temperature=0) | StrOutputParser()

    if provider == "huggingface":
        hf_client = InferenceClient(provider="featherless-ai",
                                    api_key=os.environ["HF_TOKEN"])
        return RunnableLambda(partial(call_llm, llm_model=model_name, client=hf_client))

    raise ValueError(f"Provider not supported: {provider}")



def run_task2(ds_name, d, vector_store, llm, top_k, max_queries=None):

    answers_path = os.path.join(RESULTS_PATH, f"{ds_name.lower()}_answers.csv")
    if os.path.exists(answers_path):
        done = pd.read_csv(answers_path)
    else:
        done = pd.DataFrame(columns=["query_id", "answer", "retrieved_doc_ids"])
    done_ids = set(done["query_id"])

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    chain = custom_rag_prompt | llm

    query_ids = [q for q in d["qrels"]["query_id"].unique() if q not in done_ids]
    if max_queries:
        query_ids = query_ids[:max_queries]

    for query_id in query_ids:
        question = d["queries"][d["queries"]["_id"] == query_id].iloc[0]["text"]
        docs = retriever.invoke(question)
        try:
            answer = chain.invoke({"context": format_docs(docs), "question": question})
        except Exception as e:
            print(f"{ds_name}: mi fermo su {query_id}: {e}")
            break
        done.loc[len(done)] = [query_id, answer,
                               ";".join(str(x.metadata.get("doc_id")) for x in docs)]
        done.to_csv(answers_path, index=False)  

    return done



def main():
    args = parse_args()

    data = load_datasets(DATA_PATH, DATASETS_TO_USE)
    embedding_model = build_embedding_model(args.embedding_model)
    text_splitter = build_text_splitter(args.splitter)

    llm = None
    if args.task in ("2", "both"):
        llm = build_llm(args.api_provider, args.llm or DEFAULT_LLM[args.api_provider])

    ndcg_scores = {}
    for ds_name in DATASETS_TO_USE:
        d = data[ds_name]
        if d["corpus"] is None:
            print(f"{ds_name}: corpus missing, skipping")
            continue
        vector_store = get_vector_store(ds_name, d["corpus"], embedding_model, text_splitter, CHROMA_PATH)

        if args.task in ("1", "both"):
            ndcg_scores[ds_name] = run_task1(ds_name, d, vector_store, args.top_k)
        if args.task in ("2", "both"):
            run_task2(ds_name, d, vector_store, llm, args.top_k, args.max_queries)

    if ndcg_scores:
        with open(RESULTS_TXT, "w") as f:
            for name, score in ndcg_scores.items():
                f.write(f"{name}: NDCG@{args.top_k} = {score:.4f}\n")
            mean = sum(ndcg_scores.values()) / len(ndcg_scores)
            f.write(f"OVERALL (mean): NDCG@{args.top_k} = {mean:.4f}\n")   


         
 

if __name__ == "__main__":
   main()