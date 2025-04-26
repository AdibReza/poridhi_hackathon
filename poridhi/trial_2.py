# # Install dependencies
# # !pip install datasets sentence-transformers faiss-cpu pandas

# from datasets import load_dataset
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import pandas as pd

# # Step 1: Load dataset (choose variant: cameras_small, cameras_medium, cameras_large, etc.)
# dataset = load_dataset("wdc/products-2017", "cameras_small")  # change variant as needed
# df = pd.DataFrame(dataset['train'])  # train, validation, or test split

# # Step 2: Combine left and right product info
# def build_corpus(df):
#     left = df[['id_left', 'title_left', 'description_left']].drop_duplicates(subset='id_left')
#     right = df[['id_right', 'title_right', 'description_right']].drop_duplicates(subset='id_right')

#     left.columns = ['id', 'title', 'description']
#     right.columns = ['id', 'title', 'description']

#     combined = pd.concat([left, right]).drop_duplicates(subset='id')
#     combined['text'] = (combined['title'].fillna('') + ' ' + combined['description'].fillna('')).str.strip()

#     return combined.reset_index(drop=True)

# corpus = build_corpus(df)

# # Step 3: Embed text using Sentence-BERT
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(corpus['text'].tolist(), convert_to_numpy=True, show_progress_bar=True)

# # Step 4: Build FAISS index
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(embeddings)

# # Step 5: Define search function
# def search(query, model, index, corpus, k=5):
#     query_vec = model.encode([query], convert_to_numpy=True)
#     D, I = index.search(query_vec, k)
#     return corpus.iloc[I[0]][['id', 'title', 'description']]

# # Example query
# query = "camera 4k video"
# results = search(query, model, index, corpus)
# print(results)



# # Install Hugging Face datasets if not already installed
# # !pip install datasets

# Install dependencies (run once)
# pip install datasets sentence-transformers faiss-cpu pandas

# trial.py

# Install required packages (uncomment if running first time)
# !pip install datasets sentence-transformers faiss-cpu pandas

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import re

# ================================
# STEP 1: Load and Process Dataset
# ================================
def load_and_prepare_dataset():
    dataset = load_dataset("wdc/products-2017", "cameras_small")
    df = pd.DataFrame(dataset['train'])  # choose 'train', 'validation', or 'test'
    left = df[['id_left', 'title_left', 'description_left']].drop_duplicates(subset='id_left')
    right = df[['id_right', 'title_right', 'description_right']].drop_duplicates(subset='id_right')
    left.columns = ['id', 'title', 'description']
    right.columns = ['id', 'title', 'description']
    combined = pd.concat([left, right]).drop_duplicates(subset='id')
    combined['text'] = (combined['title'].fillna('') + ' ' + combined['description'].fillna('')).str.strip()
    return combined.reset_index(drop=True)

corpus = load_and_prepare_dataset()

# ================================
# STEP 2: Embed Text & Build Index
# ================================
print("🔄 Embedding text...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(corpus['text'].tolist(), convert_to_numpy=True, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ====================
# STEP 3: Intent Logic
# ====================
def classify_intent(query):
    query = query.lower()
    if "compare" in query:
        return "compare"
    elif "under" in query or "less than" in query or "below" in query:
        return "filter_price"
    elif any(word in query for word in ["4k", "waterproof", "image stabilization", "zoom", "wireless", "night vision"]):
        return "filter_description"
    else:
        return "search"

# ====================
# STEP 4: Search Logic
# ====================
def search_semantic(query, k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k)
    return corpus.iloc[I[0]][['id', 'title', 'description']]

def filter_by_price(results, query):
    price_match = re.search(r'(\d+)', query)
    if price_match:
        max_price = float(price_match.group(1))
        if 'price' in results.columns:
            results['price'] = pd.to_numeric(results['price'], errors='coerce')
            return results[results['price'] < max_price]
    return results

def filter_by_description(results, query):
    query_keywords = query.lower().split()
    return results[results['description'].fillna('').apply(
        lambda desc: all(keyword in desc.lower() for keyword in query_keywords)
    )]

def compare_products(product_names, corpus):
    comparisons = []
    for name in product_names:
        result = search_semantic(name, k=1)
        comparisons.append(result)
    return pd.concat(comparisons)

def extract_product_names(query):
    query = query.lower().replace("compare", "").strip()
    parts = re.split(" and | vs | with ", query)
    return [p.strip() for p in parts if p]


def handle_query(query):
    intent = classify_intent(query)
    print(f"\n🧠 Intent Detected: {intent}")
    
    if intent == "compare":
        product_names = extract_product_names(query)
        return compare_products(product_names, corpus)
    elif intent == "filter_price":
        results = search_semantic(query, k=10)
        return filter_by_price(results, query)
    elif intent == "filter_description":
        results = search_semantic(query, k=10)
        return filter_by_description(results, query)
    else:
        return search_semantic(query, k=5)

if __name__ == "__main__":
    while True:
        user_input = input("\n🔎 Enter your query (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        result_df = handle_query(user_input)
        print("\n📦 Top Results:\n", result_df[['title', 'description']].head(5))
