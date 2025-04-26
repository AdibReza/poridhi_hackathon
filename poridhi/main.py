from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from typing import Optional
import time
import re  # Added for the enhanced search functionality

app = FastAPI(title="Product Search API")

# Configuration
MAX_RESULTS = 10
MODEL_NAME = 'all-MiniLM-L6-v2'

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def load_and_prepare_dataset():
    """Load and prepare the dataset by combining left and right products"""
    dataset = load_dataset("wdc/products-2017", "cameras_small")
    df = pd.DataFrame(dataset['train'])
    left = df[['id_left', 'title_left', 'description_left']].drop_duplicates(subset='id_left')
    right = df[['id_right', 'title_right', 'description_right']].drop_duplicates(subset='id_right')
    left.columns = ['id', 'title', 'description']
    right.columns = ['id', 'title', 'description']
    combined = pd.concat([left, right]).drop_duplicates(subset='id')
    combined['text'] = (combined['title'].fillna('') + ' ' + combined['description'].fillna('')).str.strip()
    return combined.reset_index(drop=True)

def classify_intent(query):
    """Classify the user's intent based on their query"""
    query = query.lower()
    if "compare" in query:
        return "compare"
    elif "under" in query or "less than" in query or "below" in query:
        return "filter_price"
    elif any(word in query for word in ["4k", "waterproof", "image stabilization", "zoom", "wireless", "night vision"]):
        return "filter_description"
    else:
        return "search"

def extract_product_names(query):
    """Extract product names from a comparison query"""
    query = query.lower().replace("compare", "").strip()
    parts = re.split(" and | vs | with ", query)
    return [p.strip() for p in parts if p]

def filter_by_price(results, query):
    """Filter results by price based on the query"""
    price_match = re.search(r'(\d+)', query)
    if price_match:
        max_price = float(price_match.group(1))
        if 'price' in results.columns:
            results['price'] = pd.to_numeric(results['price'], errors='coerce')
            return results[results['price'] < max_price]
    return results

def filter_by_description(results, query):
    """Filter results by description keywords"""
    query_keywords = query.lower().split()
    return results[results['description'].fillna('').apply(
        lambda desc: all(keyword in desc.lower() for keyword in query_keywords)
    )]

def compare_products(product_names, corpus, model, index):
    """Compare multiple products by name"""
    comparisons = []
    for name in product_names:
        result = search_semantic(name, model, index, corpus, k=1)
        comparisons.append(result)
    return pd.concat(comparisons)

def search_semantic(query, model, index, corpus, k=5):
    """Basic semantic search function"""
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k)
    return corpus.iloc[I[0]][['id', 'title', 'description']]

def handle_query(query, model, index, corpus):
    """Handle the query based on detected intent"""
    intent = classify_intent(query)
    print(f"Detected intent: {intent}")
    
    if intent == "compare":
        product_names = extract_product_names(query)
        return compare_products(product_names, corpus, model, index)
    elif intent == "filter_price":
        results = search_semantic(query, model, index, corpus, k=10)
        return filter_by_price(results, query)
    elif intent == "filter_description":
        results = search_semantic(query, model, index, corpus, k=10)
        return filter_by_description(results, query)
    else:
        return search_semantic(query, model, index, corpus, k=5)

@app.on_event("startup")
async def startup_event():
    """Initialize resources when the app starts"""
    try:
        print("Loading dataset and building corpus...")
        start_time = time.time()
        
        # Load and prepare dataset
        app.state.corpus = load_and_prepare_dataset()
        
        # Load model
        print("Loading sentence transformer model...")
        app.state.model = SentenceTransformer(MODEL_NAME)
        
        # Build index
        print("Building FAISS index...")
        embeddings = app.state.model.encode(
            app.state.corpus['text'].tolist(), 
            convert_to_numpy=True, 
            show_progress_bar=True
        )
        app.state.index = faiss.IndexFlatL2(embeddings.shape[1])
        app.state.index.add(embeddings)
        
        print(f"Initialization completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search")
async def search_query(q: str, limit: Optional[int] = 5):
    try:
        if not q or len(q.strip()) < 2:
            raise HTTPException(status_code=400, detail="Query too short")
        
        if limit > MAX_RESULTS:
            limit = MAX_RESULTS
            
        # Use the enhanced search functionality
        results_df = handle_query(q, app.state.model, app.state.index, app.state.corpus)
        
        # Limit results to the requested number
        results_df = results_df.head(limit)
        
        results_list = results_df.to_dict('records')
        
        return {
            "query": q, 
            "results": results_list, 
            "count": len(results_list),
            "status": "success"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "query": q, 
                "results": [], 
                "status": "error", 
                "message": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
