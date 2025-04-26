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

app = FastAPI(title="Product Search API")

# Configuration
MAX_RESULTS = 10
MODEL_NAME = 'all-MiniLM-L6-v2'

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    """Initialize resources when the app starts"""
    try:
        print("Loading dataset and building corpus...")
        start_time = time.time()
        
        # Load dataset
        dataset = load_dataset("wdc/products-2017", "cameras_small")
        df = pd.DataFrame(dataset['train'])
        
        # Build corpus
        def build_corpus(df):
            left = df[['id_left', 'title_left', 'description_left']].drop_duplicates(subset='id_left')
            right = df[['id_right', 'title_right', 'description_right']].drop_duplicates(subset='id_right')

            left.columns = ['id', 'title', 'description']
            right.columns = ['id', 'title', 'description']

            combined = pd.concat([left, right]).drop_duplicates(subset='id')
            combined['text'] = (combined['title'].fillna('') + ' ' + combined['description'].fillna('')).str.strip()
            return combined.reset_index(drop=True)

        app.state.corpus = build_corpus(df)
        
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
            
        query_vec = app.state.model.encode([q], convert_to_numpy=True)
        D, I = app.state.index.search(query_vec, limit)
        
        results = app.state.corpus.iloc[I[0]][['id', 'title', 'description']]
        results_list = results.to_dict('records')
        
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