import polars as pl
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import os
from dotenv import load_dotenv
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from .cache import Cache
import sys
from .wikispecies_utils import RateLimiter
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger("pipeline")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def setup_pipeline_logging(log_file_path: Path):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    logger.info(f"Logging setup: {log_file_path}")

def get_dynamic_run_base_path(model_name: str, max_val: Optional[Union[int, str]], script_dir: Path) -> Path:
    clean_model = model_name.replace("/", "_").replace(":", "_").replace(".", "-")
    max_str = str(max_val) if max_val is not None and str(max_val).lower() != "all" else "all"
    
    folder_name = f"{clean_model}_{max_str}"
    return script_dir / "runs" / folder_name

def load_data_with_offset(fname, skip=0, max_rows=1000):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    if fname == "all_abstracts.parquet":
        paths = [
            os.path.join("/app", fname),
            os.path.join(os.path.expanduser("~"), "Desktop", fname),
            os.path.join(curr_dir, fname),
            os.path.join(os.path.dirname(curr_dir), fname),
            os.path.join("/Users/kittsonhamill/Desktop", fname)
        ]
        
        fpath = None
        for p in paths:
            if os.path.exists(p):
                fpath = p
                break
                
        if fpath is None:
            raise FileNotFoundError(f"Can't find {fname} in: {paths}")
    else:
        fpath = os.path.join(curr_dir, fname)
    
    print(f"Loading: {fpath} (skip {skip}, max {max_rows})")
    
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        pf = pq.ParquetFile(fpath)
        total = pf.metadata.num_rows
        
        if skip >= total:
            return pl.DataFrame()  
        
        batches = pf.iter_batches(batch_size=1024)
        needed = max_rows
        skipped = 0
        arrow_batches = []
        
        for batch in batches:
            batch_len = batch.num_rows
            if skipped + batch_len <= skip:
                skipped += batch_len
                continue  
            
            # figure out start pos
            start = max(0, skip - skipped)
            available = batch_len - start
            take = min(available, needed)
            sliced = batch.slice(start, take)
            arrow_batches.append(sliced)
            needed -= take
            skipped += batch_len
            
            if needed <= 0:
                break
        
        if arrow_batches:
            table = pa.Table.from_batches(arrow_batches)
            df = pl.from_arrow(table)
        else:
            df = pl.DataFrame()
            
    except Exception as e:
        print(f"PyArrow failed: {e}")
        print("Using basic polars")
        df = pl.read_parquet(fpath)
        if skip >= len(df):
            return pl.DataFrame()
        end = min(skip + max_rows, len(df))
        df = df[skip:end]
    
    # drop nulls
    df = df.drop_nulls(["title", "abstract", "doi"])
    print(f"{len(df)} rows after cleaning")
    return df


def setup_llm():
    load_dotenv()
    return {
        'cache': Cache(),
        'model': "google/gemini-flash-1.5", 
        'api_rate_limiter': RateLimiter(rpm=30, ollama_mode=False),
        'species_model': "google/gemini-flash-1.5", 
        'threat_model': "google/gemini-flash-1.5",   
        'impact_model': "google/gemini-flash-1.5",   
        'use_openrouter': True
    }


def cluster_threats(triplets, model, figures_dir: Path, n_clusters=8):
    if not EMBEDDINGS_AVAILABLE or model is None:
        print("embeddings and model not available")
        return {}
        
    threat_texts = []
    for _, _, obj, _doi in triplets:
        desc = obj
        if "[IUCN:" in obj:
            desc = obj.split("[IUCN:")[0].strip()
        threat_texts.append(desc)
    
    if not threat_texts:
        print("No threats found")
        return {}
        
    print(f"Clustering {len(threat_texts)} threats...")
    
    try:
        emb = model.encode(threat_texts, show_progress_bar=True)
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(threat_texts)))
        clusters = kmeans.fit_predict(emb)
        clustered = defaultdict(list)
        for i, (text, cid) in enumerate(zip(threat_texts, clusters)):
            clustered[int(cid)].append((i, text))
        summaries = {}
        for cid, threats in clustered.items():
            center = kmeans.cluster_centers_[cid]
            dists = []
            
            for idx, text in threats:
                dist = np.linalg.norm(emb[idx] - center)
                dists.append((idx, text, dist))
            
            sorted_threats = sorted(dists, key=lambda x: x[2])
            examples = [text for _, text, _ in sorted_threats[:3]]
            summaries[cid] = {
                "count": len(threats),
                "examples": examples,
                "threat_indices": [idx for idx, _text in threats]
            }
            
            print(f"\nCluster {cid} ({len(threats)} threats):")
            for ex in examples:
                print(f"  - {ex[:100]}...")
        
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            emb_2d = reducer.fit_transform(emb)
            
            plt.figure(figsize=(12, 10))
            
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            
            for cid in range(n_clusters):
                pts = [i for i, c in enumerate(clusters) if c == cid]
                plt.scatter(
                    emb_2d[pts, 0], 
                    emb_2d[pts, 1],
                    s=50, 
                    c=[colors[cid]],
                    label=f'Cluster {cid} ({len(pts)})'
                )
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title('Threat Clusters')
            
            figures_dir.mkdir(parents=True, exist_ok=True)
            outpath = figures_dir / "threat_clusters.png"

            plt.savefig(outpath, bbox_inches='tight')
            plt.close()
            print(f"\nSaved cluster viz: {outpath}")

        except ImportError:
            print("No UMAP - install umap-learn for viz")
        except Exception as e:
            print(f"Viz failed: {e}")
        
        return summaries
    except Exception as e:
        print(f"Clustering error: {e}")
        return {}
    

def setup_vector_search(abstracts, model):
    if not EMBEDDINGS_AVAILABLE or model is None:
        print("embeddings and model not available")
        return None
        
    try:
        emb = model.encode(abstracts, show_progress_bar=True)
        
        store = {
            'embeddings': emb,
            'abstracts': abstracts
        }
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = Path(curr_dir) / "models"
        out_dir.mkdir(exist_ok=True)
        
        with open(out_dir / "vector_store.pkl", 'wb') as f:
            pickle.dump(store, f)
            
        print(f"Vector store ready: {len(abstracts)} abstracts")
        return store
    except Exception as e:
        print(f"Vector setup failed: {e}")
        return None
