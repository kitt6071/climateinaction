import os
import json
import requests
import logging
import config
from config import (
    DATA_SOURCES, DATA_PATH, PARQUET_SOURCES, PARQUET_PATH, initialize_analyzer
)

import torch
import polars as pl
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def download_data_from_url(url, local_path):
    #Download data file from URL to local path
    try:
        # Ensure the target directory exists before writing the file
        parent_dir = os.path.dirname(local_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        logger.info(f"Downloading data from {url}")
        response = requests.get(url, stream=True, timeout=1800)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (200 * 1024 * 1024) == 0:
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded/(1024*1024):.0f}/{total_size/(1024*1024):.0f} MB)")
                        else:
                            logger.info(f"Downloaded: {downloaded/(1024*1024):.0f} MB")
        
        logger.info(f"Successfully downloaded data to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download from {url}: {e}")
        return False

def load_data_if_needed():
    if config.data_loaded:
        return True
    
    data_file_path = None
    for source in DATA_SOURCES:
        if os.path.exists(source):
            data_file_path = source
            logger.info(f"Found local webapp data file at {data_file_path}")
            break

    if not data_file_path:
        logger.error("FATAL: No webapp data file found. Was the download script run?")
        return False
    
    logger.info(f"Loading data from {data_file_path}...")
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            app_data = json.load(f)
        
        config.triplets_data = app_data.get("triplets", [])
        
        for triplet in config.triplets_data:
            if 'embedding' in triplet and triplet['embedding'] is not None:
                triplet['embedding_tensor'] = torch.tensor(triplet['embedding'])
            else:
                triplet['embedding_tensor'] = None
        logger.info(f"Data loaded: {len(config.triplets_data)} triplets available.")
        
        logger.info("Initializing knowledge graph...")
        from models import EnhancedKnowledgeGraph
        config.enhanced_kg = EnhancedKnowledgeGraph()
        config.kg_results = config.enhanced_kg.build_enriched_graph(config.triplets_data, load_ecological=False)
        logger.info("Knowledge graph initialized.")

        # Initialize analyzer
        initialize_analyzer()
        
        config.data_loaded = True
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file_path}")
        return False
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from {data_file_path}")
        return False
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False

    if not config.parquet_loaded:
        parquet_file_path = None
        for source in PARQUET_SOURCES:
            if source.startswith("http"):
                logger.info(f"Attempting to download parquet from {source}")
                if download_data_from_url(source, PARQUET_PATH):
                    parquet_file_path = PARQUET_PATH
                    logger.info(f"Successfully downloaded parquet data from cloud storage")
                    break
            else:
                if os.path.exists(source):
                    parquet_file_path = source
                    logger.info(f"Found local parquet file at {parquet_file_path}")
                    break
        
        if parquet_file_path:
            try:
                logger.info(f"Loading abstracts from {parquet_file_path}...")
                
                file_size = os.path.getsize(parquet_file_path)
                logger.info(f"Parquet file size: {file_size / (1024*1024):.1f} MB")
                
                if file_size == 0:
                    logger.error(f"Parquet file {parquet_file_path} is empty (0 bytes)")
                    return config.data_loaded
                
                logger.info("Extracting only needed abstracts based on triplet DOIs...")
                
                needed_dois = {t.get('doi', '').lower().strip() for t in config.triplets_data if t.get('doi')}
                logger.info(f"Need abstracts for {len(needed_dois)} DOIs from triplets")
                
                lf = pl.scan_parquet(parquet_file_path)
                subset_df = lf.filter(
                    pl.col("doi").str.to_lowercase().is_in(list(needed_dois))
                ).with_columns(
                    pl.col("doi").str.to_lowercase().alias("doi_lower")
                ).collect()
                
                config.abstracts_df = subset_df
                config.parquet_loaded = True
                logger.info(f"Loaded {len(subset_df)} abstracts (subset) instead of entire 2.7GB file")
            except Exception as e:
                logger.error(f"❌ Error preparing lazy parquet reader for {parquet_file_path}: {e}", exc_info=True)
        else:
            logger.error("No parquet data file found from any source.")

    return config.data_loaded

def load_data_chunked(file_path=DATA_PATH, chunk_size=500):
    #Loads data in chunks from a JSON file to be more memory-efficient.
    logger.info("Starting chunked data loading...")
    
    try:
        with open(file_path, 'r') as f:
            all_data = json.load(f)

        total_items = len(all_data)
        logger.info(f"Total items to process: {total_items}")

        temp_triplets_data = []
        for i in range(0, total_items, chunk_size):
            chunk = all_data[i:i + chunk_size]
            for triplet in chunk:
                if 'embedding' in triplet and triplet['embedding'] is not None:
                    try:
                        triplet['embedding_tensor'] = torch.tensor(triplet['embedding'], dtype=torch.float32)
                    except Exception as e:
                        logger.warning(f"Skipping triplet due to embedding conversion error: {e}")
                        triplet['embedding_tensor'] = None
                else:
                    triplet['embedding_tensor'] = None
                temp_triplets_data.append(triplet)
            logger.info(f"Processed chunk {i // chunk_size + 1}/{(total_items + chunk_size -1) // chunk_size}")

        config.triplets_data = temp_triplets_data
        
        # Initialize analyzer
        initialize_analyzer()
        
        config.data_loaded = True
        logger.info("Data loading process completed successfully.")
        return True
        
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}")
        return False
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during chunked data loading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False 