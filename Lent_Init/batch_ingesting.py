import json
import os
import sys
import asyncio
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import argparse

from .setup import setup_llm

BATCH_CONFIG = {
    'summary_batch_size': 5, 
    'triplet_batch_size': 3,   
    'max_summary_workers': 20, 
    'max_triplet_workers': 15, 
    'max_enrichment_workers': 10, 
    'enable_batch_processing': True,
    'processing_batch_size': 500
}

logger = logging.getLogger("pipeline")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available")
    EMBEDDINGS_AVAILABLE = False

def load_classifier_components(vectorizer_path: Path, classifier_path: Path):
    """Load classifier TF-IDF vectorizer and relevance classifier from files"""
    vectorizer = None
    classifier = None
    
    if vectorizer_path.exists():
        try:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            logger.info(f"Loaded vectorizer from {vectorizer_path}")
        except Exception as e:
            logger.error(f"ERROR: Couldn't load vectorizer: {e}")
    
    if classifier_path.exists():
        try:
            with open(classifier_path, 'rb') as f:
                classifier = pickle.load(f)
            logger.info(f"Loaded classifier from {classifier_path}")
        except Exception as e:
            logger.error(f"ERROR: Couldn't load classifier: {e}")
    
    return vectorizer, classifier

def predict_relevance_local(text, vectorizer, classifier):
    if not vectorizer or not classifier:
        return False
    
    try:
        vec_text = vectorizer.transform([text])
        pred = classifier.predict(vec_text)[0]
        conf = classifier.predict_proba(vec_text)[0].max()
        
        if conf > 0.8:
            logger.debug(f"High confidence prediction: {pred}")
        
        return bool(pred)
    except:
        return False

async def classify_abstract_relevance_ollama(title: str, abstract: str, llm_setup: Dict) -> bool:
    cache_key = f"relevance:{hashlib.md5((title + abstract).encode()).hexdigest()}"
    
    if 'refinement_cache' in llm_setup:
        cached = llm_setup['refinement_cache'].get(cache_key)
        if cached is not None:
            return cached
    
    prompt = f"""
            Analyze this research abstract and determine if it's relevant to climate change impacts on wildlife/biodiversity.

            Title: {title}
            Abstract: {abstract}

            Respond with ONLY 'YES' or 'NO':
            - YES: If the abstract discusses climate change effects on species, ecosystems, or biodiversity
            - NO: If it's primarily about other topics (pollution, habitat loss from non-climate causes, etc.)
            """
    
    try:
        from .llm_api_utility import llm_generate
        
        response = await llm_generate(
            prompt=prompt,
            system="You are an expert at classifying research abstracts for relevance to climate change impacts on wildlife.",
            model=llm_setup.get('model', 'google/gemini-flash-1.5'),
            temp=0.1,
            llm_setup=llm_setup
        )
        result = response.strip().upper() == 'YES'
        
        if 'refinement_cache' in llm_setup:
            llm_setup['refinement_cache'].set(cache_key, result)
        
        logger.debug(f"LLM relevance classification for '{title[:30]}...': {result}")
        return result
    except Exception as e:
        logger.error(f"LLM classification error: {e}")
        return False

def setup_embedding_classifier(models_path: Path):
    if not EMBEDDINGS_AVAILABLE:
        return None, None
    
    model = None
    classifier = None
    
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return None, None
    
    clf_path = models_path / "embedding_classifier.pkl"
    if clf_path.exists():
        try:
            with open(clf_path, 'rb') as f:
                classifier = pickle.load(f)
            logger.info(f"Loaded pre-trained embedding classifier from {clf_path}")
        except Exception as e:
            logger.error(f"Error loading embedding classifier: {e}")
            classifier = None
    else:
        logger.info(f"No pre-trained embedding classifier found at {clf_path}")
    
    return model, classifier

def predict_relevance_embeddings(abstract_text: str, model, classifier) -> bool:
    """Predict relevance using embeddings and trained classifier."""
    if not model or not classifier:
        logger.warning("Embedding classifier components not available")
        return False
        
    try:
        embedding = model.encode([abstract_text])
        pred = classifier.predict(embedding)[0]
        conf = classifier.predict_proba(embedding)[0].max()
    
        logger.debug(f"Embedding classifier prediction: {pred} (confidence: {conf:.3f})")
        return bool(pred)
    except Exception as e:
        logger.error(f"Error in embedding relevance prediction: {e}")
        return False

if __name__ == "__main__":
    from .main_pipeline import run_main_pipeline_logic, run_batch_enabled_pipeline, run_wikispecies_verification_logic, run_taxonomy_comparison_logic

    parser = argparse.ArgumentParser(description="Process abstracts for species-threat analysis")
    
    parser.add_argument('--run-main-pipeline', action='store_true', 
                       help='Run main pipeline')
    parser.add_argument('--enable-batch-processing', action='store_true',
                       help='Enable batch processing')
    parser.add_argument('--batch-config', type=str,
                       help='JSON config override')
    parser.add_argument('--taxonomy', type=str,
                       help='Filter by taxonomic term')
    parser.add_argument('--max', type=str,
                       help='Max abstracts to process')
    parser.add_argument('--verify-species-wikispecies', type=str, metavar='FILEPATH',
                       help='Run wikispecies verification')
    parser.add_argument('--compare-taxonomies', action='store_true',
                       help='Compare taxonomy data')
    parser.add_argument('--target_model_name', type=str,
                       help='Target model name')
    parser.add_argument('--target_max_results', type=str,
                       help='Target max results')
    parser.add_argument('--query', type=str,
                       help='Semantic search query')

    args = parser.parse_args()

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    batch_enabled = BATCH_CONFIG.get('enable_batch_processing', True)
    if args.enable_batch_processing:
        batch_enabled = True
    elif not args.enable_batch_processing:
        batch_enabled = False

    if args.batch_config:
        try:
            override = json.loads(args.batch_config)
            BATCH_CONFIG.update(override)
            logger.info(f"Updated config: {BATCH_CONFIG}")
        except:
            logger.error("Bad JSON config")
    
    BATCH_CONFIG['enable_batch_processing'] = batch_enabled
    if args.verify_species_wikispecies:
        run_wikispecies_verification_logic(args)
    elif args.compare_taxonomies:
        run_taxonomy_comparison_logic(args)
    else: 
        if batch_enabled:
             run_batch_enabled_pipeline(args) 
        else:
             asyncio.run(run_main_pipeline_logic(args))
