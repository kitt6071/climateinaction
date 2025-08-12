import nltk
from typing import List, Tuple, Dict
from pathlib import Path
import os
from collections import defaultdict
import asyncio
import sys
import logging
import time
from .cache import Cache, SimpleCache
from .setup import setup_pipeline_logging, get_dynamic_run_base_path, load_data_with_offset
from .batch_ingesting import (BATCH_CONFIG, EMBEDDINGS_AVAILABLE, load_classifier_components,
                               predict_relevance_local, classify_abstract_relevance_ollama,
                               setup_embedding_classifier, predict_relevance_embeddings, predict_relevance_embeddings_batch)
from .iucn_refinement import get_iucn_classification_json, parse_and_validate_object, cache_enriched_triples, classify_threat_for_subject, detect_threat_content_in_abstract
from .triplet_extraction import verify_triplets, normalize_species_names, convert_to_summary, extract_entities_concurrently, generate_relationships_concurrently, consolidate_triplets
from .llm_api_utility import enable_metrics_tracking, log_metrics_summary, llm_generate
from .graph_analysis import (build_global_graph, analyze_graph_detailed, 
                           enrich_graph_with_embeddings, 
                           create_embedding_visualization, analyze_hub_node,
                           visualize_triplet_sentence_embeddings_batch_ingest)
from .wikispecies_utils import verify_species_with_wikispecies_concurrently, compare_and_log_taxonomy_discrepancies

from .setup import setup_llm, setup_vector_search

logger = logging.getLogger("pipeline")

def has_shorebird_keywords(text):
    text_lower = text.lower()
    
    if any(exclude in text_lower for exclude in [
        'root-knot', 'root knot', 'nematode', 'nematodos', 'plover cove', 
        'plover lake', 'plover point', 'plover bay', 'reservoir', 
        'virtual client', 'ceramic sherd', 'escurrimiento', 'sedimentos', 
        'comunidades de a', 'haematococcus pluvialis', 'knot sandpiper', 'great knot'
    ]):
        return False
        
    import re

    specific_terms = [
'shorebird', 'shore bird', 'wader', 'wading bird', 'shorebirds'

# Major Shorebird Families (Common Names)
'sandpiper', 'plover', 'godwit', 'curlew', 'turnstone', 'oystercatcher',
'avocet', 'stilt', 'phalarope', 'snipe', 'woodcock', 'jacana', # Though Jacanas are sometimes grouped, they're distinct
'thick-knee', # Often associated with waders
'pratincole', 'courser', # Sometimes grouped with plovers/waders

# Specific Genera (Scientific Names - often useful for more precise searching)
'Calidris',      # Sandpipers (e.g., Dunlin, Sanderling, Knot, Stints)
'Tringa',        # Yellowlegs, Tattlers, Godwits (some species)
'Charadrius',    # Plovers (e.g., Ringed Plover, Kentish Plover, Snowy Plover)
'Pluvialis',     # Golden Plovers, Grey Plover
'Numenius',      # Curlews, Whimbrel
'Limosa',        # Godwits
'Arenaria',      # Turnstones
'Haematopus',    # Oystercatchers
'Recurvirostra', # Avocets
'Himantopus',    # Stilts
'Phalaropus',    # Phalaropes
'Gallinago',     # Snipes
'Scolopax',      # Woodcocks
'Actitis',       # Sandpipers (e.g., Common Sandpiper, Spotted Sandpiper)
'Limnodromus',   # Dowitchers
'Aphriza',       # Surfbird
'Heteroscelus',  # Wandering Tattler, Grey-tailed Tattler
'Xenus',         # Terek Sandpiper
'Prosobonia',    # Polynesian Sandpipers (extinct/endangered)

# Specific Species (Common Names - a good selection of diverse types)
'sanderling', 'dunlin', 'knot', 'stint', 'little stint', 'temminck\'s stint',
'semipalmated sandpiper', 'western sandpiper', 'least sandpiper',
'peep', # Collective term for small Calidris sandpipers
'yellowlegs', 'greater yellowlegs', 'lesser yellowlegs',
'dowitcher', 'long-billed dowitcher', 'short-billed dowitcher',
'common snipe', 'jack snipe', 'great snipe',
'eurasian curlew', 'whimbrel', 'bristle-thighed curlew',
'bar-tailed godwit', 'black-tailed godwit', 'marbled godwit', 'hudsonian godwit',
'red knot', 'great knot',
'ruddy turnstone', 'black turnstone',
'common ringed plover', 'kentish plover', 'snowy plover', 'piping plover',
'killdeer', 'dotterel', 'mountain plover', 'pacific golden plover', 'european golden plover',
'grey plover', 'black-bellied plover',
'eurasian oystercatcher', 'american oystercatcher', 'black oystercatcher',
'pied avocet', 'american avocet',
'black-necked stilt', 'pied stilt',
'red-necked phalarope', 'grey phalarope', 'wilson\'s phalarope',
'common sandpiper', 'spotted sandpiper',
'wood sandpiper', 'green sandpiper', 'marsh sandpiper',
'greenshank', 'redshank', 'spotted redshank',
'terek sandpiper', 'surfbird', 'wandering tattler', 'grey-tailed tattler',
'stone-curlew', 'eurasian thick-knee',
'cream-colored courser', 'collared pratincole',

# Unique Characteristics/Behaviors
'long-legged', 'long-billed', 'short-billed', 'upturned bill', 'downcurved bill']
    
    for term in specific_terms:
        if re.search(r'\b' + term + r's?\b', text_lower):
            return True
    return False

async def check_for_primary_evidence(abstract: str, llm_setup: dict) -> dict:
    import json
    
    gate_schema = {
        "type": "object",
        "properties": {
            "primary_evidence_found": {
                "type": "boolean",
                "description": "True only if the abstract contains at least one sentence presenting a specific finding or result from a direct study."
            },
            "strongest_evidence_sentence": {
                "type": "string",
                "description": "The single best sentence from the abstract that represents a primary finding. If none, this must be an empty string."
            }
        },
        "required": ["primary_evidence_found", "strongest_evidence_sentence"]
    }

    system_prompt = f"""You are an efficient data screener. Your only job is to determine if an abstract contains primary research findings.

Scan the text for sentences that state a specific, quantitative, or observational result from the authors' own study. Look for phrases like "we found that," "results indicate," or sentences that directly report a measured outcome.

Your entire decision is based on one question: Does this text contain a sentence that looks like a primary finding?

Return a JSON object conforming to this schema.
Schema:
{json.dumps(gate_schema, indent=2)}"""

    user_prompt = f"Abstract:\n\"\"\"\n{abstract}\n\"\"\"\n"

    try:
        response = await llm_generate(
            prompt=user_prompt,
            system=system_prompt,
            model=llm_setup.get("model", "qwen/qwen3-30b-a3b"),
            temp=0.1,
            format=gate_schema,
            llm_setup=llm_setup
        )
        
        if response:
            try:
                evidence_data = json.loads(response)
                return evidence_data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse evidence gate JSON response: {response[:100]}")
                return {"primary_evidence_found": False, "strongest_evidence_sentence": ""}
        else:
            return {"primary_evidence_found": False, "strongest_evidence_sentence": ""}
            
    except Exception as e:
        logger.error(f"Error in evidence gate: {e}")
        return {"primary_evidence_found": False, "strongest_evidence_sentence": ""}

async def check_for_impact_conservation_evidence(abstract: str, llm_setup: dict) -> bool:
    import json
    
    gate_schema = {
        "type": "object",
        "properties": {
            "impact_conservation_found": {
                "type": "boolean",
                "description": "True only if the abstract contains sentences describing measured impacts on species/ecosystems or conservation outcomes."
            },
            "strongest_impact_sentence": {
                "type": "string", 
                "description": "The single best sentence showing impact on species/ecosystems or conservation results. If none, this must be an empty string."
            }
        },
        "required": ["impact_conservation_found", "strongest_impact_sentence"]
    }

    system_prompt = f"""You are scanning for specific types of evidence about impacts on wildlife or conservation outcomes.

Look for sentences that describe:
- Measured effects on species populations, behavior, or survival
- Changes in habitat or ecosystem conditions  
- Conservation intervention results or outcomes
- Environmental stressor impacts on wildlife

Find sentences that report specific impacts or conservation results, such as "population declined by X%," "habitat loss reduced nesting success," or "conservation efforts increased breeding pairs."

Your task: Does this abstract contain a sentence describing a measured impact on species/ecosystems or a conservation outcome?

Return a JSON object conforming to this schema.
Schema:
{json.dumps(gate_schema, indent=2)}"""

    user_prompt = f"Abstract:\n\"\"\"\n{abstract}\n\"\"\"\n"

    try:
        response = await llm_generate(
            prompt=user_prompt,
            system=system_prompt,
            model=llm_setup.get("model", "qwen/qwen3-30b-a3b"),
            temp=0.1,
            format=gate_schema,
            llm_setup=llm_setup
        )
        
        if response:
            try:
                evidence_data = json.loads(response)
                return evidence_data.get("impact_conservation_found", False)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse impact gate JSON response: {response[:100]}")
                return False
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error in impact/conservation gate: {e}")
        return False

async def check_relevance_batch_optimized(batch_items, llm_setup, embed_model, embed_classifier, vectorizer, legacy_classifier):
    if not batch_items:
        return []
        
    logger.info(f"Batch relevance check for {len(batch_items)} abstracts")
    
    if embed_classifier and embed_model and EMBEDDINGS_AVAILABLE:
        logger.info(f"Using batch embeddings classification for {len(batch_items)} abstracts")
        abstracts_for_ml = [item['abstract'] for item in batch_items]
        batch_results = await predict_relevance_embeddings_batch(abstracts_for_ml, embed_model, embed_classifier, threshold=0.70)
        
        results = []
        for i, (item, (is_relevant, score)) in enumerate(zip(batch_items, batch_results)):
            if not is_relevant and has_shorebird_keywords(f"{item['title']} {item['abstract']}"):
                logger.info(f"KEYWORD RESCUE: Low relevance ({score:.3f}) but has shorebird keywords: '{item['title'][:50]}...'")
                results.append((True, score))
            else:
                results.append((is_relevant, score))
        return results
        
    elif legacy_classifier and vectorizer:
        logger.info(f"Using TF-IDF batch classification for {len(batch_items)} abstracts")
        results = []
        for item in batch_items:
            is_relevant = predict_relevance_local(item['abstract'], vectorizer, legacy_classifier)
            vec_text = vectorizer.transform([item['abstract']])
            probabilities = legacy_classifier.predict_proba(vec_text)[0]
            relevance_score = probabilities[1]
            
            if not is_relevant and has_shorebird_keywords(f"{item['title']} {item['abstract']}"):
                logger.info(f"KEYWORD RESCUE: Low relevance ({relevance_score:.3f}) but has shorebird keywords: '{item['title'][:50]}...'")
                results.append((True, relevance_score))
            else:
                results.append((is_relevant, relevance_score))
        return results
    else:
        logger.info(f"Using LLM fallback for {len(batch_items)} abstracts")
        llm_tasks = []
        for item in batch_items:
            llm_tasks.append(classify_abstract_relevance_ollama(item['title'], item['abstract'], llm_setup))
        
        if llm_tasks:
            llm_results = await asyncio.gather(*llm_tasks)
            results = []
            for item, is_relevant in zip(batch_items, llm_results):
                relevance_score = 1.0 if is_relevant else 0.0
                if not is_relevant and has_shorebird_keywords(f"{item['title']} {item['abstract']}"):
                    logger.info(f"KEYWORD RESCUE: Low relevance ({relevance_score:.3f}) but has shorebird keywords: '{item['title'][:50]}...'")
                    results.append((True, relevance_score))
                else:
                    results.append((is_relevant, relevance_score))
            return results
        else:
            return [(False, 0.0)] * len(batch_items)

async def process_relevance_parallel_batches(batch_items, llm_setup, embed_model, embed_classifier, vectorizer, legacy_classifier):
    if not batch_items:
        return []
    
    RELEVANCE_BATCH_SIZE = 100
    MAX_RELEVANCE_WORKERS = 5
    
    model_pool = []
    if embed_model and embed_classifier and EMBEDDINGS_AVAILABLE:
        logger.info(f"Creating pool of embeddings models for concurrent processing")
        try:
            from sentence_transformers import SentenceTransformer
            pool_size = min(MAX_RELEVANCE_WORKERS, 5)
            model_pool.append((embed_model, embed_classifier))
            
            for i in range(pool_size - 1):
                try:
                    model_instance = SentenceTransformer(embed_model.model_name_or_path if hasattr(embed_model, 'model_name_or_path') else 'all-mpnet-base-v2')
                    model_pool.append((model_instance, embed_classifier))
                    logger.info(f"Created model instance {i+2}/{pool_size}")
                except Exception as e:
                    logger.warning(f"Failed to create model instance {i+2}: {e}")
                    model_pool.append((embed_model, embed_classifier))
            logger.info(f"Model pool complete: {len(model_pool)} instances")
        except Exception as e:
            logger.warning(f"Failed to create model pool: {e}, falling back to shared model")
            model_pool = [(embed_model, embed_classifier)]
    else:
        model_pool = [(embed_model, embed_classifier)]
    
    import itertools
    model_cycle = itertools.cycle(model_pool)
    
    sub_batches = []
    for i in range(0, len(batch_items), RELEVANCE_BATCH_SIZE):
        sub_batch = batch_items[i:i + RELEVANCE_BATCH_SIZE]
        sub_batches.append(sub_batch)
    
    logger.info(f"Split {len(batch_items)} abstracts into {len(sub_batches)} sub-batches of ~{RELEVANCE_BATCH_SIZE} each")
    
    semaphore = asyncio.Semaphore(MAX_RELEVANCE_WORKERS)
    
    async def process_sub_batch(sub_batch, worker_model, worker_classifier):
        async with semaphore:
            logger.info(f"Worker processing batch of {len(sub_batch)} abstracts")
            return await check_relevance_batch_optimized(sub_batch, llm_setup, worker_model, worker_classifier, vectorizer, legacy_classifier)
    
    tasks = []
    for sub_batch in sub_batches:
        worker_model, worker_classifier = next(model_cycle)
        tasks.append(process_sub_batch(sub_batch, worker_model, worker_classifier))
    
    logger.info(f"Starting {len(tasks)} parallel batch workers (max {MAX_RELEVANCE_WORKERS} concurrent)")
    sub_results = await asyncio.gather(*tasks)
    
    all_results = []
    for sub_result in sub_results:
        all_results.extend(sub_result)
    
    logger.info(f"Parallel batch processing complete: {len(all_results)} results")
    return all_results

async def run_main_pipeline_logic(args):
    start_time = time.time()
    
    # basic path setup
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    llm_sys = setup_llm() 
    model_name = os.getenv('MODEL_NAME_FOR_RUN', llm_sys["model"])
    
    # Enable metrics tracking for detailed usage and cost analysis
    llm_sys = enable_metrics_tracking(llm_sys)

    max_from_args = getattr(args, 'max', None) 
    max_env = os.getenv('MAX_RESULTS', 'all')
    max_from_env = None
    if str(max_env).lower() == 'all':
        max_from_env = "all"
    elif str(max_env).isdigit():
        max_from_env = int(max_env)
    
    max_setup = max_from_args if max_from_args is not None else max_from_env
    if max_setup is None: 
        max_setup = "all"

    # figure out the limit
    max_limit = float('inf')
    if isinstance(max_setup, int):
        max_limit = max_setup
    elif str(max_setup).lower() != 'all': 
        try:
            max_limit = int(max_setup)
        except ValueError:
            logging.warning(f"Invalid value '{max_setup}'. Defaulting to all abstracts")

    run_base = get_dynamic_run_base_path(model_name, max_setup, script_dir)
    logs_path = run_base / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / "main_pipeline.log"
    setup_pipeline_logging(log_file) 

    logger.info("Starting pipeline")
    logger.info(f"Logs: {log_file}")
    logger.info(f"Base dir: {run_base}")
    logger.info(f"Max abstracts: {max_limit if max_limit != float('inf') else 'all'}, chunk size: {BATCH_CONFIG['processing_batch_size']}")

    llm_setup = llm_sys
    llm_setup['model'] = model_name
    llm_setup['species_model'] = model_name  
    llm_setup['threat_model'] = model_name

    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        logger.critical(f"NLTK error: {e}", exc_info=True)
        return run_base if 'run_base' in locals() else script_dir

    embed_model, embed_classifier = None, None
    if EMBEDDINGS_AVAILABLE:
        logger.info("Setting up embeddings")
        embed_model_path = run_base / "models"
        embed_model_path.mkdir(parents=True, exist_ok=True)
        embed_model, embed_classifier = setup_embedding_classifier(models_path=embed_model_path)
    
    results_path = run_base / "results"
    figures_path = run_base / "figures"
    cache_path = run_base / "cache"
    models_path = run_base / "models" 

    for p in [results_path, figures_path, cache_path, models_path]:
        p.mkdir(parents=True, exist_ok=True)

    # try to load pre-trained stuff, seperate script process generates and this is copy pasted into models_path folder location manually right now
    vectorizer_path = models_path / "tfidf_vectorizer.pkl"
    legacy_classifier_path = models_path / "relevance_classifier.pkl"
    vectorizer, legacy_classifier = load_classifier_components(vectorizer_path, legacy_classifier_path)
    classifier_ready = bool(vectorizer and legacy_classifier)
    if classifier_ready:
        logger.info("TF-IDF classifier loaded")
    else:
        logger.info("No TF-IDF classifier found")

    # embedding classifier
    if EMBEDDINGS_AVAILABLE:
        logger.info(f"Loading from: {models_path}")
        embed_model, embed_classifier = setup_embedding_classifier(models_path) 
        if embed_model and embed_classifier:
            logger.info("Got embedding model + classifier")
        elif embed_model:
            logger.info("Got model but no classifier")
        else:
            logger.warning("No embedding model available")
    else:
        embed_model, embed_classifier = None, None
        logger.warning("No sentence-transformers")

    llm_setup['cache'] = Cache(cache_dir=str(cache_path))
    refinement_cache_dir = cache_path / "refinement_cache"
    refinement_cache = SimpleCache(refinement_cache_dir)
    
    taxonomic_filter = args.taxonomy if hasattr(args, 'taxonomy') and args.taxonomy else os.getenv('TAXONOMY_FILTER', '')
    VERIFICATION_THRESHOLD = 0.75

    all_data = []
    norm_triplets = []
    taxo_map = {}
    
    chunk = []
    
    batch_size = 1000
    skip_rows = 0
    processed_count = 0 
    total_scanned = 0
    logger.info(f"Starting data load from parquet (batch: {batch_size}, max: {max_limit if max_limit != float('inf') else 'all'}, chunk: {BATCH_CONFIG['processing_batch_size']})")
    irrelevant_file = results_path / "irrelevant_abstracts.jsonl"



    async def check_relevance(title, abstract, llm_setup, embed_model, embed_classifier, vectorizer, legacy_classifier):
        # try embedding classifier first
        if embed_classifier and embed_model and EMBEDDINGS_AVAILABLE:
            logger.debug(f"Using embedding classifier for '{title[:30]}...'")
            is_relevant = predict_relevance_embeddings(abstract, embed_model, embed_classifier, threshold=0.70)
            # Get the actual probability score for logging
            probabilities = embed_classifier.predict_proba(embed_model.encode([abstract]))[0]
            relevance_score = probabilities[1]
            if not is_relevant and has_shorebird_keywords(f"{title} {abstract}"):
                logger.info(f"KEYWORD RESCUE: Low relevance ({relevance_score:.3f}) but has shorebird keywords: '{title[:50]}...'")
                return True, relevance_score  # override decision to True

            return is_relevant, relevance_score
        elif legacy_classifier and vectorizer:
            logger.debug(f"Using TF-IDF for '{title[:30]}...'")
            is_relevant = predict_relevance_local(abstract, vectorizer, legacy_classifier)
            # Get the actual probability score for logging
            vec_text = vectorizer.transform([abstract])
            probabilities = legacy_classifier.predict_proba(vec_text)[0]
            relevance_score = probabilities[1]
            if not is_relevant and has_shorebird_keywords(f"{title} {abstract}"):
                logger.info(f"KEYWORD RESCUE: Low relevance ({relevance_score:.3f}) but has shorebird keywords: '{title[:50]}...'")
                return True, relevance_score  # override decision to True

            return is_relevant, relevance_score
        # fallback to LLM
        logger.debug(f"Using LLM for '{title[:30]}...'")
        is_relevant = await classify_abstract_relevance_ollama(title, abstract, llm_setup)
        relevance_score = 1.0 if is_relevant else 0.0

        if not is_relevant and has_shorebird_keywords(f"{title} {abstract}"):
            logger.info(f"KEYWORD RESCUE: Low relevance ({relevance_score:.3f}) but has shorebird keywords: '{title[:50]}...'")
            return True, relevance_score  # override decision to True

        return is_relevant, relevance_score

    while True:
        if processed_count >= max_limit:
            logger.info(f"Hit limit ({max_limit})")
            break

        logger.info(f"Loading batch: skip={skip_rows}, max={batch_size} (shorebird papers found so far: {processed_count})")
        df_batch = load_data_with_offset("shorebirds.parquet", skip_rows, batch_size)
        
        if len(df_batch) == 0:
            logger.info("No more data")
            break
        
        actual_rows = len(df_batch)
        total_scanned += actual_rows
        
        batch_items = []
        for i, row_data in enumerate(df_batch.iter_rows(named=True)):
            abstract_text = row_data["abstract"]
            title_text = row_data["title"]
            doi_text = row_data.get("doi")
            if not doi_text: continue
            if "captivity" in abstract_text.lower() or len(abstract_text) < 50:
                continue
            batch_items.append({'title': title_text, 'abstract': abstract_text, 'doi': doi_text, 'idx': skip_rows + i})

        skip_rows += actual_rows

        if taxonomic_filter:
            logger.info(f"Filtering by '{taxonomic_filter}' on {len(batch_items)} items")
            filtered = []
            for item in batch_items:
                if (taxonomic_filter.lower() in item['title'].lower() or 
                    taxonomic_filter.lower() in item['abstract'].lower()):
                    filtered.append(item)
            batch_items = filtered
            logger.info(f"After filter: {len(batch_items)}")

        if not batch_items:
            logger.info("Nothing left after filtering")
            if max_limit == float('inf') and total_scanned >= MAX_PARQUET_ROWS_TO_SCAN_IF_NO_MAX_RESULTS:
                 logger.warning(f"Scanned {total_scanned} rows, stopping")
                 break
            continue
        
        if batch_items:
            logger.info(f"Checking relevance for {len(batch_items)} abstracts using parallel batch workers")
            results = await process_relevance_parallel_batches(batch_items, llm_setup, embed_model, embed_classifier, vectorizer, legacy_classifier)
            logger.info("Parallel batch relevance check done")

            for i, (is_relevant, relevance_score) in enumerate(results):
                if is_relevant:
                    relevant_item = batch_items[i]
                    logger.info(f"RELEVANT #{processed_count + 1} (score: {relevance_score:.3f}): '{relevant_item['title']}'")
                    chunk.append(relevant_item)
                    processed_count += 1 

                    if len(chunk) >= BATCH_CONFIG['processing_batch_size'] or processed_count >= max_limit:
                        logger.info(f"Processing chunk of {len(chunk)} abstracts (total so far: {processed_count})")
                        
                        chunk_triplets, chunk_taxo = await process_abstract_chunk(
                            chunk, 
                            llm_setup, 
                            refinement_cache
                        )
                        logger.info(f"Got {len(chunk_triplets)} triplets, {len(chunk_taxo)} taxonomy entries")                        
                        should_backfill = (processed_count >= max_limit)
                        
                        if should_backfill:
                            logger.info("Target reached - triggering backfill optimization")                            
                            target_successful_abstracts = max_limit if max_limit != float('inf') else len(chunk)
                            successful_abstracts = []
                            failed_abstracts = []
                            
                            triplets_by_doi = {}
                            for triplet in chunk_triplets:
                                doi = triplet[3]
                                if doi not in triplets_by_doi:
                                    triplets_by_doi[doi] = []
                                triplets_by_doi[doi].append(triplet)
                            
                            for abstract in chunk:
                                doi = abstract['doi']
                                if doi in triplets_by_doi and len(triplets_by_doi[doi]) > 0:
                                    successful_abstracts.append(abstract)
                                    logger.info(f"SUCCESS: {doi} produced {len(triplets_by_doi[doi])} triplets")
                                else:
                                    failed_abstracts.append(abstract)
                                    logger.info(f"FAILED: {doi} produced 0 triplets")
                            
                            logger.info(f"Result: {len(successful_abstracts)} successful abstracts, {len(failed_abstracts)} failed abstracts")
                            
                            backfill_attempts = 0
                            max_backfill_attempts = 10
                            
                            while len(successful_abstracts) < target_successful_abstracts and backfill_attempts < max_backfill_attempts:
                                remaining_relevant = []
                                processed_dois = {abs_data['doi'] for abs_data in chunk}
                                
                                for i, (is_relevant, relevance_score) in enumerate(results):
                                    if is_relevant and batch_items[i]['doi'] not in processed_dois:
                                        remaining_relevant.append(batch_items[i])
                                
                                if not remaining_relevant:
                                    logger.info(f"Current batch exhausted, scanning for more relevant abstracts (attempt {backfill_attempts + 1})")
                                    
                                    target_relevant_per_attempt = max_limit if max_limit != float('inf') else batch_size
                                    found_relevant_count = 0
                                    backfill_batch_size = 2000
                                    
                                    while found_relevant_count < target_relevant_per_attempt:
                                        backfill_df = load_data_with_offset("shorebirds.parquet", skip_rows, backfill_batch_size)
                                        if len(backfill_df) == 0:
                                            logger.info("No more data available for backfill")
                                            break
                                        
                                        logger.info(f"Backfill scanning batch of {len(backfill_df)} abstracts (found {found_relevant_count}/{target_relevant_per_attempt} relevant so far)")
                                        
                                        backfill_items = []
                                        for i, row_data in enumerate(backfill_df.iter_rows(named=True)):
                                            abstract_text = row_data["abstract"]
                                            title_text = row_data["title"]
                                            doi_text = row_data.get("doi")
                                            if not doi_text: continue
                                            if "captivity" in abstract_text.lower() or len(abstract_text) < 50:
                                                continue
                                            backfill_items.append({'title': title_text, 'abstract': abstract_text, 'doi': doi_text, 'idx': skip_rows + i})
                                        
                                        if backfill_items:
                                            backfill_tasks = []
                                            for item in backfill_items:
                                                backfill_tasks.append(
                                                    check_relevance(item['title'], item['abstract'], llm_setup, embed_model, embed_classifier, vectorizer, legacy_classifier)
                                                )
                                            
                                            backfill_results = await asyncio.gather(*backfill_tasks)
                                            
                                            for i, (is_relevant, relevance_score) in enumerate(backfill_results):
                                                if is_relevant:
                                                    remaining_relevant.append(backfill_items[i])
                                                    found_relevant_count += 1
                                                    logger.info(f"BACKFILL RELEVANT #{found_relevant_count}: '{backfill_items[i]['title'][:50]}...'")
                                                    
                                                    if found_relevant_count >= target_relevant_per_attempt:
                                                        break
                                        
                                        skip_rows += len(backfill_df)
                                        total_scanned += len(backfill_df)
                                        
                                        if found_relevant_count >= target_relevant_per_attempt:
                                            break
                                    
                                    logger.info(f"Backfill attempt complete: found {found_relevant_count} relevant abstracts")
                                
                                if not remaining_relevant:
                                    logger.info(f"No more relevant abstracts available for replacement (attempt {backfill_attempts + 1})")
                                    break
                                
                                needed_replacements = target_successful_abstracts - len(successful_abstracts)
                                replacement_count = min(len(remaining_relevant), needed_replacements * 2)
                                replacement_abstracts = remaining_relevant[:replacement_count]
                                
                                backfill_attempts += 1
                                logger.info(f"Backfill attempt #{backfill_attempts}: Need {needed_replacements} successful abstracts, trying {len(replacement_abstracts)} replacements")

                                replacement_triplets, replacement_taxo = await process_abstract_chunk(
                                    replacement_abstracts,
                                    llm_setup,
                                    refinement_cache
                                )
                                
                                replacement_triplets_by_doi = {}
                                for triplet in replacement_triplets:
                                    doi = triplet[3]
                                    if doi not in replacement_triplets_by_doi:
                                        replacement_triplets_by_doi[doi] = []
                                    replacement_triplets_by_doi[doi].append(triplet)
                                
                                new_successes = 0
                                for abstract in replacement_abstracts:
                                    doi = abstract['doi']
                                    if doi in replacement_triplets_by_doi and len(replacement_triplets_by_doi[doi]) > 0:
                                        successful_abstracts.append(abstract)
                                        new_successes += 1
                                        logger.info(f"REPLACEMENT SUCCESS: {doi} produced {len(replacement_triplets_by_doi[doi])} triplets")
                                    else:
                                        failed_abstracts.append(abstract)
                                        logger.info(f"REPLACEMENT FAILED: {doi} produced 0 triplets")
                                
                                chunk_triplets.extend(replacement_triplets)
                                chunk_taxo.update(replacement_taxo)
                                chunk.extend(replacement_abstracts)
                                
                                logger.info(f"Backfill #{backfill_attempts}: Got {new_successes} new successful abstracts, total successful: {len(successful_abstracts)}")
                            
                            final_successful = len(successful_abstracts)
                            final_failed = len(failed_abstracts)
                            final_triplets = len(chunk_triplets)
                            
                            logger.info(f"FINAL RESULT: {final_successful} successful abstracts, {final_failed} failed abstracts, {final_triplets} total triplets")
                            
                            norm_triplets.extend(chunk_triplets)
                            taxo_map.update(chunk_taxo) 
                            all_data.extend(chunk)
                            logger.info(f"Total: {len(norm_triplets)} triplets, {len(taxo_map)} taxonomy entries")

                            chunk = []
                        else:
                            logger.info(f"Regular chunk processed: {len(chunk_triplets)} triplets from {len(chunk)} abstracts")
                            norm_triplets.extend(chunk_triplets)
                            taxo_map.update(chunk_taxo)
                            all_data.extend(chunk)
                            logger.info(f"Total so far: {len(norm_triplets)} triplets, {len(taxo_map)} taxonomy entries")
                            chunk = []
                        
                        if processed_count >= max_limit:
                            logger.info(f"Hit limit in inner loop ({max_limit})")
                            break 
                    else:
                        item = batch_items[i]
                        has_keywords = has_shorebird_keywords(f"{item['title']} {item['abstract']}")
                        with open(irrelevant_file, 'a', encoding='utf-8') as f:
                            import json
                            f.write(json.dumps({
                                "title": item['title'], 
                                "abstract": item['abstract'], 
                                "doi": item['doi'],
                                "relevance_score": float(relevance_score),
                                "has_shorebird_keywords": has_keywords,
                                "rejection_reason": "low_relevance_no_keywords" if not has_keywords else "low_relevance_with_keywords"
                            }) + '\n')
        
        if processed_count >= max_limit:
            logger.info(f"Hit limit in outer loop ({max_limit})")
            break
        
        if len(df_batch) == 0:
            logger.info("File ended")
            if chunk:
                logger.info(f"Processing final chunk of {len(chunk)} abstracts")
                chunk_triplets, chunk_taxo = await process_abstract_chunk(
                    chunk, llm_setup, refinement_cache
                )
                logger.info(f"Final chunk: {len(chunk_triplets)} triplets, {len(chunk_taxo)} taxonomy")
                norm_triplets.extend(chunk_triplets)
                taxo_map.update(chunk_taxo)
                all_data.extend(chunk)
                logger.info(f"Final total: {len(norm_triplets)} triplets, {len(taxo_map)} taxonomy")
                chunk = []
            break

        if max_limit == float('inf') and total_scanned >= MAX_PARQUET_ROWS_TO_SCAN_IF_NO_MAX_RESULTS:
            logger.warning(f"Scanned {total_scanned} rows, processing final chunk")
            if chunk:
                logger.info(f"Final chunk due to scan limit: {len(chunk)}")
                chunk_triplets, chunk_taxo = await process_abstract_chunk(
                    chunk, llm_setup, refinement_cache
                )
                logger.info(f"Scan limit chunk: {len(chunk_triplets)} triplets, {len(chunk_taxo)} taxonomy")
                norm_triplets.extend(chunk_triplets)
                taxo_map.update(chunk_taxo)
                all_data.extend(chunk)
                logger.info(f"Scan limit total: {len(norm_triplets)} triplets, {len(taxo_map)} taxonomy")
                chunk = []
            break
            
    logger.info(f"Collected {processed_count} relevant abstracts total. {len(norm_triplets)} triplets generated")

    if not norm_triplets:
        logger.warning("No triplets generated")
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Total pipeline execution time: {elapsed_time:.2f} seconds")
        print(f"\nTotal pipeline execution time: {elapsed_time:.2f} seconds")
        return run_base

    # save results
    logger.info(f"Caching {len(norm_triplets)} triplets")
    cache_enriched_triples(norm_triplets, taxo_map, results_path)

    if EMBEDDINGS_AVAILABLE and embed_model and all_data:
        logger.info("Setting up vector search")
        print("\nSetting up vector search")
        abstracts_text = [item['abstract'] for item in all_data]
        vector_store = setup_vector_search(abstracts_text, embed_model)
    
    logger.info("Building graphs")
    print("\nBuilding graphs")
    basic_graph = build_global_graph(norm_triplets)
    
    if EMBEDDINGS_AVAILABLE and embed_model:
        logger.info("Graph embeddings")
        print("\nGraph embeddings")
        potential_connections = enrich_graph_with_embeddings(basic_graph, embed_model, results_path)
        
        if potential_connections:
            with open(results_path / "potential_connections.txt", 'w') as f:
                f.write("Potential connections not in graph:\n\n")
                for node1, node2, similarity in potential_connections:
                    f.write(f"{node1} -- {node2} (similarity: {similarity:.3f})\\n")
            
        create_embedding_visualization(basic_graph, embed_model, figures_path)

    logger.info("Creating visualizations")
    analyze_graph_detailed(basic_graph, figures_path)
    analyze_hub_node(basic_graph, figures_path)

    # species verification list
    # get subject names from taxonomy map
    species_names = sorted(list(taxo_map.keys()))

    if species_names:
        species_file = results_path / "species_to_verify_with_wikispecies.txt"
        with open(species_file, 'w', encoding='utf-8') as f:
            for name in species_names:
                f.write(f"{name}\\n")
        logger.info(f"Species list saved: {species_file}")
        print(f"\\nSpecies list saved: {species_file}")
        print(f"Total species: {len(species_names)}")
        lookup_path = Path(os.path.dirname(os.path.abspath(__file__))) / "wikispecies_taxonomy_lookup.json"
        print(f"Results will go to: {lookup_path}")
    else:
        logger.warning("No species found for verification")
        print("\\nNo species found for verification")

    print("\\nPipeline complete!")
    print(f"Results: {results_path}")
    print(f"Figures: {figures_path}")
    print("\nNext steps:")
    print("1. Run Wikispecies verification")
    print("2. Run taxonomy comparison")
    print(f"\nFeatures used:")
    print(f"- Batch processing: ✓ ({batch_size} per batch)")
    print(f"- Classifier loading: ✓")
    print(f"- Relevance filtering: ✓")
    print(f"- IUCN refinement: ✓")
    print(f"- Verification threshold: ✓ ({VERIFICATION_THRESHOLD})")

    # t-SNE viz
    if basic_graph and EMBEDDINGS_AVAILABLE and embed_model:
        logger.info("Making t-SNE plot")
        try:
            await visualize_triplet_sentence_embeddings_batch_ingest(
                basic_graph, 
                embed_model, 
                figures_path,
                filename="triplet_sentences_tsne_batch_ingest.png"
            )
        except Exception as e_tsne:
            logger.error(f"t-SNE error: {e_tsne}", exc_info=True)
    else:
        logger.warning("Skipping t-SNE: missing requirements")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total pipeline execution time: {elapsed_time:.2f} seconds")
    print(f"\nTotal pipeline execution time: {elapsed_time:.2f} seconds")
    
    return run_base


async def process_abstract_chunk(
    chunk: List[Dict], 
    llm_setup, 
    refinement_cache
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, Dict]]:
    logger.info(f"Processing chunk of {len(chunk)} abstracts")
    dois = [d.get('doi', 'N/A') for d in chunk]
    logger.debug(f"DOIs: {dois}")

    logger.info(f"Pre-filtering {len(chunk)} abstracts for impact/conservation evidence with cheap model")
    cheap_llm_setup = llm_setup.copy()
    cheap_llm_setup['model'] = 'qwen/qwen3-30b-a3b'
    
    threat_filter_tasks = []
    for abstract_data in chunk:
        abstract_text = abstract_data['abstract']
        threat_filter_tasks.append(
            check_for_impact_conservation_evidence(
                abstract_text, 
                cheap_llm_setup
            )
        )
    
    if threat_filter_tasks:
        threat_filter_results = await asyncio.gather(*threat_filter_tasks)
        threat_containing_abstracts = []
        
        for i, is_impact_relevant in enumerate(threat_filter_results):
            if is_impact_relevant:
                threat_containing_abstracts.append(chunk[i])
                logger.info(f"IMPACT/CONSERVATION EVIDENCE DETECTED in abstract {chunk[i]['doi']}")
            else:
                logger.info(f"NO IMPACT/CONSERVATION EVIDENCE in abstract {chunk[i]['doi']} - FILTERED OUT")
        
        logger.info(f"Impact filter: {len(threat_containing_abstracts)}/{len(chunk)} abstracts contain impact/conservation evidence")
        
        if not threat_containing_abstracts:
            logger.warning("No abstracts contain impact/conservation evidence after filtering")
            return [], {}
        
        # Replace chunk with filtered abstracts
        chunk = threat_containing_abstracts

    logger.info(f"EVIDENCE GATE: Checking {len(chunk)} abstracts for primary research evidence")
    evidence_filter_tasks = []
    for abstract_data in chunk:
        evidence_filter_tasks.append(
            check_for_primary_evidence(
                abstract_data['abstract'], 
                cheap_llm_setup
            )
        )
    
    if evidence_filter_tasks:
        evidence_results = await asyncio.gather(*evidence_filter_tasks)
        primary_evidence_abstracts = []
        
        for i, evidence_result in enumerate(evidence_results):
            if evidence_result.get("primary_evidence_found", False):
                primary_evidence_abstracts.append(chunk[i])
                strongest_sentence = evidence_result.get("strongest_evidence_sentence", "")
                logger.info(f"PRIMARY EVIDENCE FOUND in {chunk[i]['doi']}: '{strongest_sentence[:100]}...'")
            else:
                logger.info(f"NO PRIMARY EVIDENCE in {chunk[i]['doi']} - FILTERED OUT")
        
        logger.info(f"Evidence gate: {len(primary_evidence_abstracts)}/{len(chunk)} abstracts contain primary research evidence")
        
        if not primary_evidence_abstracts:
            logger.warning("No abstracts contain primary evidence after filtering")
            return [], {}
        
        chunk = primary_evidence_abstracts

    summary_tasks = []
    details = [] 

    for abstract_data in chunk:
        summary_tasks.append(convert_to_summary(abstract_data['abstract'], llm_setup))
        details.append({
            'abstract_text': abstract_data['abstract'],
            'doi': abstract_data['doi'],
            'title': abstract_data['title']
        })
    
    # create DOI to abstract mapping for IUCN classification
    doi_to_abstract = {detail['doi']: detail['abstract_text'] for detail in details}
    
    raw_triplets = []
    if summary_tasks:
        logger.info(f"Generating summaries for {len(summary_tasks)} abstracts")
        summaries = await asyncio.gather(*summary_tasks)
        logger.info("Summary generation done")

        p2_tasks = []
        for i, summary_text in enumerate(summaries):
            if i < len(details):
                current = details[i]
                abs_text = current['abstract_text']
                doi = current['doi']

                if summary_text:
                    async def process_single(abstract_content, doi_val, llm_s):
                        logger.info(f"Extracting entities for {doi_val}")
                        entities = await extract_entities_concurrently(abstract_content, llm_s)
                        if entities and entities.get("species") and entities.get("threats"):
                            logger.info(f"Generating relationships for {doi_val} ({len(entities['species'])} species, {len(entities['threats'])} threats)")
                            trips = await generate_relationships_concurrently(abstract_content, entities["species"], entities["threats"], llm_s, doi_val)
                            return trips
                        else:
                            logger.warning(f"No entities for {doi_val}: {abstract_content[:50]}")
                            return []
                    text_for_extraction = summary_text if summary_text and summary_text.strip() else abs_text
                    p2_tasks.append(process_single(text_for_extraction, doi, llm_setup))
                else:
                    logger.warning(f"No summary for {doi}")
            else:
                logger.error(f"Index mismatch at {i}")
        
        if p2_tasks:
            logger.info(f"Running entity extraction for {len(p2_tasks)} abstracts")
            p2_results = await asyncio.gather(*p2_tasks)
            logger.info("Entity extraction done")
            for result_list in p2_results:
                if result_list: 
                    raw_triplets.extend(result_list)
    
    logger.info(f"Extracted {len(raw_triplets)} raw triplets")

    if not raw_triplets:
        logger.warning("No raw triplets extracted")
        return [], {}

    logger.info("Deferring IUCN classification until after verification/sentiment")
    enriched_triplets = list(raw_triplets)

    if not enriched_triplets:
        logger.warning("No enriched triplets")
        return [], {}

    # verification
    logger.info(f"Verifying {len(enriched_triplets)} triplets")
    
    triplets_by_doi = defaultdict(list)
    doi_to_abstract = {data['doi']: data['abstract'] for data in chunk if 'doi' in data and 'abstract' in data}

    for s, p, o, d in enriched_triplets:
        if d in doi_to_abstract: 
            triplets_by_doi[d].append((s, p, o, d)) 
        else:
            logger.warning(f"DOI {d} not in current chunk, skipping verification")

    verified_triplets = []
    verify_tasks = []

    for doi, triplets_for_doi in triplets_by_doi.items():
        abstract = doi_to_abstract.get(doi)
        if abstract and triplets_for_doi:
            verify_tasks.append(
                verify_triplets(
                    triplets_for_doi, 
                    abstract, 
                    llm_setup, 
                    verification_cutoff=0.75
                )
            )
        elif not abstract:
            logger.warning(f"No abstract for DOI {doi}, skipping {len(triplets_for_doi)} triplets")

    if verify_tasks:
        logger.info(f"Running verification for {len(verify_tasks)} abstracts")
        verify_results = await asyncio.gather(*verify_tasks, return_exceptions=True)
        logger.info("Verification done")
        
        dois_list = list(triplets_by_doi.keys())
        for i, result in enumerate(verify_results):
            doi = dois_list[i] if i < len(dois_list) else "Unknown"

            if isinstance(result, Exception):
                logger.error(f"Verification error for {doi}: {result}")
                continue
            
            if result and isinstance(result, tuple) and len(result) == 2:
                verified, counts = result
                logger.info(f"{doi} - submitted: {counts.get('submitted',0)}, yes: {counts.get('verified_yes',0)}, no: {counts.get('verified_no',0)}, errors: {counts.get('errors',0)}")
                if verified:
                    verified_triplets.extend(verified)
            else:
                 logger.error(f"Bad verification result for {doi}: {result}")

    logger.info(f"Total verified: {len(verified_triplets)}")

    if not verified_triplets: 
        logger.warning("No triplets survived verification")
        return [], {}

    logger.info(f"Classifying threat sentiment for {len(verified_triplets)} triplets.")
    threat_classification_tasks = []
    for s, p, o, d in verified_triplets:
        threat_desc, _, _, _ = parse_and_validate_object(o)
        threat_classification_tasks.append(
            classify_threat_for_subject(s, p, threat_desc, llm_setup, refinement_cache)
        )

    threat_classification_results = await asyncio.gather(*threat_classification_tasks)

    triplets_surviving_threat_check = []
    for i, result in enumerate(threat_classification_results):
        if result:
            triplets_surviving_threat_check.append(verified_triplets[i])
            classification = result.get('classification', 'N/A')
            logger.info(f"Threat check PASSED for triplet. Classification: {classification}")
        else:
            logger.info(f"Threat check filtered out triplet (was positive/very positive).")
            
    logger.info(f"{len(triplets_surviving_threat_check)}/{len(verified_triplets)} triplets survived threat sentiment check.")
    
    if not triplets_surviving_threat_check:
        logger.warning("No triplets survived threat sentiment check")
        return [], {}

    # normalization
    logger.info(f"Normalizing species names for {len(triplets_surviving_threat_check)} triplets")
    normalized_triplets, taxonomy_map = await normalize_species_names(
        triplets_surviving_threat_check, 
        llm_setup
    )
    logger.info(f"Normalization done: {len(normalized_triplets)} triplets, {len(taxonomy_map)} taxonomy entries")

    post_iucn_items = []
    post_pre_enriched = {}
    doi_to_abstract_post = {data['doi']: data['abstract'] for data in chunk if 'doi' in data and 'abstract' in data}

    for idx, (s, p, o, d) in enumerate(normalized_triplets):
        desc, code, name, is_valid = parse_and_validate_object(o)
        final_desc = desc if desc else o
        needs_iucn = not is_valid or not (code and name) or code == "12.1"

        if needs_iucn:
            abstract_text = doi_to_abstract_post.get(d, None)
            cache_key = f"iucn_classify_json_schema:{final_desc}|context:{s}|{p}|abstract:{bool(abstract_text)}"
            cached = refinement_cache.get(cache_key)
            if cached:
                cached_code, cached_name = cached
                refined_o = f"{final_desc} [IUCN: {cached_code} {cached_name}]"
                post_pre_enriched[idx] = (s, p, refined_o, d)
            else:
                post_iucn_items.append((s, p, final_desc, o, idx, abstract_text))
        else:
            refined_o = f"{final_desc} [IUCN: {code} {name}]"
            post_pre_enriched[idx] = (s, p, refined_o, d)

    post_iucn_tasks = [
        get_iucn_classification_json(item[0], item[1], item[2], llm_setup, refinement_cache, item[5])
        for item in post_iucn_items
    ]

    final_triplets: List[Tuple[str, str, str, str]] = [None] * len(normalized_triplets)  # type: ignore

    if post_iucn_tasks:
        logger.info(f"IUCN classification for {len(post_iucn_tasks)} items (post-filter)")
        post_iucn_results = await asyncio.gather(*post_iucn_tasks)
        logger.info("IUCN classification done (post-filter)")
        for i, (code, name) in enumerate(post_iucn_results):
            s_iucn, p_iucn, desc_iucn, _orig_o, orig_idx, _abstract = post_iucn_items[i]
            refined_o = f"{desc_iucn} [IUCN: {code} {name}]"
            final_triplets[orig_idx] = (s_iucn, p_iucn, refined_o, normalized_triplets[orig_idx][3])

    for idx, triplet in post_pre_enriched.items():
        final_triplets[idx] = triplet

    # fill any gaps (keep original object if IUCN missing)
    for idx in range(len(normalized_triplets)):
        if final_triplets[idx] is None:  # type: ignore
            s, p, o, d = normalized_triplets[idx]
            final_triplets[idx] = (s, p, o, d)

    # compact None
    final_triplets = [t for t in final_triplets if t is not None]
    logger.info(f"Post-filter IUCN enrichment complete: {len(final_triplets)} triplets")

    logger.info(f"Chunk complete: returning {len(final_triplets)} triplets, {len(taxonomy_map)} taxonomy entries")

    log_metrics_summary(llm_setup, logger)

    return final_triplets, taxonomy_map


async def run_batch_pipeline_logic(args):
    logger.info("Running batch pipeline")
    return await run_main_pipeline_logic(args)

def run_batch_enabled_pipeline(args):
    logger.info("Starting batch-enabled pipeline")
    return asyncio.run(run_batch_pipeline_logic(args))


def run_wikispecies_verification_logic(args):
    # basic setup
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    model = args.target_model_name if hasattr(args, 'target_model_name') and args.target_model_name else os.getenv('MODEL_NAME_FOR_RUN', "google/gemini-2.5-flash-preview")
    max_str = args.target_max_results if hasattr(args, 'target_max_results') and args.target_max_results else os.getenv('MAX_RESULTS', "all")
    
    max_path = "all"
    if str(max_str).lower() == 'all':
        max_path = "all"
    elif str(max_str).isdigit():
        max_path = int(max_str)

    base_dir = get_dynamic_run_base_path(model, max_path, script_dir)
    logs_path = base_dir / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / "wikispecies_verification.log"
    setup_pipeline_logging(log_file) 

    logger.info("Starting Wikispecies verification")
    logger.info(f"Logs: {log_file}")
    
    try:
        species_file = Path(args.verify_species_wikispecies)
        if not species_file.is_file():
            logger.error(f"Species file not found: {species_file}")
            sys.exit(1)
        
        with open(species_file, 'r', encoding='utf-8') as f:
            species = [line.strip() for line in f if line.strip()]
        
        if not species:
            logger.error("No species in file")
            sys.exit(1)
        
        print(f"Verifying {len(species)} species from {species_file}")
        print(f"Results will go to: {base_dir / 'results'}")
        
        try:
            asyncio.run(verify_species_with_wikispecies_concurrently(species, base_dir / 'results'))
        except Exception as e:
            logger.error(f"Verification error: {e}", exc_info=True)
            sys.exit(1)
        logger.info("Wikispecies verification done")
    except Exception as e:
        logger.error(f"Error in verification: {e}", exc_info=True)
        sys.exit(1)

def run_taxonomy_comparison_logic(args):
    # setup paths
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    model = args.target_model_name if hasattr(args, 'target_model_name') and args.target_model_name else os.getenv('MODEL_NAME_FOR_RUN', "google/gemini-2.5-flash-preview")
    max_str = args.target_max_results if hasattr(args, 'target_max_results') and args.target_max_results else os.getenv('MAX_RESULTS', "all")

    max_path = "all"
    if str(max_str).lower() == 'all':
        max_path = "all"
    elif str(max_str).isdigit():
        max_path = int(max_str)
            
    base_dir = get_dynamic_run_base_path(model, max_path, script_dir)
    logs_path = base_dir / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / "taxonomy_comparison.log"
    setup_pipeline_logging(log_file) 

    logger.info("Starting taxonomy comparison")
    logger.info(f"Logs: {log_file}")

    try:
        enriched_file = base_dir / "results" / "enriched_triplets.json" 
        lookup_file = script_dir / "wikispecies_taxonomy_lookup.json"
        output_file = base_dir / "results" / "taxonomy_discrepancy_details.log.json" 

        if not enriched_file.exists():
            logger.error(f"Enriched triplets not found: {enriched_file}. Run main pipeline first")
            return
        if not lookup_file.exists():
            logger.error(f"Wikispecies lookup not found: {lookup_file}. Run verification first")
            return

        print(f"Using triplets: {enriched_file}")
        print(f"Using lookup: {lookup_file}")
        compare_and_log_taxonomy_discrepancies(
            enriched_file,
            lookup_file,
            output_file
        )
        print("Taxonomy comparison done")
    except Exception as e:
        logger.critical(f"Error in taxonomy comparison: {e}", exc_info=True)


# safety constant for scanning parquet without max
MAX_PARQUET_ROWS_TO_SCAN_IF_NO_MAX_RESULTS = 50000
