import nltk
from typing import List, Tuple, Dict
from pathlib import Path
import os
from collections import defaultdict
import asyncio
import sys
import logging
from .cache import Cache, SimpleCache
from .setup import setup_pipeline_logging, get_dynamic_run_base_path, load_data_with_offset
from .batch_ingesting import (BATCH_CONFIG, EMBEDDINGS_AVAILABLE, load_classifier_components, 
                             predict_relevance_local, classify_abstract_relevance_ollama, 
                             setup_embedding_classifier, predict_relevance_embeddings)
from .iucn_refinement import get_iucn_classification_json, parse_and_validate_object, cache_enriched_triples
from .triplet_extraction import verify_triplets, normalize_species_names, convert_to_summary, extract_entities_concurrently, generate_relationships_concurrently
from .graph_analysis import (build_global_graph, analyze_graph_detailed, 
                           enrich_graph_with_embeddings, 
                           create_embedding_visualization, analyze_hub_node,
                           visualize_triplet_sentence_embeddings_batch_ingest)
from .wikispecies_utils import verify_species_with_wikispecies_concurrently, compare_and_log_taxonomy_discrepancies

from .setup import setup_llm, setup_vector_search

logger = logging.getLogger("pipeline")

async def run_main_pipeline_logic(args):
    
    # basic path setup
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    llm_sys = setup_llm() 
    model_name = os.getenv('MODEL_NAME_FOR_RUN', llm_sys["model"])

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

    # try to load pre-trained stuff
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

    async def check_relevance(title, abstract, llm_setup, embed_model, embed_classifier, vectorizer, legacy_classifier):
        # try embedding classifier first
        if embed_classifier and embed_model and EMBEDDINGS_AVAILABLE:
            logger.debug(f"Using embedding classifier for '{title[:30]}...'")
            return predict_relevance_embeddings(abstract, embed_model, embed_classifier)
        elif legacy_classifier and vectorizer:
            logger.debug(f"Using TF-IDF for '{title[:30]}...'")
            return predict_relevance_local(abstract, vectorizer, legacy_classifier)
        # fallback to LLM
        logger.debug(f"Using LLM for '{title[:30]}...'")
        return await classify_abstract_relevance_ollama(title, abstract, llm_setup)

    while True:
        if processed_count >= max_limit:
            logger.info(f"Hit limit ({max_limit})")
            break

        logger.info(f"Loading batch: skip={skip_rows}, max={batch_size}")
        df_batch = load_data_with_offset("all_abstracts.parquet", skip_rows, batch_size)
        
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
        
        tasks = []
        for item in batch_items:
            tasks.append(
                check_relevance(item['title'], item['abstract'], llm_setup, embed_model, embed_classifier, vectorizer, legacy_classifier)
            )
        
        if tasks:
            logger.info(f"Checking relevance for {len(tasks)} abstracts")
            results = await asyncio.gather(*tasks)
            logger.info("Relevance check done")

            for i, is_relevant in enumerate(results):
                if is_relevant:
                    relevant_item = batch_items[i]
                    chunk.append(relevant_item)
                    processed_count += 1 

                    if len(chunk) >= BATCH_CONFIG['processing_batch_size'] or \
                       processed_count >= max_limit:
                        
                        logger.info(f"Processing chunk of {len(chunk)} abstracts (total so far: {processed_count})")
                        
                        chunk_triplets, chunk_taxo = await process_abstract_chunk(
                            chunk, 
                            llm_setup, 
                            refinement_cache
                        )
                        logger.info(f"Got {len(chunk_triplets)} triplets, {len(chunk_taxo)} taxonomy entries")
                        
                        norm_triplets.extend(chunk_triplets)
                        taxo_map.update(chunk_taxo) 
                        all_data.extend(chunk)
                        logger.info(f"Total: {len(norm_triplets)} triplets, {len(taxo_map)} taxonomy entries")

                        chunk = [] # reset
                    
                    if processed_count >= max_limit:
                        logger.info(f"Hit limit in inner loop ({max_limit})")
                        break 
        
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

    return run_base


async def process_abstract_chunk(
    chunk: List[Dict], 
    llm_setup, 
    refinement_cache
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, Dict]]:
    """Process chunk of abstracts through summary, extraction, IUCN, and normalization"""
    logger.info(f"Processing chunk of {len(chunk)} abstracts")
    dois = [d.get('doi', 'N/A') for d in chunk]
    logger.debug(f"DOIs: {dois}")

    summary_tasks = []
    details = [] 

    for abstract_data in chunk:
        summary_tasks.append(convert_to_summary(abstract_data['abstract'], llm_setup))
        details.append({
            'abstract_text': abstract_data['abstract'],
            'doi': abstract_data['doi'],
            'title': abstract_data['title']
        })
    
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
                    p2_tasks.append(process_single(abs_text, doi, llm_setup))
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

    iucn_items = [] 
    pre_enriched = {} 

    for idx, (s, p, original_o, d) in enumerate(raw_triplets):
        desc, code, name, is_valid = parse_and_validate_object(original_o)
        final_desc = desc if desc else original_o
        needs_iucn = not is_valid or not (code and name) or code == "12.1"

        if needs_iucn:
            cache_key = f"iucn_classify_json_schema:{final_desc}|context:{s}|{p}"
            cached = refinement_cache.get(cache_key)
            if cached:
                cached_code, cached_name = cached
                refined_o = f"{final_desc} [IUCN: {cached_code} {cached_name}]"
                pre_enriched[idx] = (s, p, refined_o, d)
            else:
                iucn_items.append((s, p, final_desc, original_o, idx))
        else:
            refined_o = f"{final_desc} [IUCN: {code} {name}]"
            pre_enriched[idx] = (s, p, refined_o, d)
    
    iucn_tasks = [
        get_iucn_classification_json(item[0], item[1], item[2], llm_setup, refinement_cache) 
        for item in iucn_items
    ]
    
    enriched_triplets = [None] * len(raw_triplets)

    if iucn_tasks:
        logger.info(f"IUCN classification for {len(iucn_tasks)} items")
        iucn_results = await asyncio.gather(*iucn_tasks)
        logger.info("IUCN classification done")
        for i, (code, name) in enumerate(iucn_results):
            s_iucn, p_iucn, desc_iucn, _orig_o, orig_idx = iucn_items[i]
            refined_o = f"{desc_iucn} [IUCN: {code} {name}]"
            enriched_triplets[orig_idx] = (s_iucn, p_iucn, refined_o, raw_triplets[orig_idx][3])
    
    for idx, triplet in pre_enriched.items():
        enriched_triplets[idx] = triplet
    
    # fill in any gaps
    for idx in range(len(raw_triplets)):
        if enriched_triplets[idx] is None:
            s, p, o, d = raw_triplets[idx]
            logger.warning(f"Triplet {idx} ({s[:20]}) missed IUCN, using original: {o[:30]}")
            enriched_triplets[idx] = (s, p, o, d)
            
    enriched_triplets = [t for t in enriched_triplets if t is not None]
    logger.info(f"IUCN enrichment complete: {len(enriched_triplets)} triplets")

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

    # normalization
    logger.info(f"Normalizing species names for {len(verified_triplets)} triplets")
    normalized_triplets, taxonomy_map = await normalize_species_names(
        verified_triplets, 
        llm_setup
    )
    logger.info(f"Normalization done: {len(normalized_triplets)} triplets, {len(taxonomy_map)} taxonomy entries")
    
    logger.info(f"Chunk complete: returning {len(normalized_triplets)} triplets, {len(taxonomy_map)} taxonomy entries")
    return normalized_triplets, taxonomy_map


async def run_batch_pipeline_logic(args):
    """Placeholder - calls main pipeline for now"""
    logger.info("Running batch pipeline (placeholder)")
    return await run_main_pipeline_logic(args)

def run_batch_enabled_pipeline(args):
    """Entry point for batch pipeline"""
    logger.info("Starting batch-enabled pipeline")
    return asyncio.run(run_batch_pipeline_logic(args))


def run_wikispecies_verification_logic(args):
    # basic setup
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    model = args.target_model_name if hasattr(args, 'target_model_name') and args.target_model_name else os.getenv('MODEL_NAME_FOR_RUN', "google/gemini-flash-1.5")
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
    model = args.target_model_name if hasattr(args, 'target_model_name') and args.target_model_name else os.getenv('MODEL_NAME_FOR_RUN', "google/gemini-flash-1.5")
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
