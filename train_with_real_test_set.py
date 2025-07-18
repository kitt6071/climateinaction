import json
import pickle
import numpy as np
import logging
import os
from pathlib import Path
import random
from typing import List, Dict

from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
    exit(1)

def integrate_and_retrain_balanced(targeted_negatives: List[str], positive_synthetics: List[str], existing_training_data: List[Dict], pipeline_id: str, current_round: int = 1):
    try:
        # convert targeted negatives to training format
        new_training_examples = []
        for abstract in targeted_negatives:
            new_training_examples.append({
                'text': abstract,
                'label': 0,  # negative examples
                'source': 'targeted_synthetic_negatives'
            })
        
        # convert positive synthetics to training format
        for abstract in positive_synthetics:
            new_training_examples.append({
                'text': abstract,
                'label': 1,  # positive examples
                'source': 'targeted_synthetic_positives'
            })
        
        # Combine with existing training data
        combined_training_data = existing_training_data + new_training_examples
        
        logger.info(f"Combined training data: {len(combined_training_data)} examples")
        logger.info(f"  - Original: {len(existing_training_data)}")
        logger.info(f"  - New negative synthetics: {len(targeted_negatives)}")
        logger.info(f"  - New positive synthetics: {len(positive_synthetics)}")
        
        # Update the training cache
        training_cache_path = f".cache/{pipeline_id}_training_data.pkl"
        with open(training_cache_path, "wb") as f:
            pickle.dump(combined_training_data, f)
        
        # Clear embedding cache to force regeneration
        train_embeddings_cache = f".cache/{pipeline_id}_train_embeddings.npy"
        if Path(train_embeddings_cache).exists():
            Path(train_embeddings_cache).unlink()
            logger.info("Cleared training embeddings cache (will regenerate)")
        
        # Recursively call the main training function for next round
        logger.info("Starting next training round")
        return train_classifier_with_real_test_set(round_num=current_round + 1)
        
    except Exception as e:
        logger.error(f"Error in integrate_and_retrain_balanced: {e}")
        return None

def analyze_abstract_relevance(title: str, abstract: str) -> Dict:
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
        
        load_dotenv()
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not found")
            return {}
        
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        model = "deepseek/deepseek-r1-0528"
        
        analysis_schema = {
            "type": "object",
            "properties": {
                "is_shorebird_relevant": {
                    "type": "boolean",
                    "description": "True if this abstract is about shorebirds (Charadriiformes order)"
                },
                "main_subject": {
                    "type": "string", 
                    "description": "The primary subject"
                },
                "species": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "A list of species explicitly mentioned in the abstract"
                },
                "methodology": {
                    "type": "string",
                    "description": "The research approach used (e.g., 'radio tracking', 'genetic analysis', 'behavioral observation')"
                },
                "habitat": {
                    "type": "string",
                    "description": "The environment studied (e.g., 'tropical forest', 'marine environment', 'arctic tundra')"
                },
                "research_focus": {
                    "type": "string",
                    "description": "What the study investigates (e.g., 'space use patterns', 'population genetics', 'feeding behavior')"
                }
            },
            "required": ["is_shorebird_relevant", "main_subject", "methodology", "habitat", "research_focus"]
        }
        
        prompt = f"""Analyze this research abstract and extract key information:

                Title: {title}
                Abstract: {abstract[:800]}...

                Identify:
                1. Whether this studies shorebirds (order Charadriiformes - includes sandpipers, plovers, turnstones, etc.)
                2. The main subject/species studied
                3. A list species explicitly mentioned
                3. The research methodology 
                4. The habitat/environment
                5. The primary research focus"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a scientific literature analyst. Extract key information from research abstracts accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "abstract_analysis",
                    "schema": analysis_schema
                }
            }
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        logger.error(f"Error in abstract analysis: {e}")
        return {}

def generate_similar_abstract(analysis: Dict, original_title: str) -> str:
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
        import time
        
        load_dotenv()
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            return ""
        
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        model = "deepseek/deepseek-r1-0528"
        
        response_schema = {
            "type": "object",
            "properties": {
                "abstract": {
                    "type": "string",
                    "description": "A scientific abstract similar in style and methodology"
                }
            },
            "required": ["abstract"]
        }
        
        # Natural generation prompt without mentioning what NOT to do
        prompt = f"""Generate a scientific abstract for a study with these characteristics:

                Subject: {analysis.get('main_subject', 'wildlife')}
                Species: {', '.join(analysis.get('species', ['bird']))}
                Methodology: {analysis.get('methodology', 'field study')}
                Habitat: {analysis.get('habitat', 'natural environment')}
                Research Focus: {analysis.get('research_focus', 'behavior and ecology')}

                Write a realistic academic abstract that:
                - Studies the SAME SPECIES MENTIONED using similar methods
                - Uses the same research approach and habitat type
                - Investigates a similar research question
                - Follows standard scientific abstract format
                - Is 400-600 words long"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert scientific writer. Generate realistic research abstracts that follow academic conventions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "similar_abstract",
                    "schema": response_schema
                }
            }
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("abstract", "")
        
    except Exception as e:
        logger.error(f"Error generating similar abstract: {e}")
        return ""

def generate_targeted_synthetic_negatives(problem_cases: Dict, num_negatives: int = 20) -> List[str]:    
    false_positives = problem_cases.get('false_positives', [])
    uncertain_texts = problem_cases.get('uncertain_texts', [])
    
    if not false_positives and not uncertain_texts:
        logger.info("No problem cases found")
        return []
    
    logger.info(f"-------Analyzing and generating improved synthetic negatives--------")
    logger.info(f"Processing {len(false_positives)} false positives and {len(uncertain_texts)} uncertain cases")
    
    all_cases = []
    # Add false positives
    for abstract, title in false_positives:
        all_cases.append({'abstract': abstract, 'title': title, 'type': 'false_positive'})
    
    # split uncertain cases based on shorebird keywords
    missed_positives = []
    true_negatives = []
    
    for abstract, title in uncertain_texts:
        if is_shorebird_paper_llm(title, abstract):
            missed_positives.append({'abstract': abstract, 'title': title, 'type': 'missed_positive'})
        else:
            true_negatives.append({'abstract': abstract, 'title': title, 'type': 'uncertain'})
    
    logger.info(f"Uncertain cases split: {len(missed_positives)} missed positives, {len(true_negatives)} true negatives")
    
    # add true negatives to processing (for negative synthetic generation)
    for case in true_negatives:
        all_cases.append(case)
    
    generated_abstracts = []
    positive_synthetics = []
    
    # first generate positive synthetic examples from missed positives (top 50%)
    num_positives_to_generate = max(1, len(missed_positives))  # at least 1, up to 100%
    logger.info(f"Generating {num_positives_to_generate} positive synthetics from {len(missed_positives)} missed positives")
    
    for i, case in enumerate(missed_positives[:num_positives_to_generate]):
        try:
            logger.info(f"Generating positive synthetic from missed shorebird case {i+1}")
            analysis = analyze_abstract_relevance(case['title'], case['abstract'])
            
            if analysis:
                logger.info(f"Missed positive - {analysis.get('main_subject', 'unknown')}: {case['title'][:100]}...")
                logger.info(f'Analysis species list: {', '.join(analysis.get('species', ['bird']))} ')
                logger.info(f"Original abstract: {case['abstract']}")
                # Gen similar shorebird abstract using same method but for positive class
                similar_abstract = generate_similar_abstract(analysis, case['title'])
                
                if similar_abstract and len(similar_abstract) > 200:
                    positive_synthetics.append(similar_abstract)
                    logger.info(f"Generated positive synthetic {len(positive_synthetics)} ({len(similar_abstract)} chars)")
                    logger.info(f"Generated content: {similar_abstract}")
                
        except Exception as e:
            logger.error(f"Error generating positive synthetic {i+1}: {e}")
    
    logger.info(f"Generated {len(positive_synthetics)} positive synthetic examples")
    
    # Then generate negative synthetic examples from true negatives (top 50%)
    num_negatives_to_generate = max(1, len(all_cases))  # at least 1, up to 100%
    logger.info(f"Generating {num_negatives_to_generate} negative synthetics from {len(all_cases)} total cases")
    
    for i, case in enumerate(all_cases[:num_negatives_to_generate]):
        try:
            logger.info(f"Processing case {i+1}/{num_negatives_to_generate}: {case['type']}")
            
            # 1: Analyze the abstract neutrally
            analysis = analyze_abstract_relevance(case['title'], case['abstract'])
            
            if not analysis:
                logger.warning(f"Failed to analyze case {i+1}")
                continue
                
            logger.info(f"Analysis: {analysis.get('main_subject', 'unknown')} using {analysis.get('methodology', 'unknown')}")
            logger.info(f'Analysis species list: {', '.join(analysis.get('species', ['bird']))} ')
            
            # 2: Only generate if it's not actually about shorebirds
            if not analysis.get('is_shorebird_relevant', False):
                # 3: Generate similar abstract naturally (no negation prompts)
                logger.info(f"Original abstract: {case['abstract']}")
                similar_abstract = generate_similar_abstract(analysis, case['title'])
                
                if similar_abstract and len(similar_abstract) > 200:
                    generated_abstracts.append(similar_abstract)
                    logger.info(f"Generated similar non-shorebird abstract {len(generated_abstracts)} ({len(similar_abstract)} chars)")
                    logger.info(f"Generated content: {similar_abstract}")
                else:
                    logger.warning(f"Generated abstract too short or empty for case {i+1}")
            else:
                logger.info(f"Case {i+1} is actually about shorebirds - skipping generation")
                logger.info(f"Original Abstract: {case['abstract']}")
                
        except Exception as e:
            logger.error(f"Error processing case {i+1}: {e}")
            continue
    
    logger.info(f"Generated {len(generated_abstracts)} improved targeted negatives using natural generation")
    return generated_abstracts, positive_synthetics

def is_shorebird_paper_llm(title: str, abstract: str) -> bool:
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
        
        load_dotenv()
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not found, falling back to keyword matching")
            return has_shorebird_keywords_fallback(title, abstract)
        
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        model = "deepseek/deepseek-r1-0528"
        
        # Truncate abstract if too long to avoid token limits
        truncated_abstract = abstract[:1500] if len(abstract) > 1500 else abstract
        
        prompt = f"""You are an expert ornithologist. Analyze this research paper and determine if it is primarily about shorebirds.

SHOREBIRDS are birds in the order Charadriiformes. Key indicators include:

COMMON NAMES: sandpiper, plover, oystercatcher, godwit, curlew, turnstone, phalarope, avocet, stilt, snipe, dowitcher, yellowlegs, redshank, greenshank, dunlin, sanderling, killdeer, whimbrel

SCIENTIFIC NAMES: Calidris (sandpipers), Charadrius (plovers), Limosa (godwits), Numenius (curlews), Arenaria (turnstones), Haematopus (oystercatchers), Tringa (shanks), Pluvialis (golden plovers), Gallinago (snipes), Limnodromus (dowitchers), Phalaropus (phalaropes), Recurvirostra (avocets), Himantopus (stilts)

FAMILIES: Scolopacidae, Charadriidae, Haematopodidae, Recurvirostridae, Phalaropodidae

IMPORTANT: Papers about Red Knots (Calidris canutus), Sanderlings (Calidris alba), Black-tailed Godwits (Limosa limosa), Whimbrels (Numenius phaeopus), and other species listed above ARE shorebird papers.

NOT SHOREBIRDS: ducks, geese, swans, gulls, terns, cormorants, pelicans, songbirds, raptors, gamebirds

Title: {title}

Abstract: {truncated_abstract}

Is this paper primarily focused on studying shorebird species?"""

        response_schema = {
            "type": "object",
            "properties": {
                "is_shorebird_paper": {
                    "type": "boolean",
                    "description": "True if this paper is primarily about shorebird species (order Charadriiformes), False otherwise"
                },
                "reasoning": {
                    "type": "string", 
                    "description": "Brief explanation of why this is or isn't about shorebirds"
                }
            },
            "required": ["is_shorebird_paper", "reasoning"]
        }

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert ornithologist specializing in bird taxonomy and classification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "shorebird_classification",
                    "schema": response_schema
                }
            }
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("is_shorebird_paper", False)
        
    except Exception as e:
        logger.warning(f"Error in LLM shorebird detection: {e}, falling back to keywords")
        return has_shorebird_keywords_fallback(title, abstract)

def has_shorebird_keywords_fallback(title: str, abstract: str) -> bool:
    """Check if title or abstract contains shorebird-specific keywords"""
    shorebird_keywords = [
        # Order and explicit names
        'charadriiformes', 'shorebird', 'shorebirds', 'wader', 'waders',
        # Species names
        'sandpiper', 'sandpipers', 'plover', 'plovers', 'turnstone', 'turnstones',
        'oystercatcher', 'oystercatchers', 'godwit', 'godwits', 'curlew', 'curlews',
        'dunlin', 'sanderling', 'redshank', 'greenshank', 'yellowlegs', 'dowitcher',
        'phalarope', 'avocet', 'stilt', 'killdeer',
        # Scientific genus names
        'calidris', 'charadrius', 'limosa', 'numenius', 'arenaria', 'haematopus',
        'tringa', 'actitis', 'gallinago', 'limnodromus', 'phalaropus', 'recurvirostra',
        'himantopus', 'pluvialis', 'vanellus'
    ]
    exclusion_keywords = [
        'duck', 'ducks', 'goose', 'geese', 'swan', 'swans', 'gull', 'gulls',
        'tern', 'terns', 'petrel', 'albatross', 'cormorant', 'pelican'
    ]
    
    text_to_search = f"{title.lower()} {abstract.lower()}"
    
    # check for exclusions first
    if any(keyword in text_to_search for keyword in exclusion_keywords):
        return False
    
    # then check for shorebird keywords
    return any(keyword in text_to_search for keyword in shorebird_keywords)

def find_shorebird_abstracts_from_parquet(shorebird_max: int = 100) -> List[Dict]:
    """Search parquet file for shorebird-relevant abstracts using keyword matching"""
    
    # Comprehensive shorebird keywords
    shorebird_keywords = [
        # Order and common names
        'charadriiformes', 'shorebird', 'shorebirds', 'wader', 'waders',
        
        # Common species
        'sandpiper', 'sandpipers', 'plover', 'plovers', 'turnstone', 'turnstones',
        'oystercatcher', 'oystercatchers', 'godwit', 'godwits', 'curlew', 'curlews',
        'dunlin', 'sanderling', 'redshank', 'greenshank', 'yellowlegs', 'dowitcher',
        'knot', 'stint', 'phalarope', 'avocet', 'stilt', 'killdeer',
        
        # Scientific genus names
        'calidris', 'charadrius', 'limosa', 'numenius', 'arenaria', 'haematopus',
        'tringa', 'actitis', 'gallinago', 'limnodromus', 'phalaropus', 'recurvirostra',
        'himantopus', 'pluvialis', 'vanellus'
    ]
    
    logger.info(f"Searching parquet for shorebird abstracts (target: {shorebird_max})")
    
    try:
        import polars as pl
        parquet_path = Path("/Users/kittsonhamill/Desktop/all_abstracts.parquet")
        
        if not parquet_path.exists():
            logger.warning("Parquet file not found - skipping shorebird search")
            return []
        
        found_shorebirds = []
        batch_size = 5000
        offset = 0
        
        while len(found_shorebirds) < shorebird_max:
            df_batch = pl.scan_parquet(parquet_path).slice(offset, batch_size).collect()
            if len(df_batch) == 0:
                break
                
            logger.info(f"Searching batch {offset}-{offset+len(df_batch)} for shorebird papers...")
            batch_data = df_batch.to_dicts()
            
            for row in batch_data:
                title = str(row.get('title', '')).lower()
                abstract = str(row.get('abstract', '')).lower()
                doi = row.get('doi', '')
                
                # check if any shorebird keywords appear in title or abstract
                text_to_search = f"{title} {abstract}"
                if any(keyword in text_to_search for keyword in shorebird_keywords):
                    found_shorebirds.append({
                        'title': row.get('title', ''),
                        'abstract': row.get('abstract', ''),
                        'doi': doi,
                        'source': 'parquet_shorebird_search'
                    })
                    
                    if len(found_shorebirds) >= shorebird_max:
                        break
            
            offset += len(df_batch)
            
            # Stop if we've searched enough, around 4000000 cap
            if offset > 1000000:
                logger.info(f"Searched {offset} papers, stopping search")
                break
        
        logger.info(f"Found {len(found_shorebirds)} shorebird-relevant abstracts from parquet")
        return found_shorebirds
        
    except Exception as e:
        logger.error(f"Error searching parquet: {e}")
        return []

def train_classifier_with_real_test_set(round_num=1):
    logger.info(f"------- TRAINING CLASSIFIER ROUND {round_num} --------")
    
    # create cache directory
    os.makedirs(".cache", exist_ok=True)
    pipeline_id = "shorebird_classifier"
    
    # try to load cached training data first
    training_cache_path = f".cache/{pipeline_id}_training_data.pkl"
    if os.path.exists(training_cache_path):
        logger.info("Loading training data from cache")
        with open(training_cache_path, "rb") as f:
            training_data = pickle.load(f)
    else:
        # load training data (synthetic)
        training_file = Path('data_to_review/synthetic_training_data.json')
        if not training_file.exists():
            logger.error(f"Training data not found at {training_file}")
            return
        
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        # cache the training data
        with open(training_cache_path, "wb") as f:
            pickle.dump(training_data, f)
        logger.info("Training data cached")
    
    logger.info(f"Loaded {len(training_data)} synthetic training examples")
    
    # check if we have targeted synthetic examples
    targeted_count = sum(1 for item in training_data if item.get('source') == 'targeted_synthetic_v2')
    if targeted_count > 0:
        logger.info(f"Found {targeted_count} targeted synthetic examples in training data")
    else:
        logger.info("No targeted synthetic examples found (this is round 1)")
    
    # prepare training data
    X_train_texts = [item['text'] for item in training_data]
    y_train = [item['label'] for item in training_data]
    
    # try to load cached embeddings
    train_embeddings_cache = f".cache/{pipeline_id}_train_embeddings.npy"
    
    if os.path.exists(train_embeddings_cache):
        logger.info("Loading training embeddings from cache")
        X_train_embeddings = np.load(train_embeddings_cache)
        logger.info("Training embeddings loaded from cache")
    else:
        # generate embeddings
        logger.info("Loading embedding model")
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        logger.info("Generating embeddings for training data")
        X_train_embeddings = embedding_model.encode(X_train_texts, show_progress_bar=True)
        
        # cache the embeddings
        np.save(train_embeddings_cache, X_train_embeddings)
        logger.info("Training embeddings cached")
    
    # train classifier
    logger.info("Training classifier")
    classifier = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='liblinear', random_state=42)
    classifier.fit(X_train_embeddings, y_train)
    
    # validate using corpus-level validation
    logger.info("Creating enhanced corpus sample with shorebird papers for validation")
    
    try:
        import polars as pl
        parquet_path = Path("/Users/kittsonhamill/Desktop/all_abstracts.parquet")
        if parquet_path.exists():
            # First, get shorebird papers from parquet (with caching)
            shorebird_cache_path = ".cache/shorebird_papers_corpus.pkl"
            if os.path.exists(shorebird_cache_path):
                logger.info("Loading shorebird papers from cache...")
                with open(shorebird_cache_path, "rb") as f:
                    shorebird_papers = pickle.load(f)
                logger.info(f"Loaded {len(shorebird_papers)} cached shorebird papers")
            else:
                logger.info("Finding shorebird papers to include in corpus validation...")
                shorebird_papers = find_shorebird_abstracts_from_parquet(shorebird_max=2500)
                # Cache the results
                with open(shorebird_cache_path, "wb") as f:
                    pickle.dump(shorebird_papers, f)
                logger.info(f"Cached {len(shorebird_papers)} shorebird papers for future use")
            
            # Then load regular sample from parquet
            remaining_needed = 5000 - len(shorebird_papers)
            logger.info(f"Loading {remaining_needed} additional papers from parquet...")
            df_sample = pl.read_parquet(parquet_path, n_rows=remaining_needed)
            df_sample = df_sample.drop_nulls(subset=["title", "abstract"])
            
            # Combine shorebird papers with regular sample
            corpus_texts = []
            corpus_titles = []
            
            # Add shorebird papers first
            for paper in shorebird_papers:
                corpus_texts.append(paper['abstract'])
                corpus_titles.append(paper['title'])
            
            # Add regular sample
            regular_texts = df_sample["abstract"].to_list()
            regular_titles = df_sample["title"].to_list()
            corpus_texts.extend(regular_texts)
            corpus_titles.extend(regular_titles)
            
            logger.info(f"Enhanced corpus sample created:")
            logger.info(f"  - Shorebird papers: {len(shorebird_papers)}")
            logger.info(f"  - Regular papers: {len(regular_texts)}")
            logger.info(f"  - Total corpus: {len(corpus_texts)}")
            logger.info(f"Scoring {len(corpus_texts)} real abstracts from enhanced corpus")
            
            # generate embeddings for corpus sample
            corpus_embeddings_cache = f".cache/{pipeline_id}_corpus_embeddings.npy"
            if os.path.exists(corpus_embeddings_cache):
                logger.info("Loading corpus embeddings from cache")
                corpus_embeddings = np.load(corpus_embeddings_cache)
            else:
                embedding_model = SentenceTransformer('all-mpnet-base-v2')
                corpus_embeddings = embedding_model.encode(corpus_texts, show_progress_bar=True)
                np.save(corpus_embeddings_cache, corpus_embeddings)
                logger.info("Corpus embeddings cached")
            
            # score all papers
            all_scores = classifier.predict_proba(corpus_embeddings)[:, 1]
            
            # find top-scoring papers
            top_idxs = np.argsort(all_scores)[::-1][:50]
            top_scores = all_scores[top_idxs]
            
            logger.info("Top scoring papers from real corpus:")
            logger.info("")
            shorebird_count = 0
            for i, (idx, score) in enumerate(zip(top_idxs, top_scores)):
                title = corpus_titles[idx]
                abstract = corpus_texts[idx]
                
                # try keyword matching first (much faster than LLM)
                is_shorebird_keywords = has_shorebird_keywords_fallback(title, abstract)
                
                if is_shorebird_keywords:
                    # found shorebird keywords
                    is_shorebird = True
                    marker = "shorebird"
                else:
                    # no obvious keywords found, so ask the LLM to be sure
                    is_shorebird = is_shorebird_paper_llm(title, abstract)
                    marker = "shorebird" if is_shorebird else "other"
                
                if is_shorebird:
                    shorebird_count += 1
                
                logger.info(f"{i+1:2d}. [{marker}] Score: {score:.3f} | {title}")
            
            logger.info(f"Corpus validation summary:")
            logger.info(f"Top 50 papers: {shorebird_count} likely shorebird papers ({shorebird_count/50*100:.1f}%)")
            logger.info(f"Average score of top 10: {np.mean(top_scores[:10]):.3f}")
            logger.info(f"Average score of top 50: {np.mean(top_scores):.3f}")
            
            # identify problems for next training round
            logger.info("--------Identifying training targets-------")
            
            # find uncertain cases (scores around 0.5) - get 20% closest to 0.5
            uncertain_mask = (all_scores > 0.3) & (all_scores < 0.7)
            uncertain_idxs = np.where(uncertain_mask)[0]
            
            # Get the 20% of uncertain cases closest to 0.5
            if len(uncertain_idxs) > 0:
                uncertain_distances = np.abs(all_scores[uncertain_idxs] - 0.5)
                num_to_keep = max(1, int(len(uncertain_idxs) * 0.2))  # 20% but at least 1
                closest_idxs = np.argsort(uncertain_distances)[:num_to_keep]
                uncertain_idxs = uncertain_idxs[closest_idxs]
            
            logger.info(f"Found {len(uncertain_idxs)} uncertain papers closest to 0.5 (from {np.sum(uncertain_mask)} in range 0.3-0.7)")
            logger.info(f"All papers >0.70 {np.sum(all_scores>=0.7)}") #89, 108, 131, 141
            logger.info(f"All papers 0.30<x<0.70 {np.sum(uncertain_mask)}")#598, 581, 584, 590
            logger.info(f"All papers <0.30 {np.sum(all_scores<=0.3)}")#4313, 4311, 4285

            
            # find high-scoring non-shorebirds (false positives)
            high_scoring_non_shorebirds = []
            corrected_labels = []
            
            for idx in top_idxs[:20]:  # Check top 20
                title = corpus_titles[idx]
                abstract = corpus_texts[idx]
                score = all_scores[idx]
                
                if score > 0.7:
                    # check if this high-scoring paper might be misclassified
                    is_shorebird_keywords = has_shorebird_keywords_fallback(title, abstract)
                    
                    if is_shorebird_keywords:
                        # has shorebird keywords, so it's probably a shorebird paper
                        corrected_labels.append((idx, score, title, 'shorebird'))
                    else:
                        # no keywords, but high score is suspicious so ask llm for clarity
                        is_shorebird_llm = is_shorebird_paper_llm(title, abstract)
                        
                        if is_shorebird_llm:
                            # LLM returned shorebird, meaning this was a keyword miss
                            corrected_labels.append((idx, score, title, 'shorebird'))
                        else:
                            # both methods agree not about shorebirds, genuine false positive
                            high_scoring_non_shorebirds.append((idx, score, title))
            
            if corrected_labels:
                logger.info(f"Found {len(corrected_labels)} high-scoring papers that are actually about shorebirds:")
                for i, (idx, score, title, new_label) in enumerate(corrected_labels[:3]):
                    logger.info(f"  {i+1}. Score: {score:.3f} | {title}")
            
            # Show any papers that scored high but aren't about shorebirds
            if high_scoring_non_shorebirds:
                logger.info(f"Found {len(high_scoring_non_shorebirds)} papers scoring high but not about shorebirds (false positives):")
                for i, (idx, score, title) in enumerate(high_scoring_non_shorebirds[:5]):
                    logger.info(f"{i+1}. Score: {score:.3f} | {title}")
                    logger.info(f"Abstract: {corpus_texts[idx][:200]}...")
            else:
                logger.info("No false positives found above 0.7 threshold")
            
            # generate synthetic negatives if we have ANY problem cases (false positives OR uncertain cases)
            if high_scoring_non_shorebirds:
                
                # save these for synthetic data generation
                problems_cache = f".cache/{pipeline_id}_problem_cases.pkl"
                problem_data = {
                    'uncertain_indices': uncertain_idxs[:50].tolist(),  # Sample of uncertain cases
                    'false_positives': [(corpus_texts[idx], title) for idx, score, title in high_scoring_non_shorebirds],
                    'uncertain_texts': [(corpus_texts[idx], corpus_titles[idx]) for idx in uncertain_idxs[:20]]
                }
                with open(problems_cache, 'wb') as f:
                    pickle.dump(problem_data, f)
                logger.info(f"Saved problem cases to {problems_cache} for next training round")
                
                # generate targeted synthetic examples based on problem cases
                logger.info("Generating targeted synthetic negatives")
                targeted_negatives, positive_synthetics = generate_targeted_synthetic_negatives(problem_data, num_negatives=20)
                
                if targeted_negatives or positive_synthetics:
                    # save targeted negatives
                    if targeted_negatives:
                        targeted_cache = f".cache/{pipeline_id}_targeted_negatives.json"
                        with open(targeted_cache, 'w') as f:
                            json.dump(targeted_negatives, f, indent=2)
                        logger.info(f"Saved {len(targeted_negatives)} targeted negatives to {targeted_cache}")
                    
                    # save positive synthetics  
                    if positive_synthetics:
                        positives_cache = f".cache/{pipeline_id}_positive_synthetics.json"
                        with open(positives_cache, 'w') as f:
                            json.dump(positive_synthetics, f, indent=2)
                        logger.info(f"Saved {len(positive_synthetics)} positive synthetics to {positives_cache}")
                    
                    # automatically integrate and start next round
                    logger.info("Integrating targeted synthetics (both positive and negative)")
                    next_result = integrate_and_retrain_balanced(targeted_negatives, positive_synthetics, training_data, pipeline_id, round_num)
                    logger.info(f"Next result: {next_result}")
                    return next_result
                else:
                    logger.info("No targeted negatives generated")
            else:
                logger.info("🎉 MODEL HAS CONVERGED! 🎉")
                logger.info("No false positives >0.7 and no uncertain cases found")
                logger.info("Training complete - model is performing well!")
        
        else:
            logger.warning("Parquet file not found")
        
    except Exception as e:
        logger.error(f"Error in corpus validation: {e}")
        logger.info("Continuing with traditional validation only")
    
    # save model
    models_dir = Path("trained_relevance_models") / "real_test_classifier"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / "embedding_classifier.pkl", 'wb') as f:
        pickle.dump(classifier, f)
    
    model_info = {
        'model_name': 'real_test_classifier',
        'training_examples': len(training_data),
        'training_method': 'corpus_validated_synthetic_train',
        'validation_method': 'enhanced_corpus_with_shorebird_papers'
    }
    with open(models_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    logger.info(f"Saved classifier and metrics to: {models_dir}")
    
    return True  # indicate successful completion

if __name__ == "__main__":
    train_classifier_with_real_test_set() 