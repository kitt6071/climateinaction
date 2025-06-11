import json
import time
import polars as pl
import pickle
from pathlib import Path
import os
from dotenv import load_dotenv
import requests
import argparse
from openai import OpenAI
import logging
import hashlib
from typing import Optional, Union, List, Dict
import shutil
import difflib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("trainer")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

def setup_trainer_logging(log_file_path: Path):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)
    logger.propagate = False
    logger.info(f"Logging to {log_file_path}")

def get_dynamic_run_base_path(model_name: str, max_r_val: Optional[Union[int, str]], current_script_dir: Path, base_folder_name: str = "runs") -> Path:
    model_name_sanitized = model_name.replace("/", "_").replace(":", "_").replace("", "-")
    max_r_str = str(max_r_val) if max_r_val is not None and str(max_r_val).lower() != "all" else "all"
    
    run_folder_name = f"{model_name_sanitized}_{max_r_str}"
    return current_script_dir / base_folder_name / run_folder_name

class RateLimiter:
    def __init__(self, rpm: int = 2, is_ollama: bool = False):
        self.rpm = rpm
        self.last_call = 0
        self.interval = 60.0 / self.rpm
        self.backoff_time = 0
        self.is_ollama = is_ollama
        self.min_wait = 0

    def wait(self):
        if self.is_ollama:
            return
        now = time.time()
        elapsed = now - self.last_call
        wait_time = max(self.interval - elapsed, self.min_wait)
        if self.backoff_time > 0:
            wait_time = max(wait_time, self.backoff_time)
            self.backoff_time *= 0.5
        if wait_time > 0:
            logger.info(f"Waiting {wait_time:.1f} seconds before next request")
            time.sleep(wait_time)
            self.last_call = time.time()

    def handle_rate_limit(self):
        self.backoff_time = max(60, self.backoff_time * 2)
        self.wait()

class Cache:
    def __init__(self, cache_dir: str = "cache"):
        current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.cache_dir = current_script_dir / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _make_hash_key(self, *args: str) -> str:
        combined = ":".join(str(arg) for arg in args)
        encoded = combined.encode('utf-8', errors='replace')
        return hashlib.md5(encoded).hexdigest()
        
    def get(self, key_parts: Union[str, List[str]]):
        if not isinstance(key_parts, list):
            key_parts = [key_parts]
        cache_key = self._make_hash_key(*key_parts)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {e}")
                return None
        return None
        
    def set(self, key_parts: Union[str, List[str]], result):
        if not isinstance(key_parts, list):
            key_parts = [key_parts]
        cache_key = self._make_hash_key(*key_parts)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.error(f"Error writing to cache file {cache_file}: {e}")

def setup_llm(model_name: Optional[str] = None):
    load_dotenv()
    effective_default_model = "google/gemini-2.0-flash-001"
    
    labeling_specific_model = model_name or os.getenv('OPENROUTER_LABELING_MODEL', effective_default_model)
    
    logger.info(f"Using LLM: '{labeling_specific_model}'")

    return {
        'cache': Cache(cache_dir="training_cache"), 
        'model': labeling_specific_model,
        'api_rate_limiter': RateLimiter(rpm=30, is_ollama=False),
        'use_openrouter': True 
    }

def strip_markdown_json(response_text: str) -> str:
    if response_text is None:
        return ""
    stripped_text = response_text.strip()
    if stripped_text.startswith("```json") and stripped_text.endswith("```"):
        stripped_text = stripped_text[7:-3].strip()
    elif stripped_text.startswith("```") and stripped_text.endswith("```"):
        stripped_text = stripped_text[3:-3].strip()
    return stripped_text

def llm_generate(prompt: str, system: str, model: str, temperature: float = 0.1, 
                       timeout: int = 120, format_schema=None, llm_setup=None) -> str:
    raw_response_content = ""
    try:
        if llm_setup and llm_setup.get('use_openrouter', False):
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                logger.error("OPENROUTER_API_KEY not found in environment variables for llm_generate")
                raise ValueError("OPENROUTER_API_KEY not found for llm_generate")

            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            
            if llm_setup.get('api_rate_limiter'):
                llm_setup['api_rate_limiter'].wait()

            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            
            request_params = {
                "model": model, 
                "messages": messages, 
                "temperature": temperature, 
                "timeout": timeout
            }
            
            if format_schema and isinstance(format_schema, dict):
                request_params["response_format"] = {
                    "type": "json_schema", 
                    "json_schema": {
                        "name": "response_schema",
                        "strict": True,
                        "schema": format_schema
                    }
                }
            elif format_schema == "json": 
                request_params["response_format"] = {"type": "json_object"}

            logger.debug(f"OpenRouter Request Params (sync): {json.dumps({k: v for k, v in request_params.items() if k != 'api_key'}, indent=2)}")
            response_obj = client.chat.completions.create(**request_params)
            raw_response_content = response_obj.choices[0].message.content
        
        else:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
            payload = {"model": model, "prompt": prompt, "system": system, "stream": False, "options": {"temperature": temperature}}
            if format_schema == "json" or isinstance(format_schema, dict): payload["format"] = "json"
            
            if llm_setup and llm_setup.get('api_rate_limiter') and llm_setup['api_rate_limiter'].is_ollama:
                 llm_setup['api_rate_limiter'].wait()

            try:
                response = requests.post(ollama_url, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                raw_response_content = result.get("response", "")
            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama API error for model {model}: {e}", exc_info=True)
                raw_response_content = ""
            except json.JSONDecodeError as e_json:
                logger.error(f"Ollama API JSON decode error for model {model}: {e_json}. Response text: {response.text if 'response' in locals() else 'N/A'}", exc_info=True)
                raw_response_content = ""

    except Exception as e:
        logger.error(f"Outer error in llm_generate for model {model}: {e}", exc_info=True)
        raw_response_content = ""
    return strip_markdown_json(raw_response_content)

def load_data_with_offset(file_name, skip_rows=0, max_rows=1000):
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    possible_paths = [
        Path("/app") / file_name,
        Path.home() / "Desktop" / file_name,
        current_dir / file_name,
        current_dir.parent / file_name,
    ]

    file_path = None
    for path in possible_paths:
        if path.exists():
            file_path = path
            break
            
    if file_path is None:
        raise FileNotFoundError(f"Could not find {file_name} in any of the expected locations: {possible_paths}")
    
    logger.info(f"Loading data from: {file_path} (Skipping {skip_rows} rows, loading {max_rows} rows)")
    
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        
        if skip_rows >= total_rows:
            return pl.DataFrame()
        
        batches = parquet_file.iter_batches(batch_size=1024)
        rows_needed = max_rows
        rows_skipped = 0
        arrow_batches = []
        
        for batch in batches:
            batch_len = batch.num_rows
            if rows_skipped + batch_len <= skip_rows:
                rows_skipped += batch_len
                continue
            
            start_in_batch = max(0, skip_rows - rows_skipped)
            available = batch_len - start_in_batch
            to_take = min(available, rows_needed)
            sliced_batch = batch.slice(start_in_batch, to_take)
            arrow_batches.append(sliced_batch)
            rows_needed -= to_take
            rows_skipped += batch_len
            
            if rows_needed <= 0:
                break
        
        if arrow_batches:
            table = pa.Table.from_batches(arrow_batches)
            df = pl.from_arrow(table)
        else:
            df = pl.DataFrame()
            
    except Exception as e:
        logger.error(f"Error with PyArrow during data loading: {e}. Falling back to basic Polars read")
        df = pl.read_parquet(file_path)
        if skip_rows >= len(df):
            return pl.DataFrame()
        end_idx = min(skip_rows + max_rows, len(df))
        df = df[skip_rows:end_idx]
    
    df = df.drop_nulls(["title", "abstract", "doi"])
    logger.info(f"Loaded {len(df)} rows after dropping nulls from {file_name}")
    return df

def convert_to_summary(abstract: str, llm_setup) -> str:
    summary_cache_key = [f"summary_v1:{hashlib.md5(abstract.encode('utf-8')).hexdigest()}"]
    cached = llm_setup["cache"].get(summary_cache_key)
    if cached:
        logger.debug("Summary cache hit")
        return cached

    system_prompt = """
    You are a scientific knowledge summarizer. Convert the following text into a structured summary that:
    1. Focuses on species-specific impacts and threats
    2. Clearly states causal mechanisms and relationships
    3. Includes quantitative data when available
    4. Emphasizes HOW impacts occur, not just WHAT happened
    5. Use scientific names (Latin binomial) when mentioned in the abstract
    6. If a group of species is mentioned, look for any specific examples in the abstract
    7. If no specific species are named, use the most specific taxonomic group mentioned
    8. Never use vague terms like "birds", "larger species", or "# bird species"
    9. Do not include phrases like "spp" or number of species
    10. Each species or taxonomic group should not be a phrase
    Summarize this scientific abstract focusing on specific species and their threats. 
    Format the summary with clear sections:
    - Species Affected
    - Threat Mechanisms
    - Quantitative Findings
    - Causal Relationships
    """
    try:
        full_prompt = f"Text to summarize:\n{abstract}\n\nStructured Summary:"
        summary = llm_generate(
            prompt=full_prompt,
            system=system_prompt,
            model=llm_setup["model"],
            temperature=0.1,
            timeout=120,
            llm_setup=llm_setup
        )
        logger.debug(f"Generated Summary (first 100 chars): {summary[:100]}")
        if len(summary) < 50:
            logger.warning("Summary seems too short, might not be valid")
            return ""
        llm_setup["cache"].set(summary_cache_key, summary)
        return summary
    except Exception as e:
        logger.error(f"ERROR generating summary: {e}", exc_info=True)
        return ""

def get_relevance_label_for_abstract_summary(
    abstract_text: str, 
    title: str, 
    llm_setup_dict: dict
) -> bool:
    logger.debug(f"Getting relevance label for '{title[:50]}...'")

    summary = convert_to_summary(abstract_text, llm_setup_dict)
    if not summary:
        logger.warning(f"No summary for '{title[:50]}...'. Marking as not relevant")
        return False

    iucn_context_for_prompt = """
    General IUCN Threat Categories:
    1. Residential & commercial development
    2. Agriculture & aquaculture
    3. Energy production & mining
    4. Transportation & service corridors
    5. Biological resource use (logging, fishing, hunting)
    6. Human intrusions & disturbance
    7. Natural system modifications (dams, fire suppression)
    8. Invasive species, diseases
    9. Pollution
    10. Geological events
    11. Climate change & severe weather
    """

    system_prompt = (
        """You are an expert scientific analyst. You will be given a summary of a scientific abstract. 
        Your task is to determine if this summary indicates that the original abstract likely contains specific information about an ENTIRE species being negatively impacted by threats 
        that would fall under the general scope of international conservation threat categories (like IUCN's). Use the IUCN context provided to determine if a
        species is being negatively impacted by a threat that would fall under one of the categories, and is therefore relevant.

        General IUCN Threat Categories:
    1. Residential & commercial development
    2. Agriculture & aquaculture
    3. Energy production & mining
    4. Transportation & service corridors
    5. Biological resource use (logging, fishing, hunting)
    6. Human intrusions & disturbance
    7. Natural system modifications (dams, fire suppression)
    8. Invasive species, diseases
    9. Pollution
    10. Geological events
    11. Climate change & severe weather

        An abstract is RELEVANT if it:
        1. Mentions specific species or taxonomic groups (e.g., "Adelie penguins", "coral reefs")
        2. Describes specific threats or stressors (e.g., "climate change", "habitat loss") relating to one of the IUCN categories
        3. Explains how these threats affect the species (mechanisms, impacts)

        An abstract is NOT RELEVANT if it:
        1. Only discusses methodology without species impacts
        2. Focuses on conservation solutions without describing threats
        3. Is about species distribution without mentioning threats
        4. Discusses only human impacts without specific species effects

        Abstracts detailing how a specific threat (e.g., 'oil spills', 'habitat loss due to deforestation', 'invasive predator X') negatively affects a specific species (e.g., 'Pygoscelis adeliae', 'forest elephants') ARE relevant. 
        Respond with a JSON object: {\"is_relevant\": true/false, \"reasoning\": \"Your brief reasoning focusing on the presence/absence of specific species-threat links in the summary.\"}"""
    )
    
    user_prompt = f"""Abstract Summary:
{summary}

General IUCN Threat Categories:
{iucn_context_for_prompt}

Based on the summary, does the original abstract seem RELEVANT to species-threat information (specific species, specific negative impacts/threats)?
"""
    
    relevance_schema = {
        "type": "object", 
        "properties": {"is_relevant": {"type": "boolean"}, "reasoning": {"type": "string"}},
        "required": ["is_relevant"]
    }

    label_cache_key_parts = [f"relevance_label_v2:{title[:50]}:{hashlib.md5(summary.encode('utf-8')).hexdigest()}"]
    cached_label = llm_setup_dict["cache"].get(label_cache_key_parts)
    if cached_label is not None:
        logger.debug(f"Relevance label cache hit for '{title[:50]}...': {cached_label}")
        return cached_label

    response_str = llm_generate(
        prompt=user_prompt,
        system=system_prompt,
        model=llm_setup_dict["model"], 
        temperature=0.0,
        format_schema=relevance_schema,
        llm_setup=llm_setup_dict
    )

    try:
        if not response_str:
            logger.warning(f"LLM call for relevance label returned empty for '{title[:50]}...'. Defaulting to False")
            llm_setup_dict["cache"].set(label_cache_key_parts, False)
            return False
        result_json = json.loads(response_str)
        is_relevant = result_json.get("is_relevant", False)
        reasoning = result_json.get("reasoning", "No reasoning provided")
        logger.info(f"Relevance for '{title[:50]}...': {is_relevant}. Reasoning: {reasoning}")
        llm_setup_dict["cache"].set(label_cache_key_parts, is_relevant)
        return is_relevant
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode relevance label JSON for '{title[:50]}...': {e}. Response: '{response_str}'. Defaulting to False")
        llm_setup_dict["cache"].set(label_cache_key_parts, False)
        return False
    except Exception as e_final:
        logger.error(f"Error in relevance label determination for '{title[:50]}...': {e_final}", exc_info=True)
        llm_setup_dict["cache"].set(label_cache_key_parts, False)
        return False

def setup_embedding_classifier(models_dir: Path):
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Cannot setup embedding classifier: sentence-transformers not installed")
        return None, None
        
    try:
        model_name = "all-mpnet-base-v2"
        embedding_model = SentenceTransformer(model_name)
        logger.info(f"Embedding model '{model_name}' loaded")
        
        classifier_path = models_dir / "embedding_classifier.pkl"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        classifier = None
        if classifier_path.exists():
            try:
                with open(classifier_path, 'rb') as f:
                    classifier = pickle.load(f)
                logger.info(f"Loaded existing embedding-based classifier from {classifier_path}")
            except Exception as e:
                logger.error(f"Error loading embedding classifier from {classifier_path}: {e}. Will attempt to train a new one")
        else:
            logger.info(f"No pre-trained embedding classifier found at {classifier_path}. Will attempt to train a new one if data is provided")
        
        return embedding_model, classifier
    except Exception as e:
        logger.error(f"Error setting up embedding model: {e}", exc_info=True)
        return None, None

def train_embedding_classifier(training_data: List[Dict[str, Union[str, bool]]], 
                               embedding_model: SentenceTransformer, 
                               models_dir: Path) -> Optional[LogisticRegression]:
    if not EMBEDDINGS_AVAILABLE or embedding_model is None:
        logger.error("Cannot train embedding classifier: Model not available or sentence-transformers not installed")
        return None
    
    if not training_data:
        logger.warning("No training data provided for embedding classifier")
        return None
        
    try:
        logger.info(f"Training embedding classifier with {len(training_data)} examples")
        texts = [item['text'] for item in training_data]
        labels = [1 if item['label'] else 0 for item in training_data]
        
        logger.info("Generating embeddings for training data")
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        
        classifier = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
        classifier.fit(embeddings, labels)
        
        classifier_path = models_dir / "embedding_classifier.pkl"
        models_dir.mkdir(parents=True, exist_ok=True)

        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier, f)
        
        logger.info(f"Trained and saved embedding-based classifier to {classifier_path}")
        return classifier
    except Exception as e:
        logger.error(f"Error training embedding classifier: {e}", exc_info=True)
        return None

DEFAULT_TARGET_SAMPLES_PER_CLASS = 50
DEFAULT_MAX_TOTAL_ABSTRACTS_TO_SCAN = 10000
DEFAULT_MODEL_FOR_RELEVANCE_LABELS = "google/gemini-2.0-flash-001"
CENTRAL_MODELS_BASE_DIR_NAME = "trained_relevance_models_central"
ABSTRACT_FILE_NAME = "all_abstracts.parquet"
FILE_BATCH_SIZE = 500

def collect_and_train_classifier(args):
    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_script_dir.parent
    
    log_file = project_root / "logs" / "classifier_training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_trainer_logging(log_file)

    logger.info("--- Starting Relevance Classifier Training Script ---")

    target_samples = args.target_samples if args.target_samples else DEFAULT_TARGET_SAMPLES_PER_CLASS
    max_scan = args.max_scan if args.max_scan else DEFAULT_MAX_TOTAL_ABSTRACTS_TO_SCAN
    
    labeling_model_name = args.labeling_model if args.labeling_model else DEFAULT_MODEL_FOR_RELEVANCE_LABELS
    logger.info(f"Using LLM '{labeling_model_name}' for initial relevance labeling")

    sanitized_labeling_model_name = labeling_model_name.replace("/", "_").replace(":", "_").replace("", "-")
    
    central_model_specific_dir = project_root / CENTRAL_MODELS_BASE_DIR_NAME / sanitized_labeling_model_name
    central_model_specific_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Trained classifier will be saved centrally to: {central_model_specific_dir}")

    llm_s = setup_llm(model_name=labeling_model_name)
    training_run_cache_dir = central_model_specific_dir / "run_cache"
    llm_s['cache'] = Cache(cache_dir=str(training_run_cache_dir)) 

    training_data = []
    relevant_collected = 0
    irrelevant_collected = 0
    current_file_skip_rows = 0
    total_abstracts_scanned = 0

    logger.info(f"Goal: Collect {target_samples} relevant and {target_samples} non-relevant abstracts for training")
    logger.info(f"Max abstracts to scan from Parquet: {max_scan}")

    while (relevant_collected < target_samples or irrelevant_collected < target_samples) and total_abstracts_scanned < max_scan:
        logger.info(f"Loading abstract batch: Skip={current_file_skip_rows}, Size={FILE_BATCH_SIZE}")
        df_batch = load_data_with_offset(ABSTRACT_FILE_NAME, current_file_skip_rows, FILE_BATCH_SIZE)

        if len(df_batch) == 0:
            logger.info("No more data in abstract file")
            break
        
        current_file_skip_rows += len(df_batch)
        
        for i, row in enumerate(df_batch.iter_rows(named=True)):
            total_abstracts_scanned += 1
            if total_abstracts_scanned > max_scan:
                logger.info(f"Reached max scan limit of {max_scan} abstracts")
                break

            title = row["title"]
            abstract = row["abstract"]
            is_relevant = get_relevance_label_for_abstract_summary(abstract, title, llm_s)
            
            detail = {'text': abstract, 'title': title}
            if is_relevant and relevant_collected < target_samples:
                training_data.append({'text': detail['text'], 'label': True, 'title': detail['title']})
                relevant_collected += 1
                logger.info(f"Collected RELEVANT example #{relevant_collected}/{target_samples}: '{detail['title'][:50]}...'")
            elif not is_relevant and irrelevant_collected < target_samples:
                training_data.append({'text': detail['text'], 'label': False, 'title': detail['title']})
                irrelevant_collected += 1
                logger.info(f"Collected IRRELEVANT example #{irrelevant_collected}/{target_samples}: '{detail['title'][:50]}...'")
            
            if relevant_collected >= target_samples and irrelevant_collected >= target_samples:
                logger.info("Target samples collected for both classes")
                break
        
        if (relevant_collected >= target_samples and irrelevant_collected >= target_samples) or \
           total_abstracts_scanned >= max_scan or \
           len(df_batch) == 0: 
            break

    logger.info("--- Data Collection Phase Complete ---")
    logger.info(f"Total abstracts scanned: {total_abstracts_scanned}")
    logger.info(f"Collected {relevant_collected} relevant and {irrelevant_collected} irrelevant examples")
    logger.info(f"Total training examples: {len(training_data)}")

    training_data_save_path = central_model_specific_dir / f"collected_training_data_{relevant_collected}R_{irrelevant_collected}I.json"
    try:
        with open(training_data_save_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2)
        logger.info(f"Saved collected training data to {training_data_save_path}")
    except Exception as e:
        logger.error(f"Error saving training data: {e}")

    if relevant_collected >= 10 and irrelevant_collected >= 10:
        logger.info("--- Starting Classifier Training Phase ---")
        
        texts_for_splitting = [d['text'] for d in training_data]
        labels_for_splitting = [1 if d['label'] else 0 for d in training_data]

        if len(set(labels_for_splitting)) < 2:
            logger.error("Not enough class diversity to perform train/test split. Need at least one sample from each class. Aborting training")
        else:
            logger.info(f"Splitting data: {1.0 - args.test_split_ratio:.0%} train, {args.test_split_ratio:.0%} test")
            X_train_texts, X_test_texts, y_train, y_test = train_test_split(
                texts_for_splitting, 
                labels_for_splitting, 
                test_size=args.test_split_ratio, 
                random_state=42, 
                stratify=labels_for_splitting
            )
            logger.info(f"Training set size: {len(X_train_texts)}, Test set size: {len(X_test_texts)}")

            embedding_model, _ = setup_embedding_classifier(central_model_specific_dir)
            if embedding_model:
                logger.info("Generating embeddings for training and test sets")
                X_train_embeddings = embedding_model.encode(X_train_texts, show_progress_bar=True)
                X_test_embeddings = embedding_model.encode(X_test_texts, show_progress_bar=True)

                logger.info(f"Training a '{args.classifier_type}' classifier")
                
                if args.classifier_type == "logistic":
                    current_training_split_data = []
                    for i in range(len(X_train_texts)):
                        current_training_split_data.append({'text': X_train_texts[i], 'label': bool(y_train[i])})
                    
                    trained_classifier = train_embedding_classifier(current_training_split_data, embedding_model, central_model_specific_dir)
                
                elif args.classifier_type == "knn":
                    trained_classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
                    trained_classifier.fit(X_train_embeddings, y_train)
                    knn_model_path = central_model_specific_dir / "knn_embedding_classifier.pkl"
                    try:
                        with open(knn_model_path, 'wb') as f:
                            pickle.dump(trained_classifier, f)
                        logger.info(f"k-NN classifier saved to {knn_model_path}")
                    except Exception as e_save:
                        logger.error(f"Error saving k-NN classifier: {e_save}")
                        trained_classifier = None
                else:
                    logger.error(f"Unsupported classifier type: {args.classifier_type}. Aborting training")
                    trained_classifier = None

                if trained_classifier:
                    model_save_filename = "embedding_classifier.pkl" if args.classifier_type == "logistic" else "knn_embedding_classifier.pkl"
                    logger.info(f"SUCCESS: '{args.classifier_type}' classifier trained and saved to {central_model_specific_dir / model_save_filename}")
                    
                    logger.info("--- Evaluating Classifier on Test Set ---")
                    y_pred = trained_classifier.predict(X_test_embeddings)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, target_names=['Non-Relevant', 'Relevant'])
                    
                    logger.info(f"Test Set Accuracy: {accuracy:.4f}")
                    logger.info(f"Test Set Classification Report:\n{report}")

                    metrics_file_path = central_model_specific_dir / "evaluation_metrics.txt"
                    try:
                        with open(metrics_file_path, 'w') as f:
                            f.write(f"Test Set Accuracy: {accuracy:.4f}\n\n")
                            f.write("Test Set Classification Report:\n")
                            f.write(report)
                        logger.info(f"Saved evaluation metrics to {metrics_file_path}")
                    except Exception as e_metrics_save:
                        logger.error(f"Error saving evaluation metrics: {e_metrics_save}")

                    logger.info("Attempting to auto-copy trained classifier and metrics to relevant run directories")
                    runs_dir = project_root / "Lent_Init" / "runs"
                    copied_to_count = 0
                    if runs_dir.is_dir():
                        all_run_folder_paths = [d for d in runs_dir.iterdir() if d.is_dir()]
                        matched_run_folders = []

                        for run_folder_path in all_run_folder_paths:
                            run_folder_name = run_folder_path.name
                            model_part_to_compare = run_folder_name

                            if run_folder_name.endswith("_all"):
                                model_part_to_compare = run_folder_name[:-4]
                            elif '_' in run_folder_name:
                                parts = run_folder_name.rsplit('_', 1)
                                if parts[1].isdigit():
                                    model_part_to_compare = parts[0]
                            
                            if difflib.get_close_matches(sanitized_labeling_model_name, [model_part_to_compare], n=1, cutoff=0.85):
                                matched_run_folders.append(run_folder_path)
                                logger.info(f"Matched run folder '{run_folder_name}' for model '{sanitized_labeling_model_name}'")
                            else:
                                logger.debug(f"No close match for run folder '{run_folder_name}' with target model '{sanitized_labeling_model_name}'")

                        if not matched_run_folders:
                             logger.warning(f"No run folders in '{runs_dir}' matched model name '{sanitized_labeling_model_name}' for auto-copying")

                        for run_folder in matched_run_folders:
                            target_model_dir_in_run = run_folder / "models"
                            target_model_dir_in_run.mkdir(parents=True, exist_ok=True)
                            
                            source_classifier_filename = "embedding_classifier.pkl" if args.classifier_type == "logistic" else "knn_embedding_classifier.pkl"
                            
                            source_classifier_file = central_model_specific_dir / source_classifier_filename
                            dest_classifier_file = target_model_dir_in_run / source_classifier_filename
                            
                            if source_classifier_file.exists():
                                try:
                                    shutil.copy2(source_classifier_file, dest_classifier_file)
                                    logger.info(f"  Copied {source_classifier_filename} to {dest_classifier_file}")
                                    copied_to_count += 1

                                    generic_dest_file = target_model_dir_in_run / "embedding_classifier.pkl"
                                    if dest_classifier_file != generic_dest_file:
                                        shutil.copy2(source_classifier_file, generic_dest_file)
                                        logger.info(f"  Also copied as generic embedding_classifier.pkl to {generic_dest_file}")
                                    
                                    source_metrics_file = central_model_specific_dir / "evaluation_metrics.txt"
                                    if source_metrics_file.exists():
                                        dest_metrics_file = target_model_dir_in_run / "evaluation_metrics.txt"
                                        shutil.copy2(source_metrics_file, dest_metrics_file)
                                        logger.info(f"  Copied evaluation_metrics.txt to {dest_metrics_file}")
                                    else:
                                        logger.warning(f"  Evaluation metrics file not found for copying")

                                except Exception as e_copy:
                                    logger.error(f"  Failed to copy classifier or metrics to {target_model_dir_in_run}: {e_copy}")
                            else:
                                logger.warning(f"Source classifier file {source_classifier_file} not found for copying")
                        logger.info(f"Auto-copy complete. Relevant files copied to {copied_to_count} matched run directories")
                    else:
                        logger.warning(f"Skipping auto-copy: Runs directory not found at {runs_dir}")
                else:
                    logger.error("FAILURE: Classifier training failed or was not produced")
            else:
                logger.error("FAILURE: Could not set up embedding model. Aborting training and evaluation")
    else:
        logger.warning("Not enough data collected for one or both classes. Skipping classifier training")
    
    logger.info("--- Relevance Classifier Training Script Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data and train a relevance classifier for abstracts")
    parser.add_argument("--target_samples", type=int, help=f"Target number of samples per class (default: {DEFAULT_TARGET_SAMPLES_PER_CLASS})")
    parser.add_argument("--max_scan", type=int, help=f"Maximum total abstracts to scan from Parquet (default: {DEFAULT_MAX_TOTAL_ABSTRACTS_TO_SCAN})")
    parser.add_argument("--labeling_model", type=str, help=f"LLM model name to use for initial labeling (default: {DEFAULT_MODEL_FOR_RELEVANCE_LABELS})")
    parser.add_argument("--test_split_ratio", type=float, default=0.2, help="Ratio of data to use for the test set (default: 0.2)")
    parser.add_argument("--classifier_type", type=str, default="logistic", choices=["logistic", "knn"], help="Type of classifier to train: 'logistic' or 'knn' (default: logistic)")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parsed_args = parser.parse_args()

    collect_and_train_classifier(parsed_args)
