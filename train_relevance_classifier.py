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
import asyncio
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
    effective_default_model = "google/gemini-2.5-pro"
    
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


async def check_for_primary_evidence_training(abstract: str, llm_setup: dict) -> dict:
    """
    Identical to the pipeline's check_for_primary_evidence function
    """
    import json
    
    gate_schema = {
        "type": "object",
        "properties": {
            "is_primary_finding": {
                "type": "boolean",
                "description": "True only if ALL criteria are met: field study on wild populations, specific quantified result reported, biological subject focus"
            },
            "strongest_evidence_sentence": {
                "type": "string",
                "description": "The single sentence from the abstract that best supports a 'true' decision. Empty string if false."
            }
        },
        "required": ["is_primary_finding", "strongest_evidence_sentence"]
    }

    system_prompt = f"""You are a precision scientific screener. Your task is to determine if a provided abstract reports primary, field-based research findings on a biological population.

Carefully evaluate the abstract against the following criteria. To return `true`, **ALL** three conditions must be clearly met:

1. **Study Type**: The findings are from a **field study** on **wild populations**.
   * `FAIL` if the study is a review, meta-analysis, theoretical model, lab experiment, or on captive/domesticated populations.

2. **Result Presence**: The abstract reports a **specific, quantified result** from the authors' own work.
   * Look for conclusions like "we found," "results showed," or statements with numbers, percentages, or clear comparative terms (e.g., "higher," "decreased," "significant").
   * `FAIL` if the text only states aims, hypotheses, or methods without reporting an outcome.

3. **Subject Focus**: The result is about a **biological species** or taxonomic group.
   * `FAIL` if the findings are purely environmental (e.g., measuring water temperature with no mention of an organism).

Based on this evaluation, return a single, valid JSON object only. If the abstract fails any check, `is_primary_finding` must be `false`.

Schema:
{json.dumps(gate_schema, indent=2)}"""

    user_prompt = f"Abstract:\n\"\"\"\n{abstract}\n\"\"\"\n"

    cache_key_parts = [f"primary_evidence_v2:{hashlib.md5(abstract.encode('utf-8')).hexdigest()}"]
    cached_result = llm_setup["cache"].get(cache_key_parts)
    if cached_result is not None:
        logger.debug(f"Primary evidence cache hit for abstract hash.")
        return cached_result

    try:
        response = llm_generate(
            prompt=user_prompt,
            system=system_prompt,
            model=llm_setup.get("model", "google/gemini-2.5-pro"),
            temperature=0.0,
            format_schema=gate_schema,
            llm_setup=llm_setup
        )
        
        if response:
            try:
                evidence_data = json.loads(response)
                llm_setup["cache"].set(cache_key_parts, evidence_data)
                return evidence_data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse evidence gate JSON response: {response[:100]}")
                result = {"is_primary_finding": False, "strongest_evidence_sentence": ""}
                llm_setup["cache"].set(cache_key_parts, result)
                return result
        else:
            result = {"is_primary_finding": False, "strongest_evidence_sentence": ""}
            llm_setup["cache"].set(cache_key_parts, result)
            return result
            
    except Exception as e:
        logger.error(f"Error in evidence gate: {e}")
        result = {"is_primary_finding": False, "strongest_evidence_sentence": ""}
        llm_setup["cache"].set(cache_key_parts, result)
        return result

async def check_for_impact_conservation_evidence_training(abstract: str, llm_setup: dict) -> dict:
    import json
    
    gate_schema = {
        "type": "object",
        "properties": {
            "impact_or_conservation_found": {
                "type": "boolean",
                "description": "True only if ALL criteria are met: measured outcome, biological subject, negative impact OR conservation outcome, field context"
            },
            "strongest_impact_sentence": {
                "type": "string", 
                "description": "The single sentence from the abstract that best supports a 'true' decision. Leave as an empty string if false."
            }
        },
        "required": ["impact_or_conservation_found", "strongest_impact_sentence"]
    }

    system_prompt = f"""You are a precision data screener for conservation science. Your task is to determine if an abstract reports a concrete, measured impact on a species or a measured conservation outcome from a field study.

---
### Decision Rule

To return `true`, **ALL** of the following four conditions must be clearly satisfied:

1. **Measured Outcome**: The abstract reports a measured or observed outcome using numbers, percentages, or clear comparative terms (e.g., "significant," "increased," "decreased").
2. **Biological Subject**: The outcome is directly tied to a species, a taxonomic group, or an ecosystem condition explicitly affecting a species.
3. **Finding Type**: The outcome is either:
   * A **negative impact** on the species (affecting survival, reproduction, abundance, etc.), OR
   * A **positive/negative conservation outcome** (i.e., a human intervention produced a measured change).
4. **Field Context**: The finding is from a **field study on wild populations**.

---
### Strict Exclusions

Return `false` if the abstract is primarily about:
* Survey methods, sampling bias, or species detectability.
* Correlations described without a clear cause-and-effect impact.
* Simple exposure to a substance without a measured biological outcome.
* General natural history or basic ecology with no measured effect.
* Reviews, meta-analyses, theoretical models, or lab/captive-only studies.
* Stating aims or hypotheses without reporting corresponding results.

---
### Output Format

Based on this evaluation, return a single, valid JSON object only. If uncertain, default to `false`.

Schema:
{json.dumps(gate_schema, indent=2)}"""

    user_prompt = f"Abstract:\n\"\"\"\n{abstract}\n\"\"\"\n"

    cache_key_parts = [f"impact_conservation_v2:{hashlib.md5(abstract.encode('utf-8')).hexdigest()}"]
    cached_result = llm_setup["cache"].get(cache_key_parts)
    if cached_result is not None:
        logger.debug(f"Impact/conservation cache hit for abstract hash.")
        return cached_result

    try:
        response = llm_generate(
            prompt=user_prompt,
            system=system_prompt,
            model=llm_setup.get("model", "google/gemini-2.5-pro"),
            temperature=0.0,
            format_schema=gate_schema,
            llm_setup=llm_setup
        )
        
        if response:
            try:
                evidence_data = json.loads(response)
                llm_setup["cache"].set(cache_key_parts, evidence_data)
                return evidence_data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse impact gate JSON response: {response[:100]}")
                result = {"impact_or_conservation_found": False, "strongest_impact_sentence": ""}
                llm_setup["cache"].set(cache_key_parts, result)
                return result
        else:
            result = {"impact_or_conservation_found": False, "strongest_impact_sentence": ""}
            llm_setup["cache"].set(cache_key_parts, result)
            return result
            
    except Exception as e:
        logger.error(f"Error in impact/conservation gate: {e}")
        result = {"impact_or_conservation_found": False, "strongest_impact_sentence": ""}
        llm_setup["cache"].set(cache_key_parts, result)
        return result

async def get_comprehensive_relevance_label(
    abstract_text: str, 
    title: str, 
    llm_setup_dict: dict,
    use_gatekeeping: bool = True
) -> tuple[bool, dict]:
    """
    Enhanced relevance labeling with multi-stage gatekeeping and quality scoring.
    Returns (is_relevant, quality_info)
    """
    logger.debug(f"Getting comprehensive relevance label for '{title[:50]}...'")
    
    quality_info = {
        "iucn_threat_relevant": False,
        "has_primary_evidence": False,
        "has_impact_conservation": False,
        "quality_score": 0.0,
        "strongest_sentences": {}
    }

    # Stage 1: Basic IUCN threat relevance (same as before)
    iucn_relevant = get_iucn_threat_relevance_label(abstract_text, title, llm_setup_dict)
    quality_info["iucn_threat_relevant"] = iucn_relevant
    
    if not iucn_relevant:
        logger.debug(f"'{title[:50]}...' failed IUCN threat relevance check")
        return False, quality_info
    
    if not use_gatekeeping:
        quality_info["quality_score"] = 1.0
        return True, quality_info
    
    # Stage 2: Primary evidence check (from pipeline)
    logger.debug(f"Checking primary evidence for '{title[:50]}...'")
    primary_result = await check_for_primary_evidence_training(abstract_text, llm_setup_dict)
    quality_info["has_primary_evidence"] = primary_result.get("is_primary_finding", False)
    if primary_result.get("strongest_evidence_sentence"):
        quality_info["strongest_sentences"]["primary_evidence"] = primary_result["strongest_evidence_sentence"]
    
    # Stage 3: Impact/conservation evidence check (from pipeline)
    logger.debug(f"Checking impact/conservation evidence for '{title[:50]}...'")
    impact_result = await check_for_impact_conservation_evidence_training(abstract_text, llm_setup_dict)
    quality_info["has_impact_conservation"] = impact_result.get("impact_or_conservation_found", False)
    if impact_result.get("strongest_impact_sentence"):
        quality_info["strongest_sentences"]["impact_conservation"] = impact_result["strongest_impact_sentence"]
    
    gate_results_log = {
        "iucn": quality_info["iucn_threat_relevant"],
        "primary": quality_info["has_primary_evidence"],
        "impact": quality_info["has_impact_conservation"]
    }
    logger.info(f"Gate results for '{title[:50]}...': {gate_results_log}")

    # Calculate quality score
    score = 0.0
    if quality_info["iucn_threat_relevant"]:
        score += 0.4
    if quality_info["has_primary_evidence"]:
        score += 0.3
    if quality_info["has_impact_conservation"]:
        score += 0.3
    
    quality_info["quality_score"] = score
    
    is_relevant = (quality_info["iucn_threat_relevant"] and 
                   quality_info["has_primary_evidence"] and 
                   quality_info["has_impact_conservation"])
    
    logger.info(f"Comprehensive relevance for '{title[:50]}...': {is_relevant} (score: {score:.2f})")
    return is_relevant, quality_info

def get_iucn_threat_relevance_label(
    abstract_text: str, 
    title: str, 
    llm_setup_dict: dict
) -> bool:
    """
    Determine if an abstract contains information about direct threats to species 
    that would be classifiable under IUCN-CMP Direct Threats Classification v4.0.
    """
    logger.debug(f"Getting IUCN threat relevance label for '{title[:50]}...'")

    system_prompt = """
You are an expert conservation analyst. Your task is to determine if a scientific abstract contains information about direct threats to a species that would be classifiable under the IUCN-CMP Direct Threats Classification v4.0.

An abstract is RELEVANT if it describes a specific, direct threat causing a negative impact on a specific species or ecosystem. The threat should clearly map to one of the categories below.

An abstract is NOT RELEVANT if it ONLY discusses:
- Conservation successes without detailing the original threat.
- Purely methodological studies or population genetics.
- Species distribution or habitat descriptions without a clear, negative stressor.
- Intrinsic factors like natural predation or competition without human influence.

---
**IUCN-CMP Direct Threats Classification v4.0 - COARSE CATEGORIES**

1. **Residential, Commercial & Recreation Areas**: Urbanization, industrial sites, tourism infrastructure.
2. **Agriculture & Aquaculture**: Farming, livestock ranching, plantations, aquaculture operations.
3. **Energy Production & Mining**: Oil and gas drilling, mining, renewable energy infrastructure (e.g., wind farms).
4. **Transportation, Service & Security Corridors**: Roads, shipping lanes, utility lines causing collisions or habitat fragmentation.
5. **Biological Resource Use & Control**: Hunting, fishing, logging, harvesting of wild resources.
6. **Human Intrusions & Disturbances**: Recreational activities, conflict, research that disturbs wildlife.
7. **Natural System Management & Modifications**: Dams, fire suppression, habitat alteration.
8. **Invasive / Other Problematic Species, Genes & Pathogens**: Invasive species, problematic native species, pathogens introduced or exacerbated by human activity.
9. **Pollution**: Water-borne effluents, garbage/solid waste (plastics), air-borne pollutants, energy emissions (noise/light).
10. **Natural Disasters**: Geological events or severe weather where human activity has increased species' vulnerability.
11. **Climate Change**: Long-term changes in temperature, precipitation, and sea-level rise.
---

Analyze the abstract and respond with a JSON object with a single key "is_relevant" (true/false) and a brief "reasoning".
"""
    
    user_prompt = f"""
Title: {title}
Abstract: {abstract_text}

Does this abstract describe direct threats to species that would be classifiable under the IUCN-CMP Direct Threats Classification v4.0?
"""
    
    relevance_schema = {
        "type": "object", 
        "properties": {"is_relevant": {"type": "boolean"}, "reasoning": {"type": "string"}},
        "required": ["is_relevant", "reasoning"]
    }

    # Cache key based on abstract content, not summary
    label_cache_key_parts = [f"iucn_threat_relevance_v1:{title[:50]}:{hashlib.md5(abstract_text.encode('utf-8')).hexdigest()}"]
    cached_label = llm_setup_dict["cache"].get(label_cache_key_parts)
    if cached_label is not None:
        logger.debug(f"IUCN threat relevance cache hit for '{title[:50]}...': {cached_label}")
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
            logger.warning(f"LLM call for IUCN threat relevance returned empty for '{title[:50]}...'. Defaulting to False")
            llm_setup_dict["cache"].set(label_cache_key_parts, False)
            return False
        result_json = json.loads(response_str)
        is_relevant = result_json.get("is_relevant", False)
        reasoning = result_json.get("reasoning", "No reasoning provided")
        logger.info(f"IUCN threat relevance for '{title[:50]}...': {is_relevant}. Reasoning: {reasoning}")
        llm_setup_dict["cache"].set(label_cache_key_parts, is_relevant)
        return is_relevant
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode IUCN threat relevance JSON for '{title[:50]}...': {e}. Response: '{response_str}'. Defaulting to False")
        llm_setup_dict["cache"].set(label_cache_key_parts, False)
        return False
    except Exception as e_final:
        logger.error(f"Error in IUCN threat relevance determination for '{title[:50]}...': {e_final}", exc_info=True)
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
DEFAULT_MODEL_FOR_RELEVANCE_LABELS = "google/gemini-2.5-pro"
CENTRAL_MODELS_BASE_DIR_NAME = "trained_relevance_models"
ABSTRACT_FILE_NAME = "Lent_Init/shorebirds.parquet"
FILE_BATCH_SIZE = 500

async def collect_and_train_classifier(args):
    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_script_dir.parent
    
    log_file = project_root / "logs" / "classifier_training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_trainer_logging(log_file)

    logger.info("--- Starting Enhanced Relevance Classifier Training Script ---")

    target_samples = args.target_samples if args.target_samples else DEFAULT_TARGET_SAMPLES_PER_CLASS
    max_scan = args.max_scan if args.max_scan else DEFAULT_MAX_TOTAL_ABSTRACTS_TO_SCAN
    
    labeling_model_name = args.labeling_model if args.labeling_model else DEFAULT_MODEL_FOR_RELEVANCE_LABELS
    use_gatekeeping = getattr(args, 'use_gatekeeping', True)
    training_rounds = getattr(args, 'training_rounds', 1)
    
    logger.info(f"Using LLM '{labeling_model_name}' for relevance labeling")
    logger.info(f"Gatekeeping models: {'ENABLED' if use_gatekeeping else 'DISABLED'}")
    logger.info(f"Training rounds: {training_rounds}")

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
    quality_stats = {"high_quality": 0, "medium_quality": 0, "low_quality": 0}

    if args.load_from_file:
        logger.info(f"--- SKIPPING DATA COLLECTION: Loading training data from {args.load_from_file} ---")
        try:
            with open(args.load_from_file, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            logger.info(f"Successfully loaded {len(training_data)} examples from file.")
            relevant_collected = sum(1 for d in training_data if d.get('label'))
            irrelevant_collected = len(training_data) - relevant_collected
        except Exception as e:
            logger.error(f"Failed to load training data from {args.load_from_file}: {e}")
            return
    else:
        for round_num in range(1, training_rounds + 1):
            logger.info(f"=== TRAINING ROUND {round_num}/{training_rounds} ===")
            
            training_data = []
            relevant_collected = 0
            irrelevant_collected = 0
            current_file_skip_rows = 0
            total_abstracts_scanned = 0
            quality_stats = {"high_quality": 0, "medium_quality": 0, "low_quality": 0}

            logger.info(f"Goal: Collect {target_samples} relevant and {target_samples} non-relevant abstracts for training")
            logger.info(f"Max abstracts to scan from Parquet: {max_scan}")

            while (relevant_collected < target_samples or irrelevant_collected < target_samples) and total_abstracts_scanned < max_scan:
                logger.info(f"Loading abstract batch: Skip={current_file_skip_rows}, Size={FILE_BATCH_SIZE}")
                df_batch = load_data_with_offset(ABSTRACT_FILE_NAME, current_file_skip_rows, FILE_BATCH_SIZE)

                if len(df_batch) == 0:
                    logger.info("No more data in abstract file")
                    break
                
                current_file_skip_rows += len(df_batch)
                
                # Process batch with async gatekeeping
                batch_items = []
                for i, row in enumerate(df_batch.iter_rows(named=True)):
                    total_abstracts_scanned += 1
                    if total_abstracts_scanned > max_scan:
                        logger.info(f"Reached max scan limit of {max_scan} abstracts")
                        break

                    title = row["title"]
                    abstract = row["abstract"]
                    batch_items.append((title, abstract, i))
                
                if not batch_items:
                    continue
                    
                # Process batch with comprehensive labeling
                logger.info(f"Processing batch of {len(batch_items)} abstracts with comprehensive labeling")
                batch_tasks = []
                for title, abstract, idx in batch_items:
                    batch_tasks.append(get_comprehensive_relevance_label(abstract, title, llm_s, use_gatekeeping))
                
                batch_results = await asyncio.gather(*batch_tasks)
                
                for i, (title, abstract, idx) in enumerate(batch_items):
                    is_relevant, quality_info = batch_results[i]
                    
                    # Quality classification for statistics
                    score = quality_info["quality_score"]
                    if score >= 0.8:
                        quality_stats["high_quality"] += 1
                    elif score >= 0.5:
                        quality_stats["medium_quality"] += 1
                    else:
                        quality_stats["low_quality"] += 1
                    
                    detail = {
                        'text': abstract, 
                        'title': title, 
                        'quality_info': quality_info,
                        'quality_score': score
                    }
                    
                    if is_relevant and relevant_collected < target_samples:
                        training_data.append({'text': detail['text'], 'label': True, 'title': detail['title'], 
                                            'quality_info': detail['quality_info'], 'quality_score': detail['quality_score']})
                        relevant_collected += 1
                        logger.info(f"Round {round_num} - Collected RELEVANT example #{relevant_collected}/{target_samples} (quality: {score:.2f}): '{detail['title'][:50]}...'")
                    elif not is_relevant and irrelevant_collected < target_samples:
                        training_data.append({'text': detail['text'], 'label': False, 'title': detail['title'], 
                                            'quality_info': detail['quality_info'], 'quality_score': detail['quality_score']})
                        irrelevant_collected += 1
                        logger.info(f"Round {round_num} - Collected IRRELEVANT example #{irrelevant_collected}/{target_samples} (quality: {score:.2f}): '{detail['title'][:50]}...'")
                    
                    if relevant_collected >= target_samples and irrelevant_collected >= target_samples:
                        logger.info("Target samples collected for both classes")
                        break
                
                if (relevant_collected >= target_samples and irrelevant_collected >= target_samples) or \
                   total_abstracts_scanned >= max_scan or \
                   len(df_batch) == 0: 
                    break

            logger.info(f"=== ROUND {round_num} DATA COLLECTION COMPLETE ===")
            logger.info(f"Total abstracts scanned: {total_abstracts_scanned}")
            logger.info(f"Collected {relevant_collected} relevant and {irrelevant_collected} irrelevant examples")
            logger.info(f"Quality distribution - High: {quality_stats['high_quality']}, Medium: {quality_stats['medium_quality']}, Low: {quality_stats['low_quality']}")
            logger.info(f"Total training examples: {len(training_data)}")

            if relevant_collected >= 10 and irrelevant_collected >= 10:
                training_data_save_path = central_model_specific_dir / f"collected_training_data_round{round_num}_{relevant_collected}R_{irrelevant_collected}I.json"
                try:
                    with open(training_data_save_path, 'w', encoding='utf-8') as f:
                        json.dump(training_data, f, indent=2)
                    logger.info(f"Saved collected training data to {training_data_save_path}")
                except Exception as e:
                    logger.error(f"Error saving training data: {e}")

                logger.info(f"=== ROUND {round_num} CLASSIFIER TRAINING ===")
                
                texts_for_splitting = [d['text'] for d in training_data]
                labels_for_splitting = [1 if d['label'] else 0 for d in training_data]

                if len(set(labels_for_splitting)) < 2:
                    logger.error(f"Round {round_num}: Not enough class diversity to perform train/test split. Skipping this round.")
                    continue
                
                logger.info(f"Splitting data: {1.0 - args.test_split_ratio:.0%} train, {args.test_split_ratio:.0%} test")
                X_train_texts, X_test_texts, y_train, y_test = train_test_split(
                    texts_for_splitting, 
                    labels_for_splitting, 
                    test_size=args.test_split_ratio, 
                    random_state=42 + round_num,
                    stratify=labels_for_splitting
                )
                logger.info(f"Training set size: {len(X_train_texts)}, Test set size: {len(X_test_texts)}")

                embedding_model, _ = setup_embedding_classifier(central_model_specific_dir)
                if embedding_model:
                    logger.info("Generating embeddings for training and test sets")
                    X_train_embeddings = embedding_model.encode(X_train_texts, show_progress_bar=True)
                    X_test_embeddings = embedding_model.encode(X_test_texts, show_progress_bar=True)

                    logger.info(f"Training a '{args.classifier_type}' classifier for round {round_num}")
                    
                    round_model_dir = central_model_specific_dir / f"round_{round_num}"
                    round_model_dir.mkdir(parents=True, exist_ok=True)
                    
                    if args.classifier_type == "logistic":
                        current_training_split_data = [{'text': t, 'label': bool(l)} for t, l in zip(X_train_texts, y_train)]
                        trained_classifier = train_embedding_classifier(current_training_split_data, embedding_model, round_model_dir)
                    
                    elif args.classifier_type == "knn":
                        trained_classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
                        trained_classifier.fit(X_train_embeddings, y_train)
                        knn_model_path = round_model_dir / "knn_embedding_classifier.pkl"
                        try:
                            with open(knn_model_path, 'wb') as f:
                                pickle.dump(trained_classifier, f)
                            logger.info(f"k-NN classifier saved to {knn_model_path}")
                        except Exception as e_save:
                            logger.error(f"Error saving k-NN classifier: {e_save}")
                            trained_classifier = None
                    else:
                        logger.error(f"Unsupported classifier type: {args.classifier_type}. Aborting training.")
                        trained_classifier = None

                    if trained_classifier:
                        model_save_filename = "embedding_classifier.pkl" if args.classifier_type == "logistic" else "knn_embedding_classifier.pkl"
                        logger.info(f"Round {round_num} SUCCESS: '{args.classifier_type}' classifier trained and saved to {round_model_dir / model_save_filename}")
                        
                        logger.info(f"=== ROUND {round_num} MODEL EVALUATION ===")
                        y_pred = trained_classifier.predict(X_test_embeddings)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, target_names=['Non-Relevant', 'Relevant'])
                        
                        logger.info(f"Round {round_num} Test Set Accuracy: {accuracy:.4f}")
                        logger.info(f"Round {round_num} Test Set Classification Report:\n{report}")

                        metrics_file_path = round_model_dir / "evaluation_metrics.txt"
                        try:
                            with open(metrics_file_path, 'w') as f:
                                f.write(f"Round {round_num} Results\n")
                                f.write(f"Training Samples: {len(training_data)}\n")
                                f.write(f"Quality Distribution - High: {quality_stats['high_quality']}, Medium: {quality_stats['medium_quality']}, Low: {quality_stats['low_quality']}\n")
                                f.write(f"Test Set Accuracy: {accuracy:.4f}\n\n")
                                f.write("Test Set Classification Report:\n")
                                f.write(report)
                            logger.info(f"Saved round {round_num} evaluation metrics to {metrics_file_path}")
                        except Exception as e_metrics_save:
                            logger.error(f"Error saving evaluation metrics: {e_metrics_save}")
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model_path = round_model_dir / model_save_filename
                            logger.info(f"NEW BEST MODEL: Round {round_num} with accuracy {accuracy:.4f}")
                            
                            try:
                                shutil.copy2(best_model_path, central_model_specific_dir / model_save_filename)
                                shutil.copy2(metrics_file_path, central_model_specific_dir / "evaluation_metrics.txt")
                                logger.info(f"Copied best model to main directory")
                            except Exception as e_copy:
                                logger.error(f"Error copying best model: {e_copy}")
                        else:
                            logger.info(f"Round {round_num} accuracy {accuracy:.4f} did not improve over best {best_accuracy:.4f}")
                    
                    else:
                        logger.error(f"Round {round_num}: Classifier training failed")
                else:
                    logger.error(f"Round {round_num}: Could not set up embedding model")
            else:
                logger.warning(f"Round {round_num}: Not enough data collected for training. Skipping this round.")
        
        # Final summary
        logger.info("=== TRAINING COMPLETE - FINAL SUMMARY ===")
        logger.info(f"Completed {training_rounds} training rounds")
        if best_model_path:
            logger.info(f"Best model achieved accuracy: {best_accuracy:.4f}")
            logger.info(f"Best model saved to: {central_model_specific_dir}")
        else:
            logger.warning("No successful models were trained")
        
        logger.info("--- Enhanced Relevance Classifier Training Script Finished ---")

    if not training_data or relevant_collected < 10 or irrelevant_collected < 10:
        logger.warning("Not enough data to proceed with training. Need at least 10 examples of each class.")
        return

    logger.info("--- Starting Classifier Training Phase ---")
    
    texts_for_splitting = [d['text'] for d in training_data]
    labels_for_splitting = [1 if d['label'] else 0 for d in training_data]

    if len(set(labels_for_splitting)) < 2:
        logger.error("Not enough class diversity for train/test split. Aborting.")
        return
        
    logger.info(f"Splitting data: {1.0 - args.test_split_ratio:.0%} train, {args.test_split_ratio:.0%} test")
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts_for_splitting, 
        labels_for_splitting, 
        test_size=args.test_split_ratio, 
        random_state=42,
        stratify=labels_for_splitting
    )

    embedding_model, _ = setup_embedding_classifier(central_model_specific_dir)
    if embedding_model:
        logger.info("Generating embeddings for training and test sets")
        X_train_embeddings = embedding_model.encode(X_train_texts, show_progress_bar=True)
        X_test_embeddings = embedding_model.encode(X_test_texts, show_progress_bar=True)

        logger.info(f"Training a '{args.classifier_type}' classifier")
        
        trained_classifier = train_embedding_classifier(
            [{'text': t, 'label': bool(l)} for t, l in zip(X_train_texts, y_train)],
            embedding_model, 
            central_model_specific_dir
        )

        if trained_classifier:
            logger.info(f"SUCCESS: Classifier trained and saved.")
            # ... (evaluation logic as before)
        else:
            logger.error("FAILURE: Classifier training failed.")
    else:
        logger.error("FAILURE: Could not set up embedding model.")

    logger.info("--- Enhanced Relevance Classifier Training Script Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced relevance classifier training with multi-stage gatekeeping and iterative rounds")
    parser.add_argument("--target_samples", type=int, help=f"Target number of samples per class (default: {DEFAULT_TARGET_SAMPLES_PER_CLASS})")
    parser.add_argument("--max_scan", type=int, help=f"Maximum total abstracts to scan from Parquet (default: {DEFAULT_MAX_TOTAL_ABSTRACTS_TO_SCAN})")
    parser.add_argument("--labeling_model", type=str, help=f"LLM model name to use for initial labeling (default: {DEFAULT_MODEL_FOR_RELEVANCE_LABELS})")
    parser.add_argument("--test_split_ratio", type=float, default=0.2, help="Ratio of data to use for the test set (default: 0.2)")
    parser.add_argument("--classifier_type", type=str, default="logistic", choices=["logistic", "knn"], help="Type of classifier to train: 'logistic' or 'knn' (default: logistic)")
    parser.add_argument("--use_gatekeeping", action="store_true", default=True, help="Use pipeline gatekeeping models for enhanced filtering (default: True)")
    parser.add_argument("--no_gatekeeping", action="store_false", dest="use_gatekeeping", help="Disable gatekeeping models, use only basic IUCN relevance")
    parser.add_argument("--training_rounds", type=int, default=1, help="Number of iterative training rounds to perform (default: 1)")
    parser.add_argument("--load_from_file", type=str, help="Path to a JSON file with pre-collected data. Skips data collection and goes straight to training.")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parsed_args = parser.parse_args()

    asyncio.run(collect_and_train_classifier(parsed_args))
