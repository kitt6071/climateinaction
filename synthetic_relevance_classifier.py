import asyncio
import json
import random
from pathlib import Path
import logging
from typing import List, Dict
import os
import sys
import time
import pickle
import hashlib
from dotenv import load_dotenv

current_dir = Path(__file__).parent
lent_init_dir = current_dir / "Lent_Init"
sys.path.append(str(lent_init_dir))

from openai import OpenAI
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
    def get(self, key_parts):
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
        
    def set(self, key_parts, result):
        if not isinstance(key_parts, list):
            key_parts = [key_parts]
        cache_key = self._make_hash_key(*key_parts)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.error(f"Error writing to cache file {cache_file}: {e}")

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
                logger.error("OPENROUTER_API_KEY not found")
                raise ValueError("OPENROUTER_API_KEY not found")

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

            logger.debug(f"OpenRouter Request Params: {json.dumps({k: v for k, v in request_params.items() if k != 'api_key'}, indent=2)}")
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

SHOREBIRD_SPECIES = [
    # Sandpipers and allies
    "Shorebird",
    "Dunlin (Calidris alpina)",
    "Sanderling (Calidris alba)", 
    "Red Knot (Calidris canutus)",
    "Semipalmated Sandpiper (Calidris pusilla)",
    "Least Sandpiper (Calidris minutilla)",
    "Stilt Sandpiper (Calidris himantopus)",
    "Pectoral Sandpiper (Calidris melanotos)",
    "Purple Sandpiper (Calidris maritima)",
    "Curlew Sandpiper (Calidris ferruginea)",
    "Western Sandpiper (Calidris mauri)",
    
    # Plovers
    "Piping Plover (Charadrius melodus)",
    "Killdeer (Charadrius vociferus)",
    "Black-bellied Plover (Pluvialis squatarola)",
    "American Golden-Plover (Pluvialis dominica)",
    "Semipalmated Plover (Charadrius semipalmatus)",
    "Snowy Plover (Charadrius nivosus)",
    "Wilson's Plover (Charadrius wilsonia)",
    "Mountain Plover (Charadrius montanus)",
    "Common Ringed Plover (Charadrius hiaticula)",
    "Kentish Plover (Charadrius alexandrinus)",
    
    # Curlews and Godwits
    "Whimbrel (Numenius phaeopus)",
    "Long-billed Curlew (Numenius americanus)",
    "Eurasian Curlew (Numenius arquata)",
    "Bar-tailed Godwit (Limosa lapponica)",
    "Hudsonian Godwit (Limosa haemastica)",
    "Marbled Godwit (Limosa fedoa)",
    "Black-tailed Godwit (Limosa limosa)",
    
    # Turnstones and Surfbirds
    "Ruddy Turnstone (Arenaria interpres)",
    "Black Turnstone (Arenaria melanocephala)",
    "Surfbird (Calidris virgata)",
    "Rock Sandpiper (Calidris ptilocnemis)",
    
    # Oystercatchers and Stilts
    "American Oystercatcher (Haematopus palliatus)",
    "Eurasian Oystercatcher (Haematopus ostralegus)",
    "Black Oystercatcher (Haematopus bachmani)",
    "Black-necked Stilt (Himantopus mexicanus)",
    "American Avocet (Recurvirostra americana)",
    "Pied Avocet (Recurvirostra avosetta)",
    
    # Dowitchers and Snipe
    "Short-billed Dowitcher (Limnodromus griseus)",
    "Long-billed Dowitcher (Limnodromus scolopaceus)",
    "Wilson's Snipe (Gallinago delicata)",
    "Common Snipe (Gallinago gallinago)",
    
    # Phalaropes (often grouped with shorebirds)
    "Wilson's Phalarope (Phalaropus tricolor)",
    "Red-necked Phalarope (Phalaropus lobatus)",
    "Red Phalarope (Phalaropus fulicarius)",
    
    # Yellowlegs and Tattlers
    "Greater Yellowlegs (Tringa melanoleuca)",
    "Lesser Yellowlegs (Tringa flavipes)",
    "Solitary Sandpiper (Tringa solitaria)",
    "Spotted Sandpiper (Actitis macularius)",
    "Wandering Tattler (Tringa incana)",
    "Willet (Tringa semipalmata)",
    "Wood Sandpiper (Tringa glareola)",
    "Green Sandpiper (Tringa ochropus)",
    "Common Sandpiper (Actitis hypoleucos)",
    
    # Jacanas (often grouped with shorebirds)
    "Northern Jacana (Jacana spinosa)",
    "Wattled Jacana (Jacana jacana)",
    
    # Thick-knees (Stone-curlews)
    "Double-striped Thick-knee (Burhinus bistriatus)",
    "Peruvian Thick-knee (Burhinus superciliaris)"
]

RESEARCH_CONTEXTS = [
    "migration ecology",
    "breeding biology", 
    "foraging behavior",
    "habitat selection",
    "population dynamics",
    "conservation genetics",
    "climate change impacts",
    "pollution effects",
    "human disturbance",
    "predator-prey relationships",
    "energy expenditure",
    "stopover site usage",
    "molt migration", 
    "nest site selection",
    "parental care",
    "juvenile survival",
    "demographic analysis",
    "stable isotope analysis",
    "telemetry studies",
    "bioacoustics",
    "morphological variation",
    "community ecology",
    "wetland management",
    "coastal habitat loss",
    "invasive species interactions"
]

THREAT_CATEGORIES = {
    "habitat_loss": [
        "coastal development",
        "sea level rise", 
        "wetland drainage",
        "shoreline armoring",
        "salt marsh conversion",
        "mudflat reclamation",
        "beach nourishment",
        "dredging activities"
    ],
    "climate_change": [
        "rising temperatures",
        "altered precipitation patterns",
        "extreme weather events",
        "shifting food availability",
        "phenological mismatch",
        "ocean acidification",
        "changing storm patterns"
    ],
    "pollution": [
        "plastic ingestion",
        "chemical contamination",
        "oil spills",
        "heavy metal exposure",
        "pesticide poisoning",
        "nutrient pollution",
        "microplastics"
    ],
    "human_disturbance": [
        "recreational activities",
        "vehicle traffic",
        "dog disturbance",
        "kite flying",
        "photography pressure",
        "nest trampling",
        "noise pollution"
    ],
    "invasive_species": [
        "invasive plant encroachment",
        "introduced predators",
        "competitive exclusion",
        "parasitic infections",
        "disease transmission"
    ],
    "overexploitation": [
        "egg collection",
        "hunting pressure",
        "bycatch mortality",
        "shell collection impact"
    ]
}

def setup_llm():
    load_dotenv()
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    model = "deepseek/deepseek-r1-0528"
    logger.info(f"Using LLM: '{model}' via OpenRouter")
    
    return {
        'model': model,
        'api_rate_limiter': RateLimiter(rpm=30, is_ollama=False),
        'use_openrouter': True
    }

def generate_shorebird_abstracts(num_abstracts: int = 100) -> List[str]:
    
    KEYWORD_ABSTRACT_LENGTH = 512
    llm_setup = setup_llm()
    abstracts = []
    generated_species = []  # track species to avoid repetition
    
    system_prompt = """You are an expert in generating synthetic academic literature. Your task is to create highly specific and realistic research abstracts about shorebirds. The abstracts must focus on species from the Charadriiformes order (plovers, sandpipers, oystercatchers, etc.) and should be indistinguishable from genuine scientific research. Use a formal, academic tone, include quantitative details (e.g., sample sizes, p-values, confidence intervals), specific geographic locations, and clear methodologies and conservation implications. Ensure the content is narrowly focused on shorebird ecology, behavior, or conservation."""
    
    logger.info(f"Generating {num_abstracts} highly-specific shorebird abstracts...")
    
    schema = {
        "type": "object",
        "properties": {
            "abstract": {"type": "string"}
        },
        "required": ["abstract"]
    }
    
    for i in range(num_abstracts):
        species = random.choice(SHOREBIRD_SPECIES)
        context = random.choice(RESEARCH_CONTEXTS)
        threat_category = random.choice(list(THREAT_CATEGORIES.keys()))
        specific_threat = random.choice(THREAT_CATEGORIES[threat_category])
        
        # Add variety instruction
        variety_instruction = ""
        if len(generated_species) > 5:
            recent_species = generated_species[-5:]
            variety_instruction = f" Ensure variety - avoid similarity to recent abstracts about: {', '.join(recent_species)}."
        
        prompt = f"""Generate one highly realistic and specific shorebird research abstract about {species}.
                    Focus on the context of {context} in relation to the threat of {specific_threat}.
                    The abstract must be technical, academic, and specific to the Charadriiformes order.
                    Length: {KEYWORD_ABSTRACT_LENGTH//2}-{KEYWORD_ABSTRACT_LENGTH} characters.{variety_instruction}

                    Example abstract structure:
                    1.  Introduction: Briefly introduce the species, context, and threat.
                    2.  Methods: Describe the study location, duration, and methods used (e.g., GPS tracking, stable isotope analysis, population modeling).
                    3.  Results: Present key findings with quantitative data (e.g., "nest success was 15% lower in high-disturbance areas (p < 0.05)").
                    4.  Conclusion: State the implications for conservation or understanding of shorebird ecology.

                    Respond with ONLY a JSON object containing the 'abstract' field."""
                    
        try:
            response = llm_generate(
                prompt=prompt,
                system=system_prompt,
                model=llm_setup["model"],
                temperature=0.8,
                timeout=120,
                format_schema=schema,
                llm_setup=llm_setup
            )
            
            if response:
                try:
                    result = json.loads(response)
                    abstract = result.get("abstract", "").strip()
                    if abstract and len(abstract) > 150:
                        abstracts.append(abstract)
                        generated_species.append(species.split('(')[0].strip())
                        logger.info(f"Generated positive abstract {i+1}/{num_abstracts}: {species}")
                    else:
                        logger.warning(f"Generated positive abstract {i+1} too short, skipping.")
                except json.JSONDecodeError:
                    if len(response.strip()) > 150:
                        abstracts.append(response.strip())
                        generated_species.append(species.split('(')[0].strip())
                        logger.info(f"Generated positive abstract {i+1}/{num_abstracts} (non-JSON): {species}")
            
        except Exception as e:
            logger.error(f"Error generating positive abstract {i+1}: {e}")
            continue
    
    logger.info(f"Successfully generated {len(abstracts)} shorebird abstracts")
    return abstracts

def generate_challenging_negatives(num_negatives: int = 150) -> List[str]:
    
    KEYWORD_ABSTRACT_LENGTH = 512
    llm_setup = setup_llm()
    abstracts = []
    generated_species = []  # track species to avoid repetition
    
    # refined list focusing on common confusion species for shorebirds
    challenging_species_by_class = {
        "Gulls & Terns": [
            "Ring-billed Gull (Larus delawarensis)", "Herring Gull (Larus argentatus)", "Laughing Gull (Leucophaeus atricilla)",
            "Forster's Tern (Sterna forsteri)", "Least Tern (Sternula antillarum)", "Caspian Tern (Hydroprogne caspia)",
            "Black Skimmer (Rynchops niger)"
        ],
        "Herons & Egrets": [
            "Great Blue Heron (Ardea herodias)", "Great Egret (Ardea alba)", "Snowy Egret (Egretta thula)",
            "Tricolored Heron (Egretta tricolor)", "Black-crowned Night-Heron (Nycticorax nycticorax)",
        ],
        "Waterfowl (Coastal)": [
            "Mallard (Anas platyrhynchos)", "Northern Pintail (Anas acuta)", "American Black Duck (Anas rubripes)",
            "Brant (Branta bernicla)", "Common Eider (Somateria mollissima)"
        ],
        "Rails & Coots": [
            "Clapper Rail (Rallus crepitans)", "Virginia Rail (Rallus limicola)", "American Coot (Fulica americana)"
        ],
        "Other Coastal Birds": [
            "Brown Pelican (Pelecanus occidentalis)", "Double-crested Cormorant (Phalacrocorax auritus)",
            "Osprey (Pandion haliaetus)", "Belted Kingfisher (Megaceryle alcyon)"
        ]
    }
    
    system_prompt = """You are an expert in generating synthetic academic literature. Your task is to create realistic research abstracts about birds that are explicitly NOT shorebirds (i.e., not from the Charadriiformes order). These abstracts should be challenging negative examples that discuss bird-threat relationships but for non-shorebird species. Include threats like habitat loss, climate change, pollution, and human disturbance affecting songbirds, waterfowl, raptors, seabirds, and other bird groups. The goal is to train a classifier to distinguish shorebird research from other bird research."""
    
    abstracts = []
    schema = {
        "type": "object",
        "properties": {
            "abstract": {"type": "string"}
        },
        "required": ["abstract"]
    }
    
    logger.info(f"Generating {num_negatives} new, more challenging negative examples...")
    
    for i in range(num_negatives):
        bird_class = random.choice(list(challenging_species_by_class.keys()))
        species = random.choice(challenging_species_by_class[bird_class])
        context = random.choice(RESEARCH_CONTEXTS)
        
        # Add variety instruction
        variety_instruction = ""
        if len(generated_species) > 5:
            recent_species = generated_species[-5:]
            variety_instruction = f" Ensure the topic is distinct from recent abstracts about: {', '.join(recent_species)}."
        
        prompt = f"""Generate one realistic research abstract about the {bird_class}, specifically focusing on {species}. The context is {context}.
                    The abstract MUST NOT be about shorebirds (order Charadriiformes), but should use similar scientific language and methodologies.
                    It must be a challenging negative example for a shorebird relevance classifier.
                    Length: {KEYWORD_ABSTRACT_LENGTH//2}-{KEYWORD_ABSTRACT_LENGTH} characters.{variety_instruction}

                    Example topics for non-shorebirds:
                    - Foraging ecology of Laughing Gulls in urban coastal environments.
                    - Migratory patterns of Great Blue Herons using satellite telemetry.
                    - Effects of oil spills on Brown Pelican breeding success.

                    Respond with ONLY a JSON object containing the 'abstract' field."""
        
        try:
            response = llm_generate(
                prompt=prompt,
                system=system_prompt,
                model=llm_setup["model"],
                temperature=0.8,
                timeout=120,
                format_schema=schema,
                llm_setup=llm_setup
            )
            
            if response:
                try:
                    result = json.loads(response)
                    abstract = result.get("abstract", "").strip()
                    if abstract and len(abstract) > 150:
                        abstracts.append(abstract)
                        generated_species.append(species.split('(')[0].strip())
                        logger.info(f"Generated negative {i+1}/{num_negatives}: {species} ({bird_class})")
                    else:
                        logger.warning(f"Generated negative {i+1} was too short or empty, skipping.")
                except json.JSONDecodeError:
                    if len(response.strip()) > 150:
                        abstracts.append(response.strip())
                        generated_species.append(species.split('(')[0].strip())
                        logger.info(f"Generated negative {i+1}/{num_negatives} (non-JSON): {species} ({bird_class})")
            
        except Exception as e:
            logger.error(f"Error generating negative {i+1}: {e}")
            continue
    
    logger.info(f"Successfully generated {len(abstracts)} challenging negative examples")
    return abstracts

def load_real_shorebird_abstracts_from_parquet(num_positives: int = 50) -> List[str]:
    try:
        import polars as pl
        import re

        parquet_path = Path("/Users/kittsonhamill/Desktop/all_abstracts.parquet")
        if not parquet_path.exists():
            logger.warning(f"Parquet file not found at {parquet_path}")
            return []

        logger.info(f"Loading real shorebird abstracts from parquet file...")
        df = pl.read_parquet(parquet_path, n_rows=100000)  # scan first 100k rows
        df = df.drop_nulls(subset=["title", "abstract"])

        # shorebird keywords for positive identification
        shorebird_keywords = [
            "shorebird", "plover", "sandpiper", "oystercatcher", "turnstone", "godwit", "curlew", 
            "yellowleg", "dowitcher", "avocet", "stilt", "dunlin", "sanderling", "killdeer",
            "whimbrel", "ruddy turnstone", "semipalmated", "charadriiformes", "charadrius", 
            "calidris", "haematopus", "numenius", "limosa", "arenaria", "tringa", "actitis", 
            "limnodromus", "recurvirostra", "himantopus"
        ]

        # find abstracts mentioning shorebird terms
        shorebird_pattern = "|".join([re.escape(keyword) for keyword in shorebird_keywords])
        
        df_positives = df.filter(
            (pl.col("abstract").str.to_lowercase().str.contains(shorebird_pattern, literal=False)) |
            (pl.col("title").str.to_lowercase().str.contains(shorebird_pattern, literal=False))
        )
        
        if len(df_positives) < num_positives:
            logger.warning(f"Found only {len(df_positives)} real shorebird abstracts. Using all of them.")
            num_positives = len(df_positives)
        
        df_sampled = df_positives.sample(n=num_positives, seed=42)
        real_shorebird_abstracts = df_sampled["abstract"].to_list()
        
        logger.info(f"Successfully loaded {len(real_shorebird_abstracts)} real shorebird abstracts.")
        return real_shorebird_abstracts
        
    except Exception as e:
        logger.error(f"Error loading real shorebird abstracts: {e}", exc_info=True)
        return []

def load_real_bird_negatives_from_parquet(num_negatives: int = 100) -> List[str]:
    try:
        import polars as pl
        import re

        parquet_path = Path("/Users/kittsonhamill/Desktop/all_abstracts.parquet")
        if not parquet_path.exists():
            logger.warning(f"Parquet file not found at {parquet_path}")
            return []

        logger.info(f"Loading real bird negative examples from parquet file...")
        df = pl.read_parquet(parquet_path, n_rows=100000)  # scan first 100k rows
        df = df.drop_nulls(subset=["title", "abstract"])

        # bird keywords (but not shorebirds)
        general_bird_keywords = [
            "songbird", "waterfowl", "duck", "goose", "swan", "heron", "egret", "falcon", "hawk", 
            "eagle", "owl", "crow", "raven", "sparrow", "warbler", "finch", "wren", "swallow", 
            "flycatcher", "thrush", "robin", "blackbird", "cardinal", "blue jay", "woodpecker",
            "hummingbird", "pelican", "cormorant", "gull", "tern", "albatross", "petrel", 
            "penguin", "seabird", "raptor", "passerine", "galliformes", "anseriformes", 
            "falconiformes", "strigiformes", "piciformes", "passeriformes", "procellariiformes"
        ]

        # Shorebird terms to exclude
        shorebird_exclude_keywords = [
            "shorebird", "plover", "sandpiper", "oystercatcher", "turnstone", "godwit", "curlew", 
            "yellowleg", "dowitcher", "avocet", "stilt", "charadriiformes"
        ]

        bird_pattern = "|".join([re.escape(keyword) for keyword in general_bird_keywords])
        shorebird_pattern = "|".join([re.escape(keyword) for keyword in shorebird_exclude_keywords])

        # find bird abstracts that don't mention shorebirds
        df_bird_negatives = df.filter(
            ((pl.col("abstract").str.to_lowercase().str.contains(bird_pattern, literal=False)) |
             (pl.col("title").str.to_lowercase().str.contains(bird_pattern, literal=False))) &
            (~pl.col("abstract").str.to_lowercase().str.contains(shorebird_pattern, literal=False)) &
            (~pl.col("title").str.to_lowercase().str.contains(shorebird_pattern, literal=False))
        )
        
        if len(df_bird_negatives) < num_negatives:
            logger.warning(f"Found only {len(df_bird_negatives)} real bird negatives. Using all of them.")
            num_negatives = len(df_bird_negatives)
        
        df_sampled = df_bird_negatives.sample(n=num_negatives, seed=42)
        real_bird_negatives = df_sampled["abstract"].to_list()
        
        logger.info(f"Successfully loaded {len(real_bird_negatives)} real bird negative examples.")
        return real_bird_negatives
        
    except Exception as e:
        logger.error(f"Error loading real bird negatives: {e}", exc_info=True)
        return []

def load_negatives_from_parquet(num_hard_negatives: int = 200, num_easy_negatives: int = 100) -> List[str]:
    try:
        import polars as pl
        import re

        parquet_path = Path("/Users/kittsonhamill/Desktop/all_abstracts.parquet")
        if not parquet_path.exists():
            logger.warning(f"Parquet file not found at {parquet_path}")
            return []

        df = pl.read_parquet(parquet_path, n_rows=250000) # Scan a larger portion
        df = df.drop_nulls(subset=["title", "abstract"])

        # 1. get easy negatives (from the start of the file)
        easy_negatives = []
        if num_easy_negatives > 0:
            logger.info(f"Loading {num_easy_negatives} easy negatives from the start of the parquet file.")
            easy_negatives_df = df.head(num_easy_negatives)
            easy_negatives = easy_negatives_df["abstract"].to_list()
            logger.info(f"Loaded {len(easy_negatives)} easy negatives.")

        # 2. perform hard negative mining
        hard_negatives = []
        if num_hard_negatives > 0:
            logger.info("Starting hard negative mining...")
            shorebird_common_names = [name.split('(')[0].strip().lower() for name in SHOREBIRD_SPECIES]
            shorebird_name_pattern = "|".join([re.escape(name) for name in shorebird_common_names])
            hard_negative_keywords = ["bird", "avian", "coastal", "wetland", "ecology", "migration", "foraging", "seabird"]
            keyword_pattern = "|".join(hard_negative_keywords)

            df_hard = df.filter(
                (pl.col("abstract").str.contains(keyword_pattern, literal=False) |
                 pl.col("title").str.contains(keyword_pattern, literal=False)) &
                (~pl.col("abstract").str.to_lowercase().str.contains(shorebird_name_pattern, literal=False)) &
                (~pl.col("title").str.to_lowercase().str.contains(shorebird_name_pattern, literal=False))
            )
            
            if len(df_hard) < num_hard_negatives:
                logger.warning(f"Found only {len(df_hard)} hard negatives. Using all of them.")
            
            df_sampled = df_hard.sample(n=min(num_hard_negatives, len(df_hard)), seed=42)
            hard_negatives = df_sampled["abstract"].to_list()
            logger.info(f"Successfully mined {len(hard_negatives)} hard negative examples.")

        return easy_negatives + hard_negatives
        
    except Exception as e:
        logger.error(f"Error during negative example loading: {e}", exc_info=True)
        return []

def main():
    output_dir = Path("data_to_review")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Creating real test set")
    # create a test set of real abstracts from parquet
    real_shorebird_test = load_real_shorebird_abstracts_from_parquet(num_positives=50)
    real_bird_negatives_test = load_real_bird_negatives_from_parquet(num_negatives=100)
    general_negatives_test = load_negatives_from_parquet(num_hard_negatives=25, num_easy_negatives=25)
    
    # create test set
    test_data = []
    
    for abstract in real_shorebird_test:
        test_data.append({
            "text": abstract,
            "label": 1,
            "category": "real_shorebird_test"
        })
    
    for abstract in real_bird_negatives_test:
        test_data.append({
            "text": abstract,
            "label": 0,
            "category": "real_bird_negative_test"
        })
    
    for abstract in general_negatives_test:
        test_data.append({
            "text": abstract,
            "label": 0,
            "category": "general_negative_test"
        })
    
    random.shuffle(test_data)
    
    test_file = output_dir / "real_test_data.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created real test set: {len(real_shorebird_test)} shorebird + {len(real_bird_negatives_test)} bird negatives + {len(general_negatives_test)} general negatives = {len(test_data)} total")
    
    logger.info("Generating synthetic training data")
    # generate synthetic training data (larger amounts to compensate for domain gap)
    synthetic_positives = generate_shorebird_abstracts(num_abstracts=200)  # More synthetic positives
    synthetic_negatives = generate_challenging_negatives(num_negatives=400)  # More challenging negatives
    general_negatives_train = load_negatives_from_parquet(num_hard_negatives=100, num_easy_negatives=50)
    
    # create training data (all synthetic + some real negatives)
    training_data = []
    
    for abstract in synthetic_positives:
        training_data.append({
            "text": abstract,
            "label": 1,
            "category": "synthetic_shorebird"
        })
    
    for abstract in synthetic_negatives:
        training_data.append({
            "text": abstract,
            "label": 0,
            "category": "synthetic_bird_negative"
        })
    
    for abstract in general_negatives_train:
        training_data.append({
            "text": abstract,
            "label": 0,
            "category": "general_negative_train"
        })
    
    random.shuffle(training_data)
    
    # save training and test data separately
    training_file = output_dir / "synthetic_training_data.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # also save individual components for inspection
    positive_file = output_dir / "shorebird_positives.json"
    with open(positive_file, 'w', encoding='utf-8') as f:
        json.dump(synthetic_positives, f, indent=2, ensure_ascii=False)
    
    negative_file = output_dir / "shorebird_negatives.json"
    with open(negative_file, 'w', encoding='utf-8') as f:
        json.dump(synthetic_negatives + general_negatives_train, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Final dataset summary:")
    logger.info(f"TRAINING SET (synthetic):")
    logger.info(f"  - Synthetic shorebird positives: {len(synthetic_positives)}")
    logger.info(f"  - Synthetic bird negatives: {len(synthetic_negatives)}")
    logger.info(f"  - General negatives: {len(general_negatives_train)}")
    logger.info(f"  - Total training examples: {len(training_data)}")
    logger.info(f"TEST SET (real abstracts):")
    logger.info(f"  - Real shorebird positives: {len(real_shorebird_test)}")
    logger.info(f"  - Real bird negatives: {len(real_bird_negatives_test)}")
    logger.info(f"  - General negatives: {len(general_negatives_test)}")
    logger.info(f"  - Total test examples: {len(test_data)}")
    logger.info(f"Saved training data to {training_file}")
    logger.info(f"Saved test data to {test_file}")

if __name__ == "__main__":
    main() 