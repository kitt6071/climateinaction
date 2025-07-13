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
    
    model = "deepseek/deepseek-r1"
    logger.info(f"Using LLM: '{model}' via OpenRouter")
    
    return {
        'model': model,
        'api_rate_limiter': RateLimiter(rpm=30, is_ollama=False),
        'use_openrouter': True
    }

def generate_shorebird_abstracts(num_abstracts: int = 100) -> List[str]:
    
    llm_setup = setup_llm()
    abstracts = []
    
    system_prompt = """
    You are a scientific abstract generator specializing in shorebird research. Generate realistic, detailed abstracts for shorebird studies that cover diverse research topics, methodologies, and findings.

    Requirements:
    1. Use scientific language and terminology appropriate for ornithological research
    2. Include specific quantitative details (sample sizes, effect sizes, statistical values)
    3. Mention specific locations, seasons, or time periods
    4. Include methodology details (field observations, telemetry, genetic analysis, etc.)
    5. Present clear findings and conservation implications
    6. Vary the writing style and structure while maintaining scientific rigor
    7. Include both positive and negative findings (not all studies show dramatic effects)
    8. Cover different life stages (breeding, migration, wintering, juveniles, adults)
    9. Include both individual-level and population-level studies
    10. Vary the scale (local, regional, flyway-wide, global)

    Generate exactly ONE abstract of 150-250 words. Do not include a title or any other text.
    """
    
    logger.info(f"Generating {num_abstracts} diverse shorebird abstracts...")
    
    for i in range(num_abstracts):
        species = random.choice(SHOREBIRD_SPECIES)
        context = random.choice(RESEARCH_CONTEXTS)
        threat_category = random.choice(list(THREAT_CATEGORIES.keys()))
        specific_threat = random.choice(THREAT_CATEGORIES[threat_category])
        
        prompt_templates = [
            f"Generate an abstract about {species} focusing on {context} in relation to {specific_threat}.",
            f"Create an abstract examining how {specific_threat} affects {species} {context}.",
            f"Write an abstract about a {context} study of {species} dealing with {specific_threat}.",
            f"Generate an abstract investigating {species} response to {specific_threat} through {context} research.",
            f"Create an abstract about {species} showing {context} patterns influenced by {specific_threat}."
        ]
        
        prompt = random.choice(prompt_templates)
        
        try:
            abstract = llm_generate(
                prompt=prompt,
                system=system_prompt,
                model=llm_setup["model"],
                temperature=0.7,
                timeout=120,
                llm_setup=llm_setup
            )
            
            if abstract and len(abstract.strip()) > 100:
                abstracts.append(abstract.strip())
                logger.info(f"Generated abstract {i+1}/{num_abstracts}: {species}")
            else:
                logger.warning(f"Generated abstract {i+1} too short, skipping")
                
        except Exception as e:
            logger.error(f"Error generating abstract {i+1}: {e}")
            continue
    
    logger.info(f"Successfully generated {len(abstracts)} shorebird abstracts")
    return abstracts

def generate_challenging_negatives(num_negatives: int = 150) -> List[str]:
    
    llm_setup = setup_llm()
    abstracts = []
    
    challenging_species = [
        # Waterfowl (often in same habitats)
        "Mallard (Anas platyrhynchos)",
        "Northern Pintail (Anas acuta)",
        "Blue-winged Teal (Spatula discors)",
        "Canvasback (Aythya valisineria)",
        "Redhead (Aythya americana)",
        "Ring-necked Duck (Aythya collaris)",
        "Bufflehead (Bucephala albeola)",
        "Common Goldeneye (Bucephala clangula)",
        
        # Similar Habitat Herons
        "Great Blue Heron (Ardea herodias)",
        "Great Egret (Ardea alba)",
        "Snowy Egret (Egretta thula)",
        "Tricolored Heron (Egretta tricolor)",
        "Green Heron (Butorides virescens)",
        "Black-crowned Night-Heron (Nycticorax nycticorax)",
        "American Bittern (Botaurus lentiginosus)",
        "Least Bittern (Ixobrychus exilis)",
        
        # Rails and coots (marshland species)
        "Virginia Rail (Rallus limicola)",
        "Sora (Porzana carolina)",
        "King Rail (Rallus elegans)",
        "Clapper Rail (Rallus crepitans)",
        "American Coot (Fulica americana)",
        "Common Gallinule (Gallinula galeata)",
        
        # Gulls and terns (coastal, but not shorebirds)
        "Ring-billed Gull (Larus delawarensis)",
        "Herring Gull (Larus argentatus)",
        "Laughing Gull (Leucophaeus atricilla)",
        "Forster's Tern (Sterna forsteri)",
        "Least Tern (Sternula antillarum)",
        "Caspian Tern (Hydroprogne caspia)",
        
        # Raptors
        "Bald Eagle (Haliaeetus leucocephalus)",
        "Osprey (Pandion haliaetus)",
        "Northern Harrier (Circus hudsonius)",
        "Red-tailed Hawk (Buteo jamaicensis)",
        "Cooper's Hawk (Accipiter cooperii)",
        "American Kestrel (Falco sparverius)",
        "Peregrine Falcon (Falco peregrinus)",
        
        # Songbirds
        "Red-winged Blackbird (Agelaius phoeniceus)",
        "Yellow Warbler (Setophaga petechia)",
        "Common Yellowthroat (Geothlypis trichas)",
        "Marsh Wren (Cistothorus palustris)",
        "Sedge Wren (Cistothorus platensis)",
        "Savannah Sparrow (Passerculus sandwichensis)",
        "Swamp Sparrow (Melospiza georgiana)",
        
        # Seabirds
        "Brown Pelican (Pelecanus occidentalis)",
        "Double-crested Cormorant (Phalacrocorax auritus)",
        "Northern Gannet (Morus bassanus)",
        "Common Loon (Gavia immer)",
        "Pied-billed Grebe (Podilymbus podiceps)",
        "Horned Grebe (Podiceps auritus)",
        
        # Game birds
        "Wild Turkey (Meleagris gallopavo)",
        "Northern Bobwhite (Colinus virginianus)",
        "Ring-necked Pheasant (Phasianus colchicus)",
        "Ruffed Grouse (Bonasa umbellus)"
    ]
    
    system_prompt = """
    You are a scientific abstract generator. Generate realistic abstracts for bird research that are NOT about shorebirds. 
    
    Focus on:
    1. Waterfowl (ducks, geese, swans)
    2. Wading birds (herons, egrets, ibises)
    3. Raptors (hawks, eagles, owls)
    4. Songbirds (warblers, sparrows, finches)
    5. Seabirds (pelicans, cormorants, gannets)
    6. Game birds (turkey, quail, grouse)
    7. Rails and coots
    8. Gulls and terns
    
    Use scientific language and methodology similar to shorebird research to create challenging negative examples.
    Include quantitative details, specific locations, and conservation implications.
    Generate exactly ONE abstract of 150-250 words. Do not include a title.
    """
    
    abstracts = []
    
    logger.info(f"Generating {num_negatives} challenging negative examples...")
    
    for i in range(num_negatives):
        species = random.choice(challenging_species)
        context = random.choice(RESEARCH_CONTEXTS)
        
        prompt = f"Generate an abstract about {species} focusing on {context} research."
        
        try:
            abstract = llm_generate(
                prompt=prompt,
                system=system_prompt,
                model=llm_setup["model"],
                temperature=0.7,
                timeout=120,
                llm_setup=llm_setup
            )
            
            if abstract and len(abstract.strip()) > 100:
                abstracts.append(abstract.strip())
                logger.info(f"Generated negative {i+1}/{num_negatives}: {species}")
            else:
                logger.warning(f"Generated negative {i+1} too short, skipping")
                
        except Exception as e:
            logger.error(f"Error generating negative {i+1}: {e}")
            continue
    
    logger.info(f"Successfully generated {len(abstracts)} challenging negative examples")
    return abstracts

def load_negatives_from_parquet(num_negatives: int = 200) -> List[str]:
    try:
        import polars as pl
        
        current_dir = Path(__file__).parent
        parquet_path = current_dir / "all_abstracts.parquet"
        
        if not parquet_path.exists():
            logger.warning(f"Parquet file not found at {parquet_path}")
            return []
        
        logger.info(f"Loading {num_negatives} negatives from beginning of parquet")
        
        df = pl.read_parquet(parquet_path).head(num_negatives)
        df = df.drop_nulls(["title", "abstract"])
        
        real_negatives = []
        for i, row in enumerate(df.iter_rows(named=True)):
            abstract = row["abstract"]
            title = row["title"] 
            if abstract and len(abstract.strip()) > 50:
                real_negatives.append(abstract.strip())
                logger.info(f"negative {i+1}: '{title[:50]}...'")
        
        logger.info(f"Loaded {len(real_negatives)} negative examples from parquet")
        return real_negatives
        
    except Exception as e:
        logger.error(f"Error loading negatives: {e}")
        return []
def main():
    output_dir = Path("data_to_review")
    output_dir.mkdir(exist_ok=True)
    
    real_negatives = load_negatives_from_parquet(num_negatives=200)
    
    positive_abstracts = generate_shorebird_abstracts(num_abstracts=100)
    synthetic_negatives = generate_challenging_negatives(num_negatives=150)
    
    # Combine all negatives
    all_negatives = real_negatives + synthetic_negatives
    
    positive_file = output_dir / "shorebird_positives.json"
    with open(positive_file, 'w', encoding='utf-8') as f:
        json.dump(positive_abstracts, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(positive_abstracts)} positive examples to {positive_file}")
    
    negative_file = output_dir / "shorebird_negatives.json"
    with open(negative_file, 'w', encoding='utf-8') as f:
        json.dump(all_negatives, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(all_negatives)} negative examples to {negative_file} ({len(real_negatives)} real + {len(synthetic_negatives)} synthetic)")
    
    training_data = []
    
    for abstract in positive_abstracts:
        training_data.append({
            "text": abstract,
            "label": 1,
            "category": "shorebird_relevant"
        })
    
    for abstract in real_negatives:
        training_data.append({
            "text": abstract,
            "label": 0,
            "category": "real_negative"
        })
    
    for abstract in synthetic_negatives:
        training_data.append({
            "text": abstract,
            "label": 0,
            "category": "synthetic_negative"
        })
    
    random.shuffle(training_data)
    
    training_file = output_dir / "training_data.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Positive examples: {len(positive_abstracts)}")
    logger.info(f"Negative examples: {len(all_negatives)} ({len(real_negatives)} real + {len(synthetic_negatives)} synthetic)")
    logger.info(f"Total training examples: {len(training_data)}")
    logger.info(f"Saved combined dataset to {training_file}")

if __name__ == "__main__":
    main() 