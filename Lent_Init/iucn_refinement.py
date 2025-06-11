import json
import asyncio
import logging
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import os
import re
from .cache import SimpleCache
from .llm_api_utility import llm_generate

logger = logging.getLogger("pipeline")

async def get_iucn_classification_json(subject: str, predicate: str, threat_desc: str, llm_setup, cache: SimpleCache) -> tuple[str, str]: 
    cache_key = f"iucn_classify_json_schema:{threat_desc}|context:{subject}|{predicate}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"IUCN cache hit: '{threat_desc[:50]}...'")
        return cached_result

    logger.info(f"Classifying threat: '{threat_desc[:50]}...'")
    
    iucn_schema = {
        "type": "object",
        "properties": {
            "iucn_code": {"type": "string", "description": "IUCN code like '5.3' or '11.1'"},
            "iucn_name": {"type": "string", "description": "IUCN category name"}
        },
        "required": ["iucn_code", "iucn_name"]
    }

    prompt = f"""
            Context:
            Subject (Species): {subject}
            Predicate (Impact): {predicate}

            Threat to classify:
            {threat_desc}

            Determine the most appropriate IUCN category based on the threat description.
            Focus on the underlying cause.
            """
                
    response_str = await llm_generate(
        prompt=prompt,
        system=IUCN_THREAT_PROMPT_SYSTEM,
        model=llm_setup['model'], 
        temp=0.0, 
        format=iucn_schema,
        llm_setup=llm_setup
    )

    if response_str:
        try:
            result_json = json.loads(response_str) 
            code = result_json.get("iucn_code")
            name = result_json.get("iucn_name")
            
            # basic validation
            if isinstance(code, str) and isinstance(name, str) and code.strip() and name.strip() and re.match(r"^\d+(\.\d+)?$", code.strip()):
                code = code.strip()
                name = name.strip()
                logger.info(f"Classified as: {code} - {name}")
                result = (code, name)
                cache.set(cache_key, result)
                return result
            else:
                 logger.warning(f"Invalid response: Code='{code}', Name='{name}'")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed: {e}")
            logger.error(f"Response string: '{response_str}'")
        except Exception as e:
             logger.error(f"Response processing failed: {e}")
             
    else:
        logger.warning("LLM call failed or empty response")

    # fallback to "other"
    result = ("12.1", "Other threat")
    cache.set(cache_key, result)
    return result


IUCN_CATEGORIES_TEXT = """
            **IUCN THREAT CATEGORIES:**
            1 Residential & commercial development (1.1 Housing & urban areas, 1.2 Commercial & industrial areas, 1.3 Tourism & recreation areas)
            2 Agriculture & aquaculture (2.1 Annual & perennial non-timber crops, 2.2 Wood & pulp plantations, 2.3 Livestock farming & ranching, 2.4 Marine & freshwater aquaculture)
            3 Energy production & mining (3.1 Oil & gas drilling, 3.2 Mining & quarrying, 3.3 Renewable energy)
            4 Transportation & service corridors (4.1 Roads & railroads, 4.2 Utility & service lines, 4.3 Shipping lanes, 4.4 Flight paths)
            5 Biological resource use (5.1 Hunting & collecting terrestrial animals, 5.2 Gathering terrestrial plants, 5.3 Logging & wood harvesting, 5.4 Fishing & harvesting aquatic resources)
            6 Human intrusions & disturbance (6.1 Recreational activities, 6.2 War, civil unrest & military exercises, 6.3 Work & other activities)
            7 Natural system modifications (7.1 Fire & fire suppression, 7.2 Dams & water management/use, 7.3 Other ecosystem modifications)
            8 Invasive & other problematic species, genes & diseases (8.1 Invasive non-native/alien species/diseases, 8.2 Problematic native species/diseases, 8.3 Introduced genetic material, 8.4 Problematic species/diseases of unknown origin, 8.5 Viral/prion-induced diseases, 8.6 Diseases of unknown cause)
            9 Pollution (9.1 Domestic & urban waste water, 9.2 Industrial & military effluents, 9.3 Agricultural & forestry effluents, 9.4 Garbage & solid waste, 9.5 Air-borne pollutants, 9.6 Excess energy)
            10 Geological events (10.1 Volcanoes, 10.2 Earthquakes/tsunamis, 10.3 Avalanches/landslides)
            11 Climate change & severe weather (11.1 Habitat shifting & alteration, 11.2 Droughts, 11.3 Temperature extremes, 11.4 Storms & flooding, 11.5 Other impacts)
            12 Other options (12.1 Other threat)
            """

IUCN_THREAT_PROMPT_SYSTEM = f"""
            You are an expert ecological threat classifier. Your task is to assign the single most appropriate IUCN threat category to a given threat description, considering the context of the species and the impact mechanism.

            {IUCN_CATEGORIES_TEXT}

            **Instructions:**
            1. Analyze the provided Threat Description in the context of the Subject (Species) and Predicate (Impact Mechanism).
            2. Identify the *underlying cause* of the threat.
            3. Select the single *most specific and relevant* IUCN category code and name from the list above that best represents the *underlying cause*.
            4. **Avoid 12.1 Other threat** unless no other category is remotely applicable. Think critically about the root cause.
            5. Return ONLY a valid JSON object containing the selected code and name.
            """

def parse_and_validate_object(object_str: str) -> tuple[str, Optional[str], Optional[str], bool]:
    if not isinstance(object_str, str):
        return str(object_str), None, None, False
    
    # regex to match [IUCN: code name] format
    pattern = r"^(.*?)\s*\[IUCN:\s*([\d\.]+)\s*(.*?)\]$"
    match = re.match(pattern, object_str, re.DOTALL)
    if match:
        description = match.group(1).strip()
        code = match.group(2).strip()
        name = match.group(3).strip()
        
        # check code format
        if re.match(r"^\d+(\.\d+)?$", code):
            return description, code, name if name else None, True 
        else:
            return object_str.strip(), None, None, False
    else:
        return object_str.strip(), None, None, False


def cache_enriched_triples(triplets: List[Tuple[str, str, str, str]], llm_taxonomy_map_by_original_name: Dict[str, Dict], output_dir: Path) -> None:
    output_path = output_dir
    output_path.mkdir(exist_ok=True)
    
    # build lookup for bird taxonomy data
    canon_llm_taxo = {}
    for _original_name, tax_data in llm_taxonomy_map_by_original_name.items():
        if tax_data.get('is_bird', False):
            canonical_form = tax_data.get('canonical_form')
            if canonical_form: 
                canon_llm_taxo[canonical_form] = tax_data

    triplets_to_json = []
    for canonical_subject, predicate, obj, doi_val in triplets:
        subject_taxo = canon_llm_taxo.get(canonical_subject, {
            'error': f'No taxonomy for: {canonical_subject}',
            'is_bird': True
        })
        
        triplets_to_json.append({
            'subject': canonical_subject,
            'predicate': predicate,
            'object': obj,
            'doi': doi_val,
            'taxonomy': subject_taxo
        })
    
    # filter for birds only
    filtered_taxo_info = {
        original_name: tax_data
        for original_name, tax_data in llm_taxonomy_map_by_original_name.items()
        if tax_data.get('is_bird', False)
    }

    enriched_data = {
        'triplets': triplets_to_json,
        'taxonomic_info': filtered_taxo_info
    }
    
    with open(output_path / "enriched_triplets.json", "w", encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2)
    
    print("Enriched triplets saved to enriched_triplets.json")

    # separate file for taxonomies
    if filtered_taxo_info:
        with open(output_path / "llm_bird_taxonomies.json", "w", encoding='utf-8') as f:
            json.dump(filtered_taxo_info, f, indent=2)
        print("Bird taxonomies saved to llm_bird_taxonomies.json")
    else:
        print("No bird taxonomies to save")
