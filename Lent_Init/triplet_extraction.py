import asyncio
import json
import logging
from typing import List, Tuple, Optional, Dict
import hashlib
import pickle
from pathlib import Path
from thefuzz import fuzz
import sys
import os
from .llm_api_utility import llm_generate

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


logger = logging.getLogger("pipeline")

# Based on the knowledge graph paper, they did a multi step process of getting a summary with key facts and a title then extracting triples
# Following that, this asks the llm to generate a summary, though more basic, and send it off to generate triples
async def convert_to_summary(abstract: str, llm_setup) -> str:
    cache_result = llm_setup["cache"].get(abstract, "summary")
    if cache_result:
        return cache_result

    # maybe overkill but works well enough
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
    9. Do not include phrases like "spp." or number of species
    10. Each species or taxonomic group should not be a phrase
    Summarize this scientific abstract focusing on specific species and their threats. 
     
    Format the summary with clear sections:
    - Species Affected
    - Threat Mechanisms
    - Specific Impacts
    - Causal Relationships
    
    Be specific and detailed about the mechanisms described."""

    try:
        summary_response = await llm_generate(
            prompt=f"Text to summarize:\n{abstract}\n\nStructured Summary:",
            system=system_prompt,
            model=llm_setup["model"],
            temp=0.1,
            timeout=120,
            llm_setup=llm_setup
        )
        
        summary = summary_response.strip()
        
        if len(summary) < 50:
            logger.warning("summary looks too short")
            return ""
            
        llm_setup["cache"].set(abstract, "summary", summary)
        return summary
        
    except Exception as e:
        logger.error(f"summary generation failed: {e}")
        return ""


# ollama structured output link: https://ollama.com/blog/structured-outputs#:~:text=Ollama%20now%20supports%20structured%20outputs,Parsing%20data%20from%20documents
async def extract_triplets(summary: str, llm_setup, doi: str) -> List[Tuple[str, str, str, str]]:
    logger.info("Generating triplets (bypassing triplet cache)...")

    try:
        # Step 1: Get species from summary
        logger.info("1: Extracting species from summary...")
        
        species_schema = {
            "type": "array",
            "items": {
                "type": "object", 
                "properties": {
                    "name": {"type": "string"},
                    "scientific_name": {"type": "string"},
                    "confidence": {"type": "string"}
                },
                "required": ["name", "confidence"]
            }
        }
        
        species_system_prompt = """
        Extract all specific species or taxonomic groups mentioned in the text.

    Rules:
        1. Only include species or taxonomic groups that are DIRECTLY mentioned in the text
        2. Keep scientific names exactly as written
        3. Each entry must be a single species or specific taxonomic group
        4. Never combine multiple species into one entry (e.g., not "# bird species")
        5. Remove any qualifiers like "spp." or species counts
        6. If a scientific name is provided in the text, include it
        7. Assign a confidence level (high, medium, low) based on how clearly the species is mentioned
        """
        
        species_prompt = f"Extract all species or taxonomic groups mentioned in this text:\n\n{summary}"
        
        species_response = await llm_generate(
            prompt=species_prompt,
            system=species_system_prompt,
            model=llm_setup["species_model"],
            temp=0.1,
            format=species_schema,
            llm_setup=llm_setup
        )
        
        species_list = []
        try:
            parsed_json = json.loads(species_response)
            if isinstance(parsed_json, dict) and "value" in parsed_json and isinstance(parsed_json["value"], list):
                species_data_actual = parsed_json["value"]
            elif isinstance(parsed_json, list):
                species_data_actual = parsed_json
            else:
                logger.error(f"Unexpected JSON structure for species: {type(parsed_json)}")
                species_data_actual = []

            for s_item in species_data_actual:
                if isinstance(s_item, dict) and s_item.get('confidence', '').lower() != 'low':
                    species_list.append(s_item['name'])
                    
        except json.JSONDecodeError as e_json:
            logger.error(f"Error parsing species JSON: {e_json}")
            json_start = species_response.find('[')
            json_end = species_response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                try:
                    species_json = species_response[json_start:json_end]
                    species_data = json.loads(species_json)
                    
                    species_list = []
                    for s in species_data:
                        if isinstance(s, dict) and 'name' in s and s.get('confidence', '').lower() != 'low':
                            species_list.append(s['name'])
                except:
                    species_list = []
                    for line in species_response.split('\n'):
                        if '*' in line:
                            species = line.split('*')[1].strip()
                            if species and len(species) > 2:
                                species_list.append(species)
        
        if not species_list:
            logger.info("No species found in the summary.")
            return []
        
        logger.info(f"Extracted {len(species_list)} species:")
        for i, species in enumerate(species_list, 1):
            logger.info(f"{i}. {species}")
        
        # Step 2: Find threats for each species
        logger.info("2: Identifying threats for each species...")
        
        threats_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "species_name": {"type": "string"},
                    "threats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "threat_description": {"type": "string"},
                                "confidence": {"type": "string"}
                            },
                            "required": ["threat_description", "confidence"]
                        }
                    }
                },
                "required": ["species_name", "threats"]
            }
        }
        
        threats_system_prompt = """
        For each species mentioned in the text, identify the specific NEGATIVE threats, stressors, or CAUSES OF HARM described as impacting them.

        **Rules:**
        1. Focus ONLY on factors that HARM or NEGATIVELY impact the species.
        2. Extract the *specific description of the threat or stressor* (e.g., "drowning in oil pits", "habitat loss from logging", "increasing shoreline development", "competition from invasive species").
        3. **DO NOT extract protective factors or beneficial conditions** (e.g., do not extract "protected by vegetated shorelines").
        4. Only include threats DIRECTLY mentioned as impacting the species in the text.
        5. Do NOT attempt to classify the threat using IUCN categories here.
        6. Assign a confidence level (high, medium, low) based on how clearly the text links the threat description to the species.

        **Output Format:** Respond with ONLY a valid JSON array matching the required schema.
        """
        
        threats_prompt = f"Identify threats for each species mentioned in this text:\n\n{summary}\n\nSpecies list: {json.dumps(species_list)}"
        
        threats_response = await llm_generate(
            prompt=threats_prompt,
            system=threats_system_prompt,
            model=llm_setup["threat_model"],
            temp=0.1,
            format=threats_schema,
            llm_setup=llm_setup
        )
        
        threats_data_parsed = None
        try:
            threats_data_parsed = json.loads(threats_response)
        except Exception as e:
            logger.error(f"error parsing threats JSON with schema: {e}. raw response: '{threats_response}'")

        species_threat_pairs = []
        threats_list_to_process = []

        if isinstance(threats_data_parsed, list):
            threats_list_to_process = threats_data_parsed
            logger.info(f"Received list of {len(threats_list_to_process)} species entries.")
        elif isinstance(threats_data_parsed, dict):
            logger.warning("Received single dict instead of list for threats. Wrapping in list.")
            
            if "species" in threats_data_parsed and isinstance(threats_data_parsed["species"], list):
                logger.info(f"Found alternative format with 'species' key containing {len(threats_data_parsed['species'])} species.")
                converted_list = []
                for species_item in threats_data_parsed["species"]:
                    if isinstance(species_item, dict):
                        converted_species = {
                            "species_name": species_item.get("name", ""),
                            "threats": []
                        }
                        for threat_entry in species_item.get("threats", []):
                            if isinstance(threat_entry, dict):
                                converted_threat = {
                                    "threat_description": threat_entry.get("description", ""),
                                    "confidence": threat_entry.get("confidence", "low")
                                }
                                converted_species["threats"].append(converted_threat)
                            elif isinstance(threat_entry, str) and threat_entry.strip():
                                threat_text = threat_entry.strip()
                                if threat_text.lower() != "unknown":
                                    converted_threat = {
                                        "threat_description": threat_text,
                                        "confidence": "medium"
                                    }
                                    converted_species["threats"].append(converted_threat)
                        
                        if converted_species["threats"]:
                            converted_list.append(converted_species)
                
                threats_list_to_process = converted_list
                logger.info(f"Converted to {len(threats_list_to_process)} standardized species entries.")
                
                if not threats_list_to_process:
                    logger.info("No valid species with threats found. Skipping this abstract.")
                    return []
            else:
                threats_list_to_process = [threats_data_parsed]
        else:
            logger.warning(f"Got {type(threats_data_parsed)}, can't processthreats.")
             
        for species_threat in threats_list_to_process:
            if not isinstance(species_threat, dict):
                logger.warning(f"error in format: got {type(species_threat)}. skipping item: {species_threat}")
                continue 
                
            species_name = species_threat.get("species_name", "")
            threats_inner_list = species_threat.get("threats", [])
            
            if not isinstance(threats_inner_list, list):
                logger.warning(f"error in format: got {type(threats_inner_list)}. skipping threats for {species_name}")
                continue
                
            if not threats_inner_list:
                logger.info(f"empty threats list for species: {species_name}. skipping.")
                continue
            
            for threat_detail in threats_inner_list:
                if isinstance(threat_detail, dict):
                    confidence = threat_detail.get("confidence", "").lower()
                    if confidence == "low":
                        continue
                        
                    threat_desc = threat_detail.get("threat_description")
                    if not threat_desc:
                        continue
                        
                    if species_name and threat_desc:
                        species_threat_pairs.append({
                             "species": species_name,
                             "threat": threat_desc, 
                        })
                else:
                    logger.warning(f"Expected dict for threat, got {type(threat_detail)}: {str(threat_detail)[:50]}")
                    
        if not species_threat_pairs:
            logger.info("No valid species-threat pairs were identified. Skipping this abstract.")
            return []
        
        logger.info(f"Found {len(species_threat_pairs)} potential species-threat pairs:")
        for i, pair in enumerate(species_threat_pairs, 1):
            logger.info(f"{i}. {pair['species']} potentially affected by '{pair['threat']}'")
        
        # Step 3: Get impact mechanisms for each species-threat pair
        logger.info("STAGE 3: Determining impact mechanisms...")
        
        impacts_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "species_name": {"type": "string"},
                    "threat_name": {"type": "string"},
                    "mechanisms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "confidence": {"type": "string"}
                            },
                            "required": ["description", "confidence"]
                        }
                    }
                },
                "required": ["species_name", "threat_name", "mechanisms"]
            }
        }
        
        impacts_system_prompt = """
        For each species-threat pair provided, identify the specific NEGATIVE impact mechanism described in the text. Focus on HOW the threat DIRECTLY HARMS the species.

        Rules:
        1. Describe the harmful consequence CAUSED BY the threat. Do NOT describe the benefits of habitat or resources that are lost or affected.
        2. Focus ONLY on the negative impact mechanism (e.g., 'reduces nesting success', 'causes poisoning', 'increases predation risk', 'blocks migration route').
        3. Include specific biological, physiological, or ecological processes involved in the harm.
        4. Include quantitative measures of the negative impact when available (e.g., "reduces breeding success by 45%").
        5. Provide direct evidence or strong inference from the text for the mechanism.
        6. Assign a confidence level (high, medium, low) based on how clearly the negative impact mechanism is described.
        7. If multiple distinct negative mechanisms exist for the same species-threat pair, list them separately.

        Example:
        - Text mentions: "Shoreline development leads to loss of vegetated nesting sites crucial for Wood Ducks."
        - Threat (from Stage 2): Shoreline development
        - Species: Wood Ducks
        - Correct Mechanism: "loss of crucial vegetated nesting sites" or "reduces availability of nesting habitat"
        - Incorrect Mechanism: "benefit from vegetated nesting sites"
        """
        
        pair_strings = []
        for pair in species_threat_pairs:
            pair_strings.append(f"{pair['species']} - {pair['threat']}")
            
        impacts_prompt = f"Identify how each threat affects each species in this text:\n\n{summary}\n\nPairs to analyze: {json.dumps(pair_strings)}"
        
        impacts_response = await llm_generate(
            prompt=impacts_prompt,
            system=impacts_system_prompt,
            model=llm_setup["impact_model"],
            temp=0.1,
            format=impacts_schema,
            llm_setup=llm_setup
        )
        
        impacts_data_parsed_list = []
        try:
            parsed_json = json.loads(impacts_response)
            if isinstance(parsed_json, dict) and "items" in parsed_json and isinstance(parsed_json["items"], list):
                logger.info("Impacts JSON is a dict with an 'items' key containing the data list.")
                impacts_data_parsed_list = parsed_json["items"]
            elif isinstance(parsed_json, dict) and "value" in parsed_json and isinstance(parsed_json["value"], list):
                logger.info("Impacts JSON is a dict with a 'value' key containing the data list.")
                impacts_data_parsed_list = parsed_json["value"]
            elif isinstance(parsed_json, list):
                logger.info("Impacts JSON is a direct list of data.")
                impacts_data_parsed_list = parsed_json
            else:
                logger.error(f"unexpected JSON structure: {type(parsed_json)}. raw response: {impacts_response}")
        except json.JSONDecodeError as e_json:
            logger.error(f"error parsing impacts JSON: {e_json}")
        except Exception as e_general:
            logger.error(f"error processing impacts data: {e_general}")

        if not impacts_data_parsed_list and species_threat_pairs: 
            logger.warning("Impacts data parsing failed, creating fallback structure.")
            temp_fallback_list = []
            for pair in species_threat_pairs:
                temp_fallback_list.append({
                    "species_name": pair["species"],
                    "threat_name": pair["threat"],
                    "mechanisms": [
                        {
                            "description": f"negatively impacts {pair['species']} population",
                            "confidence": "medium"
                        }
                    ]
                })
            impacts_data_parsed_list = temp_fallback_list
        
        logger.info("final stage: assembling triplets...")
        
        raw_triplets = []
        for impact_item in impacts_data_parsed_list: 
            if not isinstance(impact_item, dict):
                logger.warning(f"Skipping non-dict item during triplet assembly: {impact_item}")
                continue 
            species = impact_item.get("species_name", "")
            threat_obj_desc_only = impact_item.get("threat_name", "") 
            
            # Create triplet for each mechanism
            for mechanism in impact_item.get("mechanisms", []):
                if isinstance(mechanism, dict) and mechanism.get("confidence", "").lower() != "low":
                    predicate = mechanism.get("description", "")
                    if species and predicate and threat_obj_desc_only:
                        raw_triplets.append((species, predicate, threat_obj_desc_only, doi))
        
        logger.info("Extracted Raw Triplets (before refinement/consolidation):")
        for i, (subject, predicate, obj, d) in enumerate(raw_triplets, 1):
            logger.info(f"{i}. {subject} | {predicate} | {obj} | DOI: {d}")
        
        # Consolidate similar triplets
        consolidated_triplets = consolidate_triplets(raw_triplets)
        
        logger.info("Consolidated Raw Triplets:")
        for subject, predicate, obj, d in consolidated_triplets:
            logger.info(f"• {subject} | {predicate} | {obj} (DOI: {d})")
        logger.info(f"Number of raw triplets: {len(consolidated_triplets)}")
        
        return consolidated_triplets 
        
    except Exception as e:
        logger.error(f"ERROR extracting triplets: {e}")
        return []

# Extract entities (species & threats) from abstract in single call
async def extract_entities_concurrently(abstract_text: str, llm_setup) -> Optional[Dict[str, List[str]]]:
    logger.info(f"P2.1: Extracting entities for abstract starting: {abstract_text[:50]}...")
    
    entity_extraction_schema = {
        "type": "object",
        "properties": {
            "species": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific species or taxonomic groups mentioned"
            },
            "threats": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "List of distinct threat phrases or negative impacts"
            }
        },
        "required": ["species", "threats"]
    }
    
    original_species_system_prompt = """
        Extract all specific species or taxonomic groups mentioned in the text.
        Only select species that are victims of threats mentioned in this abstract and strictly follow the rules below.

    Rules:
        1. Only include species or taxonomic groups that are DIRECTLY mentioned in the text
        2. Keep scientific names exactly as written
        3. Each entry must be a single species or specific taxonomic group
        4. Never combine multiple species into one entry (e.g., not "# bird species")
        5. Remove any qualifiers like "spp." or species counts
        6. If a scientific name is provided in the text, include it
        7. Assign a confidence level (high, medium, low) based on how clearly the species is mentioned
        """
    
    general_threat_extraction_rules = """
        Based on the abstract, identify all distinct phrases describing specific NEGATIVE threats, stressors, or CAUSES OF HARM.
        Read the entire abstract to identify the most impactful threat to the ENTIRETY of this species. Only select a threat if it impacts the species as a whole. 
        Find threats from observation not just lab experimentation with things like diets and confinement.
        **Key Principle: A threat is the fundamental CAUSE or DRIVER of harm, not the symptom or consequence of that harm.**
        Strictly follow the rules below:
        **Rules for Threat Extraction:**
        1. Focus ONLY on factors that CAUSE HARM or NEGATIVELY impact species generally described in the abstract. The threat should be the *origin* of the negative effect.
        2. Extract the *specific description of the CAUSE of harm or stressor* (e.g., "drowning in oil pits", "habitat loss from logging", "increasing shoreline development", "competition from invasive species", "severe aspergillosis", "mercury (Hg) exposure", "higher temperatures").
        3. **DO NOT extract symptoms, effects, or consequences as threats.** For example, if a species "suffers mortality due to illegal hunting", the threat is "illegal hunting", NOT "mortality". If a species "experiences habitat loss leading to population decline", the threat is "habitat loss", NOT "population decline".
        4. Only include threats DIRECTLY mentioned in the text as the *cause* of a negative outcome.
        5. Do NOT attempt to classify the threat using IUCN categories here.
        6. Do not try to link these threats to specific species *yet*. That will be a subsequent step.

        **Examples of Incorrect vs. Correct Threat Identification based on provided triplets:**

        *   **Context:** "Ostrich suffer depression as a symptom of severe aspergillosis"
            *   **Incorrect Threat Identification:** "depression" (This is a symptom)
            *   **Correct Threat Identification:** "severe aspergillosis" (This is the cause of the symptom)

        *   **Context:** "Little Tern faces increased risk of overheating of eggs due to a compromise between thermal protection and camouflage, resulting from breeding later in the season when temperatures are higher"
            *   **Incorrect Threat Identification:** "overheating of eggs" (This is a consequence/effect)
            *   **Correct Threat Identification:** "breeding later in the season when temperatures are higher" or "higher temperatures" (This is the cause)

        *   **Context:** "Songbird experience impaired avian health due to mercury (Hg) exposure."
            *   **Incorrect Threat Identification:** "impairs avian health" (This is a consequence/effect)
            *   **Correct Threat Identification:** "mercury (Hg) exposure" (This is the cause)

        *   **Context:** "Passerine show altered food distribution patterns characterized by parental preference for senior offspring under food limitation..."
            *   **Correct Threat Identification:** "food limitation" (This is the cause of altered patterns)

        *   **Context:** "Northern Bobwhite experiences unreliable population density estimation due to violated assumptions..."
            *   **Note:** While "violated assumptions" leads to a problem (unreliable estimation), it's a methodological issue, not a direct environmental threat to the species' survival or well-being in the same way as predation or habitat loss. Only extract direct threats to the organism or its environment. If the text described how inaccurate estimates *led to* mismanagement causing harm, then that mismanagement could be a threat. Here, the problem is with the *estimation method itself*.
        """
    combined_system_prompt = f"""You are a scientific entity extraction expert. Perform the following two tasks based on the provided abstract:

TASK 1: SPECIES EXTRACTION
---
{original_species_system_prompt}
---
List the species found under the "species" key in your JSON output (as a list of strings).

TASK 2: THREAT EXTRACTION (General from Abstract)
---
{general_threat_extraction_rules}
---
List these general threat descriptions under the "threats" key in your JSON output (as a list of strings).

Provide your complete output *only* as a single valid JSON object matching this schema:
{json.dumps(entity_extraction_schema)}
Do not include any other explanatory text or markdown around the JSON object.
"""
    
    user_prompt = abstract_text
    try:
        response_str = await llm_generate(
            prompt=user_prompt,
            system=combined_system_prompt,
            model=llm_setup.get("model"), 
            temp=0.0, 
            format=entity_extraction_schema, 
            llm_setup=llm_setup
        )
        
        if not response_str:
            logger.error(f"P2.1: LLM returned empty response for entity extraction")
            return None
            
        entities_data = json.loads(response_str)
        
        if isinstance(entities_data, dict) and \
           isinstance(entities_data.get("species"), list) and \
           isinstance(entities_data.get("threats"), list):
            if all(isinstance(s, str) for s in entities_data.get("species")) and \
               all(isinstance(t, str) for t in entities_data.get("threats")):
                logger.info(f"P2.1: Successfully extracted {len(entities_data['species'])} species and {len(entities_data['threats'])} threats.")
                return entities_data
            else:
                logger.error(f"P2.1: Extracted lists contain non-string elements")
                return None
        elif isinstance(entities_data, dict) and "value" in entities_data and isinstance(entities_data["value"], dict):
            actual_data = entities_data["value"]
            if isinstance(actual_data.get("species"), list) and \
               isinstance(actual_data.get("threats"), list) and \
               all(isinstance(s, str) for s in actual_data.get("species")) and \
               all(isinstance(t, str) for t in actual_data.get("threats")):
                logger.info(f"P2.1: Successfully extracted {len(actual_data['species'])} species and {len(actual_data['threats'])} threats (from 'value' key).")
                return actual_data
            else:
                logger.error(f"P2.1: Unexpected structure in 'value' key")
                return None
                
        logger.error(f"P2.1: Unexpected JSON structure from entity extraction")
        return None
        
    except json.JSONDecodeError as e_json:
        logger.error(f"P2.1: JSONDecodeError in entity extraction: {e_json}")
        return None
    except Exception as e:
        logger.error(f"P2.1: Error in extract_entities_concurrently: {e}")
        return None

async def generate_relationships_concurrently(abstract_text: str, species_list: List[str], threats_list: List[str], llm_setup, doi: str) -> List[Tuple[str, str, str, str]]:
    logger.info(f"P2.2: Generating relationships for DOI: {doi}, {len(species_list)} species, {len(threats_list)} threats")
    
    if not species_list or not threats_list:
        logger.warning(f"P2.2: Missing species or threats list for DOI {doi}. Skipping relationship generation.")
        return []
        
    relationship_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Species name from provided list"},
                "predicate": {"type": "string", "description": "Relationship/impact mechanism linking subject and object"},
                "object": {"type": "string", "description": "Threat description from provided list"}
            },
            "required": ["subject", "predicate", "object"]
        }
    }
    system_prompt = (
        """You are a specialized linguistic model. Your sole task is to generate a **Predicate** phrase. 
You will be given:
1. An Abstract (the source text).
2. A Subject (a specific species name from the abstract).
3. An Object (a specific threat phrase from the abstract, which is understood to be the CAUSE of harm).
The relationship should describe only species-wide effects from the threat. Read the entire abstract to get additional context when establishing the relationship between the species and threats. 
Do not be vague or redundant and ensure the relation is a FULL PHRASE, and follow these rules strictly:

You generated Predicate must:
A. Clearly and concisely describe HOW the Subject (species) is specifically affected by, interacts with, or what the direct consequence/symptom for the Subject IS, as a result of the provided Object (threat). This information MUST be derived ONLY from the abstract.
B. **CRITICALLY: The Predicate MUST NOT restate or include the exact text of the provided 'Object (Threat)'.** The Predicate's role is to bridge the gap between the Subject and the Object by describing the *effect* or *mechanism* of the Object on the Subject.
C. Be phrased so that when combined, `(Subject) (Your Generated Predicate) (Object)` forms a grammatically correct and meaningful sentence that accurately reflects the relationship described in the abstract.
D. Focus on extracting the specific impact, mechanism of interaction, or observed effect. Avoid generic phrases like 'is affected by' or 'is impacted by' if more specific information is available.
E. If the abstract does not provide a clear, specific mechanism or impact connecting the Subject to the *given* Object in a way that allows for a predicate distinct from the Object itself, return an empty string or a concise phrase like 'experiences'.

**Examples of How to Form the Predicate (focus on NOT restating the Object):**

1.  **Abstract Snippet:** \"Ostrich suffer depression as a symptom of severe aspergillosis\"
    *   **Given Subject:** Ostrich
    *   **Given Object (Threat):** severe aspergillosis
    *   **Your Generated Predicate:** \"suffer depression as a symptom of\"
    *   *(Resulting Triplet formed by your system: Ostrich suffer depression as a symptom of severe aspergillosis)*
    *   **Incorrect Predicate (restates part/all of object):** \"is made ill by severe aspergillosis\"

2.  **Abstract Snippet:** \"Little Tern faces increased risk of overheating of eggs due to a compromise between thermal protection and camouflage, resulting from breeding later in the season when temperatures are higher\"
    *   **Given Subject:** Little Tern
    *   **Given Object (Threat):** higher temperatures
    *   **Your Generated Predicate:** \"faces increased risk of overheating of eggs due to a compromise between thermal protection and camouflage, resulting from breeding later in the season due to\"
    *   *(Resulting Triplet: Little Tern faces increased risk of overheating of eggs... due to higher temperatures)*

3.  **Abstract Snippet:** \"Songbird experience impaired avian health due to mercury (Hg) exposure.\"
    *   **Given Subject:** Songbird
    *   **Given Object (Threat):** mercury (Hg) exposure
    *   **Your Generated Predicate:** \"experience impaired avian health due to\"
    *   *(Resulting Triplet: Songbird experience impaired avian health due to mercury (Hg) exposure)*

4.  **Abstract Snippet:** \"Passerine show altered food distribution patterns characterized by parental preference for senior offspring under food limitation, potentially affecting the survival and development of junior offspring.\"
    *   **Given Subject:** Passerine
    *   **Given Object (Threat):** food limitation
    *   **Your Generated Predicate:** \"show altered food distribution patterns characterized by parental preference for senior offspring under\"
    *   *(Resulting Triplet: Passerine show altered food distribution patterns... under food limitation)*

5.  **Abstract Snippet:** \"Bird suffers mortality from direct strikes with and experiences population decline due to illegal spring killing.\"
    *   **Given Subject:** Bird
    *   **Given Object (Threat):** illegal spring killing
    *   **Your Generated Predicate:** \"suffers mortality from direct strikes with and experiences population decline due to\"
    *   *(Resulting Triplet: Bird suffers mortality... due to illegal spring killing)*

Provide ONLY the predicate string as your output. Do not include 'Predicate:' or any other explanatory text."""
        )
    user_prompt = f"""Abstract:
{abstract_text}

Identified Species:
{json.dumps(species_list)}

Identified Threats:
{json.dumps(threats_list)}

Extract relationship triplets based on the abstract, linking species to threats (ensure output is ONLY the JSON array):
"""

    raw_triplets = []
    try:
        response_str = await llm_generate(
            prompt=user_prompt,
            system=system_prompt,
            model=llm_setup.get("model"), 
            temp=0.0,
            format=relationship_schema,
            llm_setup=llm_setup
        )
        
        if not response_str:
            logger.error(f"P2.2: LLM returned empty response for relationship generation. DOI: {doi}")
            return []
            
        relationships_data = json.loads(response_str)
        
        if isinstance(relationships_data, list):
            for rel in relationships_data:
                if isinstance(rel, dict):
                    subject = rel.get("subject")
                    predicate = rel.get("predicate")
                    obj_threat = rel.get("object")
                    if subject and predicate and obj_threat and subject in species_list and obj_threat in threats_list:
                        raw_triplets.append((subject, predicate, obj_threat, doi))
                    else:
                        logger.warning(f"P2.2: Dropping invalid triplet: {rel}. DOI: {doi}")
                else:
                    logger.warning(f"P2.2: Expected dict in relationships list, got {type(rel)}. DOI: {doi}")
            logger.info(f"P2.2: Successfully parsed {len(raw_triplets)} relationships for DOI: {doi}.")
            
        elif isinstance(relationships_data, dict) and "value" in relationships_data and isinstance(relationships_data["value"], list):
            logger.info("P2.2: Relationships JSON has 'value' key with data list.")
            actual_data_list = relationships_data["value"]
            for rel in actual_data_list:
                if isinstance(rel, dict):
                    subject = rel.get("subject")
                    predicate = rel.get("predicate")
                    obj_threat = rel.get("object")
                    if subject and predicate and obj_threat and subject in species_list and obj_threat in threats_list:
                        raw_triplets.append((subject, predicate, obj_threat, doi))
                    else:
                        logger.warning(f"P2.2: Dropping invalid triplet from 'value' list: {rel}. DOI: {doi}")
                else:
                    logger.warning(f"P2.2: Expected dict in 'value' relationships list, got {type(rel)}. DOI: {doi}")
            logger.info(f"P2.2: Successfully parsed {len(raw_triplets)} relationships from 'value' key for DOI: {doi}.")
        else:
            logger.error(f"P2.2: Unexpected JSON structure for relationships. Expected list or dict with 'value'. Got {type(relationships_data)}. Raw: '{response_str}'. DOI: {doi}")
    except json.JSONDecodeError as e_json:
        logger.error(f"P2.2: JSONDecodeError in relationship generation: {e_json}. Raw: '{response_str}'. DOI: {doi}")
    except Exception as e:
        logger.error(f"P2.2: Error in generate_relationships_concurrently for DOI {doi}: {e}", exc_info=True)
    return raw_triplets


# ollama structured output link: https://ollama.com/blog/structured-outputs#:~:text=Ollama%20now%20supports%20structured%20outputs,Parsing%20data%20from%20documents
async def extract_triplets(summary: str, llm_setup, doi: str) -> List[Tuple[str, str, str, str]]:
    # Cache check commented out to force regeneration (as per previous request)
    # cached = llm_setup["cache"].get(summary, "triplets")
    # if cached:
    #     return cached

    logger.info("generating triplets...")

    try:
        # Step 1: get species
        logger.info("step 1: finding species...")
        
        # Define schema for species extraction
        species_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "scientific_name": {"type": "string"},
                    "confidence": {"type": "string"}
                },
                "required": ["name", "confidence"]
            }
        }
        
        species_system_prompt = """
        Extract all specific species or taxonomic groups mentioned in the text.

    Rules:
        1. Only include species or taxonomic groups that are DIRECTLY mentioned in the text
        2. Keep scientific names exactly as written
        3. Each entry must be a single species or specific taxonomic group
        4. Never combine multiple species into one entry (e.g., not "# bird species")
        5. Remove any qualifiers like "spp." or species counts
        6. If a scientific name is provided in the text, include it
        7. Assign a confidence level (high, medium, low) based on how clearly the species is mentioned
        """
        
        species_prompt = f"Extract all species or taxonomic groups mentioned in this text:\n\n{summary}"
        
        # Stage 1: Species extraction with schema-based formatting
        species_response = await llm_generate(
            prompt=species_prompt,
            system=species_system_prompt,
            model=llm_setup["species_model"],
            temp=0.1,
            format=species_schema, # This tells the LLM the schema we want for its *output value*
            llm_setup=llm_setup
        )
        
        species_list = []
        try:
            parsed_json = json.loads(species_response)
            # Check if the response is the schema-plus-value structure
            if isinstance(parsed_json, dict) and "value" in parsed_json and isinstance(parsed_json["value"], list):
                species_data_actual = parsed_json["value"]
            # Check if the response is directly a list (ideal case)
            elif isinstance(parsed_json, list):
                species_data_actual = parsed_json
            else:
                logger.error(f"Unexpected JSON structure for species. Expected list or dict with 'value' key. Got: {type(parsed_json)}. Raw response was: {species_response}")
                species_data_actual = []

            for s_item in species_data_actual:
                if isinstance(s_item, dict) and s_item.get('confidence', '').lower() != 'low':
                    species_list.append(s_item['name'])
                elif not isinstance(s_item, dict):
                    logger.warning(f"Skipping non-dict item in species_data: {s_item}")
                    
        except json.JSONDecodeError as e_json:
            logger.error(f"Error parsing species JSON (JSONDecodeError): {e_json}. Raw response: '{species_response}'")
        except Exception as e_general: # Catch other potential errors like AttributeError if parsing was wrong
            logger.error(f"Error processing species data: {e_general}. Raw response: '{species_response}'")
            # Fallback (already present but good to be aware of)
            json_start = species_response.find('[')
            json_end = species_response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                try:
                    species_json = species_response[json_start:json_end]
                    species_data = json.loads(species_json)
                    
                    # Filter out low confidence species
                    species_list = []
                    for s in species_data:
                        if isinstance(s, dict) and 'name' in s and s.get('confidence', '').lower() != 'low':
                            species_list.append(s['name'])
                except Exception:
                    # Final fallback: simple text parsing
                    species_list = []
                    for line in species_response.split('\n'):
                        if '*' in line:
                            species = line.split('*')[1].strip()
                            if species and len(species) > 2:
                                species_list.append(species)
            else:
                # Fallback: simple text parsing
                species_list = []
                for line in species_response.split('\n'):
                    if ':' in line and 'species' not in line.lower() and 'name' not in line.lower():
                        species = line.split(':')[1].strip()
                        if species and len(species) > 2:
                            species_list.append(species)
        
        if not species_list:
            logger.info("no species found")
            return []
        
        logger.info(f"found {len(species_list)} species:")
        for i, species in enumerate(species_list, 1):
            logger.info(f"{i}. {species}")
        
        # Step 2: find threats
        logger.info("step 2: finding threats...")
        
        threats_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "species_name": {"type": "string"},
                    "threats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "threat_description": {"type": "string"},
                                "confidence": {"type": "string"}
                            },
                            "required": ["threat_description", "confidence"]
                        }
                    }
                },
                "required": ["species_name", "threats"]
            }
        }
        
        # Simplified system prompt for Stage 2
        threats_system_prompt = """
        For each species mentioned in the text, identify the specific NEGATIVE threats, stressors, or CAUSES OF HARM described as impacting them.

        **Rules:**
        1. Focus ONLY on factors that HARM or NEGATIVELY impact the species.
        2. Extract the *specific description of the threat or stressor* (e.g., "drowning in oil pits", "habitat loss from logging", "increasing shoreline development", "competition from invasive species").
        3. **DO NOT extract protective factors or beneficial conditions** (e.g., do not extract "protected by vegetated shorelines").
        4. Only include threats DIRECTLY mentioned as impacting the species in the text.
        5. Do NOT attempt to classify the threat using IUCN categories here.
        6. Assign a confidence level (high, medium, low) based on how clearly the text links the threat description to the species.

        **Output Format:** Respond with ONLY a valid JSON array matching the required schema.
        """
        
        threats_prompt = f"Identify threats for each species mentioned in this text:\n\n{summary}\n\nSpecies list: {json.dumps(species_list)}"
        
        # Stage 2: Threat identification with simplified schema
        threats_response = await llm_generate(
            prompt=threats_prompt,
            system=threats_system_prompt,
            model=llm_setup["threat_model"],
            temp=0.1,
            format=threats_schema,
            llm_setup=llm_setup
        )
        
        threats_data_parsed = None
        try:
            threats_data_parsed = json.loads(threats_response)
        except Exception as e:
            logger.error(f"Error parsing simplified threats JSON with schema: {e}. Raw response: '{threats_response}'")
        species_threat_pairs = []
        threats_list_to_process = []
        if isinstance(threats_data_parsed, list):
            threats_list_to_process = threats_data_parsed
            logger.info(f"got {len(threats_list_to_process)} species entries")
        elif isinstance(threats_data_parsed, dict):
            logger.warning("got dict instead of list, converting...")
            
            if "species" in threats_data_parsed and isinstance(threats_data_parsed["species"], list):
                logger.info(f"found {len(threats_data_parsed['species'])} species in alt format")
                converted_list = []
                for species_item in threats_data_parsed["species"]:
                    if isinstance(species_item, dict):
                        converted_species = {
                            "species_name": species_item.get("name", ""),
                            "threats": []
                        }
                        for threat_entry in species_item.get("threats", []):
                            if isinstance(threat_entry, dict):
                                converted_threat = {
                                    "threat_description": threat_entry.get("description", ""),
                                    "confidence": threat_entry.get("confidence", "low")
                                }
                                converted_species["threats"].append(converted_threat)
                            elif isinstance(threat_entry, str) and threat_entry.strip():
                                threat_text = threat_entry.strip()
                                if threat_text.lower() != "unknown":
                                    converted_threat = {
                                        "threat_description": threat_text,
                                        "confidence": "medium"
                                    }
                                    converted_species["threats"].append(converted_threat)
                        
                        if converted_species["threats"]:
                            converted_list.append(converted_species)
                
                threats_list_to_process = converted_list
                logger.info(f"converted to {len(threats_list_to_process)} entries")
                
                if not threats_list_to_process:
                    logger.info("no valid pairs found, skipping")
                    return []
            else:
                threats_list_to_process = [threats_data_parsed]
        else:
            logger.warning(f"unexpected data type: {type(threats_data_parsed)}")
            if threats_data_parsed is not None:
                logger.warning(f"unparseable: {str(threats_data_parsed)[:200]}")
            
        for species_threat in threats_list_to_process:
            if not isinstance(species_threat, dict):
                logger.warning(f"skipping non-dict: {species_threat}")
                continue 
                
            species_name = species_threat.get("species_name", "")
            threats_inner_list = species_threat.get("threats", [])
            
            if not isinstance(threats_inner_list, list):
                logger.warning(f"expected list for {species_name}, got {type(threats_inner_list)}")
                continue
                
            if not threats_inner_list:
                logger.info(f"empty threats for {species_name}")
                continue
            
            threats_found = 0
            
            for threat_detail in threats_inner_list:
                if isinstance(threat_detail, dict):
                    confidence = threat_detail.get("confidence", "").lower()
                    if confidence == "low":
                        continue
                        
                    threat_desc = threat_detail.get("threat_description")
                    if not threat_desc:
                        continue
                        
                    if species_name and threat_desc:
                        threats_found += 1
                        species_threat_pairs.append({
                             "species": species_name,
                             "threat": threat_desc, 
                        })
                else:
                    logger.warning(f"bad threat format: {str(threat_detail)[:50]}")
                    
        if not species_threat_pairs:
            logger.info("no valid pairs found")
            return []
        
        logger.info(f"found {len(species_threat_pairs)} species-threat pairs:")
        for i, pair in enumerate(species_threat_pairs, 1):
            logger.info(f"{i}. {pair['species']} vs {pair['threat']}")
        
        # Step 3: get impact mechanisms
        logger.info("step 3: finding impact mechanisms...")
        
        # Define schema for impact mechanisms
        impacts_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "species_name": {"type": "string"},
                    "threat_name": {"type": "string"},
                    "mechanisms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "confidence": {"type": "string"}
                            },
                            "required": ["description", "confidence"]
                        }
                    }
                },
                "required": ["species_name", "threat_name", "mechanisms"]
            }
        }
        
        impacts_system_prompt = """
        For each species-threat pair provided, identify the specific NEGATIVE impact mechanism described in the text. Focus on HOW the threat DIRECTLY HARMS the species.

        Rules:
        1. Describe the harmful consequence CAUSED BY the threat. Do NOT describe the benefits of habitat or resources that are lost or affected.
        2. Focus ONLY on the negative impact mechanism (e.g., 'reduces nesting success', 'causes poisoning', 'increases predation risk', 'blocks migration route').
        3. Include specific biological, physiological, or ecological processes involved in the harm.
        4. Include quantitative measures of the negative impact when available (e.g., "reduces breeding success by 45%").
        5. Provide direct evidence or strong inference from the text for the mechanism.
        6. Assign a confidence level (high, medium, low) based on how clearly the negative impact mechanism is described.
        7. If multiple distinct negative mechanisms exist for the same species-threat pair, list them separately.

        Example:
        - Text mentions: "Shoreline development leads to loss of vegetated nesting sites crucial for Wood Ducks."
        - Threat (from Stage 2): Shoreline development
        - Species: Wood Ducks
        - Correct Mechanism: "loss of crucial vegetated nesting sites" or "reduces availability of nesting habitat"
        - Incorrect Mechanism: "benefit from vegetated nesting sites"
        """
        
        # Prepare the pairs for the prompt
        pair_strings = []
        for pair in species_threat_pairs:
            pair_strings.append(f"{pair['species']} - {pair['threat']}")
            
        # Define the impacts prompt
        impacts_prompt = f"Identify how each threat affects each species in this text:\n\n{summary}\n\nPairs to analyze: {json.dumps(pair_strings)}"
        
        # Stage 3: Impact analysis with schema-based formatting
        impacts_response = await llm_generate(
            prompt=impacts_prompt,
            system=impacts_system_prompt,
            model=llm_setup["impact_model"],
            temp=0.1,
            format=impacts_schema,
            llm_setup=llm_setup
        )
        
        impacts_data_parsed_list = []
        try:
            parsed_json = json.loads(impacts_response)
            # Scenario 1: LLM returns the schema definition AND the data array under a duplicate "items" key
            if isinstance(parsed_json, dict) and "items" in parsed_json and isinstance(parsed_json["items"], list):
                logger.info("Impacts JSON is a dict with an 'items' key containing the data list.")
                impacts_data_parsed_list = parsed_json["items"]
            # Scenario 2: LLM returns a dict with a "value" key containing the data list (like species sometimes does)
            elif isinstance(parsed_json, dict) and "value" in parsed_json and isinstance(parsed_json["value"], list):
                logger.info("Impacts JSON is a dict with a 'value' key containing the data list.")
                impacts_data_parsed_list = parsed_json["value"]
            # Scenario 3: LLM returns the data list directly (ideal)
            elif isinstance(parsed_json, list):
                logger.info("Impacts JSON is a direct list of data.")
                impacts_data_parsed_list = parsed_json
            else:
                logger.error(f"Unexpected JSON structure for impacts. Expected list or dict with 'items' or 'value' key. Got: {type(parsed_json)}. Raw response: {impacts_response}")
        except json.JSONDecodeError as e_json:
            logger.error(f"Error parsing impacts JSON (JSONDecodeError): {e_json}. Raw response: '{impacts_response}'")
        except Exception as e_general:
            logger.error(f"impacts parsing error: {e_general}")

        if not impacts_data_parsed_list and species_threat_pairs: 
            logger.warning("impacts failed, using fallback")
            temp_fallback_list = []
            for pair in species_threat_pairs:
                temp_fallback_list.append({
                    "species_name": pair["species"],
                    "threat_name": pair["threat"],
                    "mechanisms": [
                        {
                            "description": f"negatively impacts {pair['species']} population",
                            "confidence": "medium"
                        }
                    ]
                })
            impacts_data_parsed_list = temp_fallback_list
        
        logger.info("assembling triplets...")
        
        raw_triplets = []
        for impact_item in impacts_data_parsed_list: 
            if not isinstance(impact_item, dict):
                logger.warning(f"Skipping non-dict item during triplet assembly: {impact_item}")
                continue 
            species = impact_item.get("species_name", "")
            threat_obj_desc_only = impact_item.get("threat_name", "") 
            for mechanism in impact_item.get("mechanisms", []):
                if isinstance(mechanism, dict) and mechanism.get("confidence", "").lower() != "low":
                    predicate = mechanism.get("description", "")
                    if species and predicate and threat_obj_desc_only:
                        raw_triplets.append((species, predicate, threat_obj_desc_only, doi))
        logger.info("raw triplets:")
        for i, (subject, predicate, obj, d) in enumerate(raw_triplets, 1):
            logger.info(f"{i}. {subject} | {predicate} | {obj}")
        consolidated_triplets = consolidate_triplets(raw_triplets)
        logger.info("consolidated triplets:")
        for subject, predicate, obj, d in consolidated_triplets:
            logger.info(f"• {subject} | {predicate} | {obj}")
        logger.info(f"final count: {len(consolidated_triplets)}")
        return consolidated_triplets 
        
    except Exception as e:
        logger.error(f"triplet extraction failed: {e}")
        return []


# fuzzy matching for similar terms
def are_terms_similar(term1: str, term2: str, threshold: int = 85) -> bool:
    ratio = fuzz.ratio(term1.lower(), term2.lower())
    return ratio >= threshold

# merge similar triplets
def consolidate_triplets(triplet_list: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
    if not triplet_list:
        return []
    
    consolidated = []
    processed_indices = set()
    
    for i, (subj1, pred1, obj1, doi1) in enumerate(triplet_list):
        if i in processed_indices:
            continue
            
        similar_group = [(subj1, pred1, obj1, doi1)]
        processed_indices.add(i)
        
        # find similar ones
        for j, (subj2, pred2, obj2, doi2) in enumerate(triplet_list[i+1:], i+1):
            if j in processed_indices:
                continue
                
            # Check if subjects and objects are similar
            if (are_terms_similar(subj1, subj2) and 
                are_terms_similar(obj1, obj2)):
                similar_group.append((subj2, pred2, obj2, doi2))
                processed_indices.add(j)
        
        # If we found similar triplets, combine them
        if len(similar_group) > 1:
            combined_subj = similar_group[0][0]
            combined_obj = similar_group[0][2]
            combined_doi = similar_group[0][3]
            
            # Combine unique predicates
            predicates = list(set(t[1] for t in similar_group))
            if len(predicates) > 1:
                combined_pred = " and ".join(predicates)
            else:
                combined_pred = predicates[0]
            
            consolidated.append((combined_subj, combined_pred, combined_obj, combined_doi))
            print(f"\nMerged triplets:")
            for t in similar_group:
                print(f"  {t[0]} | {t[1]} | {t[2]}")
            print(f"Into: {combined_subj} | {combined_pred} | {combined_obj}\n")
        else:
            consolidated.append((subj1, pred1, obj1, doi1))
    
    return consolidated

# normalize species names
async def normalize_species_names(triplet_list: List[Tuple[str, str, str, str]], llm_setup) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, Dict]]:
    logger.info(f"normalizing {len(triplet_list)} triplets")
    
    unique_subjects = sorted(list(set(t[0] for t in triplet_list)))
    logger.info(f"processing {len(unique_subjects)} unique species")

    normalization_schema = {
        "type": "object",
        "properties": {
            "canonical_form": {"type": "string"},
            "scientific_name": {"type": "string"},
            "kingdom": {"type": "string"},
            "phylum": {"type": "string"},
            "class": {"type": "string"},
            "order": {"type": "string"},
            "family": {"type": "string"},
            "genus": {"type": "string"},
            "is_bird": {"type": "boolean"}
        },
        "required": ["canonical_form", "is_bird"]
    }

    system_prompt = """You are a taxonomic expert. For the given species or group name:
        1. Provide the canonical form (standard, singular common name, e.g., "Mallard" for "mallards", "Bird" for "birds").
        2. Provide the scientific name if available. For specific species, this is the Latin binomial. For broader groups, it's the taxon name (e.g., "Aves" for birds).
        3. Provide the taxonomic classification (Kingdom, Phylum, Class, Order, Family, Genus) as specifically as possible based on the input.
        4. Determine if the input refers to a bird (i.e., belongs to Class Aves) and set 'is_bird' to true or false.

        Important: Only set 'is_bird' to true if the species/group belongs to Class Aves (birds).
        Respond with valid JSON matching the required schema."""

    species_taxonomy_cache = {}
    
    # Create tasks for all unique subjects
    tasks = []
    for subject in unique_subjects:
        species_for_llm = subject
        if subject.lower() == "birds":
            species_for_llm = "Bird"
        
        async def get_taxonomy_for_subject(s_name, s_llm_name):
            try:
                response_json_str = await llm_generate(
                    prompt=f"Normalize this species name: {s_llm_name}",
                    system=system_prompt,
                    model=llm_setup["model"],
                    temp=0.1,
                    format=normalization_schema,
                    llm_setup=llm_setup
                )
                if not response_json_str:
                    logger.error(f"Error normalizing '{s_name}': LLM returned empty response.")
                    return s_name, {
                        'original_query': s_name,
                        'canonical_form': s_name,
                        'is_bird': False,
                        'source': 'Fallback_empty_llm_response'
                    }
                norm_data = json.loads(response_json_str)
                is_bird = norm_data.get("is_bird", False) or (
                    norm_data.get("class") and "aves" in norm_data.get("class", "").lower()
                )
                
                return s_name, {
                    'original_query': s_name,
                    'canonical_form': norm_data.get("canonical_form", s_llm_name),
                    'scientific_name': norm_data.get("scientific_name"),
                    'kingdom': norm_data.get("kingdom"),
                    'phylum': norm_data.get("phylum"),
                    'class': norm_data.get("class"),
                    'order': norm_data.get("order"),
                    'family': norm_data.get("family"),
                    'genus': norm_data.get("genus"),
                    'species': norm_data.get("scientific_name") if is_bird else None,
                    'is_bird': is_bird,
                    'rank_hierarchy': [],
                    'llm_enriched': True,
                    'source': 'LLM_normalization'
                }
            except json.JSONDecodeError as e_json:
                logger.error(f"Error normalizing '{s_name}': {e_json}")
                return s_name, {
                    'original_query': s_name,
                    'canonical_form': s_name,
                    'is_bird': False,
                    'source': 'Fallback_json_decode_error'
                }
            except Exception as e:
                logger.error(f"Error normalizing '{s_name}': {e}")
                return s_name, {
                    'original_query': s_name,
                    'canonical_form': s_name,
                    'is_bird': False,
                    'source': 'Fallback_general_exception'
                }
        tasks.append(get_taxonomy_for_subject(subject, species_for_llm))

    # Run all normalization tasks concurrently
    if tasks:
        logger.info("running normalization...")
        results = await asyncio.gather(*tasks)
        for subject_name, tax_data in results:
            species_taxonomy_cache[subject_name] = tax_data
            logger.info(f"{subject_name} -> {tax_data.get('canonical_form', subject_name)}")
        logger.info("normalization done")
    else:
        logger.info("nothing to normalize")

    # Filter triplets and build taxonomy map
    normalized_triplets = []
    llm_taxonomy_map = {}
    
    for subject, predicate, obj, doi in triplet_list:
        tax_data = species_taxonomy_cache.get(subject)
        
        if tax_data:
            normalized_triplets.append((tax_data['canonical_form'], predicate, obj, doi))
            llm_taxonomy_map[subject] = tax_data

    logger.info(f"normalization complete:")
    logger.info(f"  original: {len(triplet_list)}")
    logger.info(f"  normalized: {len(normalized_triplets)}")
    logger.info(f"  taxonomy entries: {len(llm_taxonomy_map)}")
    
    return normalized_triplets, llm_taxonomy_map

# verify triplets
async def verify_triplets(
    triplet_list: List[Tuple[str, str, str, str]], 
    abstract: str, 
    llm_setup,
    verification_cutoff: float = 0.75
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, int]]:
    """check triplets against original text"""
    verified_triplets_for_abstract = []
    counts = {
        'submitted': len(triplet_list),
        'verified_yes': 0,
        'verified_no': 0,
        'errors': 0
    }

    if not abstract:
        counts['errors'] = len(triplet_list)
        return [], counts

    verification_schema = {
        "type": "object",
        "properties": {
            "verification": {"type": "string", "enum": ["YES", "NO"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["verification", "confidence"]
    }
    
    system_prompt = (
        "You are a precise scientific fact checker. "
        "Based on the provided abstract, verify if the relationship is true and the threat is correctly identified in the triplet. "
        "ADDITIONALLY, answer NO if the species in the relationship is not a type of bird. "
        "Respond ONLY with a valid JSON object matching the specified schema. "
        "The JSON object must contain two keys: 'verification' (string: \"YES\" or \"NO\") "
        "and 'confidence' (float: 0.0 to 1.0 representing your confidence in the verification)."
    )
    abstract_hash_part = hashlib.md5(abstract.encode('utf-8', errors='replace')).hexdigest()[:16]
    cache_key_text = f"verify_json_confidence_batch_async:{abstract_hash_part}:{verification_cutoff}:{len(triplet_list)}" 
    cache_key_hash = hashlib.md5(cache_key_text.encode('utf-8', errors='replace')).hexdigest()
    cache_file_path = Path(llm_setup['cache'].cache_dir) / f"{cache_key_hash}.pkl" 

    if cache_file_path.exists():
        try:
            with open(cache_file_path, 'rb') as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, tuple) and len(cached_data) == 2:
                cached_triplets_list, cached_counts_dict = cached_data
                if isinstance(cached_triplets_list, list) and isinstance(cached_counts_dict, dict):
                    logger.info(f"cache hit for {abstract_hash_part}")
                    return cached_triplets_list, cached_counts_dict
        except Exception as e:
            logger.warning(f"cache read failed: {e}")
            if cache_file_path.exists(): cache_file_path.unlink(missing_ok=True)

    async def verify_single_triplet_task(subject, predicate, obj, doi_val, p_llm_setup, p_system_prompt, p_verification_schema):
        prompt = f"""Abstract:
                {abstract}

                Relationship to verify:
                Subject: "{subject}"
                Predicate: "{predicate}"
                Object: "{obj}"

                Is this relationship true based on the abstract (and is the subject a bird)? Provide your answer in the specified JSON format."""
                        
        response_str = None
        try:
            response_str = await llm_generate(
                prompt=prompt, 
                system=p_system_prompt, 
                model=p_llm_setup["model"],
                temp=0.0, 
                format=p_verification_schema, 
                llm_setup=p_llm_setup
            )
            if not response_str:
                return (subject, predicate, obj, doi_val), "ERROR_EMPTY_RESPONSE", 0.0

            result_json = json.loads(response_str)
            verification_decision = result_json.get("verification")
            confidence_score = result_json.get("confidence")

            if isinstance(verification_decision, str) and isinstance(confidence_score, (float, int)):
                return (subject, predicate, obj, doi_val), verification_decision.upper(), confidence_score
            else:
                return (subject, predicate, obj, doi_val), "ERROR_INVALID_JSON_CONTENT", 0.0
        
        except json.JSONDecodeError:
            logger.error(f"JSONDecodeError for triplet: {subject}|{predicate}|{obj}. Response: {response_str}")
            return (subject, predicate, obj, doi_val), "ERROR_JSON_DECODE", 0.0
        except Exception as e:
            logger.error(f"Exception in verify_single_triplet_task for {subject}|{predicate}|{obj}: {e}")
            return (subject, predicate, obj, doi_val), f"ERROR_LLM_CALL: {str(e)[:50]}", 0.0

    tasks = []
    for subject, predicate, obj, doi_val in triplet_list:
        tasks.append(verify_single_triplet_task(subject, predicate, obj, doi_val, llm_setup, system_prompt, verification_schema))
    
    if not tasks: return [], counts

    logger.info(f"verifying {len(tasks)} triplets...")
    verification_results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"verification done")

    for i, res_tuple_or_exc in enumerate(verification_results):
        original_triplet = triplet_list[i]
        subject, predicate, obj, doi_val = original_triplet

        if isinstance(res_tuple_or_exc, Exception):
            counts['errors'] += 1
            logger.error(f"verification error: {res_tuple_or_exc}")
            continue
        
        if res_tuple_or_exc is None or not isinstance(res_tuple_or_exc, tuple) or len(res_tuple_or_exc) != 3:
            counts['errors'] += 1
            logger.error(f"ERROR: Unexpected result format from verify_single_triplet_task for {original_triplet}. Result: {res_tuple_or_exc}")
            continue

        _triplet_data, decision, confidence = res_tuple_or_exc

        if "ERROR" in decision:
            counts['errors'] += 1
            logger.warning(f"rejected: {subject} | {predicate} | {obj}")
        elif decision == "YES" and confidence >= verification_cutoff:
            verified_triplets_for_abstract.append(original_triplet)
            counts['verified_yes'] += 1
            logger.info(f"verified: {subject} | {predicate} | {obj}")
        else:
            counts['verified_no'] += 1
            logger.warning(f"rejected: {subject} | {predicate} | {obj} (conf: {confidence:.2f})")

    try:
        with open(cache_file_path, 'wb') as f:
            pickle.dump((verified_triplets_for_abstract, counts), f)
    except Exception as e:
        logger.error(f"cache write failed: {e}")

    return verified_triplets_for_abstract, counts
