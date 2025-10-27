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
from .llm_api_utility import llm_generate, llm_generate_with_retry, extract_content_from_result, strip_markdown_json

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
    
    CRITICAL: Keep the summary concise but informative. Maximum 300 words total.
    Be specific and detailed about the mechanisms described, but use brief, clear language."""

    try:
        summary_response = await llm_generate_with_retry(
            prompt=f"Text to summarize:\n{abstract}\n\nStructured Summary:",
            system=system_prompt,
            model=llm_setup.get("model", "qwen/qwq-32b"),
            temp=0.1,
            timeout=120,
            llm_setup=llm_setup,
            max_retries=2
        )
        
        summary = extract_content_from_result(summary_response).strip()
        
        if len(summary) < 50:
            logger.warning("summary looks too short")
            return ""
            
        llm_setup["cache"].set(abstract, "summary", summary)
        return summary
        
    except Exception as e:
        logger.error(f"summary generation failed: {e}")
        return ""

# Extract entities (species & threats) from abstract in single call
async def extract_entities_concurrently(abstract_text: str, llm_setup) -> Optional[Dict[str, List[str]]]:
    import hashlib    
    cache = llm_setup.get('refinement_cache')
    if cache:
        cache_key = f"entity_extraction:{hashlib.md5(abstract_text.encode('utf-8', errors='replace')).hexdigest()}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Entity extraction cache hit for abstract: {abstract_text[:50]}...")
            return cached_result
    
    model_name = llm_setup.get("model", "unknown")
    logger.info(f"P2.1: Extracting entities using model {model_name} for abstract: {abstract_text[:50]}...")
    
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
            },
            "evidence_sentences": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Verbatim sentences from abstract that provide clearest evidence of negative impact"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief one-sentence explanation of your extraction decisions"
            }
        },
        "required": ["species", "threats", "evidence_sentences", "reasoning"]
    }
    
    combined_system_prompt = f"""You are an expert scientific entity extractor focused on conservation biology. Your task is to analyze a scientific abstract and extract species, threats, and the evidence for them.

Follow these steps precisely:

**STEP 1: Assess Abstract Relevance**
First, determine if the abstract contains documented negative impacts on a species (e.g., reduced survival, reproduction, population size, or habitat quality from field observations).

**CRITICAL: ONLY EXTRACT CONFIRMED FINDINGS, NOT HYPOTHESES**
- Extract if the study **CONCLUDED** or **FOUND** negative impacts
- Do NOT extract if the study **HYPOTHESIZED**, **TESTED**, or **EXPLORED** potential impacts
- Do NOT extract if the study **CONCLUDED NO IMPACT** or **UNLIKELY TO BE AFFECTED**

* **If NO confirmed negative findings,** do not proceed. Return an empty JSON object: {{"species": [], "threats": [], "evidence_sentences": [], "reasoning": "Abstract does not contain documented findings of negative impact."}}
* **If YES confirmed negative findings,** proceed to Step 2.

**Do NOT extract from abstracts that are primarily about:**
- Survey methodology, detectability, or observer bias
- Method comparisons or sampling design
- Basic ecology or natural history without measured negative impact
- Hypotheses, models, or theoretical risks NOT supported by study results
- Laboratory studies or captive breeding studies

**EXAMPLE OF WHAT NOT TO EXTRACT:**
- Study abstract: "We tested whether 1080 baits pose a risk to Bush Stone-curlews... Our results indicate that reintroduction programs are **unlikely to be affected** by concurrent 1080-baiting."
-  **DO NOT EXTRACT**: The study concluded NO negative impact
-  **CORRECT ACTION**: Return empty JSON - no confirmed negative findings

---

**STEP 2: Extract Entities**
If the abstract is relevant, perform the following extractions.

**TASK A: SPECIES EXTRACTION**
Extract specific species or taxonomic groups mentioned in the abstract.
- **DO NOT infer, assume, or generate** species names that are not explicitly stated in the abstract
- Keep scientific names exactly as written in the source text
- Do not combine multiple species into one entry
- If the study groups multiple species (e.g., 'shorebirds', 'seabirds') and provides examples like 'curlew', 'dunlin', extract the aggregate term as written, but also extract any individual species mentioned in the abstract.

**TASK B: THREAT EXTRACTION**
Extract the CAUSES of negative impacts (i.e., the threats).
- **DO NOT add percentages, numbers, or statistics** that are not explicitly stated in the abstract
- **CRITICAL RULE:** A "threat" is the *origin* or *cause* of harm. DO NOT extract the symptoms, effects, or consequences.
    - **Example 1:** If a species "suffers mortality due to illegal hunting", the threat is "illegal hunting", NOT "mortality".
    - **Example 2:** If a species "experiences habitat loss leading to population decline", the threat is "habitat loss", NOT "population decline".
    - **Example 3:** If birds "suffer from severe aspergillosis", the threat is "severe aspergillosis", NOT "suffering".
- **BE SPECIFIC:** Capture the specific description of the threat exactly as written in the text (e.g., extract "habitat loss from logging", not just "habitat loss"; extract "mercury (Hg) exposure", not just "pollution").
- Only extract threats documented by field observations or monitoring in wild populations, not from lab studies or theoretical models.

**TASK C: EVIDENCE SENTENCES**
Extract the verbatim sentence(s) from the abstract that provide the clearest evidence of the negative impact and link the species to the threat.

---

**STEP 3: Format Output**
Return a single, valid JSON object with the extracted information. Provide a brief, one-sentence reasoning for your choices.

Provide your complete output *only* as a single valid JSON object matching this schema:
{json.dumps(entity_extraction_schema)}

Do not add any text or markdown before or after the JSON object.
"""
    
    user_prompt = abstract_text
    try:
        response_result = await llm_generate(
            prompt=user_prompt,
            system=combined_system_prompt,
            model=llm_setup.get("model", "qwen/qwen3-235b-a22b"), 
            temp=0.0, 
            format=entity_extraction_schema, 
            llm_setup=llm_setup,
            #extra_body={"require_parameters": True}
        )
        
        response_str = extract_content_from_result(response_result)
        if not response_str:
            model_name = llm_setup.get("model", "unknown")
            logger.error(f"P2.1: LLM ({model_name}) returned empty response for entity extraction")
            logger.error(f"P2.1: Abstract length: {len(abstract_text)} chars, Preview: {abstract_text[:50]}...")
            return None
            
        entities_data = json.loads(response_str)
        
        if isinstance(entities_data, dict) and \
           isinstance(entities_data.get("species"), list) and \
           isinstance(entities_data.get("threats"), list) and \
           isinstance(entities_data.get("evidence_sentences"), list):
            if all(isinstance(s, str) for s in entities_data.get("species")) and \
               all(isinstance(t, str) for t in entities_data.get("threats")) and \
               all(isinstance(e, str) for e in entities_data.get("evidence_sentences")):
                reasoning = entities_data.get("reasoning", "No reasoning provided")
                logger.info(f"P2.1: Successfully extracted {len(entities_data['species'])} species, {len(entities_data['threats'])} threats, {len(entities_data['evidence_sentences'])} evidence sentences.")
                logger.info(f"P2.1: Reasoning: {reasoning}")
                if cache:
                    cache.set(cache_key, entities_data)
                
                return entities_data
            else:
                logger.error(f"P2.1: Extracted lists contain non-string elements")
                return None
        elif isinstance(entities_data, dict) and "value" in entities_data and isinstance(entities_data["value"], dict):
            actual_data = entities_data["value"]
            if isinstance(actual_data.get("species"), list) and \
               isinstance(actual_data.get("threats"), list) and \
               isinstance(actual_data.get("evidence_sentences"), list) and \
               all(isinstance(s, str) for s in actual_data.get("species")) and \
               all(isinstance(t, str) for t in actual_data.get("threats")) and \
               all(isinstance(e, str) for e in actual_data.get("evidence_sentences")):
                reasoning = actual_data.get("reasoning", "No reasoning provided")
                logger.info(f"P2.1: Successfully extracted {len(actual_data['species'])} species, {len(actual_data['threats'])} threats, {len(actual_data['evidence_sentences'])} evidence sentences (from 'value' key).")
                logger.info(f"P2.1: Reasoning: {reasoning}")
                if cache:
                    cache.set(cache_key, actual_data)
                
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

#Impact relationship extraction
async def generate_relationships_concurrently(abstract_text: str, species_list: List[str], threats_list: List[str], llm_setup, doi: str) -> List[Tuple[str, str, str, str, str]]:
    import hashlib
    
    cache = llm_setup.get('refinement_cache')
    if cache:
        cache_input = f"{abstract_text}|{sorted(species_list)}|{sorted(threats_list)}"
        cache_key = f"relationships:{hashlib.md5(cache_input.encode('utf-8', errors='replace')).hexdigest()}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Relationship generation cache hit for DOI: {doi}")
            return cached_result
    
    model_name = llm_setup.get("model", "unknown")
    logger.info(f"P2.2: Generating relationships using model {model_name} for DOI: {doi}, {len(species_list)} species, {len(threats_list)} threats")
    
    if not species_list or not threats_list:
        logger.warning(f"P2.2: Missing species or threats list for DOI {doi}. Skipping relationship generation.")
        return []
        
    relationship_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Species name from provided list"},
                "predicate_summary": {"type": "string", "description": "Summary of the impact (e.g., 'experienced population decline', 'showed reduced breeding success')"},
                "predicate_evidence": {"type": "string", "description": "Direct quote or paraphrase from the abstract that provides evidence for this relationship"},
                "object": {"type": "string", "description": "Threat description from provided list"},
                "reason": {"type": "string", "description": "Brief explanation of why this relationship exists based on the abstract"}
            },
            "required": ["subject", "predicate_summary", "predicate_evidence", "object", "reason"]
        }
    }
    system_prompt = (
        """You are an expert in scientific information extraction. Your task is to read a scientific abstract and extract structured species-threat relationship triplets.

Follow these steps precisely:

## STEP 1: Assess Abstract Relevancy

First, analyze the abstract to determine if it is suitable for extraction. You will extract triplets **ONLY IF ALL** of the following conditions are met:
- The abstract reports **observed or quantified negative effects** on a species (e.g., mortality, reduced reproduction, population decline, habitat degradation).
- The causal link is clear: a specific **threat causes an impact** on a species.
- The work is based on **field observations** or documented real-world cases, not lab experiments or theoretical models.

**CRITICAL: ONLY EXTRACT CONFIRMED FINDINGS, NOT HYPOTHESES**
- Extract if the study **CONCLUDED** or **FOUND** negative impacts
- Do NOT extract if the study **CONCLUDED NO IMPACT** or **UNLIKELY TO BE AFFECTED**

**Do NOT extract triplets from abstracts that are primarily about:**
- Survey methods, species detectability, or sampling design.
- Hypotheses or predictions that were not confirmed by the study's results.
- Basic species ecology unless a specific harm is measured and attributed.

If the abstract is not relevant, stop. Otherwise, proceed to Step 2.

---

## STEP 2: Generate Relationship Triplets

For each distinct negative relationship you identify, construct a triplet with a `subject`, `predicate`, and `object`.

### Triplet Generation Rules:

1. **Subject**: The specific species or taxonomic group being harmed (must be from provided species list).
2. **Object**: The *cause* of the harm - the threat/external driver (must be from provided threats list).
3. **Predicate Structure**: Split the impact into two components:
   - **predicate_summary**: Summary of what happened (e.g., "experienced population decline", "showed reduced breeding success")
   - **predicate_evidence**: Direct quote or paraphrase from the abstract that supports this relationship

**CRITICAL PREDICATE RULES:**
* **The predicate_summary MUST NOT restate the Object (cause).** They must be distinct concepts.
    * **INCORRECT**: `{"subject": "BirdA", "predicate_summary": "experiences habitat loss", "object": "habitat loss"}`
    * **CORRECT**: `{"subject": "BirdA", "predicate_summary": "experiences population decline", "object": "widespread habitat loss"}`
* **predicate_summary**: Keep this concise and focused on the biological effect (e.g., "reduced nesting success", "increased mortality", "impaired health")
* **predicate_evidence**: Must be a direct quote or paraphrase from the abstract that demonstrates the relationship, not your interpretation or inference
* **Deduplication Rules:** 
    - If multiple similar impacts are mentioned for the same species-threat pair, create ONE comprehensive triplet that captures the primary effect.
    - If distinctly different biological processes are affected (e.g., both survival AND reproduction), create separate triplets.
    - Avoid near-identical triplets that essentially describe the same relationship.

### Examples of Structured Predicates:

* **Abstract Snippet**: "Little Tern faces increased risk of overheating of eggs, resulting from breeding later in the season when temperatures are higher."
    * **Subject**: `Little Tern`
    * **Object**: `higher temperatures`
    * **predicate_summary**: `faces increased risk of overheating of eggs`
    * **predicate_evidence**: `faces increased risk of overheating of eggs, resulting from breeding later in the season when temperatures are higher`

* **Abstract Snippet**: "Songbird populations experience impaired avian health due to mercury (Hg) exposure from industrial runoff."
    * **Subject**: `Songbird`
    * **Object**: `mercury (Hg) exposure from industrial runoff`
    * **predicate_summary**: `experience impaired avian health`
    * **predicate_evidence**: `Songbird populations experience impaired avian health due to mercury (Hg) exposure from industrial runoff`

For each relationship triplet, provide:
- subject: Species name from provided list (use exact names from provided lists from the abstractonly)
- predicate_summary: Impact summary focusing on the biological effect
- predicate_evidence: Direct quote or paraphrase from abstract that supports the relationship
- object: Threat description from provided list (use exact names from provided lists only)
- reason: Brief explanation of why this relationship exists based on the abstract text (maximum 1 sentence)

CRITICAL: 
- Predicate: **DO NOT add percentages, numbers, or statistics** that are not explicitly stated in the abstract
- Subject/Object: Use exact names from provided lists only
- Reason: Maximum 1 sentence explaining the relationship

Output as a JSON array of objects. Do not include any explanatory text outside the JSON."""
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
        response_result = await llm_generate(
            prompt=user_prompt,
            system=system_prompt,
            model=llm_setup.get("model", "deepseek/deepseek-r1"), 
            temp=0.0,
            format=relationship_schema,
            llm_setup=llm_setup,
            #extra_body={"require_parameters": True}
        )
        
        response_str = extract_content_from_result(response_result)
        if not response_str:
            model_name = llm_setup.get("model", "unknown")
            logger.error(f"P2.2: LLM ({model_name}) returned empty response for relationship generation. DOI: {doi}")
            logger.error(f"P2.2: Species: {species_list}, Threats: {threats_list}")
            logger.error(f"P2.2: Abstract length: {len(abstract_text)} chars")
            return []
        
        clean_json = strip_markdown_json(response_str)
        try:
            relationships_data = json.loads(clean_json)
        except json.JSONDecodeError as e_json:
            logger.error(f"P2.2: JSONDecodeError in relationship generation: {e_json}. DOI: {doi}")
            logger.error(f"P2.2: Raw response: '{response_str[:500]}...'")
            logger.error(f"P2.2: Cleaned JSON: '{clean_json[:500]}...'")
            return []
        
        if isinstance(relationships_data, list):
            for rel in relationships_data:
                if isinstance(rel, dict):
                    subject = rel.get("subject")
                    predicate_summary = rel.get("predicate_summary")
                    predicate_evidence = rel.get("predicate_evidence")
                    obj_threat = rel.get("object")
                    reason = rel.get("reason")
                    
                    predicate = predicate_summary
                    
                    if subject and predicate and obj_threat and subject in species_list and obj_threat in threats_list:
                        if len(predicate.split()) > 1:
                            raw_triplets.append((subject, predicate, obj_threat, doi, predicate_evidence))
                            if reason:
                                logger.info(f"P2.2: Triplet reasoning - {subject} | {predicate} | {obj_threat}: REASONING: {reason}")
                            if predicate_evidence:
                                logger.info(f"P2.2: Evidence - {subject} | {predicate} | {obj_threat}: EVIDENCE: {predicate_evidence[:100]}...")
                        else:
                            logger.warning(f"P2.2: Dropping invalid triplet (short predicate): {rel}. DOI: {doi}")
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
                    predicate_summary = rel.get("predicate_summary")
                    predicate_evidence = rel.get("predicate_evidence")
                    obj_threat = rel.get("object")
                    reason = rel.get("reason")                    
                    predicate = predicate_summary
                    
                    if subject and predicate and obj_threat and subject in species_list and obj_threat in threats_list:
                        if len(predicate.split()) > 1:
                            raw_triplets.append((subject, predicate, obj_threat, doi, predicate_evidence))
                            if reason:
                                logger.info(f"P2.2: Triplet reasoning - {subject} | {predicate} | {obj_threat}: REASONING: {reason}")
                            if predicate_evidence:
                                logger.info(f"P2.2: Evidence - {subject} | {predicate} | {obj_threat}: EVIDENCE: {predicate_evidence[:100]}...")
                        else:
                            logger.warning(f"P2.2: Dropping invalid triplet from 'value' list (short predicate): {rel}. DOI: {doi}")
                    else:
                        logger.warning(f"P2.2: Dropping invalid triplet from 'value' list: {rel}. DOI: {doi}")
                else:
                    logger.warning(f"P2.2: Expected dict in 'value' relationships list, got {type(rel)}. DOI: {doi}")
            logger.info(f"P2.2: Successfully parsed {len(raw_triplets)} relationships from 'value' key for DOI: {doi}.")
        else:
            logger.error(f"P2.2: Unexpected JSON structure for relationships. Expected list or dict with 'value'. Got {type(relationships_data)}. Raw: '{response_str}'. DOI: {doi}")
    except Exception as e:
        logger.error(f"P2.2: Error in generate_relationships_concurrently for DOI {doi}: {e}", exc_info=True)    
    if raw_triplets and cache:
        cache.set(cache_key, raw_triplets)
    
    return raw_triplets



# ollama structured output link: https://ollama.com/blog/structured-outputs#:~:text=Ollama%20now%20supports%20structured%20outputs,Parsing%20data%20from%20documents
# OLD VERSION OF EXTRACTING TRIPLETS IN THREE STEPS, one prompt for species, one for threats, one for impacts
# New version (extract_entities_concurrently) splits NER tasks into one and the impact task into another for speed and clarity.
async def extract_triplets(summary: str, llm_setup, doi: str) -> List[Tuple[str, str, str, str]]:
    # Cache check commented out to force regeneration (as per previous request)
    # cached = llm_setup["cache"].get(summary, "triplets")
    # if cached:
    #     return cached

    logger.info("generating triplets...")

    try:
        # Step 1: get species
        logger.info("Step 1: finding species...")
        
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
            llm_setup=llm_setup,
            #extra_body={"require_parameters": True}
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
        logger.info("Step 2: finding threats...")
        
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
            llm_setup=llm_setup,
            #extra_body={"require_parameters": True}
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
        logger.info("Step 3: finding impact mechanisms...")
        
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
        3. Include specific biological, physiological, or ecological processes involved in the harm described in the abstract.
        4. Provide direct evidence from the text for the mechanism.
        5. Assign a confidence level (high, medium, low) based on how clearly the negative impact mechanism is described.
        6. If multiple distinct negative mechanisms exist for the same species-threat pair, list them separately.

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
            llm_setup=llm_setup,
            #extra_body={"require_parameters": True}
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


def are_terms_similar(term1: str, term2: str, threshold: int = 80) -> bool:
    t1 = term1.lower().strip()
    t2 = term2.lower().strip()
    return fuzz.token_set_ratio(t1, t2) >= threshold

def are_threats_semantically_similar(threat1: str, threat2: str) -> bool:

    t1 = threat1.strip()
    t2 = threat2.strip()

    t1_lower = t1.lower()
    t2_lower = t2.lower()

    if t1_lower == t2_lower:
        return True

    # Token Jaccard + fuzzy token-set combo
    def tokenize(s: str) -> set[str]:
        import re as _re
        s = _re.sub(r"[^a-z0-9\s]", " ", s.lower())
        tokens = [tok for tok in s.split() if tok not in {"the","a","an","of","and","to","for","in","on","by","with","due","from"}]
        return set(tokens)

    tok1 = tokenize(t1)
    tok2 = tokenize(t2)
    if not tok1 or not tok2:
        return False

    jacc = len(tok1 & tok2) / max(1, len(tok1 | tok2))
    fuzzy = fuzz.token_set_ratio(t1_lower, t2_lower) / 100.0

    avg_len = (len(t1) + len(t2)) / 2
    base_thresh = 0.78 if avg_len > 20 else 0.85
    score = 0.6 * fuzzy + 0.4 * jacc
    return score >= base_thresh

# merge similar triplets
def consolidate_triplets(triplet_list: List[Tuple[str, str, str, str, str]]) -> List[Tuple[str, str, str, str, str]]:
    if not triplet_list:
        return []
    
    consolidated = []
    processed_indices = set()
    
    for i, (subj1, pred1, obj1, doi1, evidence1) in enumerate(triplet_list):
        if i in processed_indices:
            continue
            
        similar_group = [(subj1, pred1, obj1, doi1, evidence1)]
        processed_indices.add(i)
        
        # find similar ones
        for j, (subj2, pred2, obj2, doi2, evidence2) in enumerate(triplet_list[i+1:], i+1):
            if j in processed_indices:
                continue
                
            # Check if subjects and objects are similar using enhanced matching
            if (are_terms_similar(subj1, subj2) and 
                are_threats_semantically_similar(obj1, obj2) and
                (doi1 == doi2)):
                similar_group.append((subj2, pred2, obj2, doi2, evidence2))
                processed_indices.add(j)
        
        # If we found similar triplets, combine them
        if len(similar_group) > 1:
            combined_subj = similar_group[0][0]
            # Choose the most detailed object (threat description)
            objects = [t[2] for t in similar_group]
            combined_obj = max(objects, key=len)  # Most detailed threat description
            # If all DOIs match, keep it; otherwise fallback to the first
            doi_set = {t[3] for t in similar_group}
            combined_doi = doi1 if len(doi_set) == 1 else similar_group[0][3]
            
            predicates = list(set(t[1] for t in similar_group))
            evidence_list = [t[4] for t in similar_group if t[4]]  # Collect all evidence
            
            if len(predicates) > 1:
                def predicate_score(p: str) -> int:
                    p_low = p.lower()
                    bonus_terms = [
                        "due to", "resulting in", "leading to", "causing", "through", "via",
                        "reduced", "decreased", "mortality", "breeding", "nest", "productivity",
                        "contamination", "pollution", "disturbance", "collision", "predation"
                    ]
                    score = min(len(p), 160)
                    score += sum(2 for bt in bonus_terms if bt in p_low)
                    return score
                # choose by score, but avoid merging totally dissimilar text
                best = max(predicates, key=predicate_score)
                if any(are_terms_similar(best, p, threshold=60) for p in predicates):
                    combined_pred = best
                else:
                    combined_pred = predicates[0]
            else:
                combined_pred = predicates[0]
            
            # Combine evidence - use the longest/most detailed one
            combined_evidence = max(evidence_list, key=len) if evidence_list else evidence1
            
            consolidated.append((combined_subj, combined_pred, combined_obj, combined_doi, combined_evidence))
            print(f"\nMerged triplets:")
            for t in similar_group:
                print(f"  {t[0]} | {t[1]} | {t[2]} | Evidence: {(t[4] or '')[:50]}...")
            print(f"Into: {combined_subj} | {combined_pred} | {combined_obj} | Evidence: {(combined_evidence or '')[:50]}...\n")
        else:
            consolidated.append((subj1, pred1, obj1, doi1, evidence1))
    
    return consolidated

# normalize species names
async def normalize_species_names(triplet_list: List[Tuple[str, str, str, str, str]], llm_setup) -> Tuple[List[Tuple[str, str, str, str, str]], Dict[str, Dict]]:
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

        CRITICAL: Be extremely concise in all text fields. Use shortest possible names and classifications.
        Important: Only set 'is_bird' to true if the species/group belongs to Class Aves (birds).
        Respond with valid JSON matching the required schema. No explanatory text outside JSON."""

    species_taxonomy_cache = {}
    
    # Create tasks for all unique subjects
    tasks = []
    for subject in unique_subjects:
        species_for_llm = subject
        if subject.lower() == "birds":
            species_for_llm = "Bird"
        
        async def get_taxonomy_for_subject(s_name, s_llm_name):
            try:
                response_result = await llm_generate(
                    prompt=f"Normalize this species name: {s_llm_name}",
                    system=system_prompt,
                    model=llm_setup.get("model", "qwen/qwen3-235b-a22b"),
                    temp=0.1,
                    format=normalization_schema,
                    llm_setup=llm_setup,
                    #extra_body={"require_parameters": True}
                )
                response_json_str = extract_content_from_result(response_result)
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
    
    for subject, predicate, obj, doi, evidence in triplet_list:
        tax_data = species_taxonomy_cache.get(subject)
        
        if tax_data:
            normalized_triplets.append((tax_data['canonical_form'], predicate, obj, doi, evidence))
            llm_taxonomy_map[subject] = tax_data

    logger.info(f"normalization complete:")
    logger.info(f"  original: {len(triplet_list)}")
    logger.info(f"  normalized: {len(normalized_triplets)}")
    logger.info(f"  taxonomy entries: {len(llm_taxonomy_map)}")
    
    return normalized_triplets, llm_taxonomy_map

# verify triplets
async def verify_triplets(triplet_list: List[Tuple[str, str, str, str, str]], abstract: str, llm_setup, verification_cutoff: float = 0.75) -> Tuple[List[Tuple[str, str, str, str, str]], Dict[str, int]]:
    #check triplets against original text
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
            "decision": {"type": "string", "enum": ["KEEP", "DROP"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of issues found during review, empty if no issues"
            }
        },
        "required": ["decision", "confidence", "issues"]
    }
    
    system_prompt = (
        "You are a meticulous scientific data reviewer. Your task is to perform a quality control check on a proposed species-threat relationship (triplet), comparing it against the provided abstract and a set of rules."
        "\n\n### **Review Checklist**"
        "\n\nEvaluate the triplet based on ALL of the following criteria:"
        "\n\n1. **Factual Accuracy**: Are the species name, threat description, and impact details actually mentioned in the abstract text?"
        "\n   * `FAIL` if the subject contains species names NOT found in the abstract"
        "\n   * `FAIL` if the object contains threat details NOT found in the abstract"
        "\n   * `FAIL` if the predicate contains percentages, numbers, or specifics NOT found in the abstract"
        "\n\n2. **Evidentiary Support**: Is the relationship an observed result or conclusion explicitly stated in the abstract?"
        "\n   * `FAIL` if it's only a hypothesis, a statement from the introduction, or not mentioned."
        "\n\n3. **Subject Validity**: Is the `subject` a specific species or well-defined taxonomic group?"
        "\n   * `FAIL` if the subject is an overly broad aggregated group when specific species were named in the abstract."
        "\n\n4. **Threat Validity**: Is the `object` (the threat) conceptually sound?"
        "\n   * `FAIL` if it is a circular object (e.g., \"conservation status\"), an effect (e.g., \"mortality\"), or a simple ecological finding without measured harm (e.g., \"presence of parasites\")."
        "\n\n5. **IUCN Category Consistency**: Does the assigned `IUCN label` correctly represent the threat's underlying driver according to standard classification rules? Pay special attention to whether the threat should be anthropogenic (IUCN 1-9) or natural (IUCN 10+)."
        "\n\n### **Decision and Output**"
        "\n\nBased on your review, provide a single, valid JSON object. Do not include any other text."
        "\n\n* The `decision` should be `\"KEEP\"` only if the triplet passes ALL checks. Otherwise, it should be `\"DROP\"`."
        "\n* The `issues` list must contain a brief explanation for each failed check."
        "\n\nReturn ONLY valid JSON. No explanatory text outside the JSON."
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
                    submitted = cached_counts_dict.get('submitted', 0)
                    verified_yes = cached_counts_dict.get('verified_yes', 0)
                    verified_no = cached_counts_dict.get('verified_no', 0)
                    errors = cached_counts_dict.get('errors', 0)
                    
                    if errors > 0:
                        logger.warning(f"CACHED VERIFICATION had {errors} ERRORS out of {submitted} triplets (yes: {verified_yes}, no: {verified_no})")
                        logger.warning(f"  Error rate: {errors/submitted*100:.1f}% of triplets - clearing cache and re-running verification")
                        cache_file_path.unlink(missing_ok=True)
                    else:
                        logger.info(f"cache hit for {abstract_hash_part}")
                        logger.debug(f"CACHED VERIFICATION: {submitted} submitted, {verified_yes} verified, {verified_no} rejected, {errors} errors")
                        return cached_triplets_list, cached_counts_dict
        except Exception as e:
            logger.warning(f"cache read failed: {e}")
            if cache_file_path.exists(): cache_file_path.unlink(missing_ok=True)

    async def verify_single_triplet_task(subject, predicate, obj, doi_val, evidence, p_llm_setup, p_system_prompt, p_verification_schema):
        evidence_section = ""
        if evidence:
            evidence_section = f"""
                **SUPPORTING EVIDENCE:**
                "{evidence}"
                """
        
        prompt = f"""Abstract:
                {abstract}

                **TRIPLET TO REVIEW:**
                Subject (Species): "{subject}"
                Predicate (Impact): "{predicate}"
                Object (Threat): "{obj}"
                {evidence_section}

                **TASK:**
                Perform a quality control check on this triplet using the review checklist. Evaluate evidentiary support, subject validity, threat validity, and provide your decision with any issues identified."""
                        
        response_str = None
        
        max_attempts = 5
        base_delay = 8.0
        
        verification_model = p_llm_setup.get("model", "qwen/qwen3-235b-a22b")
        response_result = None
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt == 1:
                    logger.info(f"{verification_model} for verification with logprobs")
                elif attempt > 1:
                    logger.warning(f"Verification retry attempt {attempt}/{max_attempts} for {subject[:30]}|{predicate[:30]}|{obj[:30]}")
                
                response_result = await llm_generate_with_retry(
                    prompt=prompt, 
                    system=p_system_prompt, 
                    model=verification_model,
                    temp=0.0, 
                    format=p_verification_schema, 
                    llm_setup=p_llm_setup,
                    logprobs=True,
                    top_logprobs=5,
                    max_retries=1,
                )
                
                if response_result is None or (isinstance(response_result, tuple) and not response_result[0]):
                    raise Exception("Empty response from LLM (rate limited or server error)")
                break
                
            except Exception as e:
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(f"Verification attempt {attempt}/{max_attempts} failed: {str(e)[:100]}")
                    logger.warning(f"  Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    logger.error(f"VERIFICATION ERROR - All {max_attempts} attempts failed for {subject}|{predicate}|{obj}")
                    return (subject, predicate, obj, doi_val, evidence), f"ERROR_MAX_RETRIES: {str(e)[:50]}", 0.0
        
        try:
            
            if isinstance(response_result, tuple):
                if len(response_result) == 3:
                    response_str, logprobs_info, usage_info = response_result
                    logger.debug(f"Received response with logprobs and usage for triplet: {subject}|{predicate}|{obj}")
                elif len(response_result) == 2:
                    response_str, logprobs_info = response_result
                    logger.debug(f"Received response with logprobs for triplet: {subject}|{predicate}|{obj}")
                else:
                    response_str = response_result[0] if response_result else ""
                    logprobs_info = None
                    logger.debug(f"Received unexpected tuple format for triplet: {subject}|{predicate}|{obj}")
            else:
                response_str = response_result
                logprobs_info = None
                logger.debug(f"Received response without logprobs for triplet: {subject}|{predicate}|{obj}")
            
            if not response_str:
                logger.error(f"VERIFICATION ERROR - Empty response from LLM for triplet: {subject}|{predicate}|{obj}")
                return (subject, predicate, obj, doi_val, evidence), "ERROR_EMPTY_RESPONSE", 0.0

            clean_response = response_str.strip()
            if clean_response.startswith("```json") and clean_response.endswith("```"):
                clean_response = clean_response[7:-3].strip()
            elif clean_response.startswith("```") and clean_response.endswith("```"):
                clean_response = clean_response[3:-3].strip()
            
            result_json = json.loads(clean_response)
            verification_decision = result_json.get("decision")
            issues = result_json.get("issues", [])
            
            # Try to extract log probabilities for KEEP/DROP tokens
            log_prob_confidence = None
            if logprobs_info and logprobs_info.content:
                logger.debug(f"Analyzing logprobs for {len(logprobs_info.content)} tokens")
                keep_logprob = -float('inf')
                drop_logprob = -float('inf')
                tokens_examined = 0
                keep_tokens_found = []
                drop_tokens_found = []
                
                # Search through all tokens for KEEP/DROP
                for i, token_info in enumerate(logprobs_info.content):
                    # Handle both object and dict access patterns
                    top_logprobs = getattr(token_info, 'top_logprobs', None) or token_info.get('top_logprobs') if isinstance(token_info, dict) else None
                    if top_logprobs:
                        tokens_examined += 1
                        logger.debug(f"Token {i}: examining {len(top_logprobs)} top alternatives")
                        for top_token in top_logprobs:
                            # Handle both object and dict access patterns for tokens
                            token_text = getattr(top_token, 'token', None) or top_token.get('token') if isinstance(top_token, dict) else str(top_token)
                            token_logprob = getattr(top_token, 'logprob', None) or top_token.get('logprob') if isinstance(top_token, dict) else 0.0
                            
                            if not token_text:
                                continue
                                
                            token_lower = token_text.lower().strip()
                            logger.debug(f"   Token: '{token_text}' (normalized: '{token_lower}') -> logprob: {token_logprob:.4f}")
                            
                            if token_lower in ['keep', '"keep"']:
                                old_keep = keep_logprob
                                keep_logprob = max(keep_logprob, token_logprob)
                                keep_tokens_found.append((token_text, token_logprob))
                                if token_logprob > old_keep:
                                    logger.debug(f"Found KEEP token: '{token_text}' with logprob {token_logprob:.4f}")
                            elif token_lower in ['drop', '"drop"']:
                                old_drop = drop_logprob
                                drop_logprob = max(drop_logprob, token_logprob)
                                drop_tokens_found.append((token_text, token_logprob))
                                if token_logprob > old_drop:
                                    logger.debug(f"Found DROP token: '{token_text}' with logprob {token_logprob:.4f}")
                
                logger.debug(f"Logprob analysis complete: examined {tokens_examined} tokens")
                logger.info(f"KEEP tokens found: {len(keep_tokens_found)} (best: {keep_logprob:.4f})")
                logger.info(f"DROP tokens found: {len(drop_tokens_found)} (best: {drop_logprob:.4f})")
                
                # Use log probability as confidence if found
                if verification_decision.upper() == "KEEP" and keep_logprob > -float('inf'):
                    log_prob_confidence = keep_logprob
                    logger.debug(f"Using KEEP log probability: {log_prob_confidence:.4f} for decision '{verification_decision}'")
                elif verification_decision.upper() == "DROP" and drop_logprob > -float('inf'):
                    log_prob_confidence = drop_logprob
                    logger.debug(f"Using DROP log probability: {log_prob_confidence:.4f} for decision '{verification_decision}'")
                else:
                    logger.debug(f"Could not find matching log probability for decision '{verification_decision}'")
            else:
                logger.debug(f"No usable logprobs available")
            
            # Fall back to JSON confidence if no logprob found or if any errors happen
            if log_prob_confidence is None:
                confidence_score = result_json.get("confidence", 0.0)
                if isinstance(confidence_score, (float, int)):
                    log_prob_confidence = confidence_score
                    logger.debug(f"Falling back to JSON confidence: {log_prob_confidence:.4f}")
                else:
                    log_prob_confidence = 0.0
                    logger.debug(f"Invalid JSON confidence, using 0.0")
            else:
                logger.info(f"Successfully extracted log probability confidence: {log_prob_confidence:.4f}")

            if isinstance(verification_decision, str):
                decision = verification_decision.upper()
                issues_str = "; ".join(issues) if issues else "No issues"
                if decision == "KEEP":
                    logger.info(f"ACCEPT: {subject} | {predicate} | {obj} (conf: {log_prob_confidence:.3f}) | Issues: {issues_str}")
                else:
                    logger.info(f"REJECT: {subject} | {predicate} | {obj} (conf: {log_prob_confidence:.3f}) | Issues: {issues_str}")
                return (subject, predicate, obj, doi_val, evidence), decision, log_prob_confidence
            else:
                logger.error(f"VERIFICATION ERROR - Invalid decision type: {type(verification_decision)} for triplet: {subject}|{predicate}|{obj}. Decision: {verification_decision}")
                return (subject, predicate, obj, doi_val, evidence), "ERROR_INVALID_JSON_CONTENT", 0.0
        
        except json.JSONDecodeError as json_err:
            logger.error(f"VERIFICATION ERROR - JSONDecodeError for triplet: {subject}|{predicate}|{obj}")
            logger.error(f"  Response was: {response_str[:200]}...")
            return (subject, predicate, obj, doi_val, evidence), "ERROR_JSON_DECODE", 0.0
        except Exception as e:
            logger.error(f"VERIFICATION ERROR - Exception for {subject}|{predicate}|{obj}: {type(e).__name__}: {str(e)}")
            return (subject, predicate, obj, doi_val, evidence), f"ERROR_LLM_CALL: {str(e)[:50]}", 0.0

    tasks = []
    for subject, predicate, obj, doi_val, evidence in triplet_list:
        tasks.append(verify_single_triplet_task(subject, predicate, obj, doi_val, evidence, llm_setup, system_prompt, verification_schema))
    
    if not tasks: return [], counts

    logger.info(f"Starting verification of {len(tasks)} triplets using {llm_setup.get('model', 'qwen/qwen3-235b-a22b')} with log probability analysis...")
    verification_results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.debug(f"Verification batch complete - processing results...")

    for i, res_tuple_or_exc in enumerate(verification_results):
        original_triplet = triplet_list[i]
        subject, predicate, obj, doi_val, evidence = original_triplet

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
            logger.error(f"VERIFICATION ERROR - Triplet rejected due to error: {subject} | {predicate} | {obj}")
            logger.error(f"  Error type: {decision}")
        elif decision == "KEEP":
            # Handle both log probability and regular confidence thresholds
            if confidence < 0:  # Log probability (negative values, closer to 0 = higher confidence)
                # Logprob examples: -0.1 (very confident), -1.0 (medium), -3.0 (low confidence)
                log_prob_threshold = -2.0  # logprobs threshold (accept if >= -2.0)
                is_confident = confidence >= log_prob_threshold
                conf_type = "logprob"
                logger.info(f"Log probability threshold comparison: {confidence:.4f} >= {log_prob_threshold} = {is_confident}")
            else:  # normal conf reported from llm (0.0 to 1.0 scale)
                is_confident = confidence >= verification_cutoff
                conf_type = "conf"
                logger.info(f"Regular confidence threshold comparison: {confidence:.4f} >= {verification_cutoff} = {is_confident}")
            
            logger.info(f"Processing KEEP decision for: {subject}|{predicate}|{obj}")
            logger.info(f"Confidence type: {conf_type}, value: {confidence:.4f}, passes threshold: {is_confident}")
            
            if is_confident:
                verified_triplets_for_abstract.append(original_triplet)
                counts['verified_yes'] += 1
                logger.info(f"VERIFIED: {subject} | {predicate} | {obj} ({conf_type}: {confidence:.3f})")
            else:
                counts['verified_no'] += 1
                logger.warning(f"REJECTED (low confidence): {subject} | {predicate} | {obj} ({conf_type}: {confidence:.3f})")
        else:
            counts['verified_no'] += 1
            logger.info(f"REJECTED: {subject} | {predicate} | {obj} (decision: {decision})")

    try:
        with open(cache_file_path, 'wb') as f:
            pickle.dump((verified_triplets_for_abstract, counts), f)
    except Exception as e:
        logger.error(f"cache write failed: {e}")

    return verified_triplets_for_abstract, counts
