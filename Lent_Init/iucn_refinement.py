import json
import asyncio
import logging
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import os
import re
from .cache import SimpleCache
from .llm_api_utility import llm_generate, extract_content_from_result

logger = logging.getLogger("pipeline")

async def get_iucn_classification_json(subject: str, predicate: str, threat_desc: str, llm_setup, cache: SimpleCache, abstract: Optional[str] = None) -> tuple[str, str]: 
    cache_key = f"iucn_classify_json_schema:{threat_desc}|context:{subject}|{predicate}|abstract:{bool(abstract)}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"IUCN cache hit: '{threat_desc[:50]}...'")
        return cached_result

    if abstract:
        logger.info(f"Classifying threat: '{threat_desc[:50]}...' with abstract context ({len(abstract)} chars)")
    else:
        logger.info(f"Classifying threat: '{threat_desc[:50]}...' without abstract context")
    
    iucn_schema = {
        "type": "object",
        "properties": {
            "iucn_code": {"type": "string", "description": "IUCN code like '5.3' or '11.1'"},
            "iucn_name": {"type": "string", "description": "IUCN category name"},
            "justification": {"type": "string", "description": "1-2 sentences mapping the threat evidence to the classification rules"}
        },
        "required": ["iucn_code", "iucn_name", "justification"]
    }

    abstract_section = ""
    if abstract:
        abstract_section = f"""
**ABSTRACT:**

{abstract}

"""

    prompt = f"""
{abstract_section}**EXTRACTED RELATIONSHIP:**

Subject (Species): {subject}
Predicate (Interaction): {predicate}
Object (Threat Detail): {threat_desc}

**TASK:**
Classify this threat using the IUCN-CMP Direct Threats Classification v4.0. Identify the underlying driver of this threat, not just the immediate symptom.

**REQUIRED OUTPUT FORMAT:**
{{
  "iucn_code": "<The most specific L2 code, e.g., '1.1'>",
  "iucn_name": "<The corresponding name for the L2 code>",
  "justification": "<1-2 sentences mapping the threat evidence to the classification rules>"
}}
            """
                
    response_result = await llm_generate(
        prompt=prompt,
        system=IUCN_THREAT_PROMPT_SYSTEM,
        model=llm_setup.get("model", "qwen/qwq-32b"), 
        temp=0.0, 
        format=iucn_schema,
        llm_setup=llm_setup,
        #extra_body={"require_parameters": True}
    )

    response_str = extract_content_from_result(response_result)
    if response_str:
        try:
            result_json = json.loads(response_str) 
            code = result_json.get("iucn_code")
            name = result_json.get("iucn_name")
            justification = result_json.get("justification", "")
            
            # basic validation
            if isinstance(code, str) and isinstance(name, str) and code.strip() and name.strip() and re.match(r"^\d+(\.\d+)?$", code.strip()):
                code = code.strip()
                name = name.strip()
                if justification:
                    logger.info(f"Classified as: {code} - {name} | Justification: {justification}")
                else:
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

    # fallback to "unknown threats"
    result = ("12", "Unknown Threats")
    cache.set(cache_key, result)
    return result


IUCN_CATEGORIES_TEXT = """
**IUCN-CMP Direct Threats Classification v4.0**

Select the single most specific and relevant category from the list below that best represents the underlying cause of the threat.

**A. Use of Lands & Waters**
Human uses of land and water areas that have a substantial spatial footprint.

1. Residential, Commercial & Recreation Areas: Human settlements, industrial areas, and other non-agricultural land uses with a substantial footprint.
   1.1 Residential Areas: Cities, towns, and settlements including non-housing development typically integrated with housing. (Examples: urban areas, suburbs, vacation homes, shopping areas, offices, schools).
   1.2 Commercial & Industrial Areas: Factories and other commercial centers. (Examples: stand-alone office parks, manufacturing plants, military bases, power plants, landfills, ports, airports).
   1.3 Recreation & Tourism Areas: Tourism and recreation sites with a substantial spatial footprint. (Examples: visitor facilities in parks, campgrounds, ski areas, golf courses, marinas).

2. Agriculture & Aquaculture: Farming and ranching including agricultural expansion, intensification, or practices with a spatial footprint.
   2.1 Annual & Perennial Non-Timber Crops: Crops planted for food, fodder, fiber, or fuel. (Examples: farms, oil palm plantations, orchards, vineyards, biofuel crops).
   2.2 Wood & Pulp Plantations: Stands of trees planted for timber or fiber outside of natural forests. (Examples: teak or eucalyptus plantations, firewood lots, christmas tree farms).
   2.3 Terrestrial Animal Farming, Ranching & Herding: Domestic terrestrial animals raised for commercial purposes. (Examples: cattle feed lots, dairy farms, cattle ranching, chicken farms, goat/camel/yak herding).
   2.4 Marine & Freshwater Aquaculture: Aquatic species raised for harvest. (Examples: fish ponds, shrimp production, salmon pens, seeded shellfish beds).

3. Energy Production & Mining: Extraction of non-biological resources.
   3.1 Oil & Gas Exploration & Extraction: Exploring for and extracting petroleum and other liquid hydrocarbons. (Examples: oil wells, hydraulic fracturing, natural gas drilling).
   3.2 Mining & Quarrying: Exploring for, developing and producing minerals and rocks. (Examples: coal mines, gold mines, rock quarries, sand/salt mining, guano harvesting).
   3.3 Renewable Energy: Exploring, developing and producing renewable energy. (Examples: geothermal power, solar farms, wind farms, tidal farms).

4. Transportation, Service & Security Corridors: Linear infrastructure and the effects associated with its use.
   4.1 Roads, Trails & Railroads: Transport on roadways and dedicated tracks. (Examples: highways, logging roads, bridges, vehicle collisions with wildlife, railroads).
   4.2 Utility & Service Lines: Transport of energy & resources. (Examples: electrical & phone wires, aqueducts, oil & gas pipelines, electrocution of wildlife).
   4.3 Shipping Lanes: Transport on and in freshwater and ocean waterways. (Examples: shipping channels, canals, ships running into whales).
   4.4 Atmospheric & Space Activities: Air and space transport and other activities. (Examples: flight paths, jets impacting birds, drones).
   4.5 Fencing & Walls: Barriers to movement. (Examples: border walls, fences around farm fields, disease control fencing).

**B. Use / Management of Species & Ecosystems**
Human uses of biotic resources and disturbance from human presence or management actions in natural systems.

5. Biological Resource Use & Control: Consumptive use of "wild" biological resources.
   5.1 Hunting, Collecting & Controlling Terrestrial Animals: Hunting or trapping terrestrial wild animals. (Examples: subsistence hunting, commercial wild meat hunting, trophy hunting, pet trade, culling, killing crop-raiding animals).
   5.2 Gathering, Harvesting & Controlling Terrestrial Plants & Fungi: Harvesting plants, fungi, and other non-timber/non-animal products. (Examples: gathering wild fruit, mushrooms, herbs for medicine; rubber tapping).
   5.3 Logging, Harvesting & Controlling Trees: Harvesting trees and other woody vegetation for timber, fiber, or fuel. (Examples: clear-cutting, selective logging, pulp operations, fuel wood collection).
   5.4 Fishing, Harvesting & Controlling Aquatic Species: Harvesting aquatic wild animals or plants. (Examples: net fishing, trawling, blast fishing, whaling, seal hunting, shellfish harvesting, live coral/aquarium fish collection).

6. Human Intrusions & Disturbances: Non-consumptive activities that alter, disturb, and destroy ecosystems and species.
   6.1 Recreational Activities: People spending time in natural areas outside of established transport corridors. (Examples: hikers, off-road vehicles, motorboats, jet-skis, whale watching, pets in recreational areas).
   6.2 Conflict, Civil Unrest & Security Activities: Actions by formal or paramilitary forces without a permanent footprint. (Examples: armed conflict, military training exercises, border patrols, abandoned land mines).
   6.3 Other Human Disturbances: People spending time in natural environments for reasons other than recreation or conflict. (Examples: drug smuggling, species research, vandalism).

7. Natural System Management & Modifications: Human actions that modify ecosystem structures, composition, or regimes.
   7.1 Fire & Fire Management: Actions that either suppress or increase fire frequency/intensity. (Examples: fire suppression, inappropriate fire management, escaped agricultural fires, arson, campfires).
   7.2 Dams & Water Management / Use: Actions that modify water levels, flows, and chemistry. (Examples: dam construction/operation, levees, channelization, groundwater pumping, surface water withdrawals).
   7.3 Earth & Sediment Management: Actions that modify the geophysical environment or change sediment regimes. (Examples: dune stabilization, shoreline armoring, dredging, mine reclamation).
   7.4 Weather & Climate Management: Actions that modify atmospheric structure and processes. (Examples: cloud seeding, ocean 'fertilization').
   7.5 Biological System Management: Actions that modify biotic systems. (Examples: mowing grass, removal of snags from streams, artificial reef creation, bird feeders, electric barriers to stop invasive fish).
   7.6 Removing / Reducing Human Management: Absence or reduction of management regimes important for maintaining ecosystems. (Examples: lack of mowing of meadows, cessation of grazing, stopping predator control).

**C. Additional Sources of Stress**
Stressors in natural systems that have been altered by the effects of current or historical human actions.

8. Invasive / Other Problematic Species, Genes & Pathogens: Threats from non-native and native species, pathogens, or genetic materials that have become harmful due to human activities.
   8.1 Invasive Non-Native / Alien Species: Harmful species not originally found within the ecosystem, introduced by human activities. (Examples: rats on islands, feral horses, zebra mussels, stocking exotic fish, ballast water discharge).
   8.2 Problematic Native Species: Native species that have become "out-of-balance" due to human activities. (Examples: overabundant native deer, algal blooms, insect outbreaks).
   8.3 Introduced Genetic Material: Human-caused introduction of natural or synthetic genes. (Examples: hatchery salmon breeding with wild fish, domestic cats breeding with wild cats, genetically modified organisms).
   8.4 Pathogens: Harmful native and non-native agents causing disease. (Examples: plague, Dutch elm disease, Chytrid fungus affecting amphibians).

9. Pollution: Introduction of exotic and/or excess materials or energy.
   9.1 Water-Borne & Other Effluent Pollution: Water-borne and other liquid pollutants. (Examples: municipal waste discharge, leaking septic systems, fertilizer/pesticide run-off, oil spills, mine tailings, road salt).
   9.2 Garbage & Solid Waste: Rubbish and other solid materials. (Examples: municipal waste, litter, agricultural plastics, microplastics, ghost fishing gear, construction debris).
   9.3 Air-Borne Pollutants: Atmospheric pollutants. (Examples: acid rain, smog from vehicle emissions, smoke from fires, radioactive fallout).
   9.4 Energy Emissions: Inputs of heat, sound, light, or other wave energy. (Examples: beach lights disorienting turtles, noise from highways/airplanes, seismic exploration, sonar).

**D. Other Events & Factors**

10. Natural Disasters: Potentially catastrophic natural disturbances.
    10.1 Geological Events: Volcanoes, earthquakes, tsunamis, avalanches, landslides.
    10.2 Severe Weather Events: Storms, hurricanes, blizzards, floods, droughts (as discrete events).

11. Climate Change: Change in climate patterns resulting from increased atmospheric greenhouse gasses.
    11.1 Changes in Physical & Chemical Regimes: Broad-scale changes in abiotic conditions. (Examples: ocean acidification, changes in salinity, changes in ocean currents).
    11.2 Changes in Temperature Regimes: Broad-scale changes in temperature. (Examples: heat waves, cold spells, oceanic temperature changes, loss of glaciers/sea ice).
    11.3 Changes in Precipitation & Hydrological Regimes: Broad-scale changes in precipitation and water cycles. (Examples: changes in rainfall patterns, long-term droughts, increased severity of floods, sea-level rise).

12. Unknown Threats: Use only when no other category is applicable.
            """

IUCN_THREAT_PROMPT_SYSTEM = f"""
You are an expert ecologist specializing in threat classification. Your task is to classify a described threat using the IUCN-CMP Direct Threats Classification v4.0.

Your goal is to identify the single most appropriate threat category representing the **underlying driver** of the threat, not the immediate symptom. For example, if a species has "low productivity" (the symptom) because of "mining pollution" (the driver), the correct classification is for pollution, not something generic.

---

### **Classification Rules & Instructions**

1. **Find the Root Cause**: Always trace the impact back to the external human activity or environmental factor causing it.
2. **Select the Most Specific Category**: Choose the most detailed sub-category (e.g., '1.1') that is directly supported by the evidence.
3. **Provide Justification**: In your output, briefly explain *why* your chosen category is the correct one, referencing the evidence.

### **Specific Disambiguation Rules**

To ensure consistency, apply the following specific rules for commonly confused categories:

* **Pollution (IUCN 9.x) Disambiguation**:
    * **9.2 Garbage & Solid Waste**: Use for plastics, litter, derelict fishing gear, and other solid items causing harm (e.g., entanglement, ingestion).
    * **9.1 Water-Borne & Other Effluent Pollution**: Use for industrial/military effluents (mining tailings, toxins), agricultural effluents (pesticides, herbicides, fertilizer runoff), and domestic/urban wastewater. This covers most liquid, chemical, or non-solid pollutants.

* **Climate vs. Weather Disambiguation**:
    * **10.2 Severe Weather Events**: Use for episodic events like storms, hurricanes, heatwaves, and named events (El Niño/La Niña).
    * **11.2 Changes in Temperature Regimes / 11.3 Changes in Precipitation**: Use for long-term trends, regime shifts, and persistent pattern changes (e.g., long-term warming, multi-year droughts).

* **Energy vs. Collisions Disambiguation**:
    * **3.3 Renewable Energy**: Use when the threat is explicitly the development or presence of energy infrastructure (e.g., wind farms causing habitat loss).
    * **4.2 Utility & Service Lines / 4.1 Roads, Trails & Railroads**: Use for direct collisions with infrastructure (e.g., power line electrocution, vehicle collisions) unless the infrastructure is explicitly part of an energy project.

* **Unknown Threats (12.x)**: Use only when absolutely no other category applies and no root cause can be identified from the evidence provided.

---

{IUCN_CATEGORIES_TEXT}

---

**THREAT DEFINITION:** A threat is a direct, external factor that causes or contributes to the degradation, loss, or impairment of a species or ecosystem. Exclude:
- Natural ecological processes (predator-prey dynamics, migration, food availability changes)
- Survey methodologies or research activities (unless harmful to species)
- Intrinsic population dynamics (intraspecific competition, natural barriers)
- Neutral ecological observations (species distributions, habitat preferences)

**ECOLOGICAL TERMINOLOGY:** Use proper conservation terminology:
- "Productivity" = reproductive output (breeding success, clutch size, fecundity)
- "Low productivity" is a SYMPTOM, not a threat - identify what caused the low productivity
- Terms like "decline", "mortality", "population reduction" describe EFFECTS, not underlying threats
- Always trace back to the external human or environmental factor causing these effects

Return your answer as a single, valid JSON object only. Do not include any explanatory text outside the JSON structure.
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
            'is_bird': False
        })
        
        if subject_taxo.get('is_bird', False):
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

    # separate file for taxonomies to inspect
    if filtered_taxo_info:
        with open(output_path / "llm_bird_taxonomies.json", "w", encoding='utf-8') as f:
            json.dump(filtered_taxo_info, f, indent=2)
        print("Bird taxonomies saved to llm_bird_taxonomies.json")
    else:
        print("No bird taxonomies to save")

async def classify_threat_for_subject(subject: str, predicate: str, obj: str, llm_setup, cache: SimpleCache) -> Optional[dict]:
    cache_key = f"threat_sentiment_for_subject:{subject}|{predicate}|{obj}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Threat sentiment cache hit for: {subject}|{predicate}|{obj}")
        classification = cached_result.get("classification")
        if classification in ["negative", "very negative"]:
            return cached_result
        else:
            return None

    logger.info(f"Classifying threat sentiment for: {subject}|{predicate}|{obj}")

    classification_schema = {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": ["very positive", "positive", "neutral", "negative", "very negative"],
                "description": "The sentiment of the interaction from the perspective of the subject species.",
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["classification", "confidence"],
    }

    system_prompt = """
You are an ecologist analyzing species interactions. Classify the sentiment of the interaction from the perspective of the **subject** species.

    **EXCLUDE** irrelevant content: survey methodologies, book reviews, research knowledge gaps, or theoretical discussions that don't describe actual threats to wild populations.
    
    **EXCLUDE SYMPTOMS AS THREATS**: If the object describes a symptom/effect rather than an underlying cause (e.g., "low productivity", "population decline", "mortality", "reduced breeding success"), classify as **neutral** unless a clear external threat is specified.

    Use the following categories:
    - **very negative**: The subject is directly and severely threatened by the interaction (e.g., high mortality from a predator or disease).
    - **negative**: The subject is negatively impacted, but not necessarily severely (e.g., competition for resources).
    - **neutral**: No clear benefit or detriment (e.g., simple co-occurrence). Also use for irrelevant research content.
    - **positive**: The interaction is clearly beneficial for the subject and does not suggest it is a threat to others (e.g., finding a food source, successful nesting). The interaction is beneficial for the subject, but in a way that makes the subject a threat to its environment or other species. For example, a species' population exploding due to a new food source, which in turn harms the ecosystem.
    - **very positive**: Strong benefit (e.g., essential food source, successful nesting).
    Return a JSON object with your classification and confidence.
    """

    prompt = f"""
Analyze the following ecological interaction:

- **Subject (species of interest)**: {subject}
- **Predicate (the interaction)**: {predicate}
- **Object (what the subject interacts with)**: {obj}

    Example: If the subject is "Shorebirds", the predicate is "population increase", and the object is "due to algae blooms", the classification should be **"positive"** because while the population increase is good for the shorebirds, it indicates a potential ecological imbalance where they might be contributing to a problem.

    Based on the information, what is the threat direction for the **subject**?
    """

    response_result = await llm_generate(
        prompt=prompt,
        system=system_prompt,
        model=llm_setup.get("model", "qwen/qwen-long"),
        temp=0.1,
        format=classification_schema,
        llm_setup=llm_setup,
    )

    response_str = extract_content_from_result(response_result)
    if response_str:
        try:
            result_json = json.loads(response_str)
            classification = result_json.get("classification")
            confidence = result_json.get("confidence")

            if classification and isinstance(confidence, (int, float)):
                result = {"classification": classification, "confidence": confidence}
                cache.set(cache_key, result)
                logger.info(f"Threat sentiment for '{subject}|{predicate}|{obj}' classified as '{classification}' with confidence {confidence}")

                if classification in ["negative", "very negative"]:
                    return result
                else:
                    return None

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to decode or parse threat sentiment JSON: {e}. Response: '{response_str}'")

    fallback_result = {"classification": "neutral", "confidence": 0.0}
    cache.set(cache_key, fallback_result)
    return None


async def detect_threat_content_in_abstract(abstract_text: str, llm_setup, cache: SimpleCache) -> bool:
    """
    Neutral detection of whether an abstract contains threat-related content.
    Avoids the pink elephant problem by asking what the abstract is about, not if threats exist.
    """
    cache_key = f"abstract_threat_detection:{hash(abstract_text[:200])}"
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.info(f"Abstract threat detection cache hit")
        return cached_result
    
    detection_schema = {
        "type": "object",
        "properties": {
            "primary_focus": {
                "type": "string",
                "enum": [
                    "impact_studies", 
                    "conservation_management", 
                    "basic_ecology", 
                    "methodology_review", 
                    "theoretical_modeling"
                ],
                "description": "The primary research focus of this abstract"
            },
            "study_type": {
                "type": "string", 
                "enum": [
                    "empirical_field_study",
                    "experimental_study", 
                    "observational_survey",
                    "literature_review",
                    "methodological_paper",
                    "theoretical_analysis"
                ],
                "description": "The type of research approach used"
            }
        },
        "required": ["primary_focus", "study_type"]
    }
    
    system_prompt = """You are analyzing research abstracts to categorize their research focus and approach. Be completely objective.

Classify the PRIMARY research focus:
- **impact_studies**: Research examining effects, changes, influences, mortality, population dynamics, habitat changes, or species responses to factors
- **conservation_management**: Studies about protection, restoration, management practices, policy, or conservation interventions  
- **basic_ecology**: Fundamental ecological research, surveys, behavior, distribution, life history, or natural processes
- **methodology_review**: Methods development, literature reviews, or technical/analytical discussions
- **theoretical_modeling**: Mathematical models, simulations, or conceptual frameworks

Classify the study type:
- **empirical_field_study**: Field-based data collection and observation
- **experimental_study**: Controlled experiments or manipulative studies
- **observational_survey**: Surveys, monitoring, or descriptive observations
- **literature_review**: Synthesis of existing research or meta-analysis
- **methodological_paper**: Technical methods or analytical approaches
- **theoretical_analysis**: Models, simulations, or conceptual work

Focus only on what the research methodology and objectives actually are."""

    prompt = f"""Analyze this research abstract and categorize its research focus and methodology:

{abstract_text}

What is the primary research focus and study type?"""

    response_result = await llm_generate(
        prompt=prompt,
        system=system_prompt,
        model=llm_setup.get("model", "qwen/qwen3-30b-a3b"),
        temp=0.1,
        format=detection_schema,
        llm_setup=llm_setup,
    )

    response_str = extract_content_from_result(response_result)
    if response_str and response_str.strip():
        try:
            result_json = json.loads(response_str)
            primary_focus = result_json.get("primary_focus")
            study_type = result_json.get("study_type")
            
            if primary_focus and study_type:
                is_threat_relevant = (
                    primary_focus in ["impact_studies", "conservation_management"] and
                    study_type in ["empirical_field_study", "experimental_study", "observational_survey"]
                )
                
                cache.set(cache_key, is_threat_relevant)
                logger.info(f"Abstract focus: {primary_focus}, study type: {study_type}, relevant: {is_threat_relevant}")
                return is_threat_relevant
            else:
                logger.warning(f"Missing fields in threat detection response: focus={primary_focus}, type={study_type}")
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse threat detection JSON: {e}. Response: '{response_str[:100] if response_str else 'None'}...'")
    else:
        logger.warning(f"Empty or invalid response from threat detection LLM")
    
    logger.info("Threat detection failed - defaulting to INCLUDE abstract")
    cache.set(cache_key, True)
    return True
