import json
import asyncio
import logging
import math
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import os
import re
from .cache import SimpleCache
from .llm_api_utility import llm_generate, extract_content_from_result, strip_markdown_json

logger = logging.getLogger("pipeline")

async def get_iucn_classification_json(subject: str, predicate: str, threat_desc: str, llm_setup, cache: SimpleCache, abstract: Optional[str] = None, use_hierarchical: bool = False) -> tuple[str, str]: 
    if use_hierarchical:
        return await get_hierarchical_iucn_classification(subject, predicate, threat_desc, llm_setup, cache, abstract)
    
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
Classify this threat using the IUCN-CMP Direct Threats Classification v4.0. Identify the immediate threat affecting the species, not the ultimate underlying cause.

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
        model=llm_setup.get("model", "moonshotai/kimi-k2"), 
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
Human uses of land and water areas that have a substantial spatial footprint. Includes effects from their construction (e.g. ecosystem conversion), ongoing use, and abandonment.

This class includes both the ecosystem conversion / degradation effects of the expansion and ongoing presence of these activities, but does not include their associated pollution effects. The threats in 1. Residential, Commercial & Recreation Areas and 2. Agriculture & Aquaculture are generally tied to a defined and relatively compact area, which distinguishes them from those in 4. Transportation, Service & Security Corridors which have a long narrow footprint, and 6. Human Intrusions & Disturbances which do not have an explicit footprint. Standard land-cover classifications can often be used to assess the stresses delivered by these direct threats.

1. Residential, Commercial & Recreation Areas: Human settlements, industrial areas, and other non-agricultural land uses with a substantial footprint.
   
   There is potential overlap between the threats in these categories. For example, should a tourist hotel complex in a city be part of 1.1 or 1.3? In general, most activities inside a defined municipal area should be in 1.1 whereas 1.2 and 1.3 are more for stand-alone developments in otherwise natural spaces.
   
   1.1 Residential Areas: Cities, towns, and settlements including non-housing development typically integrated with housing. (Examples: urban areas, suburbs, vacation homes, shopping areas, offices, schools).
   
   This category dovetails somewhat arbitrarily with 1.2 Commercial & Industrial Areas. As a general rule, however, if people live in or directly around the footprint of the area in question, it should fall into this category. Tourism facilities within a municipal area should generally go here and not in 1.3 Recreation & Tourism Areas.
   
   1.2 Commercial & Industrial Areas: Factories and other commercial centers. (Examples: stand-alone office parks, manufacturing plants, military bases, power plants, landfills, ports, airports).
   
   Ports and airports fall into this category, whereas shipping lanes and flight paths fall under 4. Transportation, Service & Security Corridors. Hydropower dams are NOT included here, but are in 7.2 Dams & Water Management / Use.
   
   1.3 Recreation & Tourism Areas: Tourism and recreation sites with a substantial spatial footprint. (Examples: visitor facilities in parks, campgrounds, ski areas, golf courses, marinas).
   
   This category focuses on the spatial footprint of recreation areas and facilities while 6.1 Recreational Activities focuses on the disturbance effects posed by recreational activities. There is a fine line between residential areas and tourism/resort areas; per discussion above, if the tourism area is within municipal boundaries, it probably belongs in 1.1 Residential Areas. Trails and other linear tourism features belong in 4.1. Roads, Trails & Railroads.

2. Agriculture & Aquaculture: Farming and ranching including agricultural expansion, intensification, or practices with a spatial footprint; includes tree plantations, mariculture, and aquaculture.
   
   This order focuses on the footprint and operations of these activities. Agricultural and aquacultural pollution threats (e.g. drift of herbicides or run-off of fertilizers) should be included in the appropriate category under 9. Pollution.
   
   2.1 Annual & Perennial Non-Timber Crops: Crops planted for food, fodder, fiber, fuel, or other uses. (Examples: farms, oil palm plantations, orchards, vineyards, biofuel crops).
   
   "Shifting cultivation" refers to systems in which land is temporarily farmed and then abandoned for a period of time, not crop rotation or occasional fallow periods on annual crop lands. Crops grown in on-farm greenhouses belong here; those grown in urban greenhouses, vertical farm facilities, or in indoor 'factories' (e.g. marijuana) should be included in 1.2 Commercial & Industrial Areas.
   
   2.2 Wood & Pulp Plantations: Stands of trees planted for timber or fiber outside of natural forests, often with non-native species. (Examples: teak or eucalyptus plantations, firewood lots, christmas tree farms).
   
   If it is one or a couple timber species that are planted on a rotation cycle, it belongs here. If it is multiple species or enrichment plantings in a quasi-natural system, it belongs in 5.3 Logging, Harvesting & Controlling Trees.
   
   2.3 Terrestrial Animal Farming, Ranching & Herding: Domestic terrestrial animals raised in one location on farmed or non-local resources (farming); also domestic or semidomesticated animals allowed to roam in semi-natural areas (ranching) or the wild and supported by natural habitats (herding). (Examples: cattle feed lots, dairy farms, cattle ranching, chicken farms, goat/camel/yak herding).
   
   In farming, animals are kept in tight captivity; in ranching they are allowed to roam in larger more natural areas, and in herding they are using wild habitats. If a few animals are mixed in a subsistence cropping system, it belongs in 2.1 Annual & Perennial Non-Timber Crops. Foraging for wild resources for stall-fed animals falls under 5.2 Gathering, Collecting & Controlling Terrestrial Plants & Fungi. Growing crops for animal consumption falls under 2.1 Annual & Perennial Non-Timber Crops. Producing meat from animal cells in factories belongs in 1.2 Commercial & Industrial Areas.
   
   2.4 Marine & Freshwater Aquaculture: Aquatic species raised for harvest in artificial water bodies (analogous to terrestrial 'farming'), enclosures in natural waters (analogous to 'ranching'), or unenclosed natural waters (analogous to 'herding'). (Examples: fish ponds, shrimp production, salmon pens, seeded shellfish beds).
   
   It may seem strange to talk about 'farming, ranching, and herding" in an aquatic environment, but from a conservationist's point of view, the effects of each are generally analogous to their terrestrial counterparts. For convenience, we are including raising of aquatic plants, algae, and other non-animal species in this category. Note that producing aquatic organisms for commercial or recreational fishing belongs here and not in 5.4 Fishing, Harvesting & Controlling Aquatic Species. Producing them for conservation or restoration purposes belongs in 7.5 Biological System Management. Problems caused by escaped invasive or problematic animals, interbreeding with native species, disease transmission, and pollution from aquaculture should be coded in the appropriate categories in C. Additional Sources of Stress.

3. Energy Production & Mining: Extraction of non-biological resources, often widely dispersed across the land/seascape.
   
   This order contains activities that are generally more widely dispersed than the industrial sites in 1.2 Commercial & Industrial Areas. While they technically 'produce' energy, power plants and oil refineries are compact industrial sites that belong in 1.2 Commercial & Industrial Areas. Various forms of water use (for example, dams for hydro power) could conceivably be put in this order, but seem more related to other threats that involve alterations to hydrologic regimes. As a result, they should go in 7.2 Dams & Water Management / Use.
   
   3.1 Oil & Gas Exploration & Extraction: Exploring for and extracting petroleum and other liquid hydrocarbons. (Examples: oil wells, hydraulic fracturing, natural gas drilling).
   
   Oil refineries, LPG gas ports, and other activities with a compact footprint belong in 1.2 Commercial & Industrial Areas. Oil and gas pipelines go into 4.2 Utility & Service Lines. Oil spills that occur at the drill site or from oil tankers or pipelines should go in 9.1 Water-Borne & Other Effluent Pollution.
   
   3.2 Mining & Quarrying: Exploring for, developing and producing minerals and rocks. (Examples: coal mines, gold mines, rock quarries, sand/salt mining, guano harvesting).
   
   It is a judgement call whether deforestation caused by strip mining should be in this category or in 5.3 Logging, Harvesting & Controlling Trees - it depends on whether the primary motivation for the deforestation is access to the trees or to the minerals. Sediment or toxic chemical runoff from mining should be placed in 9.1 Water-Borne & Other Effluent Pollution if it is the major threat from a mining operation.
   
   3.3 Renewable Energy: Exploring, developing and producing renewable energy. (Examples: geothermal power, solar farms, wind farms, tidal farms).
   
   Arguably renewable energy production with a compact footprint (e.g. solar concentrators) could be put in 1.2 Commercial & Industrial Areas but we propose they belong here to keep it with other forms of renewable energy. Hydropower should be put in 7.2 Dams & Water Management / Use. Growing biofuels should be in 2.1 Annual or Perennial Non-Timber Crops or 2.2. Wood & Pulp Plantations. Wood pellet production from natural forests should be in 5.3 Logging, Harvesting & Controlling Trees.

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

Your goal is to identify the single most appropriate threat category representing the **immediate threat** affecting the species, not the ultimate underlying cause. For example, if a species has "low productivity" (the symptom) because of "mining pollution" (the threat), the correct classification is for pollution, not something generic.

---

### **Classification Rules & Instructions**

1. **Identify the Immediate Threat**: Focus on the direct threat affecting the species, not the ultimate root cause.
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
- Terms like "decline", "mortality", "population reduction" describe EFFECTS, not immediate threats
- Focus on the direct threat affecting the species, not the ultimate underlying cause

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
    
    # build lookup for all wildlife taxonomy data
    canon_llm_taxo = {}
    for _original_name, tax_data in llm_taxonomy_map_by_original_name.items():
        canonical_form = tax_data.get('canonical_form')
        if canonical_form: 
            canon_llm_taxo[canonical_form] = tax_data

    triplets_to_json = []
    for canonical_subject, predicate, obj, doi_val, evidence in triplets:
        subject_taxo = canon_llm_taxo.get(canonical_subject, {
            'error': f'No taxonomy for: {canonical_subject}',
            'is_bird': False
        })
        
        triplets_to_json.append({
            'subject': canonical_subject,
            'predicate': predicate,
            'object': obj,
            'doi': doi_val,
            'evidence': evidence,
            'taxonomy': subject_taxo
        })
    
    filtered_taxo_info = llm_taxonomy_map_by_original_name

    enriched_data = {
        'triplets': triplets_to_json,
        'taxonomic_info': filtered_taxo_info
    }
    
    target_file = output_path / "enriched_triplets.json"
    tmp_file = output_path / "enriched_triplets.json.tmp"
    with open(tmp_file, "w", encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp_file, target_file)
    print("Enriched triplets saved to enriched_triplets.json")

    # separate file for taxonomies to inspect
    if filtered_taxo_info:
        tax_target = output_path / "llm_wildlife_taxonomies.json"
        tax_tmp = output_path / "llm_wildlife_taxonomies.json.tmp"
        with open(tax_tmp, "w", encoding='utf-8') as f:
            json.dump(filtered_taxo_info, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tax_tmp, tax_target)
        print("Wildlife taxonomies saved to llm_wildlife_taxonomies.json")
    else:
        print("No wildlife taxonomies to save")

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
        model=llm_setup.get("model", "moonshotai/kimi-k2"),
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
        model=llm_setup.get("model", "moonshotai/kimi-k2"),
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

COARSE_IUCN_CATEGORIES = """
**IUCN-CMP Direct Threats Classification v4.0 - COARSE CATEGORIES**

Select the single most specific and relevant category from the list below that best represents the underlying cause of the threat.

**A. Use of Lands & Waters**
**Definition**: Human uses of land and water areas that have a substantial spatial footprint. Includes effects from their construction (e.g. ecosystem conversion), ongoing use, and abandonment.
**Exposition**: This class includes both the ecosystem conversion/degradation effects of the expansion and ongoing presence of these activities, but does not include their associated pollution effects. The threats in categories 1 and 2 are generally tied to a defined and relatively compact area, which distinguishes them from those in category 4 which have a long narrow footprint, and category 6 which do not have an explicit footprint.

1. **Residential, Commercial & Recreation Areas**: Human settlements, industrial areas, and other non-agricultural land uses with a substantial footprint.
   **Exposition**: There is potential overlap between these subcategories. In general, most activities inside a defined municipal area should be in 1.1 whereas 1.2 and 1.3 are more for stand-alone developments in otherwise natural spaces.
   (Examples: urban areas, suburbs, vacation homes, shopping areas, offices, schools, factories, manufacturing plants, military bases, power plants, landfills, ports, airports, visitor facilities in parks, campgrounds, ski areas, golf courses, marinas)

2. **Agriculture & Aquaculture**: Farming and ranching including agricultural expansion, intensification, or practices with a spatial footprint; includes tree plantations, mariculture, and aquaculture.
   **Exposition**: This order focuses on the footprint and operations of these activities. Agricultural and aquacultural pollution threats (e.g. drift of herbicides or run-off of fertilizers) should be included in the appropriate category under 9. Pollution.
   (Examples: farms, oil palm plantations, orchards, vineyards, biofuel crops, teak or eucalyptus plantations, firewood lots, christmas tree farms, cattle feed lots, dairy farms, cattle ranching, chicken farms, goat/camel/yak herding, fish ponds, shrimp production, salmon pens, seeded shellfish beds)

3. **Energy Production & Mining**: Extraction of non-biological resources, often widely dispersed across the land/sea scape.
   **Exposition**: This order contains activities that are generally more widely dispersed than the industrial sites in 1.2. While they technically 'produce' energy, power plants and oil refineries are compact industrial sites that belong in 1.2. Various forms of water use (for example, dams for hydro power) could conceivably be put in this order, but seem more related to other threats that involve alterations to hydrologic regimes and should go in 7.2.
   (Examples: oil wells, hydraulic fracturing, natural gas drilling, coal mines, gold mines, rock quarries, sand/salt mining, guano harvesting, geothermal power, solar farms, wind farms, tidal farms)

4. **Transportation, Service & Security Corridors**: Linear infrastructure such as long, narrow service or transport corridors including the effects associated with their use (e.g. mortality from vehicle collisions, restriction of species movement).
   **Exposition**: This order focuses on corridors outside of human settlements and industrial developments. These corridors create specific stresses to biodiversity including especially loss and fragmentation of habitats and direct killing of wildlife and lead to other threats including the spread of farms, invasive species, and poachers.
   (Examples: highways, logging roads, bridges, vehicle collisions with wildlife, railroads, electrical & phone wires, aqueducts, oil & gas pipelines, electrocution of wildlife, shipping channels, canals, ships running into whales, flight paths, jets impacting birds, drones, border walls, fences around farm fields, disease control fencing)

**B. Use / Management of Species & Ecosystems**
**Definition**: Human uses of biotic resources and disturbance from human presence or management actions in natural systems.
**Exposition**: Human actions in this group generally do not 'intend' to convert the ecosystem although the most destructive forms of logging, fishing or dam building can have this result. These uses typically do not have a large spatial footprint excepting again the most intense forms of resource extraction or ecosystem management.

5. **Biological Resource Use & Control**: Consumptive use of "wild" biological resources including deliberate and unintentional harvesting effects as well as persecution of specific species.
   **Exposition**: Consumptive use means that the resource is removed from the system or destroyed; multiple people cannot use the same resource as they could under 6. Human Intrusions & Disturbance. Threats in this class can affect targeted species or they can intentionally or unintentionally affect secondary 'bycatch' species and ecosystems. Persecution/control involves harming or killing species because they are considered undesirable.
   (Examples: subsistence hunting, commercial wild meat hunting, trophy hunting, pet trade, culling, killing crop-raiding animals, gathering wild fruit, mushrooms, herbs for medicine, rubber tapping, clear-cutting, selective logging, pulp operations, fuel wood collection, net fishing, trawling, blast fishing, whaling, seal hunting, shellfish harvesting, live coral/aquarium fish collection)

6. **Human Intrusions & Disturbances**: Human activities that alter, disturb, and destroy ecosystems and species associated with non-consumptive uses of biological areas and resources.
   **Exposition**: Non-consumptive use means that the resource is not removed - multiple people can use the same resource (for example, birdwatching). These threats typically do not permanently destroy ecosystems except in extremely severe manifestations. Pollution from these activities belongs in the appropriate category in 9. Pollution.
   (Examples: hikers, off-road vehicles, motorboats, jet-skis, whale watching, pets in recreational areas, armed conflict, military training exercises, border patrols, abandoned land mines, drug smuggling, species research, vandalism)

7. **Natural System Management & Modifications**: Human actions that modify ecosystem structures, composition, or regimes, generally to deliberately improve human welfare or benefit certain species. This category includes both construction of permanent or long-term structures and their operations as well as more transitory management practices.
   **Exposition**: This order deals primarily with human caused changes to natural ecosystem processes such as fire, hydrology, and sedimentation, rather than land use. Thus it does not include threats relating to infrastructure (categories 1 and 4) or agriculture (category 2). It also includes the removal of management actions on which ecosystems and species now depend.
   (Examples: fire suppression, inappropriate fire management, escaped agricultural fires, arson, campfires, dam construction/operation, levees, channelization, groundwater pumping, surface water withdrawals, dune stabilization, shoreline armoring, dredging, mine reclamation, cloud seeding, ocean 'fertilization', mowing grass, removal of snags from streams, artificial reef creation, bird feeders, electric barriers to stop invasive fish, lack of mowing of meadows, cessation of grazing, stopping predator control)

**C. Additional Sources of Stress**
**Definition**: Stressors in natural systems that have been altered by the effects of current or historical human actions.
**Exposition**: Many of the entries in this class are the result of other direct threats. For example, agricultural practices or commercial shipping could lead to invasive species, pollution from toxic chemicals, or the release of greenhouse gasses that drive climate change. But there are many situations in which invasive species, pollution, or climate change impacts might be a problem in a project area, but it is not clear what the source of these threats are.

8. **Invasive / Other Problematic Species, Genes & Pathogens**: Threats from non-native and native plants, animals, pathogens/microbes, or genetic materials that have or are predicted to have harmful effects on biodiversity following their introduction, spread and/or increase in abundance or virulence.
   **Exposition**: We restrict the use of "invasive species" to non-native species to keep things simple for policy makers. "Problematic native species" are native species that have become superabundant or otherwise cause problems due to human alterations of the ecosystem. "Pathogens" are generally microorganisms that directly cause disease in individual organisms.
   (Examples: rats on islands, feral horses, zebra mussels, stocking exotic fish, ballast water discharge, overabundant native deer, algal blooms, insect outbreaks, hatchery salmon breeding with wild fish, domestic cats breeding with wild cats, genetically modified organisms, plague, Dutch elm disease, Chytrid fungus affecting amphibians)

9. **Pollution**: Introduction of exotic and/or excess materials or energy from point and nonpoint sources.
   **Exposition**: This order deals with exotic or excess materials introduced to the environment, which often have a different spatial area of impact than their source human activity. There is obviously a fine distinction when the pollution comes from another threat - for example, should an oil spill from a pipeline be classified as 4.2 or 9.1? You will have to exercise some judgement as to which represents the direct threat in your situation.
   (Examples: municipal waste discharge, leaking septic systems, fertilizer/pesticide run-off, oil spills, mine tailings, road salt, municipal waste, litter, agricultural plastics, microplastics, ghost fishing gear, construction debris, acid rain, smog from vehicle emissions, smoke from fires, radioactive fallout, beach lights disorienting turtles, noise from highways/airplanes, seismic exploration, sonar)

**D. Other Events & Factors**

10. **Natural Disasters**: Potentially catastrophic natural disturbances that conservation practitioners may still need to consider, particularly when managing small and/or remnant species populations or ecosystems.
    **Exposition**: Even though these may be 'natural' system disturbances, nonetheless, if you are charged with conserving a small remnant species population or ecosystem, you may have to take these factors into account in planning and prioritizing threats.
    (Examples: Volcanoes, earthquakes, tsunamis, avalanches, landslides, storms, hurricanes, blizzards, floods, droughts as discrete events)

11. **Climate Change**: Change in climate patterns resulting from increased atmospheric greenhouse gasses.
    **Exposition**: Strictly speaking individual climatic events may be part of natural disturbance regimes in many ecosystems and are thus technically "stresses" and not "direct threats." But they act as a threat if a species or ecosystem is damaged from other threats and has lost its resilience and is thus vulnerable to the disturbance. As a general rule, the Conservation Measures Partnership recommends coding the most immediate impacts of climate change as direct threats.
    (Examples: ocean acidification, changes in salinity, changes in ocean currents, heat waves, cold spells, oceanic temperature changes, loss of glaciers/sea ice, changes in rainfall patterns, long-term droughts, increased severity of floods, sea-level rise)

12. **Unknown Threats**: Required by IUCN for cases in which the threats to species are unknown.
"""

async def get_coarse_iucn_classification( subject: str, predicate: str, threat_desc: str, llm_setup, cache: SimpleCache, abstract: Optional[str] = None) -> tuple[str, float, str, List[dict]]:
    cache_key = f"coarse_iucn:{threat_desc}|context:{subject}|{predicate}|abstract:{bool(abstract)}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Coarse IUCN cache hit: '{threat_desc[:50]}...'")
        return cached_result

    if abstract:
        logger.info(f"Coarse classifying threat: '{threat_desc[:50]}...' with abstract context")
    else:
        logger.info(f"Coarse classifying threat: '{threat_desc[:50]}...' without abstract context")

    coarse_schema = {
        "type": "object",
        "properties": {
            "reasoning_chain": {
                "type": "string", 
                "description": "Step-by-step reasoning: 1) What is the human activity? 2) What category does this fall under? 3) Why this category vs others?"
            },
            "top_categories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category_number": {"type": "string", "description": "Category number (1-12)"},
                        "category_name": {"type": "string", "description": "Category name"},
                        "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    },
                    "required": ["category_number", "category_name", "probability"]
                },
                "minItems": 1,
                "maxItems": 3,
                "description": "Top 1-3 categories in order of likelihood, probabilities should sum to 1.0"
            }
        },
        "required": ["reasoning_chain", "top_categories"]
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
        Use the 'Threat Mechanism First' principle to classify this threat into broad IUCN categories.

        **STEP 1: IDENTIFY THE THREAT MECHANISM (IGNORE KEYWORDS)**
        What is the fundamental mechanism causing harm to the species?

        - **Biological System Changes?** (vegetation overgrowth, encroachment, succession, habitat conversion through natural processes) → **Category 7**
        - **Biological Agent?** (invasive species, native wildlife overpopulation, pathogens) → **Category 8** 
        - **Chemical/Physical Pollutant?** (toxic substances, noise, light) → **Category 9**
        - **Habitat Transformation for Development?** (land cleared for cities, farms, infrastructure) → **Categories 1-2**
        - **Resource Extraction?** (hunting, logging, fishing) → **Category 5**
        - **Infrastructure/Transportation?** (collisions, barriers) → **Category 4**

        **STEP 2: APPLY MECHANISM TO CATEGORY**
        Based on STEP 1, which category (1-12) matches the mechanism?

        **STEP 3: VERIFY WITH KEYWORDS** 
        Do the keywords support your mechanism-based choice? If not, trust the mechanism over keywords.

        {COARSE_IUCN_CATEGORIES}

        Provide your reasoning chain and rank the top 1-3 most likely categories.
        """

    system_prompt = """You are an expert ecologist applying the IUCN-CMP Direct Threats Classification. Your primary goal is to identify the **immediate threat's fundamental mechanism** to ensure accurate classification.

        ### **MANDATORY APPROACH: Threat Mechanism First - IGNORE KEYWORDS**

        **YOU MUST CLASSIFY BY MECHANISM, NOT BY KEYWORDS.** 

        Before looking at any IUCN categories, identify the fundamental mechanism:
        - Is it a **Biological Agent**? (e.g., invasive species, problematic native algae/deer, pathogens) -> This points toward **IUCN 8**.
        - Is it a **Chemical/Physical Pollutant**? (e.g., oil spills, fertilizer runoff, road salt, noise) -> This points toward **IUCN 9**.
        - Is it **Habitat Transformation**? (e.g., clearing land for farms, building cities) -> This points toward **IUCN 1 or 2**.
        - Is it **Biological System Changes**? (e.g., vegetation overgrowth, succession, eutrophication effects, fire suppression effects) -> This points toward **IUCN 7**.
        - Is it **Direct Resource Extraction**? (e.g., hunting, logging, fishing) -> This points toward **IUCN 5**.
        - Is it **Infrastructure/Transportation**? (e.g., road collisions, power line electrocution) -> This points toward **IUCN 4**.

        **The mechanism determines the category:**
        - "Toxins" from "algal bloom" → **Biological Agent** (8), not Pollution (9)
        - "Overgrazing" from "reindeer" → **Biological Agent** (8), not Agriculture (2)
        - Any "vegetation changes/overgrowth/encroachment" → **Biological System Changes** (7)

        ### **Instructions**

        1. **Focus on the Immediate Cause**: Classify the direct threat described, not its distant, underlying cause.
        2. **Analyze the Mechanism**: Follow the "Threat Mechanism First" principle to guide your choice.
        3. **Justify Your Choice**: Your reasoning must explain *why* the threat's mechanism fits your chosen category better than the next best alternative.

        **AVOID CLASSIFYING SYMPTOMS**: "Population decline," "low productivity," or "mortality" are effects, not threats. Identify the agent or action that *causes* these effects.

        ### **Classification Rules & Instructions**

        1. **Identify the Immediate Threat**: Focus on the direct threat affecting the species, not the ultimate root cause.
        2. **Select the Most Specific Category**: Choose the most detailed category that is directly supported by the evidence.
        3. **Provide Chain-of-Reasoning**: Explain *why* your chosen category is correct, referencing the evidence.

        ### **Specific Disambiguation Rules**

        To ensure consistency, apply the following specific rules for commonly confused categories:

        * **Pollution (IUCN 9) vs. Resource Use (IUCN 5)**:
            * **9. Pollution**: Use when the threat is the pollutant itself (e.g., lead contamination, chemical runoff, plastic debris).
            * **5. Biological Resource Use**: Use when the threat is the harvesting/hunting activity that may incidentally create pollution (e.g., hunting that leaves lead shot, fishing that creates ghost nets).

        * **Energy (IUCN 3) vs. Transportation (IUCN 4) Collisions**:
            * **3. Energy Production**: Use when the threat is explicitly the development or presence of energy infrastructure (e.g., wind farms causing habitat loss).
            * **4. Transportation Corridors**: Use for direct collisions with infrastructure (e.g., power line electrocution, vehicle collisions) unless the infrastructure is explicitly part of an energy project.

        * **Climate (IUCN 11) vs. Weather (IUCN 10)**:
            * **10. Natural Disasters**: Use for episodic events like storms, hurricanes, heatwaves, and named events (El Niño/La Niña).
            * **11. Climate Change**: Use for long-term trends, regime shifts, and persistent pattern changes (e.g., long-term warming, multi-year droughts).

        * **Unknown Threats (12)**: Use only when absolutely no other category applies and no immediate threat can be identified from the evidence provided.

        **THREAT DEFINITION:** A threat is a direct, external factor that causes or contributes to the degradation, loss, or impairment of a species or ecosystem. Exclude:
        - Natural ecological processes (predator-prey dynamics, migration, food availability changes)
        - Survey methodologies or research activities (unless harmful to species)
        - Intrinsic population dynamics (intraspecific competition, natural barriers)
        - Neutral ecological observations (species distributions, habitat preferences)

        **ECOLOGICAL TERMINOLOGY:** Use proper conservation terminology:
        - "Productivity" = reproductive output (breeding success, clutch size, fecundity)
        - "Low productivity" is a SYMPTOM, not a threat - identify what caused the low productivity
        - Terms like "decline", "mortality", "population reduction" describe EFFECTS, not immediate threats
        - Focus on the direct threat affecting the species, not the ultimate underlying cause

        Chain-of-reasoning approach:
        1. Identify the immediate threat affecting the species
        2. Map it to the appropriate broad category  
        3. Consider alternative interpretations and explain why your choice is better
        4. Assign probability estimates to your top 3 choices

        Focus on the IMMEDIATE THREAT, not symptoms like "mortality", "decline", or "reduced breeding success"."""

    response_result = await llm_generate(
        prompt=prompt,
        system=system_prompt,
        model=llm_setup.get("model", "moonshotai/kimi-k2"),
        temp=0.1,
        format=coarse_schema,
        llm_setup=llm_setup,
        logprobs=True,
        top_logprobs=5
    )

    if isinstance(response_result, tuple) and len(response_result) >= 2:
        response_str, logprobs_info = response_result[0], response_result[1]
    else:
        response_str = extract_content_from_result(response_result)
        logprobs_info = None
    
    if response_str:
        clean_json = strip_markdown_json(response_str)
        try:
            result_json = json.loads(clean_json)
            reasoning = result_json.get("reasoning_chain", "")
            categories = result_json.get("top_categories", [])
            
            if categories and len(categories) > 0:
                top_category = categories[0]
                category_num = top_category.get("category_number", "")
                self_reported_confidence = top_category.get("probability", 0.0)
                logger.debug(f"Extracted from JSON: category_num='{category_num}', confidence={self_reported_confidence}")
                logprob_confidences = extract_all_category_confidences_from_logprobs(logprobs_info)
                logprob_confidence = logprob_confidences.get(category_num)
                
                actual_confidence = logprob_confidence if logprob_confidence is not None else self_reported_confidence
                
                enhanced_categories = []
                for cat in categories:
                    cat_num = cat.get("category_number", "")
                    original_prob = cat.get("probability", 0.0)
                    logprob_prob = logprob_confidences.get(cat_num, original_prob)
                    
                    enhanced_cat = cat.copy()
                    enhanced_cat["logprob_probability"] = logprob_prob
                    enhanced_cat["confidence_source"] = "logprob" if cat_num in logprob_confidences else "self-reported"
                    enhanced_categories.append(enhanced_cat)
                
                if logprob_confidences:
                    enhanced_categories.sort(key=lambda x: x.get("logprob_probability", 0), reverse=True)
                
                if enhanced_categories and logprob_confidences:
                    top_enhanced = enhanced_categories[0]
                    final_category_num = top_enhanced.get("category_number", category_num)
                    final_confidence = top_enhanced.get("logprob_probability", actual_confidence)
                else:
                    final_category_num = category_num
                    final_confidence = actual_confidence
                
                if final_category_num.isdigit() and 1 <= int(final_category_num) <= 12:
                    result = (final_category_num, final_confidence, reasoning, enhanced_categories or categories)
                    cache.set(cache_key, result)
                    
                    confidence_source = "logprob" if final_confidence == logprob_confidence else "self-reported"
                    logger.info(f"Coarse classified as: Category {final_category_num} with {final_confidence:.3f} confidence ({confidence_source})")
                    
                    if logprob_confidences:
                        top_3_logprob = sorted(logprob_confidences.items(), key=lambda x: x[1], reverse=True)[:3]
                        logger.info(f"Top 3 categories by logprob confidence:")
                        for i, (cat, conf) in enumerate(top_3_logprob, 1):
                            logger.info(f"  {i}. Category {cat}: {conf:.3f} confidence")
                        
                        all_cats = sorted(logprob_confidences.items(), key=lambda x: x[1], reverse=True)
                        logger.debug(f"All categories from logprobs: {[(cat, f'{conf:.4f}') for cat, conf in all_cats]}")
                    
                    if logprob_confidence is not None and abs(logprob_confidence - self_reported_confidence) > 0.2:
                        logger.warning(f"Confidence mismatch: logprob={logprob_confidence:.3f} vs self-reported={self_reported_confidence:.3f}")
                    
                    logger.info(f"Reasoning: {reasoning[:100]}...")
                    return result
                else:
                    logger.error(f"Invalid category number: '{final_category_num}' (type: {type(final_category_num)})")
                    logger.error(f"Top category data: {top_category}")
            else:
                logger.error(f"No categories found in JSON response. Categories: {categories}")
                logger.error(f"Full JSON: {result_json}")
                    
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Coarse classification JSON error: {e}")
            logger.error(f"Raw response: '{response_str[:200]}...'")
            logger.error(f"Cleaned JSON: '{clean_json[:200]}...'")
    
    result = ("12", 0.0, "Classification failed - defaulting to Unknown", [])
    cache.set(cache_key, result)
    return result


async def get_fine_iucn_classification( coarse_category: str, subject: str, predicate: str, threat_desc: str, llm_setup, cache: SimpleCache, abstract: Optional[str] = None) -> tuple[str, str, str]:
    cache_key = f"fine_iucn:{coarse_category}|{threat_desc}|context:{subject}|{predicate}|abstract:{bool(abstract)}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Fine IUCN cache hit for category {coarse_category}: '{threat_desc[:50]}...'")
        return cached_result
    logger.info(f"Fine classifying in category {coarse_category}: '{threat_desc[:50]}...'")

    subcategory_text = get_subcategories_for_coarse_category(coarse_category)
    
    fine_schema = {
        "type": "object",
        "properties": {
            "reasoning_chain": {
                "type": "string",
                "description": "Step-by-step reasoning: 1) What specific mechanism/activity is described? 2) Which subcategory best fits? 3) Why this specific code?"
            },
            "iucn_code": {"type": "string", "description": "Specific IUCN code like '9.1' or '5.4'"},
            "iucn_name": {"type": "string", "description": "Full name of the specific category"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Confidence in this specific classification"}
        },
        "required": ["reasoning_chain", "iucn_code", "iucn_name", "confidence"]
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

        **CONTEXT:** 
        Based on coarse classification, this threat belongs to broad category {coarse_category}. Now identify the most specific subcategory.

        **REASONING STEPS:**
        1. What is the immediate threat affecting the species?
        2. Looking at the subcategories below, which one best matches this threat?
        3. Why is this the most appropriate specific code rather than others in this category?

        {subcategory_text}

        Provide your reasoning and select the single most appropriate specific code within this category.
        """

    system_prompt = f"""You are an expert ecologist specializing in threat classification within IUCN category {coarse_category}. Your task is to select the most specific subcategory using the IUCN-CMP Direct Threats Classification v4.0.

        ### **Apply Threat Mechanism First Principle**

        Continue applying the mechanism-first approach from coarse classification:
        - **Biological System Changes** (vegetation overgrowth, succession, encroachment) → Category 7
        - **Habitat Transformation** (land conversion for development) → Category 1-2  
        - **Chemical Pollutants** (toxic substances) → Category 9
        - **Biological Agents** (invasive species, wildlife) → Category 8

        **CRITICAL: Focus on the MECHANISM, not the trigger:**

        Your goal is to identify the single most appropriate subcategory representing the **immediate threat's mechanism** affecting the species.

        ### **Classification Rules & Instructions**

        1. **Identify the Immediate Threat**: Focus on the direct threat affecting the species, not the ultimate root cause.
        2. **Select the Most Specific Subcategory**: Choose the most detailed sub-category that is directly supported by the evidence.
        3. **Provide Chain-of-Reasoning**: Explain *why* your chosen subcategory is correct, referencing the evidence.

        ### **Specific Disambiguation Rules for Category {coarse_category}**

        {get_disambiguation_rules_for_category(coarse_category)}

        **THREAT DEFINITION:** A threat is a direct, external factor that causes or contributes to the degradation, loss, or impairment of a species or ecosystem.

        Chain-of-reasoning approach:
        1. Identify the immediate threat affecting the species
        2. Match to the most appropriate subcategory based on the evidence
        3. Explain why this choice is better than alternatives within this category
        4. Provide confidence level in your classification

        Be precise - choose the most specific code that is directly supported by the evidence."""
            
    response_result = await llm_generate(
        prompt=prompt,
        system=system_prompt,
        model=llm_setup.get("model", "moonshotai/kimi-k2"),
        temp=0.0,
        format=fine_schema,
        llm_setup=llm_setup,
        logprobs=True,
        top_logprobs=5
    )

    response_str = extract_content_from_result(response_result)
    if response_str:
        clean_json = strip_markdown_json(response_str)
        try:
            result_json = json.loads(clean_json)
            reasoning = result_json.get("reasoning_chain", "")
            code = result_json.get("iucn_code", "")
            name = result_json.get("iucn_name", "")
            confidence = result_json.get("confidence", 0.0)
            
            if (isinstance(code, str) and isinstance(name, str) and 
                code.strip() and name.strip() and
                re.match(r"^\d+\.\d+$", code.strip()) and
                code.startswith(coarse_category + ".")):
                
                result = (code.strip(), name.strip(), reasoning)
                cache.set(cache_key, result)
                logger.info(f"Fine classified as: {code} - {name} (confidence: {confidence:.2f})")
                logger.info(f"Reasoning: {reasoning[:100]}...")
                return result
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Fine classification JSON error: {e}")
            logger.error(f"Response: '{response_str[:200]}...'")
    
    result = (f"{coarse_category}.0", f"Category {coarse_category} (Unspecified)", "Fine classification failed")
    cache.set(cache_key, result)
    return result


def get_subcategories_for_coarse_category(category_num: str) -> str:
    
    subcategory_map = {
        "1": """
1. Residential, Commercial & Recreation Areas:
   1.1 Residential Areas: Cities, towns, settlements, housing development
   1.2 Commercial & Industrial Areas: Factories, manufacturing, military bases, airports
   1.3 Recreation & Tourism Areas: Parks, campgrounds, ski areas, golf courses
""",
        "2": """
2. Agriculture & Aquaculture:
   2.1 Annual & Perennial Non-Timber Crops: Farms, plantations, biofuel crops
   2.2 Wood & Pulp Plantations: Tree plantations, timber operations
   2.3 Terrestrial Animal Farming: Cattle, dairy farms, ranching, livestock
   2.4 Marine & Freshwater Aquaculture: Fish ponds, salmon pens, aquaculture
""",
        "3": """
3. Energy Production & Mining:
   3.1 Oil & Gas Exploration: Oil wells, fracking, natural gas drilling
   3.2 Mining & Quarrying: Coal mines, metal mining, rock quarries, sand mining
   3.3 Renewable Energy: Solar farms, wind farms, geothermal, tidal energy
""",
        "4": """
4. Transportation, Service & Security Corridors:
   4.1 Roads, Trails & Railroads: Highways, vehicle collisions, railroads
   4.2 Utility & Service Lines: Power lines, pipelines, electrocution
   4.3 Shipping Lanes: Shipping channels, vessel strikes, canals
   4.4 Atmospheric & Space Activities: Flight paths, aircraft collisions
   4.5 Fencing & Walls: Border walls, farm fencing, barriers to movement
""",
        "5": """
5. Biological Resource Use & Control:
   5.1 Hunting & Collecting Terrestrial Animals: Hunting, trapping, pet trade
   5.2 Gathering Terrestrial Plants & Fungi: Plant harvesting, medicine collection
   5.3 Logging & Tree Harvesting: Forest logging, fuel wood collection
   5.4 Fishing & Aquatic Harvesting: Net fishing, trawling, whaling, shellfish
""",
        "6": """
6. Human Intrusions & Disturbances:
   6.1 Recreational Activities: Hikers, off-road vehicles, boats, whale watching
   6.2 Conflict & Security Activities: Armed conflict, military exercises
   6.3 Other Human Disturbances: Research activities, vandalism, smuggling
""",
        "7": """
7. Natural System Management & Modifications:
   7.1 Fire & Fire Management: Fire suppression, prescribed burns, arson
   7.2 Dams & Water Management: Dam construction, water diversions, pumping
   7.3 Earth & Sediment Management: Dredging, shoreline modification, mining reclamation
   7.4 Weather & Climate Management: Cloud seeding, geoengineering
   7.5 Biological System Management: Mowing, snag removal, artificial reefs
   7.6 Removing Human Management: Cessation of grazing, stopping predator control
""",
        "8": """
8. Invasive / Other Problematic Species:
   8.1 Invasive Non-Native Species: Introduced rats, zebra mussels, exotic fish
   8.2 Problematic Native Species: Overabundant deer, algal blooms
   8.3 Introduced Genetic Material: Hatchery fish breeding, GMOs
   8.4 Pathogens: Disease, fungal infections, plague
""",
        "9": """
9. Pollution:
   9.1 Water-Borne & Other Effluent Pollution: Water-borne and other liquid pollutants
   9.2 Garbage & Solid Waste: Rubbish and other solid materials
   9.3 Air-Borne Pollutants: Atmospheric pollutants
   9.4 Energy Emissions: Inputs of heat, sound, light, or other wave energy
""",
        "10": """
10. Natural Disasters:
    10.1 Geological Events: Volcanoes, earthquakes, tsunamis, landslides
    10.2 Severe Weather Events: Storms, hurricanes, floods, droughts (discrete events)
""",
        "11": """
11. Climate Change:
    11.1 Changes in Physical & Chemical Regimes: Ocean acidification, current changes
    11.2 Changes in Temperature Regimes: Heat waves, warming, glacial loss
    11.3 Changes in Precipitation & Hydrological Regimes: Rainfall changes, sea-level rise
""",
        "12": """
12. Unknown Threats:
    Use only when no other category is applicable and the threat cannot be identified.
"""
    }
    
    return subcategory_map.get(category_num, "No subcategories found for this category.")


def extract_all_category_confidences_from_logprobs(logprobs_info) -> Dict[str, float]:

    if not logprobs_info or not logprobs_info.content:
        return {}
    
    category_logprobs = {}
    valid_categories = [str(i) for i in range(1, 13)]
    
    try:
        for i, token_info in enumerate(logprobs_info.content):
            token_text = getattr(token_info, 'token', '') or token_info.get('token', '') if isinstance(token_info, dict) else ''
            token_logprob = getattr(token_info, 'logprob', -float('inf')) or token_info.get('logprob', -float('inf')) if isinstance(token_info, dict) else -float('inf')
            
            clean_token = token_text.strip('"\'')
            if clean_token in valid_categories:
                category_logprobs[clean_token] = max(category_logprobs.get(clean_token, -float('inf')), token_logprob)
            
            top_logprobs = getattr(token_info, 'top_logprobs', None) or token_info.get('top_logprobs') if isinstance(token_info, dict) else None
            if top_logprobs:
                for top_token in top_logprobs:
                    alt_token_text = getattr(top_token, 'token', '') or top_token.get('token', '') if isinstance(top_token, dict) else ''
                    alt_logprob = getattr(top_token, 'logprob', -float('inf')) or top_token.get('logprob', -float('inf')) if isinstance(top_token, dict) else -float('inf')
                    
                    clean_alt_token = alt_token_text.strip('"\'')
                    if clean_alt_token in valid_categories:
                        category_logprobs[clean_alt_token] = max(category_logprobs.get(clean_alt_token, -float('inf')), alt_logprob)
    
        category_probs = {}
        total_prob = 0.0
        
        for cat, logprob in category_logprobs.items():
            if logprob > -float('inf'):
                prob = math.exp(logprob)
                category_probs[cat] = prob
                total_prob += prob
        
        if total_prob > 0:
            for cat in category_probs:
                category_probs[cat] /= total_prob
        
        logger.info(f"Extracted category probabilities from logprobs: {category_probs}")
        logger.debug(f"Raw logprob data for categories: {category_logprobs}")
        return category_probs
        
    except Exception as e:
        logger.error(f"Error extracting category confidences: {e}")
        return {}


def extract_category_confidence_from_logprobs(logprobs_info, target_category: str, reasoning_text: str) -> Optional[float]:

    if not logprobs_info or not logprobs_info.content:
        return None
    
    logger.debug(f"Analyzing logprobs for category {target_category} across {len(logprobs_info.content)} tokens")
    
    target_tokens = [target_category, f'"{target_category}"', f"'{target_category}'"]
    best_logprob = -float('inf')
    
    try:
        for i, token_info in enumerate(logprobs_info.content):
            token_text = getattr(token_info, 'token', '') or token_info.get('token', '') if isinstance(token_info, dict) else ''
            token_logprob = getattr(token_info, 'logprob', -float('inf')) or token_info.get('logprob', -float('inf')) if isinstance(token_info, dict) else -float('inf')
            
            if token_text.strip('"\'') == target_category:
                best_logprob = max(best_logprob, token_logprob)
                logger.debug(f"Found target category token '{token_text}' with logprob {token_logprob:.4f}")
            
            top_logprobs = getattr(token_info, 'top_logprobs', None) or token_info.get('top_logprobs') if isinstance(token_info, dict) else None
            if top_logprobs:
                for top_token in top_logprobs:
                    alt_token_text = getattr(top_token, 'token', '') or top_token.get('token', '') if isinstance(top_token, dict) else ''
                    alt_logprob = getattr(top_token, 'logprob', -float('inf')) or top_token.get('logprob', -float('inf')) if isinstance(top_token, dict) else -float('inf')
                    
                    if alt_token_text.strip('"\'') == target_category:
                        best_logprob = max(best_logprob, alt_logprob)
                        logger.debug(f"Found target category in alternatives: '{alt_token_text}' with logprob {alt_logprob:.4f}")
        
        if best_logprob > -float('inf'):
            confidence = min(1.0, max(0.0, math.exp(best_logprob)))
            logger.debug(f"Extracted logprob confidence for category {target_category}: {confidence:.4f} (logprob: {best_logprob:.4f})")
            return confidence
        else:
            logger.debug(f"No logprob found for category {target_category}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting logprob confidence: {e}")
        return None


def get_disambiguation_rules_for_category(category_num: str) -> str:
    return """
**Apply the Threat Mechanism First principle:**

1. **Identify the fundamental mechanism** causing the threat:
   - Biological Agent (organism behavior/presence) → Usually Category 8
   - Chemical/Physical Pollutant (inert substance/energy) → Usually Category 9  
   - Habitat Transformation (land conversion for purpose) → Usually Category 1-2
   - Resource Extraction (direct harvesting/removal) → Usually Category 5
   - Infrastructure/Transportation (barriers/collisions) → Usually Category 4

2. **Focus on the DIRECT agent** causing harm, not:
   - Keywords or superficial descriptors
   - Ultimate root causes
   - Secondary effects or symptoms

3. **Examples of mechanism-first thinking**:
   - "Red tide toxins" → Biological agent (algae) causing harm → Category 8
   - "Reindeer overgrazing" → Biological agent (wildlife) causing harm → Category 8
   - "Eutrophication and vegetation overgrowth" → Biological system changes → Category 7
   - "Reforestation eliminating grassland" → Biological system changes → Category 7
   - "Mining runoff pollution" → Chemical pollutant → Category 9
   - "Road vehicle collisions" → Infrastructure-related → Category 4

**The mechanism determines the category, not the keywords.**
"""


async def get_hierarchical_iucn_classification(subject: str, predicate: str,threat_desc: str, llm_setup, cache: SimpleCache, abstract: Optional[str] = None) -> tuple[str, str]:
    logger.info(f"=== STARTING HIERARCHICAL IUCN CLASSIFICATION ===")
    logger.info(f"Subject: {subject}")
    logger.info(f"Predicate: {predicate[:100]}...")
    logger.info(f"Threat: {threat_desc[:100]}...")
    logger.info(f"Has abstract: {abstract is not None}")
    
    logger.info(f"\n--- STEP 1: COARSE CLASSIFICATION ---")
    coarse_category, confidence, reasoning, all_categories = await get_coarse_iucn_classification(
        subject, predicate, threat_desc, llm_setup, cache, abstract
    )
    
    logger.info(f"Coarse classification result: Category {coarse_category} (confidence: {confidence:.3f})")
    
    logger.info(f"Coarse confidence: {confidence:.2f}")
    if len(all_categories) > 1:
        alt_categories = [f"{cat.get('category_number')} ({cat.get('probability', 0):.2f})" for cat in all_categories[1:]]
        logger.info(f"Alternative categories available: {alt_categories}")
    
    logger.info(f"\n--- STEP 2: FINE CLASSIFICATION ---")
    logger.info(f"Proceeding with coarse category {coarse_category} for fine classification")
    
    try:
        fine_code, fine_name, fine_reasoning = await get_fine_iucn_classification(
            coarse_category, subject, predicate, threat_desc, llm_setup, cache, abstract
        )
        
        logger.info(f"Fine classification result: {fine_code} - {fine_name}")
        logger.info(f"=== HIERARCHICAL CLASSIFICATION COMPLETE ===")
        logger.info(f"FINAL RESULT: {fine_code} - {fine_name}")
        return fine_code, fine_name
        
    except Exception as e:
        logger.error(f"Fine classification failed: {e}")
        fallback_code = f"{coarse_category}.0"
        fallback_name = f"Category {coarse_category} (Unspecified)"
        logger.info(f"Using fallback: {fallback_code} - {fallback_name}")
        return fallback_code, fallback_name


def parse_structured_predicate(predicate: str) -> dict:
    return {
        "summary": predicate,
        "evidence": "",
        "original": predicate
    }