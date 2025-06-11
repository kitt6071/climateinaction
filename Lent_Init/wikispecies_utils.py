import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import aiohttp
import asyncio
import time
import requests
from bs4 import BeautifulSoup


class RateLimiter:
    def __init__(self, rpm=2, ollama_mode=False):
        self.rpm = rpm 
        self.last_call = 0 
        self.wait_time = 60.0 / self.rpm
        self.backoff = 0
        self.ollama = ollama_mode
        self.min_wait = 0
        
    def wait(self):
        if self.ollama:
            return
            
        now = time.time()
        elapsed = now - self.last_call
        
        wait_needed = max(self.wait_time - elapsed, self.min_wait)
        
        if self.backoff > 0:
            wait_needed = max(wait_needed, self.backoff)
            self.backoff *= 0.5
        
        if wait_needed > 0:
            print(f"waiting {wait_needed:.1f}s")
            time.sleep(wait_needed)
            self.last_call = time.time()
    
    def handle_rate_limit(self):
        self.backoff = max(60, self.backoff * 2)
        self.wait()

    async def async_wait(self):
        if self.ollama:
            return

        now = time.time()
        elapsed = now - self.last_call
        wait_time = max(self.wait_time - elapsed, self.min_wait)

        if self.backoff > 0:
            wait_time = max(wait_time, self.backoff)

        if wait_time > 0:
            print(f"async waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            self.last_call = time.time()
    
    def handle_async_rate_limit(self):
        self.backoff = max(60, self.backoff * 2)
        print(f"hit rate limit, backing off: {self.backoff}s")


# Wikispecies API for taxonomy
class WikispeciesClient:
    def __init__(self):
        self.base_url = "https://species.wikimedia.org/w/api.php"
        self.limiter = RateLimiter(rpm=30)
        self.results = []
        
    def search_species(self, name: str):
        self.limiter.wait()
        
        result = {
            'query_name': name,
            'found_name': None,
            'page_id': None,
            'taxonomy': None,
            'error': None,
            'raw_response': None
        }
        
        try:
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': name,
                'format': 'json'
            }
            
            print(f"searching {name}")
            resp = requests.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            result['raw_response'] = data
            
            search_results = data.get('query', {}).get('search')
            if search_results:
                # just take the first one for now
                first = search_results[0]
                result['found_name'] = first['title']
                result['page_id'] = first['pageid']
                
                content = self.get_page_content(first['title'])
                if content:
                    result['taxonomy'] = self.parse_taxonomy(content)
                
                print(f"found {result['found_name']}")
            else:
                result['error'] = "nothing found"
                print(f"no results for {name}")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"error searching {name}: {e}")
        
        self.results.append(result)
        return result
    
    def get_page_content(self, title: str):
        self.limiter.wait()
        
        params = {
            'action': 'parse',
            'page': title,
            'format': 'json',
            'prop': 'text'
        }
        
        try:
            resp = requests.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            text_content = data.get('parse', {}).get('text', {}).get('*')
            if text_content:
                return text_content
            return None
            
        except Exception as e:
            print(f"error getting page {title}: {e}")
            return None
    
    def parse_taxonomy(self, html_content: str):
        taxonomy = {
            'kingdom': None,
            'phylum': None,
            'class': None,
            'order': None,
            'family': None,
            'genus': None,
            'species': None,
            'rank_hierarchy': []
        }
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # find all paragraphs and extract taxonomy info
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.text.strip()
                lines = text.split('\n')
                for line in lines:
                    taxonomy['rank_hierarchy'].append(line.strip())
                    
                    line_lower = line.lower()
                    if 'regnum: ' in line_lower and not taxonomy['kingdom']:
                        taxonomy['kingdom'] = line_lower.split('regnum: ')[1].strip()
                    elif 'phylum: ' in line_lower and not taxonomy['phylum']:
                        taxonomy['phylum'] = line_lower.split('phylum: ')[1].strip()
                    elif 'classis: ' in line_lower and not taxonomy['class']:
                        taxonomy['class'] = line_lower.split('classis: ')[1].strip()
                    elif 'ordo: ' in line_lower and not taxonomy['order']:
                        taxonomy['order'] = line_lower.split('ordo: ')[1].strip()
                    elif 'familia: ' in line_lower and not taxonomy['family']:
                        taxonomy['family'] = line_lower.split('familia: ')[1].strip()
                    elif 'genus: ' in line_lower and not taxonomy['genus']:
                        taxonomy['genus'] = line_lower.split('genus: ')[1].strip()
                    elif 'species: ' in line_lower and not taxonomy['species']:
                        taxonomy['species'] = line_lower.split('species: ')[1].strip()
            
        except Exception as e:
            print(f"error parsing taxonomy: {e}")
        
        return taxonomy
    
    def save_results(self, output_dir="results"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = Path(current_dir) / output_dir
        output_path.mkdir(exist_ok=True)
        
        # save json results
        with open(output_path / "wikispecies_results.json", "w", encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        # quick CSV too
        csv_file = output_path / "wikispecies_summary.csv"
        with open(csv_file, "w", encoding='utf-8') as f:
            f.write("query_name,found_name,page_id,error\n")
            for result in self.results:
                query = result['query_name']
                found = result['found_name'] or ''
                page_id = result['page_id'] or ''
                error = result['error'] or ''
                f.write(f"{query},{found},{page_id},{error}\n")
        
        print(f"saved to {output_path}/")

    async def search_species_async(self, name: str, session):
        await self.limiter.async_wait() 
        
        result = {
            'query_name': name, 'found_name': None, 'page_id': None,
            'taxonomy': None, 'error': None, 'raw_response': None
        }
        
        try:
            params = {
                'action': 'query', 'list': 'search',
                'srsearch': name, 'format': 'json'
            }
            print(f"async search {name}")
            
            async with session.get(self.base_url, params=params, timeout=30) as response:
                response.raise_for_status() 
                data = await response.json()
            
            result['raw_response'] = data
            
            search_hits = data.get('query', {}).get('search')
            if search_hits:
                first = search_hits[0]
                result['found_name'] = first['title']
                result['page_id'] = first['pageid']
                
                page_html = await self.get_page_content_async(first['title'], session)
                if page_html:
                    result['taxonomy'] = self.parse_taxonomy(page_html) 
                
                print(f"async found {result['found_name']}")
            else:
                result['error'] = "nothing found"
                print(f"async no results {name}")
                
        except aiohttp.ClientResponseError as http_err:
            err_msg = f"HTTP {http_err.status}: {http_err.message}"
            result['error'] = err_msg
            print(f"http error {name}: {err_msg}")
            if http_err.status == 429: 
                self.limiter.handle_async_rate_limit()
        except asyncio.TimeoutError:
            result['error'] = "timeout"
            print(f"timeout {name}")
        except Exception as e:
            result['error'] = str(e)
            print(f"error async search {name}: {e}")
        
        return result

    async def get_page_content_async(self, title: str, session):
        await self.limiter.async_wait()
        
        params = {
            'action': 'parse', 'page': title,
            'format': 'json', 'prop': 'text'
        }
        
        try:
            async with session.get(self.base_url, params=params, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
            
            page_text = data.get('parse', {}).get('text', {}).get('*')
            if page_text:
                return page_text
            return None
            
        except aiohttp.ClientResponseError as http_err:
            print(f"http error getting page {title}: {http_err.status}")
            if http_err.status == 429:
                self.limiter.handle_async_rate_limit()
            return None
        except asyncio.TimeoutError:
            print(f"timeout getting page {title}")
            return None
        except Exception as e:
            print(f"error getting page {title}: {e}")
            return None


async def verify_species_with_wikispecies_concurrently(species_list, run_results_path):
    """
    Check species names against Wikispecies API concurrently.
    Uses a shared lookup file plus saves run-specific log.
    """
    client = WikispeciesClient()
    
    # persistent lookup file (shared across runs)
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    lookup_file = script_dir / "wikispecies_taxonomy_lookup.json"
    
    run_results_path.mkdir(parents=True, exist_ok=True)
    run_log_file = run_results_path / "wikispecies_verification_log.json"
    
    print(f"lookup file: {lookup_file}")
    print(f"run log: {run_log_file}")

    entries = {}

    # load existing data
    if lookup_file.exists():
        try:
            with open(lookup_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    if isinstance(entry, dict) and 'query_name' in entry:
                        if isinstance(entry['query_name'], str):
                            entries[entry['query_name'].lower()] = entry
            print(f"loaded {len(entries)} cached entries")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"couldn't load lookup file: {e}")
            entries = {}
        except Exception as e:
            print(f"error loading {lookup_file}: {e}")
            entries = {}
    else:
        print("no existing lookup file found")

    to_fetch = []
    seen = set()

    # figure out what to fetch
    for species in species_list:
        if not isinstance(species, str) or not species.strip():
            continue

        species_key = species.lower()
        
        if species_key in seen:
            continue
        seen.add(species_key)

        if species_key not in entries:
            print(f"need to fetch: {species}")
            to_fetch.append(species)
        else:
            print(f"using cached: {species}")

    if to_fetch:
        print(f"fetching {len(to_fetch)} new species")
        timeout = aiohttp.ClientTimeout(total=60) 
        connector = aiohttp.TCPConnector(limit_per_host=5, ssl=False) 

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [client.search_species_async(name, session) for name in to_fetch]
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    original_name = to_fetch[i]
                    entry = None

                    if isinstance(result, Exception):
                        print(f"task failed for {original_name}: {result}")
                        entry = {
                            'query_name': original_name,
                            'found_name': None, 'page_id': None, 'taxonomy': None,
                            'error': f"task failed: {str(result)}",
                            'raw_response': None
                        }
                    elif isinstance(result, dict):
                        if result.get('query_name') != original_name:
                            print(f"fixing query_name mismatch for {original_name}")
                            result['query_name'] = original_name
                        entry = result
                    else: 
                        print(f"weird result type for '{original_name}': {type(result)}")
                        entry = {
                            'query_name': original_name,
                            'error': f"Unknown result type: {type(result)}",
                            'raw_response': None, 'taxonomy': None, 'page_id': None, 'found_name': None
                        }
                    
                    if entry and 'query_name' in entry:
                        entries[entry['query_name'].lower()] = entry
    else:
        print("all species already in cache")
    
    try:
        all_entries = list(entries.values())
        
        # update lookup file
        with open(lookup_file, "w", encoding='utf-8') as f:
            json.dump(all_entries, f, indent=2)
        print(f"updated lookup with {len(all_entries)} entries")
        
        # save run log
        with open(run_log_file, "w", encoding='utf-8') as f:
            json.dump(all_entries, f, indent=2)
        print("run log saved")
        
    except Exception as e:
        print(f"ERROR writing files: {e}")

    client.results = all_entries


def parse_wikispecies_rank_hierarchy(hierarchy_list):
    """Parse rank hierarchy from Wikispecies into standard dict."""
    ranks = {
        "kingdom": None, "phylum": None, "class": None, "order": None,
        "family": None, "genus": None, "species": None
    }
    
    # mapping from wikispecies terms to our standard ones
    mappings = {
        "superregnum": "kingdom", "regnum": "kingdom", "phylum": "phylum",
        "subphylum": "phylum", "classis": "class", "subclassis": "class",
        "ordo": "order", "familia": "family", "genus": "genus", "species": "species"
    }
    
    class_vals = []
    kingdom_vals = {}

    for item in hierarchy_list:
        if ':' not in item:
            continue
        
        parts = item.split(':', 1)
        if len(parts) != 2:
            continue
            
        rank_name = parts[0].strip().lower()
        rank_value = parts[1].strip()

        standard_key = mappings.get(rank_name)
        if standard_key:
            if standard_key == "class":
                class_vals.append(rank_value)
            elif standard_key == "kingdom":
                kingdom_vals[rank_name] = rank_value
            elif standard_key == "phylum":
                if rank_name == "phylum" or not ranks.get(standard_key):
                    ranks[standard_key] = rank_value
            elif standard_key == "species":
                ranks[standard_key] = rank_value
            else:
                ranks[standard_key] = rank_value
    
    # handle kingdom priority
    if 'regnum' in kingdom_vals:
        ranks["kingdom"] = kingdom_vals['regnum']
    elif 'superregnum' in kingdom_vals:
        ranks["kingdom"] = kingdom_vals['superregnum']

    # handle class - prefer Aves if found
    aves_class = None
    for cv in class_vals:
        if "aves" in cv.lower():
            aves_class = cv 
            break
    
    if aves_class:
        ranks["class"] = aves_class
    elif class_vals:
        ranks["class"] = class_vals[-1]
        
    return ranks


def compare_and_log_taxonomy_discrepancies(enriched_path, wikispecies_path, output_path):
    """Compare LLM vs Wikispecies taxonomy data and log differences."""
    
    try:
        with open(enriched_path, 'r', encoding='utf-8') as f:
            enriched_data = json.load(f)
        with open(wikispecies_path, 'r', encoding='utf-8') as f:
            ws_data = json.load(f)
    except FileNotFoundError as e:
        print(f"missing file for comparison: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return

    if isinstance(enriched_data, dict) and 'triplets' in enriched_data:
        enriched_list = enriched_data['triplets']
    elif isinstance(enriched_data, list):
        enriched_list = enriched_data
    else:
        print(f"unexpected data format in {enriched_path}")
        return

    llm_map = {}
    for item in enriched_list:
        tax_info = item.get('taxonomy', {})
        sci_name = tax_info.get('scientific_name')
        common_name = item.get('subject') 
        
        if not sci_name and tax_info.get('species'): 
            sci_name = tax_info.get('species')

        if isinstance(sci_name, str):
            key = sci_name.lower()
            llm_map[key] = {
                'taxonomy_data': tax_info,
                'common_name_llm': common_name or tax_info.get('canonical_form')
            }

    ws_map = {}
    for item in ws_data:
        ws_tax = item.get('taxonomy')
        if not ws_tax: 
            continue

        key_name = None
        if ws_tax.get('species') and isinstance(ws_tax.get('species'), str):
            key_name = ws_tax.get('species')
        elif item.get('found_name') and isinstance(item.get('found_name'), str):
            key_name = item.get('found_name')
        
        if key_name:
            ws_tax_aug = ws_tax.copy()
            ws_tax_aug['original_wikispecies_query_name'] = item.get('query_name', 'N/A')
            
            if 'rank_hierarchy' in ws_tax_aug and ws_tax_aug['rank_hierarchy']:
                parsed = parse_wikispecies_rank_hierarchy(ws_tax_aug['rank_hierarchy'])
                
                for rank_key, rank_value in parsed.items():
                    if rank_value is not None:
                        ws_tax_aug[rank_key] = rank_value
                
                print(f"    updated ws data: kingdom='{ws_tax_aug.get('kingdom')}', phylum='{ws_tax_aug.get('phylum')}', class='{ws_tax_aug.get('class')}', species='{ws_tax_aug.get('species')}'")
            else:
                print(f"  no rank hierarchy for {item.get('query_name', 'N/A')}")
            
            ws_map[key_name.lower()] = ws_tax_aug

    discrepancies = []
    ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    print(f"\n--- Comparing Taxonomies ---")
    print(f"LLM entries: {len(llm_map)}")
    print(f"Wikispecies entries: {len(ws_map)}")

    for llm_name, llm_entry in llm_map.items():
        llm_tax = llm_entry['taxonomy_data']
        llm_common = llm_entry['common_name_llm']
        ws_match = ws_map.get(llm_name)
        match_type = "scientific_name"

        if not ws_match:
            # try fallback using common name
            for ws_item in ws_data:
                orig_query = ws_item.get('query_name', '')
                found_name = ws_item.get('found_name', '')
                potential_tax = ws_item.get('taxonomy', {})

                if isinstance(llm_common, str):
                    llm_common_lower = llm_common.lower()
                    
                    match_orig = False
                    if isinstance(orig_query, str):
                        match_orig = llm_common_lower in orig_query.lower()
                    
                    match_found = False
                    if isinstance(found_name, str):
                        match_found = llm_common_lower in found_name.lower()

                    if match_orig or match_found:
                        if potential_tax is None:
                            print(f"  found match but taxonomy was null for {orig_query}")
                            continue

                        fallback_tax = potential_tax.copy()
                        
                        if fallback_tax.get('rank_hierarchy'):
                            parsed = parse_wikispecies_rank_hierarchy(fallback_tax['rank_hierarchy'])
                            for rank_key, rank_value in parsed.items():
                                if rank_value is not None:
                                    fallback_tax[rank_key] = rank_value
                            
                            ws_match = fallback_tax
                            match_type = "common_name_fallback"
                            print(f"  fallback match for {llm_common} using {orig_query}")
                            break 
                        else:
                            print(f"  fallback match found but no hierarchy for {orig_query}")

        if not ws_match:
            discrepancies.append({
                "scientific_name_llm": llm_name,
                "common_name_llm": llm_common or "N/A",
                "discrepancy_type": "Wikispecies_Entry_Missing_Or_No_Taxonomy",
                "details": f"No taxonomy found for '{llm_name}' or fallback '{llm_common}'"
            })
            continue

        mismatches = []
        for rank in ranks:
            llm_val = None
            if rank == 'species':
                llm_val = llm_tax.get('scientific_name') or llm_tax.get('species')
            else:
                llm_val = llm_tax.get(rank)
            
            ws_val = ws_match.get(rank)
            
            llm_norm = str(llm_val).lower().strip() if llm_val is not None else None
            ws_norm = str(ws_val).lower().strip() if ws_val is not None else None
            
            if llm_norm != ws_norm:
                # special handling for class with subclass prefixes
                if rank == 'class' and ws_norm and llm_norm:
                    if llm_norm in ws_norm or ws_norm in llm_norm: 
                        continue
                 
                mismatches.append({
                    "rank": rank,
                    "llm_value": llm_val,
                    "wikispecies_value": ws_val,
                })
        
        if mismatches:
            discrepancies.append({
                "scientific_name_llm": llm_name,
                "common_name_llm": llm_common or "N/A",
                "wikispecies_original_query": ws_match.get('original_wikispecies_query_name', "N/A"),
                "discrepancy_type": "Taxonomic_Rank_Mismatch",
                "mismatches": mismatches,
                "match_type": match_type
            })

    if discrepancies:
        print(f"found discrepancies for {len(discrepancies)} species")
        for disc in discrepancies:
            if disc["discrepancy_type"] == "Wikispecies_Entry_Missing_Or_No_Taxonomy":
                print(f"  - missing WS data: {disc['scientific_name_llm']} ({disc['common_name_llm']})")
            else:
                print(f"  - mismatches: {disc['scientific_name_llm']} ({disc['common_name_llm']})")
                for mismatch in disc['mismatches']:
                    print(f"    {mismatch['rank']}: LLM='{mismatch['llm_value']}', WS='{mismatch['wikispecies_value']}'")
    else:
        print("no taxonomy discrepancies found")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(discrepancies, f, indent=2)
        print(f"comparison log saved: {output_path}")
    except IOError as e:
        print(f"error writing comparison log: {e}")

    return discrepancies
