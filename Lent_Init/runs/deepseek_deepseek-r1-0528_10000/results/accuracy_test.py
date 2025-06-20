import json
import csv
import random
import os
import sys

# --- Configuration ---
INPUT_JSON_FILE = "enriched_triplets.json"
OUTPUT_ANNOTATED_FILE = "annotated_triplets.csv"
SAMPLE_INDICES_FILE = "sample_indices.json"  # Stores the indices of the triplets chosen for the sample
SAMPLE_SIZE = 375  # As determined for 15592 population, 95% confidence, 5% margin

ERROR_CATEGORIES = {
    "A": "Incorrectness: The factual assertion is wrong.",
    "A1": "Species Incorrect: The subject entity is at the wrong taxonomic level or species/group is wrong.",
    "B": "Inconsistency: Invalid predicate usage or mismatch in abstraction levels.",
    "B1": "Unstated/Misplaced Threat/Cause: Actual threat/cause implied or buried, not clear object of causal predicate.",
    "C": "Vagueness: Subject, predicate, or object too general or imprecise.",
    "D": "Redundancy: Object largely repeats information in the predicate.",
    "E": "Metadata Error: Error in associated taxonomy block (e.g., incorrect scientific name)."
}

# --- Helper Functions ---

def load_triplets(filename=INPUT_JSON_FILE):
    """
    Loads triplets from the JSON file.
    Handles cases where JSON is a list or a dictionary containing a list of triplets.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            loaded_json = json.load(f) # Load the whole JSON structure

        triplet_data_list = None

        if isinstance(loaded_json, list): # It's already a list
            triplet_data_list = loaded_json
            print(f"INFO: Loaded {filename} as a direct list of triplets.")
        elif isinstance(loaded_json, dict): # It's a dictionary
            if not loaded_json:
                print(f"Error: {filename} contains an empty JSON object.")
                sys.exit(1)
            
            # Try to find a common key or use the first one
            possible_keys = ['triplets', 'data', 'records', 'results'] # Add more common keys if needed
            found_key_for_list = None
            for key_to_check in possible_keys:
                if key_to_check in loaded_json and isinstance(loaded_json[key_to_check], list):
                    found_key_for_list = key_to_check
                    break
            
            if found_key_for_list:
                triplet_data_list = loaded_json[found_key_for_list]
                print(f"INFO: Found list of triplets under the key '{found_key_for_list}' in {filename}.")
            else:
                # Fallback: use the value of the first key, if it's a list
                if loaded_json.keys():
                    first_key_in_dict = list(loaded_json.keys())[0]
                    if isinstance(loaded_json[first_key_in_dict], list):
                        triplet_data_list = loaded_json[first_key_in_dict]
                        print(f"INFO: Using list of triplets found under the first key '{first_key_in_dict}' in {filename}.")
                        print(f"      If this is not the correct key, please ensure your list of triplets is under a common key "
                              f"like 'triplets', 'data', 'records', or modify the script's 'possible_keys'.")
                    else:
                        print(f"Error: {filename} is a dictionary. Tried common keys and the first key ('{first_key_in_dict}'), "
                              "but its value is not a list of triplets.")
                        print(f"       Please ensure the dictionary contains a key whose value IS the list of triplets.")
                        sys.exit(1)
                else: # Should be caught by the "if not loaded_json" check, but as a safeguard
                    print(f"Error: {filename} is an empty dictionary with no keys to search for the triplet list.")
                    sys.exit(1)
        else:
            # Neither a list nor a dict at the top level, or some other unexpected type
            print(f"Error: Content of {filename} is not a JSON list or a dictionary that contains a list of triplets.")
            sys.exit(1)

        # Final validation of the extracted triplet_data_list
        if not isinstance(triplet_data_list, list):
            # This case should ideally be caught by the logic above, but as a final check.
            print(f"Error: Could not successfully extract a list of triplets from {filename}.")
            sys.exit(1)
        
        for i, item in enumerate(triplet_data_list):
            if not (isinstance(item, dict) and
                    'subject' in item and
                    'predicate' in item and
                    'object' in item):
                print(f"Error: Triplet at index {i} in the extracted list from {filename} is not a valid dictionary "
                      "with 'subject', 'predicate', and 'object' keys.")
                print(f"Problematic item: {item}")
                sys.exit(1)
        return triplet_data_list
        
    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found. Please create it with your triplets.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Please check its format.")
        sys.exit(1)


def get_or_create_sample_indices(total_population_size, sample_size_to_select, indices_filepath):
    """
    Loads sample indices from file if it exists, otherwise creates and saves them.
    Ensures the same sample is used across sessions.
    """
    if os.path.exists(indices_filepath):
        with open(indices_filepath, 'r') as f:
            sample_indices = json.load(f)
        print(f"Loaded {len(sample_indices)} sample indices from {indices_filepath}.")
        if len(sample_indices) != sample_size_to_select:
            # Adjust sample_size_to_select if the file has a different number,
            # but the number of indices in the file dictates the actual sample for this run.
            # This handles cases where SAMPLE_SIZE config changes but user wants to continue with old sample.
            print(f"Warning: Number of indices in {indices_filepath} ({len(sample_indices)}) "
                  f"differs from configured SAMPLE_SIZE ({sample_size_to_select}). "
                  f"Proceeding with {len(sample_indices)} indices from the file.")
            sample_size_to_select = len(sample_indices) # Use the count from the file
        # Ensure indices are within the bounds of the current population
        if total_population_size > 0 and sample_indices and max(sample_indices) >= total_population_size:
            print(f"Error: One or more sample indices in {indices_filepath} are out of bounds "
                  f"for the current population size ({total_population_size}). Max index found: {max(sample_indices)}.")
            print(f"Please delete {indices_filepath} to regenerate a new sample, or ensure it matches your data.")
            sys.exit(1)

        return sample_indices
    else:
        if total_population_size == 0: # Cannot sample from empty population
             print(f"Warning: Population size is 0. Cannot create a sample. Check {INPUT_JSON_FILE}.")
             return []
        if total_population_size < sample_size_to_select:
            print(f"Warning: Population size ({total_population_size}) is smaller than desired sample size ({sample_size_to_select}). "
                  "Sampling all available items.")
            sample_indices = list(range(total_population_size))
        else:
            population_indices = list(range(total_population_size))
            sample_indices = sorted(random.sample(population_indices, sample_size_to_select))
        
        with open(indices_filepath, 'w') as f:
            json.dump(sample_indices, f)
        print(f"Created and saved {len(sample_indices)} new sample indices to {indices_filepath}.")
        return sample_indices

def load_annotated_data(output_filename=OUTPUT_ANNOTATED_FILE):
    """Loads already annotated triplets to know which original_index has been processed."""
    processed_indices = set()
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'original_index' in row:
                        try:
                            processed_indices.add(int(row['original_index']))
                        except ValueError:
                            print(f"Warning: Invalid 'original_index' found in {output_filename}: {row['original_index']}")
        except Exception as e:
            print(f"Error reading existing annotations from {output_filename}: {e}")
            print("Attempting to continue without loading previous annotations for resume check.")
            processed_indices = set() # Reset to be safe
            
    return processed_indices

def save_annotation(annotation_data, output_filename=OUTPUT_ANNOTATED_FILE):
    """Appends a single annotation to the CSV file."""
    file_exists = os.path.exists(output_filename)
    fieldnames = ['original_index', 'subject', 'predicate', 'object', 'is_correct', 'error_codes', 'notes']
    
    try:
        with open(output_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(output_filename) == 0:
                writer.writeheader()
            writer.writerow(annotation_data)
    except IOError as e:
        print(f"Error saving annotation to {output_filename}: {e}")
        print("Please check file permissions and available disk space.")

def display_triplet_info(triplet, original_idx, current_count, total_sample_count):
    """Displays the triplet and progress."""
    print("\n" + "="*80)
    print(f"Annotating Triplet {current_count} of {total_sample_count} (Original Index: {original_idx})")
    print("-"*80)
    print(f"  Subject:   {triplet.get('subject', 'N/A')}")
    print(f"  Predicate: {triplet.get('predicate', 'N/A')}")
    print(f"  Object:    {triplet.get('object', 'N/A')}")
    print("-"*80)

def get_user_annotation(triplet_obj, original_idx):
    """Handles interactive prompting for annotation."""
    while True:
        correct_input = input("Is this triplet Correct? (y/n/q to quit): ").strip().lower()
        if correct_input in ['y', 'yes']:
            is_correct = True
            error_codes_str = ""
            notes = ""
            break
        elif correct_input in ['n', 'no']:
            is_correct = False
            print("\nError Categories:")
            for code, desc in ERROR_CATEGORIES.items():
                print(f"  {code}: {desc}")
            
            selected_errors = []
            while True:
                codes_input = input("Enter Error Category Code(s) (comma-separated, e.g., A1,B or 'done'): ").strip().upper()
                if codes_input.lower() == 'done':
                    break
                
                current_input_codes = []
                valid_codes_in_input = True
                for code in codes_input.split(','):
                    code = code.strip()
                    if code: # Process non-empty codes
                        if code in ERROR_CATEGORIES:
                            if code not in selected_errors and code not in current_input_codes: # avoid duplicates
                                current_input_codes.append(code)
                        else:
                            print(f"Invalid code '{code}'. Please use codes from the list.")
                            valid_codes_in_input = False
                            break # Stop processing this batch of codes
                
                if valid_codes_in_input:
                    selected_errors.extend(current_input_codes)
                    print(f"Currently selected errors: {', '.join(sorted(list(set(selected_errors)))) if selected_errors else 'None'}")
                    # Ask if done with error codes for this triplet or add more
                    another_code_batch = input("Add more error codes? (y/n, default n): ").strip().lower()
                    if another_code_batch != 'y':
                        break # Done with error codes for this triplet
            
            error_codes_str = ",".join(sorted(list(set(selected_errors))))
            notes = input("Notes (optional): ").strip()
            break
        elif correct_input in ['q', 'quit']:
            return "QUIT"
        else:
            print("Invalid input. Please enter 'y', 'n', or 'q'.")

    return {
        'original_index': original_idx,
        'subject': triplet_obj.get('subject'),
        'predicate': triplet_obj.get('predicate'),
        'object': triplet_obj.get('object'),
        'is_correct': is_correct,
        'error_codes': error_codes_str,
        'notes': notes
    }

# --- Main Script Logic ---
def main():
    print("Starting Triplet Annotation Script...")

    all_triplets = load_triplets() # This function is now updated
    total_population_size = len(all_triplets)
    print(f"Successfully loaded {total_population_size} triplets to process.")

    if total_population_size == 0:
        print("No triplets to annotate. Exiting.")
        return

    # Determine the sample: load existing sample indices or create new ones
    effective_sample_size = min(SAMPLE_SIZE, total_population_size)
    sample_indices_to_process = get_or_create_sample_indices(total_population_size, effective_sample_size, SAMPLE_INDICES_FILE)
    
    if not sample_indices_to_process and total_population_size > 0 : # Edge case if get_or_create failed or returned empty for non-empty pop
        print("Could not obtain a sample of indices to process. Exiting.")
        return
    if not sample_indices_to_process and total_population_size == 0: # Already handled, but good check
        print("No triplets to sample from. Exiting.")
        return

    actual_sample_being_processed_count = len(sample_indices_to_process) # The number of items we are actually going to try and annotate

    # Load IDs of already processed triplets from the output CSV
    processed_original_indices = load_annotated_data(OUTPUT_ANNOTATED_FILE)
    print(f"Found {len(processed_original_indices)} already annotated triplets in {OUTPUT_ANNOTATED_FILE} (based on original_index).")

    # Filter the sample_indices_to_process to get only those not yet annotated
    remaining_indices_to_annotate = [idx for idx in sample_indices_to_process if idx not in processed_original_indices]

    if not remaining_indices_to_annotate:
        print(f"All {actual_sample_being_processed_count} triplets in the defined sample have already been annotated.")
        return

    print(f"Starting annotation for {len(remaining_indices_to_annotate)} triplets (out of {actual_sample_being_processed_count} in sample).")
    
    annotated_count_this_session = 0
    # Calculate how many from *this specific sample* were already done for progress display
    annotated_in_current_sample_previously = actual_sample_being_processed_count - len(remaining_indices_to_annotate)


    for i, original_idx in enumerate(remaining_indices_to_annotate):
        if original_idx >= total_population_size : # Safety check
            print(f"Skipping original_index {original_idx} as it's out of bounds for the loaded population size {total_population_size}.")
            continue
            
        triplet_to_annotate = all_triplets[original_idx]
        
        current_progress_in_sample = annotated_in_current_sample_previously + annotated_count_this_session + 1
        
        display_triplet_info(triplet_to_annotate, original_idx, current_progress_in_sample, actual_sample_being_processed_count)
        
        annotation_result = get_user_annotation(triplet_to_annotate, original_idx)

        if annotation_result == "QUIT":
            print(f"Quitting... {annotated_count_this_session} triplets annotated in this session.")
            print(f"Total annotated for this sample so far (across all sessions): {current_progress_in_sample -1}")
            break
        
        save_annotation(annotation_result, OUTPUT_ANNOTATED_FILE)
        annotated_count_this_session += 1
        print(f"Annotation for original_index {original_idx} saved.")

    print("\n" + "="*80)
    if annotated_count_this_session > 0:
        print(f"Finished session. {annotated_count_this_session} triplets annotated in this session.")
    
    # Recalculate final count based on the output file for accuracy
    final_processed_indices_from_sample = {idx for idx in load_annotated_data(OUTPUT_ANNOTATED_FILE) if idx in sample_indices_to_process}
    final_processed_count = len(final_processed_indices_from_sample)

    remaining_to_hit_sample_target = actual_sample_being_processed_count - final_processed_count
    
    print(f"Total {final_processed_count} out of {actual_sample_being_processed_count} target sample triplets are now annotated in {OUTPUT_ANNOTATED_FILE}.")
    if remaining_to_hit_sample_target > 0:
        print(f"{remaining_to_hit_sample_target} more triplets needed to complete the sample of {actual_sample_being_processed_count}.")
        print(f"Rerun the script to continue.")
    else:
        print("Annotation of the defined sample is complete!")
    print("="*80)

if __name__ == "__main__":
    main()