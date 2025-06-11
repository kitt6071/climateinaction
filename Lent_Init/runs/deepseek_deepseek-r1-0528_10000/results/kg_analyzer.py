import pandas as pd
from scipy.stats import beta
from collections import Counter

def analyze_kg_annotations(csv_file_path, correct_column_name, error_column_name, true_value="True", false_value="False"):
    """
    Analyzes annotated knowledge graph triples from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.
        correct_column_name (str): Name of the column indicating correctness.
        error_column_name (str): Name of the column containing error codes.
        true_value (str, int, bool): The value in correct_column_name that signifies a correct triple.
        false_value (str, int, bool): The value in correct_column_name that signifies an incorrect triple.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    try:
        # Load the annotated data
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return {"error": f"File not found: {csv_file_path}"}
    except Exception as e:
        return {"error": f"Error reading CSV: {e}"}

    if correct_column_name not in df.columns:
        return {"error": f"Correctness column '{correct_column_name}' not found in CSV."}
    if error_column_name not in df.columns:
        return {"error": f"Error code column '{error_column_name}' not found in CSV."}

    # --- 1. Calculate Point Estimate of Accuracy ---
    # Convert boolean-like strings to actual booleans if necessary
    # Handle potential string 'True'/'False' or integer 1/0
    if df[correct_column_name].dtype == 'object':
        df[correct_column_name] = df[correct_column_name].astype(str).str.strip().str.lower()
        true_value_str = str(true_value).lower()
        is_correct_series = (df[correct_column_name] == true_value_str)
    elif pd.api.types.is_bool_dtype(df[correct_column_name]):
        is_correct_series = df[correct_column_name]
    elif pd.api.types.is_numeric_dtype(df[correct_column_name]):
        is_correct_series = (df[correct_column_name] == true_value)
    else:
        return {"error": "Unsupported data type for correctness column. Please use boolean, 0/1 integers, or string values like 'True'/'False'."}


    k_total = is_correct_series.sum()
    n_total = len(df)

    if n_total == 0:
        return {"error": "CSV file is empty or contains no data rows."}

    point_accuracy = k_total / n_total

    # --- 2. Calculate Error Frequencies ---
    all_error_codes = []
    # Ensure error codes are treated as strings, handle NaN for correct triples
    df[error_column_name] = df[error_column_name].astype(str).fillna('')

    for codes in df[error_column_name]:
        if pd.notna(codes) and codes.strip() and codes.lower()!= 'nan' and codes.lower()!= 'true' and codes.lower()!= 'false': # Ignore if it's just 'nan' or boolean string
            # Split by comma, strip whitespace from each code
            individual_codes = [code.strip() for code in codes.split(',') if code.strip()]
            all_error_codes.extend(individual_codes)

    error_frequencies = Counter(all_error_codes)

    # --- 3. Bayesian Credible Interval (95%) ---
    # Prior: Beta(1,1) - uninformative prior
    alpha_prior = 1
    beta_prior = 1

    # Posterior parameters
    alpha_posterior = k_total + alpha_prior
    beta_posterior = n_total - k_total + beta_prior

    # Calculate the 95% Credible Interval
    # (from 2.5th percentile to 97.5th percentile of the posterior Beta distribution)
    lower_bound = beta.ppf(0.025, alpha_posterior, beta_posterior)
    upper_bound = beta.ppf(0.975, alpha_posterior, beta_posterior)

    results = {
        "total_triples_annotated": n_total,
        "correct_triples": k_total,
        "incorrect_triples": n_total - k_total,
        "point_estimate_accuracy": point_accuracy,
        "bayesian_credible_interval_95%": (lower_bound, upper_bound),
        "error_code_frequencies": dict(error_frequencies) # Convert Counter to dict for easier display
    }

    return results

if __name__ == "__main__":
    # --- USER INPUT ---
    # Replace with the actual path to your CSV file
    csv_file_path = "annotated_triplets.csv"

    # Replace with the actual name of the column that indicates if a triple is correct
    # This column should contain values like True/False, 1/0, or "Correct"/"Incorrect"
    correctness_column = "is_correct"

    # Replace with the actual name of the column that contains your error codes
    # For correct triples, this column can be empty or NaN.
    # For incorrect triples, list error codes, separated by commas if multiple (e.g., "A1, C1")
    error_codes_column = "error_codes"

    # Define what values in your 'correctness_column' mean True and False
    # Common examples:
    # For boolean True/False: true_val = True, false_val = False
    # For string "Correct"/"Incorrect": true_val = "Correct", false_val = "Incorrect"
    # For string "True"/"False": true_val = "True", false_val = "False" (case-insensitive matching is handled)
    # For integer 1/0: true_val = 1, false_val = 0
    true_val = "True" # Adjust this if your 'True' representation is different
    false_val = "False" # Adjust this if your 'False' representation is different
    # --- END USER INPUT ---

    analysis_results = analyze_kg_annotations(csv_file_path, correctness_column, error_codes_column, true_value=true_val, false_value=false_val)

    output_filename = "kg_annotation_analysis_results.txt"

    with open(output_filename, 'w') as f:
        if "error" in analysis_results:
            error_message = f"Error: {analysis_results['error']}"
            print(error_message)
            f.write(error_message + "\n")
        else:
            output_lines = []
            output_lines.append("--- Knowledge Graph Annotation Analysis ---")
            output_lines.append(f"Total Triples Annotated: {analysis_results['total_triples_annotated']}")
            output_lines.append(f"Correct Triples: {analysis_results['correct_triples']}")
            output_lines.append(f"Incorrect Triples: {analysis_results['incorrect_triples']}")
            output_lines.append(f"Point Estimate of Accuracy: {analysis_results['point_estimate_accuracy']:.4f} ({analysis_results['point_estimate_accuracy']*100:.2f}%)")
            output_lines.append(f"95% Bayesian Credible Interval for Accuracy: [{analysis_results['bayesian_credible_interval_95%'][0]:.4f}, {analysis_results['bayesian_credible_interval_95%'][1]:.4f}]")
            output_lines.append(f"  (This means there is a 95% probability that the true accuracy of the entire KG is between {analysis_results['bayesian_credible_interval_95%'][0]*100:.2f}% and {analysis_results['bayesian_credible_interval_95%'][1]*100:.2f}%)\n")
            output_lines.append("--- Error Code Frequencies ---")
            if analysis_results['error_code_frequencies']:
                for code, freq in analysis_results['error_code_frequencies'].items():
                    output_lines.append(f"  {code}: {freq} occurrences")
            else:
                output_lines.append("  No error codes found (or all triples were correct).")
            
            for line in output_lines:
                print(line)
                f.write(line + "\n")
            print(f"\nResults saved to {output_filename}")