
TAXONOMY_ARG=""
MAX_RESULTS_ARG=""
MODEL_NAME_ARG="" 

while [[ $# -gt 0 ]]; do
  case $1 in
    --taxonomy)
      TAXONOMY_ARG="$2"
      shift 2
      ;;
    --max)
      MAX_RESULTS_ARG="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME_ARG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option for run.sh: $1"
      echo "Usage: ./run.sh [--taxonomy TERM] [--max NUMBER] [--model_name MODEL_STRING]"
      exit 1
      ;;
  esac
done

export TAXONOMY_FILTER="${TAXONOMY_ARG:-${TAXONOMY_FILTER:-}}"
export MAX_RESULTS="${MAX_RESULTS_ARG:-${MAX_RESULTS:-"all"}}"
export MODEL_NAME_FOR_RUN="${MODEL_NAME_ARG:-${MODEL_NAME_FOR_RUN:-"gemini-2.0-flash-001"}}"

DOCKER_SERVICE="threat_analysis"
APP_BASE_PATH="/app/Lent_Init"

SANITIZED_MODEL_NAME_FOR_PATH=$(echo "$MODEL_NAME_FOR_RUN" | sed 's|/|_|g' | sed 's|:|_|g' | sed 's|\.|-|g')

MAX_RESULTS_FOR_PATH_STR="${MAX_RESULTS}"
if [[ -z "$MAX_RESULTS_FOR_PATH_STR" || "$MAX_RESULTS_FOR_PATH_STR" == "all" ]]; then
  MAX_RESULTS_FOR_PATH_STR="all"
fi

DYNAMIC_RUN_BASE_PATH="${APP_BASE_PATH}/runs/${SANITIZED_MODEL_NAME_FOR_PATH}_${MAX_RESULTS_FOR_PATH_STR}"
DYNAMIC_RESULTS_PATH="${DYNAMIC_RUN_BASE_PATH}/results"

WIKI_SPECIES_LIST_FILE="${DYNAMIC_RESULTS_PATH}/species_to_verify_with_wikispecies.txt"
ENRICHED_TRIPLETS_FILE="${DYNAMIC_RESULTS_PATH}/enriched_triplets.json"
WIKISPECIES_LOG_FILE="${DYNAMIC_RESULTS_PATH}/wikispecies_verification_log.json"

mkdir -p Lent_Init/runs

PIPELINE_COMMANDS="echo '--- Step 1: Running Main Data Processing Pipeline (Enhanced Batch Processing) ---'; "
PIPELINE_COMMANDS+="python -m Lent_Init.batch_ingesting --enable-batch-processing --run-main-pipeline; "
PIPELINE_COMMANDS+="echo '--- Step 2: Running Wikispecies Verification ---'; "
PIPELINE_COMMANDS+="if [ -f \"$WIKI_SPECIES_LIST_FILE\" ]; then "
PIPELINE_COMMANDS+="python -m Lent_Init.batch_ingesting --verify-species-wikispecies \"$WIKI_SPECIES_LIST_FILE\" --target_model_name \"$MODEL_NAME_FOR_RUN\" --target_max_results \"$MAX_RESULTS_FOR_PATH_STR\"; "
PIPELINE_COMMANDS+="else echo 'Warning: Species list file ($WIKI_SPECIES_LIST_FILE) not found for this run. Skipping verification.'; fi; " # Note semicolon

PIPELINE_COMMANDS+="echo '--- Step 3: Running Final Taxonomic Comparison ---'; "
PIPELINE_COMMANDS+="if [ -f \"$ENRICHED_TRIPLETS_FILE\" ] && [ -f \"$WIKISPECIES_LOG_FILE\" ]; then "
PIPELINE_COMMANDS+="python -m Lent_Init.batch_ingesting --compare-taxonomies --target_model_name \"$MODEL_NAME_FOR_RUN\" --target_max_results \"$MAX_RESULTS_FOR_PATH_STR\"; "
PIPELINE_COMMANDS+="else echo 'Warning: Required files for taxonomy comparison not found for this run. Skipping comparison.'; fi; " # Note semicolon

PIPELINE_COMMANDS+="echo '--- Full Pipeline Complete ---'"

echo "Executing full pipeline in a single container"
echo "Run parameters: MODEL_NAME_FOR_RUN='${MODEL_NAME_FOR_RUN}', MAX_RESULTS_FOR_RUN='${MAX_RESULTS_FOR_RUN}' (env MAX_RESULTS='${MAX_RESULTS}')"
echo "Dynamic run base path (inside container): ${DYNAMIC_RUN_BASE_PATH}"
docker-compose run --rm $DOCKER_SERVICE /bin/sh -c "$PIPELINE_COMMANDS" 