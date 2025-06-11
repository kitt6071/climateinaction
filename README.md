# Climate Inaction Knowledge Graph Pipeline

This project is a data processing pipeline designed to extract, analyze, and visualize relationships between species and climate change-related threats from a vast collection of scientific abstracts. It leverages Large Language Models (LLMs) and various natural language processing (NLP) techniques to build a comprehensive knowledge graph.

## Key Features

- **Automated Triplet Extraction:** Extracts `(subject, predicate, object)` triplets (e.g., `(Polar Bear, is threatened by, sea ice loss)`) from scientific abstracts.
- **Taxonomic Verification:** Uses Wikispecies to verify and normalize species names, ensuring data accuracy.
- **Threat Classification:** Classifies extracted threats according to the IUCN threat category system.
- **Relevance Classification:** Employs both traditional machine learning and LLM-based classifiers to determine the relevance of abstracts to climate change.
- **Data Deduplication:** Includes a caching mechanism to avoid reprocessing of abstracts and entities.
- **Interactive Visualization:** Provides a web-based interface to visualize the extracted knowledge graph and analyze threat clusters.
- **Dockerized Environment:** Ensures reproducibility and ease of setup through a Dockerized environment.

## File Structure

The project is organized into several directories, each serving a specific purpose. Here is a breakdown of the key files and folders:

### Root Directory

- `Dockerfile`: Defines the Docker image for the application, ensuring a consistent and reproducible environment with all necessary dependencies.
- `app.py`: A Flask web application that serves the interactive knowledge graph visualization and provides an API for querying the data.
- `backend_triplets.py`: Contains the logic for the backend data processing, including data loading, embedding generation, and threat clustering.
- `batchrun.sh`: A shell script for running the data processing pipeline in batch mode, allowing for large-scale data ingestion.
- `docker-compose.yml`: Configures the services, networks, and volumes for the Docker application, making it easy to run the entire stack with a single command.
- `full_analysis.py`: A script for running a comprehensive analysis of the extracted data, including graph metrics and cluster analysis.
- `requirements.txt`: Lists the Python dependencies required for the project, which are installed in the Docker container.
- `train_relevance_classifier.py`: A script for training the machine learning models used to classify the relevance of abstracts.

### `Lent_Init/`

This directory contains the core logic of the data processing pipeline.

- `main_pipeline.py`: The main entry point for the data processing pipeline, orchestrating the various steps from data loading to triplet extraction and analysis.
- `setup.py`: Contains setup functions for the pipeline, including logging configuration, data loading, and LLM setup.
- `llm_api_utility.py`: Provides utility functions for interacting with LLM APIs, including rate limiting and response handling.
- `triplet_extraction.py`: Contains the core logic for extracting species, threats, and their relationships from abstracts using LLMs.
- `iucn_refinement.py`: Includes functions for classifying extracted threats according to the IUCN threat categories.
- `wikispecies_utils.py`: Provides utilities for interacting with the Wikispecies API to verify and normalize species names.
- `batch_ingesting.py`: Manages the batch ingestion of abstracts, including relevance classification and chunking.
- `graph_analysis.py`: Contains functions for analyzing the extracted knowledge graph, including degree centrality and other metrics.
- `cache.py`: Implements a simple caching mechanism to store and retrieve intermediate results, avoiding redundant processing.

### `static/`

This directory contains the static assets for the web application, including JavaScript, CSS, and other client-side files.

- `script.js`: The main JavaScript file for the web interface, handling user interactions and orchestrating the various visualization components.
- `visualize_graph.js`: Contains the core logic for rendering and interacting with the knowledge graph using the `vis.js` library.
- `analysis.js`: Handles data analysis features on the frontend, such as calculating and displaying graph metrics.
- `calc_network.js`: Responsible for calculating network properties and metrics for the graph visualization.
- `chain_build.js`: Manages the construction of query chains or other complex user interactions on the frontend.
- `chart_util.js`: Provides utility functions for creating and updating charts, likely used for displaying analysis results.
- `data_loading.js`: Handles the loading of data into the frontend, including making asynchronous requests to the backend API.
- `kg_query.js`: Manages the querying of the knowledge graph from the frontend, allowing users to search for specific entities.
- `kg_transfer.js`: Used for transferring knowledge graph data, such as exporting or importing graph structures.
- `search.js`: Implements the search functionality for the web interface, allowing users to find nodes and relationships in the graph.
- `style.css`: The main stylesheet for the web application, defining the visual appearance of all interface components.
- `embeddings_scripts/`: This subdirectory contains scripts related to the analysis and visualization of threat embeddings.
    - `click.js`: Handles click events within the embedding visualization.
    - `cluster_quality_analysis.js`: Provides frontend logic for analyzing the quality of threat clusters.
    - `clustering.js`: Manages frontend clustering operations for embeddings.
    - `dim_reduct.js`: Manages the visualization of dimensionality reduction (e.g., UMAP or t-SNE) on the frontend.
    - `embeddings_analysis.js`: The main script for the embedding analysis interface.
    - `semantic_analysis.js`: Contains logic for performing semantic analysis of embeddings on the frontend.
    - `visualization.js`: Provides general visualization scripts for the embedding analysis interface.

### `templates/`

This directory contains the HTML templates for the Flask web application.

- `index.html`: The main HTML file for the single-page web application. It provides the structure for the knowledge graph visualization and includes all necessary JavaScript and CSS files.

### `trained_relevance_models_central/deepseek_r1_runs/deepseek_deepseek-r1-0528/`

This directory stores the pre-trained machine learning models for relevance classification for Deepseek-r1-0528.

- `embedding_classifier.pkl`: The trained classifier for predicting the relevance of abstracts based on their embeddings.

## Setup and Installation

The project is designed to run in a Dockerized environment to ensure consistency and ease of setup. Follow these steps to get the application running:

1.  **Build the Docker Image and Run Three Part Pipeline With:**
    ```./batchrun.sh --max {number of abstracts} --model_name {model name}
    ```

2.  **Run the Website:**
    ```python3 app.py
    ```

REQUIREMENTS FOR RUNNING:
You will need a .env file with an OPENROUTER_API_KEY. This will let you use the LLM APIs. You will also need an all_abstracts.parquet.
I was unable to attach this because some of the files in the document are private and only for the use of the Conservation Evidence Team.

## Usage

### Running the Data Pipeline

The data processing pipeline can be run in batch mode using the `batchrun.sh` script. This script will ingest abstracts from the specified data source, process them through the pipeline, and store the results in the appropriate directories.

### Visualizing the Knowledge Graph

Once the pipeline has been run with `app.py`, you can view the interactive knowledge graph by opening your web browser and navigating to `http://localhost:5000`. The web interface allows you to explore the relationships between species and threats, search for specific entities, and analyze the structure of the graph.
