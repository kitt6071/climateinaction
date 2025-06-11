import json
import re
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime

JSON_FILE_PATH = '/Users/kittsonhamill/Desktop/dissertation/climate_inaction/Lent_Init/runs/second run/deepseek_deepseek-r1-0528_10000/results/enriched_triplets.json'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
N_CLUSTERS = 8
TOP_N_CENTRALITY = 15
TOP_N_CLUSTER_ITEMS = 10
OUTPUT_DIR = "/Users/kittsonhamill/Desktop/dissertation/climate_inaction/Lent_Init/runs/second run/deepseek_deepseek-r1-0528_10000/results/analysis_output"
REPORT_FILE_NAME = "analysis_report.txt"

IUCN_CODES_FOR_HEATMAP = [str(i) for i in range(1, 13)]

plt.style.use('default')
plt.rcParams.update({
    'font.size': 26,
    'axes.titlesize': 36,
    'axes.labelsize': 32,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 40,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('triplets', [])
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return []

def preprocess_object_node(object_str):
    match = re.match(r"^(.*?)\[IUCN: (.*?)\]$", object_str)
    if match:
        description = match.group(1).strip()
        iucn_code = match.group(2).strip()
        return description, iucn_code
    return object_str.strip(), None

def create_knowledge_graph_and_triplets(triplets_data_input):
    G = nx.DiGraph()
    processed_triplets_list = []

    general_subject_terms_to_filter = {'aves', 'bird', 'birds', 'afrotropical bird',
                                       'seabird', 'seabirds', 'waterbird', 'waterbirds',
                                       'passerine', 'passerines', 'raptor', 'raptors',
                                       'forest bird', 'forest birds'}
    filtered_triplet_count = 0

    for i, raw_triplet in enumerate(triplets_data_input):
        subject_name_raw = raw_triplet.get('subject', f"UnknownSubject_{i}")
        predicate_raw = raw_triplet.get('predicate', f"UnknownPredicate_{i}")
        object_raw_str = raw_triplet.get('object', f"UnknownObject_{i}")
        doi = raw_triplet.get('doi')
        taxonomy = raw_triplet.get('taxonomy', {})

        subject_node_label = taxonomy.get('canonical_form', subject_name_raw).strip()

        if subject_node_label.lower() in general_subject_terms_to_filter:
            filtered_triplet_count += 1
            continue

        object_node_label, iucn_code_from_obj = preprocess_object_node(object_raw_str)

        processed_triplets_list.append({
            "id": i,
            "subject": subject_node_label,
            "predicate": predicate_raw,
            "object": object_node_label,
            "iucn_code": iucn_code_from_obj,
            "doi": doi,
            "original_taxonomy": taxonomy
        })

        if subject_node_label not in G:
            G.add_node(subject_node_label, type='subject', **taxonomy)
        if object_node_label not in G:
            G.add_node(object_node_label, type='object', iucn_code_attr=iucn_code_from_obj, label=object_node_label)
        G.add_edge(subject_node_label, object_node_label, predicate=predicate_raw, doi=doi)

    if filtered_triplet_count > 0:
        print(f"Filtered out {filtered_triplet_count} triplets with general subject terms.")

    return G, processed_triplets_list

def plot_degree_distribution(G_undirected, title="Degree Distribution", report_lines=None, filename_base="degree_dist"):
    degrees = [G_undirected.degree(n) for n in G_undirected.nodes()]
    if not degrees:
        print("Cannot plot degree distribution: graph has no nodes.")
        if report_lines is not None: report_lines.append(f"{title}: Graph has no nodes.\n")
        return

    min_degree_val = min((d for d in degrees if d > 0), default=1)
    max_degree_val = max(degrees, default=1)

    plt.figure(figsize=(16, 12))
    if min_degree_val >= max_degree_val:
        bins = np.array([min_degree_val, min_degree_val + 1])
    else:
        bins=np.logspace(np.log10(min_degree_val), np.log10(max_degree_val), 50)

    plt.hist(degrees, bins=bins, alpha=0.8, color='#2E86AB', edgecolor='#A23B72', linewidth=2)
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.title(title, fontsize=40, fontweight='bold', pad=30)
    plt.xlabel("Degree (log scale)", fontsize=32, fontweight='bold')
    plt.ylabel("Frequency (log scale)", fontsize=32, fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.3, color='gray')

    plt.tight_layout()
    plot_filename = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved plot: {plot_filename}")
    if report_lines is not None:
        report_lines.append(f"{title} Plot: {plot_filename}\n")
        report_lines.append(f"  Max degree: {max(degrees, default=0)}, Min degree: {min(degrees, default=0)}, Avg degree: {np.mean(degrees) if degrees else 0:.2f}\n")

def analyze_centrality(G_undirected, G_orig_for_attrs, top_n=TOP_N_CENTRALITY, report_lines=None):
    if not G_undirected.nodes:
        if report_lines is not None: report_lines.append("Centrality Analysis: Graph is empty.\n")
        return

    degree_centrality_dict = nx.degree_centrality(G_undirected)
    all_degree_centrality_sorted = sorted(degree_centrality_dict.items(), key=lambda item: item[1], reverse=True)

    top_subject_degrees = [
        (node, centrality) for node, centrality in all_degree_centrality_sorted
        if G_orig_for_attrs.has_node(node) and G_orig_for_attrs.nodes[node].get('type') == 'subject'
    ][:top_n]

    if report_lines is not None:
        report_lines.append(f"\n--- Top {len(top_subject_degrees)} Subjects by Degree Centrality ---\n")
        for i, (node, centrality) in enumerate(top_subject_degrees):
            report_lines.append(f"{i+1}. {node}: {centrality:.4f}\n")

    if G_undirected.number_of_nodes() > 1000:
        print("Calculating betweenness centrality (sampling for large graph)...")
        full_betweenness_centrality_dict = nx.betweenness_centrality(G_undirected, k=min(500, G_undirected.number_of_nodes()), normalized=True, seed=42)
    elif G_undirected.number_of_nodes() > 0 :
        full_betweenness_centrality_dict = nx.betweenness_centrality(G_undirected, normalized=True)
    else:
        full_betweenness_centrality_dict = {}

    all_betweenness_centrality_sorted = sorted(full_betweenness_centrality_dict.items(), key=lambda item: item[1], reverse=True)

    top_subject_betweenness = [
        (node, centrality) for node, centrality in all_betweenness_centrality_sorted
        if G_orig_for_attrs.has_node(node) and G_orig_for_attrs.nodes[node].get('type') == 'subject'
    ][:top_n]

    if report_lines is not None:
        report_lines.append(f"\n--- Top {len(top_subject_betweenness)} Subjects by Betweenness Centrality ---\n")
        for i, (node, centrality) in enumerate(top_subject_betweenness):
            report_lines.append(f"{i+1}. {node}: {centrality:.4f}\n")

def analyze_communities(G_undirected, report_lines=None):
    if not G_undirected.nodes or not G_undirected.edges:
        if report_lines is not None: report_lines.append("Community Analysis: Graph empty or no edges.\n")
        return

    if report_lines is not None: report_lines.append("\n--- Community Analysis ---\n")
    try:
        communities_generator = nx.community.greedy_modularity_communities(G_undirected)
        communities = [list(c) for c in communities_generator if c]
        if not communities:
            if report_lines is not None: report_lines.append("No communities found.\n")
            return
        if report_lines is not None: report_lines.append(f"Number of communities found: {len(communities)}\n")
    except Exception as e:
        if report_lines is not None: report_lines.append(f"Error in community detection: {e}\n")

def analyze_clustering_coefficient(G_undirected, report_lines=None, context="Giant Component"):
    if not G_undirected.nodes:
        if report_lines is not None: report_lines.append(f"Clustering Coefficient ({context}): Graph empty.\n")
        return
    avg_clustering = nx.average_clustering(G_undirected)
    transitivity = nx.transitivity(G_undirected)
    if report_lines is not None:
        report_lines.append(f"\nAverage Clustering Coefficient ({context}): {avg_clustering:.4f}\n")
        report_lines.append(f"Global Clustering Coefficient (Transitivity, {context}): {transitivity:.4f}\n")

def generate_embeddings_for_items(item_sentences, model_name=EMBEDDING_MODEL_NAME):
    print(f"\nGenerating {len(item_sentences)} embeddings using {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(item_sentences, show_progress_bar=True)
    return embeddings

def plot_iucn_distribution_heatmap(all_cluster_iucn_counts, n_actual_clusters, specific_iucn_codes, report_lines, filename_suffix=""):
    if not any(all_cluster_iucn_counts):
        return

    heatmap_data = pd.DataFrame(index=[f"Cluster {i}" for i in range(n_actual_clusters)], columns=specific_iucn_codes)
    for i in range(n_actual_clusters):
        cluster_name = f"Cluster {i}"
        current_cluster_counts = all_cluster_iucn_counts[i] if i < len(all_cluster_iucn_counts) else Counter()
        for code in specific_iucn_codes:
            heatmap_data.loc[cluster_name, code] = current_cluster_counts.get(code, 0)

    plt.figure(figsize=(max(20, len(specific_iucn_codes) * 1.4), max(14, n_actual_clusters * 1.0)))

    sns.heatmap(heatmap_data.fillna(0).astype(float), annot=True, fmt=".0f", cmap="YlGnBu", linewidths=1.0, linecolor='white',
                       annot_kws={'fontsize': 48, 'fontweight': 'bold'}, cbar_kws={'label': 'Count'}, vmin=0, vmax=50)

    plt.title(f"IUCN Categories per Cluster", fontsize=72, fontweight='bold', pad=80)
    plt.ylabel("Cluster", fontsize=64, fontweight='bold')
    plt.xlabel("IUCN Code / Category", fontsize=64, fontweight='bold')
    plt.xticks(rotation=45, ha="right", fontsize=40, fontweight='bold')
    plt.yticks(rotation=0, fontsize=40, fontweight='bold')

    filename = os.path.join(OUTPUT_DIR, f"iucn_heatmap{filename_suffix}.pdf")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        if report_lines is not None: report_lines.append(f"IUCN Heatmap: {filename}\n")
        print(f"Saved IUCN heatmap to {filename}")
    except Exception as e:
        if report_lines is not None: report_lines.append(f"  Error saving IUCN heatmap to {filename}: {e}\n")
    finally:
        plt.close()

def plot_comparative_item_heatmap(all_cluster_item_counts, n_actual_clusters, report_lines, item_type="Items", filename_suffix=""):
    overall_item_counts = Counter()
    for counts in all_cluster_item_counts:
        overall_item_counts.update(counts)

    if not overall_item_counts:
        return

    top_n_items_overall = [item[0] for item in overall_item_counts.most_common(TOP_N_CLUSTER_ITEMS)]

    heatmap_data = pd.DataFrame(index=[f"Cluster {i}" for i in range(n_actual_clusters)], columns=top_n_items_overall)
    for i in range(n_actual_clusters):
        cluster_name = f"Cluster {i}"
        current_cluster_counts = all_cluster_item_counts[i] if i < len(all_cluster_item_counts) else Counter()
        for item_key in top_n_items_overall:
            heatmap_data.loc[cluster_name, item_key] = current_cluster_counts.get(item_key, 0)

    plt.figure(figsize=(max(44, len(top_n_items_overall) * 2.8), max(28, n_actual_clusters * 2.0)))

    sns.heatmap(heatmap_data.fillna(0).astype(float), annot=True, fmt=".0f", cmap="viridis", linewidths=1.0, linecolor='white',
                       annot_kws={'fontsize': 48, 'fontweight': 'bold'}, cbar_kws={'label': 'Count'}, vmin=0, vmax=50)

    title_item_type = item_type if item_type.lower().endswith('s') else f"{item_type}s"

    plt.title(f"{title_item_type} per Cluster", fontsize=72, fontweight='bold', pad=80)
    plt.ylabel("Cluster", fontsize=64, fontweight='bold')
    plt.xlabel(item_type, fontsize=64, fontweight='bold')
    plt.xticks(rotation=45, ha="right", fontsize=40, fontweight='bold')
    plt.yticks(rotation=0, fontsize=40, fontweight='bold')

    safe_item_type = item_type.lower().replace('/', '_')
    filename = os.path.join(OUTPUT_DIR, f"comparative_{safe_item_type}_heatmap{filename_suffix}.pdf")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        if report_lines is not None: report_lines.append(f"Comparative {item_type} Heatmap: {filename}\n")
        print(f"Saved {item_type} heatmap to {filename}")
    except Exception as e:
        if report_lines is not None: report_lines.append(f"  Error saving {item_type} heatmap to {filename}: {e}\n")
    finally:
        plt.close()

def generate_cluster_heatmaps(triplets_in_analysis, cluster_labels, n_actual_clusters, report_lines):
    report_lines.append(f"\n\n--- Triplet Cluster Analysis ---\n")

    all_cluster_iucn_counts = []
    all_cluster_object_counts = []

    for i in range(n_actual_clusters):
        current_cluster_triplets = [triplets_in_analysis[idx] for idx, label in enumerate(cluster_labels) if label == i]

        if not current_cluster_triplets:
            all_cluster_iucn_counts.append(Counter())
            all_cluster_object_counts.append(Counter())
            continue

        cluster_objects = Counter(t['object'] for t in current_cluster_triplets)
        cluster_iucn_codes = Counter()
        for triplet_dict in current_cluster_triplets:
            iucn_full_string = triplet_dict.get('iucn_code')
            if iucn_full_string and iucn_full_string.lower() != "none" and iucn_full_string.strip() != "":
                match_primary_code = re.match(r"^(\d+)", iucn_full_string)
                if match_primary_code:
                    primary_code = match_primary_code.group(1)
                    cluster_iucn_codes[primary_code] += 1

        all_cluster_object_counts.append(cluster_objects)
        all_cluster_iucn_counts.append(cluster_iucn_codes)

    if any(all_cluster_object_counts):
        plot_comparative_item_heatmap(all_cluster_object_counts, n_actual_clusters, report_lines,
                                      item_type="Objects/Threats", filename_suffix="_threat_heatmap")

    if any(all_cluster_iucn_counts):
        plot_iucn_distribution_heatmap(all_cluster_iucn_counts, n_actual_clusters,
                                      IUCN_CODES_FOR_HEATMAP, report_lines,
                                      filename_suffix="_primary_codes")

def save_report(report_lines, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"Analysis report saved to {filename}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    report_lines = [f"Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"]
    report_lines.append(f"Source JSON: {JSON_FILE_PATH}\n")
    report_lines.append(f"Output Directory: {OUTPUT_DIR}\n\n")

    raw_triplets_from_json = load_data(JSON_FILE_PATH)

    if not raw_triplets_from_json:
        print("No data loaded. Exiting.")
    else:
        report_lines.append(f"Loaded {len(raw_triplets_from_json)} raw triplets.\n")

        G_orig, processed_triplets_for_embedding = create_knowledge_graph_and_triplets(raw_triplets_from_json)

        report_lines.append(f"Knowledge graph: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges.\n")
        report_lines.append(f"Triplets for embedding: {len(processed_triplets_for_embedding)}.\n")

        report_lines.append("\n--- Graph Structural Analysis ---\n")
        G_undirected_for_structure = G_orig.to_undirected()

        if G_undirected_for_structure.number_of_nodes() > 0:
            if not nx.is_connected(G_undirected_for_structure):
                largest_cc_nodes = max(nx.connected_components(G_undirected_for_structure), key=len)
                G_giant_undirected_nodes_component = G_orig.subgraph(largest_cc_nodes).to_undirected()
                report_lines.append(f"Giant component: {G_giant_undirected_nodes_component.number_of_nodes()} nodes, {G_giant_undirected_nodes_component.number_of_edges()} edges.\n")
            else:
                report_lines.append("Graph is connected.\n")
                G_giant_undirected_nodes_component = G_undirected_for_structure

            report_lines.append(f"Density of giant component: {nx.density(G_giant_undirected_nodes_component):.6f}\n")
            plot_degree_distribution(G_giant_undirected_nodes_component, title="Node Degree Distribution (Giant Component)", report_lines=report_lines, filename_base="node_giant_degree_dist")
            analyze_centrality(G_giant_undirected_nodes_component, G_orig, report_lines=report_lines)
            analyze_communities(G_giant_undirected_nodes_component, report_lines=report_lines)
            analyze_clustering_coefficient(G_giant_undirected_nodes_component, report_lines=report_lines, context="Giant Component")
        else:
            report_lines.append("Graph is empty. Skipping structural analysis.\n")

        if processed_triplets_for_embedding:
            triplet_sentences = [f"{t['subject']} {t['predicate']} {t['object']}" for t in processed_triplets_for_embedding]
            triplet_embeddings = generate_embeddings_for_items(triplet_sentences)

            if triplet_embeddings.shape[0] > N_CLUSTERS:
                kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(triplet_embeddings)
                generate_cluster_heatmaps(processed_triplets_for_embedding, cluster_labels, N_CLUSTERS, report_lines)
            else:
                report_lines.append("\nNot enough triplets to form clusters.\n")

    print("\nAnalysis complete.")
    report_lines.append("\n--- Analysis Complete ---\n")
    save_report(report_lines, os.path.join(OUTPUT_DIR, REPORT_FILE_NAME))