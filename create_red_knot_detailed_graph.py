#!/usr/bin/env python3

import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import textwrap
import re

def clean_threat_description(threat_text):
    if '[IUCN:' not in threat_text:
        return threat_text
    
    desc_part = threat_text.split('[IUCN:')[0].strip()
    iucn_part = threat_text.split('[IUCN:')[1].split(']')[0].strip()
    
    iucn_code_match = re.match(r'^[0-9.]+', iucn_part)
    if iucn_code_match:
        iucn_code = iucn_code_match.group(0)
        return f"{desc_part} [IUCN: {iucn_code}]"
    
    return f"{desc_part} [IUCN: N/A]"

def get_category_color(iucn_code):
    main_cat = iucn_code.split('.')[0] if '.' in iucn_code else iucn_code
    colors = {
        '1': '#E57373',  # Red
        '2': '#81C784',  # Green
        '3': '#64B5F6',  # Blue
        '4': '#FFF176',  # Yellow
        '5': '#FFB74D',  # Orange
        '6': '#F06292',  # Pink
        '7': '#9575CD',  # Purple
        '8': '#A1887F',  # Brown
        '9': '#4DD0E1',  # Cyan
        '10': '#78909C', # Blue Grey
        '11': '#FF8A65', # Deep Orange
    }
    return colors.get(main_cat, '#BDBDBD')

def create_final_presentation_graph():    
    input_file = "red_knot_analysis/enriched_triplets.json"
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_file}'")
        return None

    G = nx.DiGraph()
    G.add_node("Red Knot", node_type="species")
    
    threat_info = {}
    edge_labels = {}
    
    for triplet in data['triplets']:
        threat_original = triplet['object']
        threat_clean = clean_threat_description(threat_original)
        
        if '[IUCN:' in threat_clean:
            iucn_code = threat_clean.split('[IUCN:')[1].split(']')[0].strip()
        else:
            iucn_code = "12"
            
        if iucn_code.startswith('12'):
            continue

        predicate_full = triplet['predicate']
        
        G.add_node(threat_clean, node_type="threat", iucn_code=iucn_code)
        G.add_edge("Red Knot", threat_clean, predicate=predicate_full)
        
        threat_info[threat_clean] = iucn_code
        
        wrapped_predicate = textwrap.fill(predicate_full, width=35)
        edge_labels[("Red Knot", threat_clean)] = wrapped_predicate
    
    plt.figure(figsize=(65, 50))
    
    center_node = ["Red Knot"]
    outer_nodes = [node for node in G.nodes() if node != "Red Knot"]
    pos = nx.shell_layout(G, [center_node, outer_nodes], scale=15)

    threat_nodes = outer_nodes
    threat_colors = [get_category_color(threat_info.get(n, '12')) for n in threat_nodes]
    
    nx.draw_networkx_nodes(G, pos, nodelist=threat_nodes, 
                          node_color=threat_colors, node_size=60000, alpha=1.0,
                          edgecolors='black', linewidths=3.5)
    
    nx.draw_networkx_nodes(G, pos, nodelist=center_node, 
                          node_color='gold', node_size=90000, alpha=1.0,
                          edgecolors='black', linewidths=8)
    
    nx.draw_networkx_edges(G, pos, edge_color='dimgray', alpha=0.9, 
                          arrows=True, arrowsize=150, width=8.0)
    
    node_labels = {node: textwrap.fill(node, width=20) for node in G.nodes()}
    node_labels["Red Knot"] = "Red Knot\n(Calidris canutus)"
    
    nx.draw_networkx_labels(G, pos, node_labels, font_size=48, font_weight='bold')
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=36,
                                font_color='black',
                                font_weight='bold',
                                label_pos=0.6,
                                bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.95, edgecolor='darkgray'))
    
    plt.title("Threats to Red Knots", 
             fontsize=100, fontweight='bold', pad=80)
    
    category_names = {
        '1': 'Residential & Commercial', '2': 'Agriculture & Aquaculture',
        '3': 'Energy & Mining', '4': 'Transportation', '5': 'Biological Resource Use',
        '6': 'Human Intrusions', '7': 'Natural System Modifications',
        '8': 'Invasive Species', '9': 'Pollution', '10': 'Geological Events',
        '11': 'Climate Change'
    }
    main_categories_present = sorted(list(set([info.split('.')[0] for info in threat_info.values()])))
    legend_elements = [patches.Patch(color=get_category_color(cat), label=f'IUCN {cat}.x: {category_names.get(cat)}') for cat in main_categories_present]
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.05, 1.0),
               fontsize=48, title="IUCN Threat Categories", title_fontsize=52)
    
    plt.axis('off')
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    output_path = Path("red_knot_analysis/figures")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "red_knot_png.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return G

if __name__ == "__main__":
    create_final_presentation_graph()
