import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

def read_PipeFile():
    """Read pipeline data from the local file"""
    # Use the local updated file
    with open('G://中控技术//blueprint//0708烟台S4_updated.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Read nodes and edges
    nodelist = data['nodelist']
    edgelist = data['linklist']
    
    # Convert to DataFrames
    nodeDF = pd.DataFrame(nodelist)
    edgeDF = pd.DataFrame(edgelist)
    
    # Use original IDs
    nodeID = nodeDF['id']
    edgeID = edgeDF['id']
    
    # Store original edge endpoints
    edgeDF['sourceid_original'] = edgeDF['sourceid']
    edgeDF['targetid_original'] = edgeDF['targetid']

    # Extract node positions and types
    nodeParas = []
    nodePSTs = []
    nodeTypes = []
    
    for i in range(len(nodeDF)):
        try:
            if isinstance(nodeDF['parameter'].iloc[i], str):
                para = json.loads(nodeDF['parameter'].iloc[i])
            else:
                para = nodeDF['parameter'].iloc[i]
            nodeParas.append(para)
            
            # Extract position information
            if 'styles' in para and 'position' in para['styles']:
                position = para['styles']['position']
                nodePSTs.append([position['x'], position['y']])
            else:
                nodePSTs.append([0, 0])
                print(f"Warning: Node {nodeDF['id'].iloc[i]} has no position info, using (0, 0)")
            
            # Extract node type
            node_type = para.get('type', 'Unknown')
            nodeTypes.append(node_type)
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing node {nodeDF['id'].iloc[i]} parameters: {e}")
            nodeParas.append({})
            nodePSTs.append([0, 0])
            nodeTypes.append('Unknown')
    
    nodePSTs = pd.Series(nodePSTs)
    nodeTypes = pd.Series(nodeTypes)

    return nodeDF, edgeDF, nodeID, edgeID, nodePSTs, nodeTypes

def make_edge_attr(edgeDF, nodeID_to_index):
    """Create edge attributes with length information"""
    edge_data = []
    
    for i in range(len(edgeDF)):
        source_id = edgeDF['sourceid_original'].iloc[i]
        target_id = edgeDF['targetid_original'].iloc[i]
        
        # Check if nodes exist
        if source_id in nodeID_to_index and target_id in nodeID_to_index:
            from_idx = nodeID_to_index[source_id]
            to_idx = nodeID_to_index[target_id]
            
            # Parse edge parameters to get length
            try:
                if isinstance(edgeDF['parameter'].iloc[i], str):
                    edge_para = json.loads(edgeDF['parameter'].iloc[i])
                else:
                    edge_para = edgeDF['parameter'].iloc[i]
                
                # Get length information
                length = edge_para.get('parameter', {}).get('Length', 1.0)
                if length is None or length <= 0:
                    length = 1.0
                    
                edge_data.append({
                    'from': from_idx,
                    'to': to_idx,
                    'length': float(length),
                    'edge_id': edgeDF['id'].iloc[i]
                })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing edge {edgeDF['id'].iloc[i]} parameters: {e}, using default length 1.0")
                edge_data.append({
                    'from': from_idx,
                    'to': to_idx,
                    'length': 1.0,
                    'edge_id': edgeDF['id'].iloc[i]
                })
        else:
            print(f"Warning: Edge {edgeDF['id'].iloc[i]} endpoints not found, skipping")
    
    return pd.DataFrame(edge_data)

def get_node_visual_properties(node_type):
    """Get visual properties (color, shape, size) for different node types"""
    type_properties = {
        'VavlePro': {'color': '#FF6B6B', 'marker': 's', 'size': 100},  # Red square
        'Stream': {'color': '#4ECDC4', 'marker': 'o', 'size': 80},     # Teal circle  
        'Tee': {'color': '#45B7D1', 'marker': '^', 'size': 120},       # Blue triangle
        'Mixer': {'color': '#96CEB4', 'marker': 'D', 'size': 110},     # Green diamond
        'Unknown': {'color': '#95A5A6', 'marker': 'o', 'size': 60}     # Gray circle
    }
    return type_properties.get(node_type, type_properties['Unknown'])

def create_directed_graph_visualization(show_edge_labels=True, label_threshold=100):
    """Main function to create directed graph visualization
    
    Args:
        show_edge_labels (bool): Whether to show edge length labels
        label_threshold (float): Only show labels for edges shorter than this length
    """
    
    # Read data
    print("Loading pipeline data...")
    nodeDF, edgeDF, nodeID, edgeID, nodePSTs, nodeTypes = read_PipeFile()
    
    print(f"Loaded {len(nodeDF)} nodes and {len(edgeDF)} edges")
    
    # Create node ID mapping
    nodeID_to_index = {val: idx for idx, val in enumerate(nodeID)}
    index_to_nodeID = {idx: val for idx, val in enumerate(nodeID)}
    
    # Create edge attributes
    edge_attr = make_edge_attr(edgeDF, nodeID_to_index)
    
    if len(edge_attr) == 0:
        print("Error: No valid edge data")
        return
    
    print(f"Processed {len(edge_attr)} valid edges")
    
    # Create DIRECTED graph
    G = nx.DiGraph()  # Changed from nx.Graph() to nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from(range(len(nodeID)))
    
    # Add edges with length attributes
    for _, edge in edge_attr.iterrows():
        G.add_edge(edge['from'], edge['to'], length=edge['length'])
    
    print(f"Directed graph created: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Create visualization
    plt.figure(figsize=(20, 16))
    
    # Use real-world coordinates for positioning
    pos = {i: nodePSTs.iloc[i] for i in range(len(nodePSTs))}
    
    # Get unique node types for legend
    unique_types = nodeTypes.unique()
    print(f"Node types found: {list(unique_types)}")
    
    # Draw nodes by type
    for node_type in unique_types:
        # Get nodes of this type
        nodes_of_type = [i for i, t in enumerate(nodeTypes) if t == node_type]
        if not nodes_of_type:
            continue
            
        # Get visual properties
        props = get_node_visual_properties(node_type)
        
        # Get positions for these nodes
        node_positions = [pos[node] for node in nodes_of_type]
        x_coords = [p[0] for p in node_positions]
        y_coords = [p[1] for p in node_positions]
        
        # Draw nodes
        plt.scatter(x_coords, y_coords, 
                   c=props['color'], 
                   marker=props['marker'], 
                   s=props['size'],
                   label=f'{node_type} ({len(nodes_of_type)})',
                   alpha=0.8, 
                   edgecolors='black', 
                   linewidth=0.5)
    
    # Draw edges with arrows and labels
    edge_count = 0
    for _, edge in edge_attr.iterrows():
        from_pos = pos[edge['from']]
        to_pos = pos[edge['to']]
        
        # Draw edge with arrow
        plt.annotate('', xy=to_pos, xytext=from_pos,
                    arrowprops=dict(arrowstyle='->', 
                                  color='gray', 
                                  alpha=0.6, 
                                  lw=0.8))
        
        # Add edge length label (only for some edges to avoid clutter)
        if show_edge_labels and edge['length'] < label_threshold and edge_count % 10 == 0:  # Show every 10th short edge
            mid_x = (from_pos[0] + to_pos[0]) / 2
            mid_y = (from_pos[1] + to_pos[1]) / 2
            
            plt.text(mid_x, mid_y, f"{edge['length']:.1f}", 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', 
                             facecolor='white', 
                             alpha=0.8, 
                             edgecolor='lightgray'))
        edge_count += 1
    
    # Customize plot
    plt.title('Directed Pipeline Network Visualization', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    
    # Add legend
    plt.legend(title='Node Types', 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left',
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # Add grid and formatting
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Tight layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('G://中控技术//blueprint//YTS4_directed_graph.png', 
                dpi=500, bbox_inches='tight')
    
    # Display statistics
    print("\n=== Graph Statistics ===")
    print(f"Total nodes: {len(G.nodes())}")
    print(f"Total edges: {len(G.edges())}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")
    print(f"Is directed: {G.is_directed()}")
    print(f"Is connected: {nx.is_weakly_connected(G)}")
    
    print("\n=== Node Type Distribution ===")
    for node_type in unique_types:
        count = sum(1 for t in nodeTypes if t == node_type)
        print(f"{node_type}: {count} nodes")
    
    print("\n=== Edge Length Statistics ===")
    lengths = edge_attr['length'].values
    print(f"Min length: {lengths.min():.2f}")
    print(f"Max length: {lengths.max():.2f}")
    print(f"Average length: {lengths.mean():.2f}")
    print(f"Median length: {np.median(lengths):.2f}")
    
    return G, pos, nodeTypes, edge_attr

if __name__ == '__main__':
    # Set font (avoid Chinese font warnings)
    plt.rcParams['font.family'] = ['DejaVu Sans']
    
    # Create the visualization
    G, pos, nodeTypes, edge_attr = create_directed_graph_visualization(
        show_edge_labels=True, 
        label_threshold=100
    )
    
    print("\nVisualization saved as 'directed_pipeline_graph.png'")
    print("Visualization complete!")
