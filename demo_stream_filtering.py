#!/usr/bin/env python3
"""
Demonstration script showing Stream node filtering functionality
"""

import matplotlib.pyplot as plt
import numpy as np
from GraphPlot_0708YTS4 import read_PipeFile, extract_node_types, filter_stream_nodes

def demonstrate_filtering():
    """Demonstrate the Stream node filtering with a simple example"""
    print("🎯 Stream Node Filtering Demonstration")
    print("=" * 50)
    
    # Read data
    nodeDF, edgeDF, nodeID, edgeID, nodePSTs = read_PipeFile(debug=True)
    
    # Extract node types
    node_types = extract_node_types(nodeDF)
    stream_indices = filter_stream_nodes(nodeDF, node_types)
    
    # Create a simple visualization showing the difference
    plt.figure(figsize=(15, 5))
    
    # Plot 1: All nodes
    plt.subplot(1, 3, 1)
    for i in range(len(nodeDF)):
        pos = nodePSTs.iloc[i]
        node_id = nodeDF['id'].iloc[i]
        node_type = node_types[node_id]
        
        if node_type == 'Stream':
            plt.scatter(pos[0], pos[1], c='red', s=30, marker='o', alpha=0.7, label='Stream' if i == 0 else "")
        elif node_type == 'VavlePro':
            plt.scatter(pos[0], pos[1], c='blue', s=20, marker='s', alpha=0.5, label='VavlePro' if i == 0 else "")
        elif node_type == 'Mixer':
            plt.scatter(pos[0], pos[1], c='green', s=20, marker='^', alpha=0.5, label='Mixer' if i == 0 else "")
        elif node_type == 'Tee':
            plt.scatter(pos[0], pos[1], c='orange', s=20, marker='d', alpha=0.5, label='Tee' if i == 0 else "")
    
    plt.title('All Nodes (Original)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    
    # Plot 2: Stream nodes only
    plt.subplot(1, 3, 2)
    for idx in stream_indices:
        pos = nodePSTs.iloc[idx]
        plt.scatter(pos[0], pos[1], c='red', s=50, marker='o', alpha=0.8)
    
    plt.title('Stream Nodes Only (Filtered)')
    plt.grid(alpha=0.3)
    plt.axis('equal')
    
    # Plot 3: Legend explanation
    plt.subplot(1, 3, 3)
    plt.text(0.1, 0.8, 'Filtering Results:', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.7, f'• Total Nodes: {len(nodeDF)}', fontsize=12)
    plt.text(0.1, 0.6, f'• Stream Nodes: {len(stream_indices)}', fontsize=12, color='red')
    plt.text(0.1, 0.5, f'• Hidden Nodes: {len(nodeDF) - len(stream_indices)}', fontsize=12, color='gray')
    
    plt.text(0.1, 0.35, 'Visualization Features:', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.25, '• ● Circle markers for Stream nodes', fontsize=12, color='red')
    plt.text(0.1, 0.15, '• ─ Solid lines: Stream↔Stream', fontsize=12)
    plt.text(0.1, 0.05, '• ┅ Dashed lines: Stream↔Hidden', fontsize=12, alpha=0.6)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Implementation Features')
    
    plt.tight_layout()
    plt.savefig('/tmp/stream_filtering_demo.png', dpi=150, bbox_inches='tight')
    print(f"📊 Demo visualization saved to: /tmp/stream_filtering_demo.png")
    
    # Summary statistics
    type_counts = {}
    for node_type in node_types.values():
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    print(f"\n📈 Summary Statistics:")
    print(f"   📍 Total nodes processed: {len(nodeDF)}")
    print(f"   🎯 Stream nodes displayed: {len(stream_indices)} ({len(stream_indices)/len(nodeDF)*100:.1f}%)")
    print(f"   🔗 Edge connections preserved: {len(edgeDF)}")
    
    print(f"\n🏷️  Node type breakdown:")
    for node_type, count in sorted(type_counts.items()):
        visibility = "Visible" if node_type == "Stream" else "Hidden"
        print(f"   {node_type}: {count} ({visibility})")

if __name__ == "__main__":
    demonstrate_filtering()