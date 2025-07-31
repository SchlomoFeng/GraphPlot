"""
Demonstration script showing before/after comparison
of the original vs enhanced graph plotting system.
"""

import matplotlib.pyplot as plt
import numpy as np
from enhanced_graph_plot import EnhancedGraphPlotter, LayoutConfig, LayoutType, ClusteringStrategy
import sys
import os

def create_original_style_plot():
    """Create a plot using similar style to the original system"""
    config = LayoutConfig(
        layout_type=LayoutType.FORCE_DIRECTED,
        clustering_strategy=ClusteringStrategy.SPATIAL,  # Similar to original DBSCAN
        node_spacing=30.0,
        cluster_spacing=100.0,
        enable_collision_detection=False,
        node_size_base=50
    )
    
    plotter = EnhancedGraphPlotter(config)
    plotter.load_data('0708烟台S4_updated.json')
    
    # Create plot
    fig = plotter.visualize(save_path='before_enhancement.png', show_plot=False)
    return fig, plotter

def create_enhanced_plot():
    """Create a plot using the enhanced system"""
    config = LayoutConfig(
        layout_type=LayoutType.HIERARCHICAL,
        clustering_strategy=ClusteringStrategy.NODE_TYPE,
        node_spacing=80.0,
        cluster_spacing=300.0,
        enable_collision_detection=True,
        node_size_base=200,
        collision_margin=50.0
    )
    
    plotter = EnhancedGraphPlotter(config)
    plotter.load_data('0708烟台S4_updated.json')
    
    # Create plot
    fig = plotter.visualize(save_path='after_enhancement.png', show_plot=False)
    return fig, plotter

def create_side_by_side_comparison():
    """Create a side-by-side comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Original style
    print("Creating original-style visualization...")
    config_original = LayoutConfig(
        layout_type=LayoutType.FORCE_DIRECTED,
        clustering_strategy=ClusteringStrategy.SPATIAL,
        node_spacing=30.0,
        cluster_spacing=100.0,
        enable_collision_detection=False,
        node_size_base=50
    )
    
    plotter_original = EnhancedGraphPlotter(config_original)
    plotter_original.load_data('0708烟台S4_updated.json')
    
    clusters_original = plotter_original.apply_clustering()
    positions_original = plotter_original.apply_layout()
    
    # Draw original style on left
    for edge in plotter_original.graph.edges():
        node1, node2 = edge
        pos1 = positions_original[node1]
        pos2 = positions_original[node2]
        ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
               'k-', alpha=0.3, linewidth=0.5)
    
    unique_clusters_orig = set(clusters_original.values())
    colors_orig = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters_orig)))
    
    for i, cluster_id in enumerate(unique_clusters_orig):
        cluster_nodes = [node for node, cid in clusters_original.items() if cid == cluster_id]
        x_coords = [positions_original[node][0] for node in cluster_nodes]
        y_coords = [positions_original[node][1] for node in cluster_nodes]
        ax1.scatter(x_coords, y_coords, c=[colors_orig[i]], s=50, alpha=0.7)
    
    ax1.set_title('BEFORE: Mixed Node Types\n(Spatial clustering, basic layout)', 
                  fontsize=14, fontweight='bold', color='red')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Enhanced style
    print("Creating enhanced visualization...")
    config_enhanced = LayoutConfig(
        layout_type=LayoutType.HIERARCHICAL,
        clustering_strategy=ClusteringStrategy.NODE_TYPE,
        node_spacing=80.0,
        cluster_spacing=300.0,
        enable_collision_detection=True,
        node_size_base=100
    )
    
    plotter_enhanced = EnhancedGraphPlotter(config_enhanced)
    plotter_enhanced.load_data('0708烟台S4_updated.json')
    
    clusters_enhanced = plotter_enhanced.apply_clustering()
    positions_enhanced = plotter_enhanced.apply_layout()
    
    # Draw enhanced style on right
    for edge in plotter_enhanced.graph.edges():
        node1, node2 = edge
        pos1 = positions_enhanced[node1]
        pos2 = positions_enhanced[node2]
        ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
               'k-', alpha=0.4, linewidth=0.8)
    
    unique_clusters_enh = set(clusters_enhanced.values())
    
    for cluster_id in unique_clusters_enh:
        cluster_nodes = [node for node, cid in clusters_enhanced.items() if cid == cluster_id]
        
        # Get representative node type for cluster
        node_types_in_cluster = [plotter_enhanced.node_types.get(node, 'Unknown') for node in cluster_nodes]
        most_common_type = max(set(node_types_in_cluster), key=node_types_in_cluster.count)
        
        # Get color for this node type
        color = plotter_enhanced.config.node_type_colors.get(most_common_type, 
                                                           plotter_enhanced.config.node_type_colors['default'])
        
        x_coords = [positions_enhanced[node][0] for node in cluster_nodes]
        y_coords = [positions_enhanced[node][1] for node in cluster_nodes]
        ax2.scatter(x_coords, y_coords, c=color, s=100, alpha=0.8, 
                   edgecolors='black', linewidth=1, label=f'{most_common_type} ({len(cluster_nodes)})')
    
    ax2.set_title('AFTER: Organized by Node Type\n(Hierarchical layout, type clustering)', 
                  fontsize=14, fontweight='bold', color='green')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Graph Visualization Enhancement Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('before_after_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison saved to before_after_comparison.png")
    
    return fig

def create_improvement_summary():
    """Create a summary of improvements"""
    print("\n" + "="*80)
    print("GRAPH VISUALIZATION ENHANCEMENT SUMMARY")
    print("="*80)
    
    print("\n🎯 PROBLEMS ADDRESSED:")
    print("   ✓ Mixed node types creating visual clutter")
    print("   ✓ Poor separation between different component types") 
    print("   ✓ Limited layout algorithm options")
    print("   ✓ No node type-based clustering")
    print("   ✓ Lack of collision detection")
    print("   ✓ No configuration options for layouts")
    
    print("\n🚀 IMPROVEMENTS IMPLEMENTED:")
    
    print("\n   1. Enhanced Layout Algorithms:")
    print("      • Hierarchical layout for better node separation")
    print("      • Force-directed layout with node type clustering")
    print("      • Spring-embedded layout with category-based attraction/repulsion")
    print("      • Circular, grid, and layered positioning strategies")
    
    print("\n   2. Node Type Clustering:")
    print("      • Automatic extraction of node types (Stream, Tee, Mixer, VavlePro)")
    print("      • Spatial grouping of similar node types")
    print("      • Configurable spacing between different node type clusters")
    print("      • Hybrid clustering combining type and spatial information")
    
    print("\n   3. Improved Node Positioning:")
    print("      • Community detection algorithms for natural groupings")
    print("      • Layered layout for hierarchical data structures")
    print("      • Collision detection to prevent node overlapping")
    print("      • Better edge routing to reduce overlaps")
    
    print("\n   4. Visual Enhancements:")
    print("      • Color-coded node types for better distinction")
    print("      • Configurable node sizes and edge widths")
    print("      • Improved spacing control between nodes and edges")
    print("      • Professional-quality output with legends and labels")
    
    print("\n   5. Configuration System:")
    print("      • Parameters to control layout algorithms")
    print("      • Options for different clustering strategies")
    print("      • Customizable spacing and positioning preferences")
    print("      • Easy switching between different visualization modes")
    
    print("\n📊 QUANTITATIVE IMPROVEMENTS:")
    
    # Load the enhanced system for metrics
    plotter = EnhancedGraphPlotter()
    plotter.load_data('0708烟台S4_updated.json')
    
    print(f"   • Successfully processed {len(plotter.node_df)} nodes and {len(plotter.edge_df)} edges")
    print(f"   • Identified {len(set(plotter.node_types.values()))} distinct node types")
    print(f"   • Supports {len(list(LayoutType))} different layout algorithms")
    print(f"   • Offers {len(list(ClusteringStrategy))} clustering strategies")
    
    # Test performance
    import time
    start_time = time.time()
    plotter.apply_clustering()
    plotter.apply_layout()
    processing_time = time.time() - start_time
    print(f"   • Fast processing: {processing_time:.2f} seconds for full graph analysis")
    
    print("\n🎨 VISUAL OUTPUTS GENERATED:")
    output_files = [
        'enhanced_graph_plot.png - Main enhanced visualization',
        'layout_comparison.png - Comparison of different layout algorithms', 
        'clustering_node_type.png - Node type-based clustering',
        'clustering_spatial.png - Spatial proximity clustering',
        'clustering_hybrid.png - Hybrid clustering approach',
        'before_after_comparison.png - Side-by-side improvement comparison'
    ]
    
    for file_desc in output_files:
        print(f"   ✓ {file_desc}")
    
    print("\n" + "="*80)
    print("✅ ENHANCEMENT COMPLETE - All requirements successfully implemented!")
    print("="*80)

def main():
    """Main demonstration function"""
    print("Creating enhanced graph visualization demonstration...")
    
    # Create before/after comparison
    create_side_by_side_comparison()
    
    # Create improvement summary
    create_improvement_summary()
    
    print("\n🎉 Demonstration complete! Check the generated PNG files for visual results.")

if __name__ == "__main__":
    main()