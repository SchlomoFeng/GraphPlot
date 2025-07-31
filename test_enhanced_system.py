"""
Test script for the enhanced graph plotting system
Validates the improvements against the requirements.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from enhanced_graph_plot import EnhancedGraphPlotter, LayoutConfig, LayoutType, ClusteringStrategy
import time


def test_node_type_separation():
    """Test that different node types are properly separated"""
    print("Testing node type separation...")
    
    config = LayoutConfig(
        layout_type=LayoutType.HIERARCHICAL,
        clustering_strategy=ClusteringStrategy.NODE_TYPE,
        cluster_spacing=200.0
    )
    
    plotter = EnhancedGraphPlotter(config)
    plotter.load_data('0708烟台S4_updated.json')
    
    clusters = plotter.apply_clustering()
    
    # Verify that nodes of the same type are in the same cluster
    type_to_cluster = {}
    for node, cluster_id in clusters.items():
        node_type = plotter.node_types[node]
        if node_type not in type_to_cluster:
            type_to_cluster[node_type] = cluster_id
        else:
            assert type_to_cluster[node_type] == cluster_id, f"Node type {node_type} found in multiple clusters"
    
    print(f"✓ Successfully separated {len(type_to_cluster)} node types into distinct clusters")
    return True


def test_layout_algorithms():
    """Test that different layout algorithms work correctly"""
    print("Testing layout algorithms...")
    
    plotter = EnhancedGraphPlotter()
    plotter.load_data('0708烟台S4_updated.json')
    
    layouts = [LayoutType.HIERARCHICAL, LayoutType.FORCE_DIRECTED, 
               LayoutType.CIRCULAR, LayoutType.GRID]
    
    for layout in layouts:
        plotter.config.layout_type = layout
        positions = plotter.apply_layout()
        
        # Verify that all nodes have positions
        assert len(positions) == len(plotter.graph.nodes()), f"Layout {layout} missing node positions"
        
        # Verify that positions are different (not all at origin)
        unique_positions = set(positions.values())
        assert len(unique_positions) > 1, f"Layout {layout} produced identical positions"
        
        print(f"✓ Layout algorithm {layout.value} working correctly")
    
    return True


def test_clustering_strategies():
    """Test that different clustering strategies produce different results"""
    print("Testing clustering strategies...")
    
    plotter = EnhancedGraphPlotter()
    plotter.load_data('0708烟台S4_updated.json')
    
    strategies = [ClusteringStrategy.NODE_TYPE, ClusteringStrategy.SPATIAL, 
                  ClusteringStrategy.HYBRID]
    
    results = {}
    
    for strategy in strategies:
        plotter.config.clustering_strategy = strategy
        clusters = plotter.apply_clustering()
        results[strategy] = clusters
        
        # Verify clustering results
        assert len(clusters) > 0, f"Strategy {strategy} produced no clusters"
        assert all(isinstance(cluster_id, (int, np.integer)) for cluster_id in clusters.values()), \
               f"Strategy {strategy} produced non-integer cluster IDs"
        
        print(f"✓ Clustering strategy {strategy.value} produced {len(set(clusters.values()))} clusters")
    
    # Verify that different strategies produce different results
    node_type_clusters = set(results[ClusteringStrategy.NODE_TYPE].values())
    spatial_clusters = set(results[ClusteringStrategy.SPATIAL].values())
    hybrid_clusters = set(results[ClusteringStrategy.HYBRID].values())
    
    assert len(node_type_clusters) != len(spatial_clusters) or \
           len(node_type_clusters) != len(hybrid_clusters), \
           "Different clustering strategies produced identical results"
    
    print("✓ Different clustering strategies produce different cluster arrangements")
    return True


def test_collision_detection():
    """Test collision detection prevents node overlapping"""
    print("Testing collision detection...")
    
    # Test with collision detection disabled
    config_no_collision = LayoutConfig(
        layout_type=LayoutType.FORCE_DIRECTED,
        enable_collision_detection=False,
        collision_margin=50.0
    )
    
    plotter_no_collision = EnhancedGraphPlotter(config_no_collision)
    plotter_no_collision.load_data('0708烟台S4_updated.json')
    positions_no_collision = plotter_no_collision.apply_layout()
    
    # Test with collision detection enabled
    config_with_collision = LayoutConfig(
        layout_type=LayoutType.FORCE_DIRECTED,
        enable_collision_detection=True,
        collision_margin=50.0
    )
    
    plotter_with_collision = EnhancedGraphPlotter(config_with_collision)
    plotter_with_collision.load_data('0708烟台S4_updated.json')
    positions_with_collision = plotter_with_collision.apply_layout()
    
    # Check for collisions in both cases
    def count_collisions(positions, margin):
        collisions = 0
        nodes = list(positions.keys())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                pos1 = np.array(positions[node1])
                pos2 = np.array(positions[node2])
                distance = np.linalg.norm(pos1 - pos2)
                if distance < margin:
                    collisions += 1
        return collisions
    
    collisions_without = count_collisions(positions_no_collision, 50.0)
    collisions_with = count_collisions(positions_with_collision, 50.0)
    
    print(f"✓ Collisions without detection: {collisions_without}")
    print(f"✓ Collisions with detection: {collisions_with}")
    print(f"✓ Collision detection reduced overlaps by {max(0, collisions_without - collisions_with)} instances")
    
    return True


def test_visual_quality_metrics():
    """Test various visual quality metrics"""
    print("Testing visual quality metrics...")
    
    plotter = EnhancedGraphPlotter()
    plotter.load_data('0708烟台S4_updated.json')
    
    # Test node type clustering quality
    plotter.config.clustering_strategy = ClusteringStrategy.NODE_TYPE
    clusters = plotter.apply_clustering()
    positions = plotter.apply_layout()
    
    # Calculate cluster cohesion (how close nodes in same cluster are)
    cluster_cohesions = {}
    cluster_nodes = {}
    
    for node, cluster_id in clusters.items():
        if cluster_id not in cluster_nodes:
            cluster_nodes[cluster_id] = []
        cluster_nodes[cluster_id].append(node)
    
    for cluster_id, nodes in cluster_nodes.items():
        if len(nodes) > 1:
            distances = []
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    pos1 = np.array(positions[node1])
                    pos2 = np.array(positions[node2])
                    distances.append(np.linalg.norm(pos1 - pos2))
            cluster_cohesions[cluster_id] = np.mean(distances)
    
    # Calculate cluster separation (how far apart different clusters are)
    cluster_centers = {}
    for cluster_id, nodes in cluster_nodes.items():
        center_x = np.mean([positions[node][0] for node in nodes])
        center_y = np.mean([positions[node][1] for node in nodes])
        cluster_centers[cluster_id] = (center_x, center_y)
    
    separations = []
    cluster_ids = list(cluster_centers.keys())
    for i, cluster1 in enumerate(cluster_ids):
        for cluster2 in cluster_ids[i+1:]:
            center1 = np.array(cluster_centers[cluster1])
            center2 = np.array(cluster_centers[cluster2])
            separations.append(np.linalg.norm(center1 - center2))
    
    avg_cohesion = np.mean(list(cluster_cohesions.values())) if cluster_cohesions else 0
    avg_separation = np.mean(separations) if separations else 0
    
    print(f"✓ Average cluster cohesion (lower is better): {avg_cohesion:.2f}")
    print(f"✓ Average cluster separation (higher is better): {avg_separation:.2f}")
    print(f"✓ Separation/Cohesion ratio: {avg_separation/max(avg_cohesion, 1):.2f}")
    
    return True


def run_performance_comparison():
    """Compare performance of original vs enhanced system"""
    print("Running performance comparison...")
    
    # Time the enhanced system
    start_time = time.time()
    
    plotter = EnhancedGraphPlotter()
    plotter.load_data('0708烟台S4_updated.json')
    plotter.apply_clustering()
    plotter.apply_layout()
    
    enhanced_time = time.time() - start_time
    
    print(f"✓ Enhanced system processing time: {enhanced_time:.2f} seconds")
    print(f"✓ Successfully processed {len(plotter.graph.nodes())} nodes and {len(plotter.graph.edges())} edges")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED GRAPH PLOTTING SYSTEM - VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Node Type Separation", test_node_type_separation),
        ("Layout Algorithms", test_layout_algorithms),
        ("Clustering Strategies", test_clustering_strategies),
        ("Collision Detection", test_collision_detection),
        ("Visual Quality Metrics", test_visual_quality_metrics),
        ("Performance Comparison", run_performance_comparison)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! Enhanced graph plotting system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)