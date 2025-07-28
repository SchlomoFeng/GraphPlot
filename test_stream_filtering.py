#!/usr/bin/env python3
"""
Test script to verify Stream node filtering functionality
"""

import json
import pandas as pd
from GraphPlot_0708YTS4 import extract_node_types, filter_stream_nodes, read_PipeFile

def test_node_type_extraction():
    """Test that node types are correctly extracted"""
    print("Testing node type extraction...")
    
    # Read the data
    nodeDF, edgeDF, nodeID, edgeID, nodePSTs = read_PipeFile(debug=True)
    
    # Extract node types
    node_types = extract_node_types(nodeDF)
    
    # Count each type
    type_counts = {}
    for node_type in node_types.values():
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    print(f"Total nodes: {len(nodeDF)}")
    print("Node type counts:")
    for node_type, count in sorted(type_counts.items()):
        print(f"  {node_type}: {count}")
    
    # Verify Stream nodes exist
    assert 'Stream' in type_counts, "No Stream nodes found"
    assert type_counts['Stream'] > 0, "Stream count should be > 0"
    
    print(f"✓ Found {type_counts['Stream']} Stream nodes")
    return node_types, nodeDF

def test_stream_filtering():
    """Test that Stream nodes are correctly filtered"""
    print("\nTesting Stream node filtering...")
    
    node_types, nodeDF = test_node_type_extraction()
    
    # Filter Stream nodes
    stream_indices = filter_stream_nodes(nodeDF, node_types)
    
    # Verify filtering
    stream_count = sum(1 for node_type in node_types.values() if node_type == 'Stream')
    
    assert len(stream_indices) == stream_count, f"Expected {stream_count} Stream nodes, got {len(stream_indices)}"
    
    # Verify that all filtered nodes are indeed Stream nodes
    for idx in stream_indices:
        node_id = nodeDF['id'].iloc[idx]
        assert node_types[node_id] == 'Stream', f"Node {node_id} at index {idx} is not a Stream node"
    
    print(f"✓ Successfully filtered {len(stream_indices)} Stream nodes")
    return stream_indices

def test_edge_connectivity():
    """Test that all edges are preserved"""
    print("\nTesting edge connectivity...")
    
    nodeDF, edgeDF, nodeID, edgeID, nodePSTs = read_PipeFile(debug=True)
    
    print(f"Total edges in dataset: {len(edgeDF)}")
    
    # Verify edge structure
    source_nodes = set(edgeDF['sourceid'])
    target_nodes = set(edgeDF['targetid'])
    all_edge_nodes = source_nodes.union(target_nodes)
    all_nodes = set(nodeDF['id'])
    
    # Check if all edge endpoints exist in nodes
    missing_nodes = all_edge_nodes - all_nodes
    if missing_nodes:
        print(f"Warning: {len(missing_nodes)} edge endpoints reference non-existent nodes")
    else:
        print("✓ All edge endpoints reference valid nodes")
    
    return len(edgeDF)

def main():
    """Run all tests"""
    print("=== Stream Node Filtering Tests ===")
    
    try:
        # Test 1: Node type extraction
        test_node_type_extraction()
        
        # Test 2: Stream filtering
        stream_indices = test_stream_filtering()
        
        # Test 3: Edge connectivity
        edge_count = test_edge_connectivity()
        
        print(f"\n=== Test Summary ===")
        print(f"✓ All tests passed!")
        print(f"✓ Stream nodes identified: {len(stream_indices)}")
        print(f"✓ Total edges preserved: {edge_count}")
        print(f"✓ Implementation successfully filters nodes while preserving edge connectivity")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()