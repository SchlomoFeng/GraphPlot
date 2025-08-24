# Stream Node Filtering Implementation

This document describes the enhanced graph visualization functionality that filters and displays only Stream type nodes while maintaining complete edge connectivity.

## Features

### 1. Node Type Extraction
- Automatically extracts node type information from the `parameter` field in the dataset
- Supports JSON string and dictionary parameter formats
- Handles missing or invalid type information gracefully

### 2. Stream Node Filtering
- Identifies and filters only nodes with `type="Stream"`
- Maintains original node indexing for proper edge connectivity
- Preserves all edge information regardless of node visibility

### 3. Enhanced Visualization
- **Circular Markers**: All Stream nodes are displayed using circular (○) markers
- **Color-coded Clustering**: Different colors represent different cluster assignments
- **Edge Differentiation**:
  - **Solid lines**: Connections between two visible Stream nodes
  - **Dashed lines**: Connections between a Stream node and a hidden non-Stream node
  - **Transparent**: Edges to hidden nodes have reduced opacity

### 4. Complete Edge Connectivity
- All edges from the original dataset are preserved and analyzed
- No edge information is lost during filtering
- Hidden nodes' connections are visualized as transparent/dashed lines

## Usage

### Basic Usage
```python
from GraphPlot_0708YTS4 import main

# Run with debug mode to use local data file
main(debug=True)
```

### API Usage
```python
from GraphPlot_0708YTS4 import read_PipeFile, extract_node_types, filter_stream_nodes

# Read data
nodeDF, edgeDF, nodeID, edgeID, nodePSTs = read_PipeFile(debug=True)

# Extract node types
node_types = extract_node_types(nodeDF)

# Filter Stream nodes
stream_indices = filter_stream_nodes(nodeDF, node_types)

print(f"Found {len(stream_indices)} Stream nodes out of {len(nodeDF)} total nodes")
```

## Output

### Console Output
```
读取到 209 个节点和 206 条边
节点类型统计:
  VavlePro: 36
  Mixer: 21
  Tee: 68
  Stream: 84
找到 84 个Stream类型节点
```

### Generated Visualizations
The system generates three clustering visualizations:
1. **DBSCAN Clustering** - Density-based clustering
2. **Hierarchical Clustering** - Agglomerative clustering  
3. **K-means Clustering** - Centroid-based clustering

Each visualization:
- Shows only Stream nodes as colored circles
- Uses solid lines for Stream-to-Stream connections
- Uses dashed/transparent lines for Stream-to-hidden connections
- Includes a legend explaining the visualization strategy

### Files Generated
- `DBSCANclustering_stream_only.png`
- `层次clustering_stream_only.png` 
- `K-meansclustering_stream_only.png`

## Data Requirements

### Input File Format
The system expects a JSON file (`0708烟台_updated.txt`) with the following structure:

```json
{
  "nodelist": [
    {
      "id": "node-uuid",
      "parameter": {
        "type": "Stream",
        "styles": {
          "position": {"x": 100, "y": 200}
        }
      }
    }
  ],
  "linklist": [
    {
      "id": "edge-uuid", 
      "sourceid": "source-node-uuid",
      "targetid": "target-node-uuid",
      "parameter": {
        "parameter": {"Length": 1.5}
      }
    }
  ]
}
```

### Node Types Supported
- `Stream`: Primary focus nodes (displayed)
- `VavlePro`: Valve components (hidden)
- `Mixer`: Mixing components (hidden)  
- `Tee`: Junction components (hidden)

## Implementation Details

### Key Functions

#### `extract_node_types(nodeDF)`
Extracts node type information from the DataFrame parameter field.

**Returns:** Dictionary mapping node IDs to their types

#### `filter_stream_nodes(nodeDF, node_types)`
Filters and returns indices of Stream type nodes.

**Returns:** List of indices for Stream nodes

#### `apply_clustering_and_visualize(...)`
Enhanced visualization function that:
- Applies clustering algorithms to all nodes
- Displays only Stream nodes with circular markers
- Renders all edges with appropriate styling
- Adds informational annotations

## Testing

Run the test suite to verify functionality:

```bash
python3 test_stream_filtering.py
```

The tests verify:
- ✓ Node type extraction accuracy
- ✓ Stream filtering correctness  
- ✓ Edge connectivity preservation
- ✓ Data integrity maintenance

## Benefits

1. **Focused Analysis**: Users can focus on Stream components without visual clutter
2. **Complete Connectivity**: No loss of network topology information
3. **Visual Clarity**: Clear distinction between visible and hidden connections
4. **Clustering Insights**: Maintains clustering analysis on complete dataset
5. **Flexible Framework**: Easy to extend for other node type filtering

This implementation successfully addresses the requirement to visualize only Stream nodes while preserving the complete network structure and connectivity information.