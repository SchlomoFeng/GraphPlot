# GraphPlot

## Directed Pipeline Network Visualization

This repository contains tools for visualizing pipeline network data as directed graphs with heterogeneous node types and enhanced visualization features.

### Features

- **Directed Graph Representation**: Shows flow direction using arrows
- **Heterogeneous Node Visualization**: Different colors, shapes, and sizes for different node types
- **Edge Attributes**: Displays edge lengths/distances as labels
- **Real-world Coordinates**: Uses actual position data for accurate spatial representation
- **Node Type Legend**: Comprehensive legend showing all node types and counts
- **Statistical Analysis**: Provides graph connectivity and edge length statistics

### Node Types

- **VavlePro**: Red squares - Valve/Flow control components (36 nodes)
- **Stream**: Teal circles - Flow streams (84 nodes)
- **Tee**: Blue triangles - Junction points (68 nodes)
- **Mixer**: Green diamonds - Mixing components (21 nodes)

### Usage

#### New Directed Graph Visualization
```bash
python GraphPlot_directed.py
```
This creates a professional directed graph visualization with:
- Flow direction arrows
- Node type differentiation
- Edge length labels
- Comprehensive statistics

#### Original Clustering Analysis
```bash
python GraphPlot_0708YTS4.py
```
The original clustering-based analysis (now fixed to work with local data).

### Data Format

The pipeline data is stored in JSON format (`0708烟台_updated.txt`) containing:
- `nodelist`: Nodes with type, name, and position parameters
- `linklist`: Edges with source/target IDs and length parameters

### Output

The visualization generates:
- High-resolution PNG image (`directed_pipeline_graph.png`)
- Statistical summary of the network
- Node type distribution analysis
- Edge length statistics

### Dependencies

```bash
pip install networkx matplotlib scikit-learn pandas numpy
```