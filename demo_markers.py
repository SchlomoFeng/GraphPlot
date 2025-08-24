#!/usr/bin/env python3
"""
Demo script showing the enhanced GraphPlot functionality with node type markers

This script demonstrates the new capability to visualize different node types
using distinct marker shapes while maintaining cluster color coding.

Node Types and Markers:
- Stream: Circle (○) - marker='o'
- Mixer: Square (■) - marker='s' 
- VavlePro/ValvePro: Triangle (▲) - marker='^'
- Tee: Diamond (◆) - marker='D'
- Unknown: Pentagon (⬟) - marker='p'
"""

import os
import sys

def main():
    """Run the enhanced graph visualization with node type markers"""
    print("GraphPlot Enhanced with Node Type Markers")
    print("=" * 50)
    
    # Import and run the main function
    from GraphPlot_0708YTS4 import main as graph_main
    
    print("Node Type Markers:")
    print("- Stream: Circle (○)")
    print("- Mixer: Square (■)")
    print("- VavlePro/ValvePro: Triangle (▲)")
    print("- Tee: Diamond (◆)")
    print("- Unknown: Pentagon (⬟)")
    print()
    
    print("Running graph clustering with enhanced visualization...")
    graph_main(debug=True)
    
    print("\nVisualization complete!")
    print("Generated files:")
    for filename in ["DBSCAN聚类结果_with_markers.png", 
                    "层次聚类结果_with_markers.png",
                    "K-means聚类结果_with_markers.png"]:
        if os.path.exists(filename):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (not found)")

if __name__ == "__main__":
    main()