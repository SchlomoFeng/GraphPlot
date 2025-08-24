/**
 * 图形可视化模块
 * 使用D3.js实现交互式管网拓扑图
 */

class GraphVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.svg = d3.select(`#${containerId}`);
        this.width = 0;
        this.height = 0;
        
        // 数据
        this.nodes = [];
        this.edges = [];
        this.nodeTypeConfig = {};
        
        // D3组件
        this.zoom = null;
        this.simulation = null;
        this.container = null;
        
        // 选择状态
        this.selectedNode = null;
        this.selectedEdge = null;
        
        // 显示设置
        this.showLabels = true;
        this.showEdgeLabels = true;
        
        this.init();
    }

    /**
     * 初始化SVG和缩放
     */
    init() {
        // 获取容器尺寸
        this.updateSize();
        
        // 清空SVG
        this.svg.selectAll('*').remove();
        
        // 设置缩放行为
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 5])
            .on('zoom', (event) => {
                this.container.attr('transform', event.transform);
                this.updateZoomLevel(event.transform.k);
            });
        
        this.svg.call(this.zoom);
        
        // 创建主容器组
        this.container = this.svg.append('g').attr('class', 'graph-container');
        
        // 创建图层
        this.edgeGroup = this.container.append('g').attr('class', 'edges');
        this.nodeGroup = this.container.append('g').attr('class', 'nodes');
        this.labelGroup = this.container.append('g').attr('class', 'labels');
        
        // 监听窗口大小变化
        window.addEventListener('resize', () => {
            this.updateSize();
            this.fitToView();
        });
    }

    /**
     * 更新容器尺寸
     */
    updateSize() {
        const rect = this.svg.node().getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;
        this.svg.attr('width', this.width).attr('height', this.height);
    }

    /**
     * 设置数据并开始可视化
     * @param {Object} data - 包含nodes和edges的数据对象
     */
    setData(data) {
        this.nodes = data.nodes || [];
        this.edges = data.edges || [];
        this.nodeTypeConfig = data.nodeTypeConfig || {};
        this.bounds = data.bounds;
        
        // 创建节点查找映射
        this.nodeMap = new Map();
        this.nodes.forEach(node => {
            this.nodeMap.set(node.id, node);
        });
        
        // 过滤有效的边（两端节点都存在）
        this.edges = this.edges.filter(edge => {
            return this.nodeMap.has(edge.source) && this.nodeMap.has(edge.target);
        });
        
        console.log(`可视化数据: ${this.nodes.length} 个节点, ${this.edges.length} 条边`);
        
        this.render();
    }

    /**
     * 渲染图形
     */
    render() {
        // 停止之前的仿真
        if (this.simulation) {
            this.simulation.stop();
        }
        
        // 清空现有元素
        this.edgeGroup.selectAll('*').remove();
        this.nodeGroup.selectAll('*').remove();
        this.labelGroup.selectAll('*').remove();
        
        // 渲染边
        this.renderEdges();
        
        // 渲染节点
        this.renderNodes();
        
        // 渲染标签
        this.renderLabels();
        
        // 创建力导向仿真（可选，主要用于没有坐标的节点）
        this.createSimulation();
        
        // 适应视图
        this.fitToView();
    }

    /**
     * 渲染边
     */
    renderEdges() {
        const edges = this.edgeGroup
            .selectAll('.edge')
            .data(this.edges)
            .enter()
            .append('line')
            .attr('class', 'edge')
            .attr('stroke', '#999')
            .attr('stroke-width', 2)
            .attr('x1', d => this.getNodeById(d.source).x)
            .attr('y1', d => this.getNodeById(d.source).y)
            .attr('x2', d => this.getNodeById(d.target).x)
            .attr('y2', d => this.getNodeById(d.target).y)
            .style('cursor', 'pointer')
            .on('click', (event, d) => this.selectEdge(d))
            .on('mouseover', (event, d) => this.showTooltip(event, this.formatEdgeTooltip(d)))
            .on('mouseout', () => this.hideTooltip());
        
        // 设置边的样式
        edges.each(function(d) {
            const edge = d3.select(this);
            if (d.parameter && d.parameter.styles) {
                const styles = d.parameter.styles;
                if (styles.color) edge.attr('stroke', styles.color);
                if (styles.width) edge.attr('stroke-width', Math.max(1, styles.width / 10));
                if (styles.dash) edge.attr('stroke-dasharray', '5,5');
            }
        });
    }

    /**
     * 渲染节点
     */
    renderNodes() {
        const nodeGroups = this.nodeGroup
            .selectAll('.node-group')
            .data(this.nodes)
            .enter()
            .append('g')
            .attr('class', 'node-group')
            .attr('transform', d => `translate(${d.x}, ${d.y})`)
            .style('cursor', 'pointer')
            .on('click', (event, d) => this.selectNode(d))
            .on('mouseover', (event, d) => this.showTooltip(event, this.formatNodeTooltip(d)))
            .on('mouseout', () => this.hideTooltip());

        // 根据节点类型绘制不同形状
        nodeGroups.each((d, i, nodes) => {
            const group = d3.select(nodes[i]);
            const config = this.nodeTypeConfig[d.type] || this.nodeTypeConfig['Unknown'];
            
            this.drawNodeShape(group, config);
        });
    }

    /**
     * 根据配置绘制节点形状
     */
    drawNodeShape(group, config) {
        const size = config.size || 10;
        const color = config.color || '#999';
        const shape = config.shape || 'circle';
        
        switch (shape) {
            case 'circle':
                group.append('circle')
                    .attr('r', size)
                    .attr('fill', color)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2);
                break;
                
            case 'rect':
                group.append('rect')
                    .attr('width', size * 2)
                    .attr('height', size * 2)
                    .attr('x', -size)
                    .attr('y', -size)
                    .attr('fill', color)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2);
                break;
                
            case 'diamond':
                const diamondPath = `M 0,${-size} L ${size},0 L 0,${size} L ${-size},0 Z`;
                group.append('path')
                    .attr('d', diamondPath)
                    .attr('fill', color)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2);
                break;
                
            case 'triangle':
                const trianglePath = `M 0,${-size} L ${size * 0.866},${size * 0.5} L ${-size * 0.866},${size * 0.5} Z`;
                group.append('path')
                    .attr('d', trianglePath)
                    .attr('fill', color)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2);
                break;
                
            case 'hexagon':
                const hexPath = this.createHexagonPath(size);
                group.append('path')
                    .attr('d', hexPath)
                    .attr('fill', color)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2);
                break;
                
            case 'cross':
                group.append('path')
                    .attr('d', `M ${-size},0 L ${size},0 M 0,${-size} L 0,${size}`)
                    .attr('stroke', color)
                    .attr('stroke-width', 4)
                    .attr('stroke-linecap', 'round');
                break;
                
            default:
                // 默认圆形
                group.append('circle')
                    .attr('r', size)
                    .attr('fill', color)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2);
        }
    }

    /**
     * 创建六边形路径
     */
    createHexagonPath(size) {
        const points = [];
        for (let i = 0; i < 6; i++) {
            const angle = (Math.PI / 3) * i;
            const x = size * Math.cos(angle);
            const y = size * Math.sin(angle);
            points.push(`${x},${y}`);
        }
        return `M ${points.join(' L ')} Z`;
    }

    /**
     * 渲染标签
     */
    renderLabels() {
        if (this.showLabels) {
            // 节点标签
            this.labelGroup
                .selectAll('.node-label')
                .data(this.nodes)
                .enter()
                .append('text')
                .attr('class', 'node-label')
                .attr('x', d => d.x)
                .attr('y', d => d.y + 20)
                .attr('text-anchor', 'middle')
                .attr('font-size', '12px')
                .attr('fill', '#333')
                .text(d => d.name)
                .style('pointer-events', 'none');
        }
        
        if (this.showEdgeLabels) {
            // 边标签
            this.labelGroup
                .selectAll('.edge-label')
                .data(this.edges)
                .enter()
                .append('text')
                .attr('class', 'edge-label')
                .attr('x', d => {
                    const source = this.getNodeById(d.source);
                    const target = this.getNodeById(d.target);
                    return (source.x + target.x) / 2;
                })
                .attr('y', d => {
                    const source = this.getNodeById(d.source);
                    const target = this.getNodeById(d.target);
                    return (source.y + target.y) / 2;
                })
                .attr('text-anchor', 'middle')
                .attr('font-size', '10px')
                .attr('fill', '#666')
                .text(d => d.name)
                .style('pointer-events', 'none');
        }
    }

    /**
     * 创建力导向仿真（用于微调布局）
     */
    createSimulation() {
        // 这里主要是为了让图形有一些动态效果，因为我们已经有坐标了
        this.simulation = d3.forceSimulation(this.nodes)
            .force('charge', d3.forceManyBody().strength(-50))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .alpha(0.1)  // 较小的alpha值，因为我们不需要太多的布局调整
            .alphaDecay(0.05);
    }

    /**
     * 获取节点通过ID
     */
    getNodeById(id) {
        return this.nodeMap.get(id) || { x: 0, y: 0 };
    }

    /**
     * 选择节点
     */
    selectNode(node) {
        // 清除之前的选择
        this.nodeGroup.selectAll('.node-group').classed('selected', false);
        this.selectedEdge = null;
        
        // 选择新节点
        this.selectedNode = node;
        
        // 高亮选中的节点
        this.nodeGroup.selectAll('.node-group')
            .filter(d => d.id === node.id)
            .classed('selected', true);
            
        // 更新信息面板
        this.updateInfoPanel(node, 'node');
    }

    /**
     * 选择边
     */
    selectEdge(edge) {
        // 清除之前的选择
        this.edgeGroup.selectAll('.edge').classed('selected', false);
        this.selectedNode = null;
        
        // 选择新边
        this.selectedEdge = edge;
        
        // 高亮选中的边
        this.edgeGroup.selectAll('.edge')
            .filter(d => d.id === edge.id)
            .classed('selected', true);
            
        // 更新信息面板
        this.updateInfoPanel(edge, 'edge');
    }

    /**
     * 适应视图
     */
    fitToView() {
        if (!this.bounds || this.nodes.length === 0) return;
        
        const padding = 50;
        const scale = Math.min(
            (this.width - padding * 2) / this.bounds.width,
            (this.height - padding * 2) / this.bounds.height
        );
        
        const centerX = this.bounds.minX + this.bounds.width / 2;
        const centerY = this.bounds.minY + this.bounds.height / 2;
        
        const transform = d3.zoomIdentity
            .translate(this.width / 2, this.height / 2)
            .scale(Math.min(scale, 1))
            .translate(-centerX, -centerY);
            
        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, transform);
    }

    /**
     * 重置视图
     */
    resetView() {
        this.fitToView();
    }

    /**
     * 设置缩放级别
     */
    setZoomLevel(level) {
        const scale = level / 100;
        const transform = d3.zoomTransform(this.svg.node());
        const newTransform = transform.scale(scale / transform.k);
        
        this.svg.transition()
            .duration(200)
            .call(this.zoom.transform, newTransform);
    }

    /**
     * 更新缩放级别显示
     */
    updateZoomLevel(scale) {
        const zoomValue = document.getElementById('zoom-value');
        if (zoomValue) {
            zoomValue.textContent = `${Math.round(scale * 100)}%`;
        }
        
        const zoomSlider = document.getElementById('zoom-level');
        if (zoomSlider) {
            zoomSlider.value = Math.round(scale * 100);
        }
    }

    /**
     * 切换标签显示
     */
    toggleLabels(show) {
        this.showLabels = show;
        this.labelGroup.selectAll('.node-label').style('display', show ? 'block' : 'none');
    }

    /**
     * 切换边标签显示
     */
    toggleEdgeLabels(show) {
        this.showEdgeLabels = show;
        this.labelGroup.selectAll('.edge-label').style('display', show ? 'block' : 'none');
    }

    /**
     * 显示工具提示
     */
    showTooltip(event, content) {
        const tooltip = document.getElementById('tooltip');
        if (tooltip) {
            tooltip.innerHTML = content;
            tooltip.style.left = `${event.pageX + 10}px`;
            tooltip.style.top = `${event.pageY - 10}px`;
            tooltip.classList.add('visible');
        }
    }

    /**
     * 隐藏工具提示
     */
    hideTooltip() {
        const tooltip = document.getElementById('tooltip');
        if (tooltip) {
            tooltip.classList.remove('visible');
        }
    }

    /**
     * 格式化节点工具提示
     */
    formatNodeTooltip(node) {
        let html = `<strong>${node.name}</strong><br>`;
        html += `类型: ${node.type}<br>`;
        html += `位置: (${node.x.toFixed(2)}, ${node.y.toFixed(2)})<br>`;
        
        if (Object.keys(node.properties).length > 0) {
            html += '<br><strong>属性:</strong><br>';
            Object.entries(node.properties).forEach(([key, value]) => {
                html += `${key}: ${value}<br>`;
            });
        }
        
        return html;
    }

    /**
     * 格式化边工具提示
     */
    formatEdgeTooltip(edge) {
        let html = `<strong>${edge.name}</strong><br>`;
        html += `类型: ${edge.type}<br>`;
        
        const source = this.getNodeById(edge.source);
        const target = this.getNodeById(edge.target);
        html += `连接: ${source.name || edge.source} → ${target.name || edge.target}<br>`;
        
        if (Object.keys(edge.properties).length > 0) {
            html += '<br><strong>属性:</strong><br>';
            Object.entries(edge.properties).forEach(([key, value]) => {
                html += `${key}: ${value}<br>`;
            });
        }
        
        return html;
    }

    /**
     * 更新信息面板
     */
    updateInfoPanel(item, type) {
        const infoContent = document.getElementById('info-content');
        if (!infoContent) return;
        
        let html = `<h4>${item.name}</h4>`;
        html += `<div class="property"><span class="property-key">类型:</span><span class="property-value">${item.type}</span></div>`;
        html += `<div class="property"><span class="property-key">ID:</span><span class="property-value">${item.id}</span></div>`;
        
        if (type === 'node') {
            html += `<div class="property"><span class="property-key">位置:</span><span class="property-value">(${item.x.toFixed(2)}, ${item.y.toFixed(2)})</span></div>`;
        } else if (type === 'edge') {
            const source = this.getNodeById(item.source);
            const target = this.getNodeById(item.target);
            html += `<div class="property"><span class="property-key">起点:</span><span class="property-value">${source.name || item.source}</span></div>`;
            html += `<div class="property"><span class="property-key">终点:</span><span class="property-value">${target.name || item.target}</span></div>`;
        }
        
        // 添加其他属性
        Object.entries(item.properties).forEach(([key, value]) => {
            html += `<div class="property"><span class="property-key">${key}:</span><span class="property-value">${value}</span></div>`;
        });
        
        infoContent.innerHTML = html;
    }

    /**
     * 导出为PNG
     */
    exportPNG() {
        const svgNode = this.svg.node();
        const svgData = new XMLSerializer().serializeToString(svgNode);
        
        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = () => {
            ctx.drawImage(img, 0, 0);
            
            const link = document.createElement('a');
            link.download = 'pipeline-graph.png';
            link.href = canvas.toDataURL();
            link.click();
        };
        
        const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        img.src = url;
    }

    /**
     * 导出为SVG
     */
    exportSVG() {
        const svgNode = this.svg.node();
        const svgData = new XMLSerializer().serializeToString(svgNode);
        
        const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.download = 'pipeline-graph.svg';
        link.href = url;
        link.click();
        
        URL.revokeObjectURL(url);
    }
}

// 导出供其他模块使用
window.GraphVisualizer = GraphVisualizer;