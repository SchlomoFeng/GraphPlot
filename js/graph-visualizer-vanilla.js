/**
 * 简化的图形可视化模块
 * 使用原生SVG和JavaScript实现交互式管网拓扑图
 */

class GraphVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.svg = document.getElementById(containerId);
        this.width = 0;
        this.height = 0;
        
        // 数据
        this.nodes = [];
        this.edges = [];
        this.nodeTypeConfig = {};
        
        // 状态
        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;
        this.isPanning = false;
        this.lastMousePos = { x: 0, y: 0 };
        
        // 选择状态
        this.selectedNode = null;
        this.selectedEdge = null;
        
        // 显示设置
        this.showLabels = true;
        this.showEdgeLabels = true;
        
        this.init();
    }

    /**
     * 初始化SVG和事件
     */
    init() {
        // 获取容器尺寸
        this.updateSize();
        
        // 清空SVG
        this.svg.innerHTML = '';
        
        // 创建主容器组
        this.container = this.createSVGElement('g', { class: 'graph-container' });
        this.svg.appendChild(this.container);
        
        // 创建图层
        this.edgeGroup = this.createSVGElement('g', { class: 'edges' });
        this.nodeGroup = this.createSVGElement('g', { class: 'nodes' });
        this.labelGroup = this.createSVGElement('g', { class: 'labels' });
        
        this.container.appendChild(this.edgeGroup);
        this.container.appendChild(this.nodeGroup);
        this.container.appendChild(this.labelGroup);
        
        // 设置缩放和平移事件
        this.setupPanZoom();
        
        // 监听窗口大小变化
        window.addEventListener('resize', () => {
            this.updateSize();
            this.fitToView();
        });
    }

    /**
     * 创建SVG元素
     */
    createSVGElement(tagName, attributes = {}) {
        const element = document.createElementNS('http://www.w3.org/2000/svg', tagName);
        Object.keys(attributes).forEach(key => {
            element.setAttribute(key, attributes[key]);
        });
        return element;
    }

    /**
     * 设置平移和缩放
     */
    setupPanZoom() {
        this.svg.addEventListener('mousedown', (e) => {
            if (e.target === this.svg || e.target === this.container) {
                this.isPanning = true;
                this.lastMousePos = { x: e.clientX, y: e.clientY };
                this.svg.style.cursor = 'grabbing';
            }
        });

        this.svg.addEventListener('mousemove', (e) => {
            if (this.isPanning) {
                const dx = e.clientX - this.lastMousePos.x;
                const dy = e.clientY - this.lastMousePos.y;
                
                this.translateX += dx;
                this.translateY += dy;
                
                this.updateTransform();
                
                this.lastMousePos = { x: e.clientX, y: e.clientY };
            }
        });

        this.svg.addEventListener('mouseup', () => {
            this.isPanning = false;
            this.svg.style.cursor = 'grab';
        });

        this.svg.addEventListener('mouseleave', () => {
            this.isPanning = false;
            this.svg.style.cursor = 'grab';
        });

        // 缩放
        this.svg.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            const rect = this.svg.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = Math.max(0.1, Math.min(5, this.scale * scaleFactor));
            
            // 计算缩放中心
            const scaleChange = newScale / this.scale;
            this.translateX = mouseX - (mouseX - this.translateX) * scaleChange;
            this.translateY = mouseY - (mouseY - this.translateY) * scaleChange;
            
            this.scale = newScale;
            this.updateTransform();
            this.updateZoomLevel(this.scale);
        });
    }

    /**
     * 更新变换
     */
    updateTransform() {
        this.container.setAttribute('transform', 
            `translate(${this.translateX}, ${this.translateY}) scale(${this.scale})`);
    }

    /**
     * 更新容器尺寸
     */
    updateSize() {
        const rect = this.svg.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;
        this.svg.setAttribute('width', this.width);
        this.svg.setAttribute('height', this.height);
    }

    /**
     * 设置数据并开始可视化
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
        
        // 过滤有效的边
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
        // 清空现有元素
        this.edgeGroup.innerHTML = '';
        this.nodeGroup.innerHTML = '';
        this.labelGroup.innerHTML = '';
        
        // 渲染边
        this.renderEdges();
        
        // 渲染节点
        this.renderNodes();
        
        // 渲染标签
        this.renderLabels();
        
        // 适应视图
        this.fitToView();
    }

    /**
     * 渲染边
     */
    renderEdges() {
        this.edges.forEach(edge => {
            const source = this.getNodeById(edge.source);
            const target = this.getNodeById(edge.target);
            
            const line = this.createSVGElement('line', {
                class: 'edge',
                x1: source.x,
                y1: source.y,
                x2: target.x,
                y2: target.y,
                stroke: '#999',
                'stroke-width': 2,
                style: 'cursor: pointer'
            });

            // 设置边的样式
            if (edge.parameter && edge.parameter.styles) {
                const styles = edge.parameter.styles;
                if (styles.color) line.setAttribute('stroke', styles.color);
                if (styles.width) line.setAttribute('stroke-width', Math.max(1, styles.width / 10));
                if (styles.dash) line.setAttribute('stroke-dasharray', '5,5');
            }

            // 添加事件监听器
            line.addEventListener('click', () => this.selectEdge(edge));
            line.addEventListener('mouseenter', (e) => this.showTooltip(e, this.formatEdgeTooltip(edge)));
            line.addEventListener('mouseleave', () => this.hideTooltip());

            this.edgeGroup.appendChild(line);
        });
    }

    /**
     * 渲染节点
     */
    renderNodes() {
        this.nodes.forEach(node => {
            const config = this.nodeTypeConfig[node.type] || this.nodeTypeConfig['Unknown'] || {
                shape: 'circle',
                color: '#999',
                size: 10,
                label: 'Unknown'
            };

            const nodeGroup = this.createSVGElement('g', {
                class: 'node-group',
                transform: `translate(${node.x}, ${node.y})`,
                style: 'cursor: pointer'
            });

            // 绘制节点形状
            const shape = this.createNodeShape(config);
            nodeGroup.appendChild(shape);

            // 添加事件监听器
            nodeGroup.addEventListener('click', () => this.selectNode(node));
            nodeGroup.addEventListener('mouseenter', (e) => this.showTooltip(e, this.formatNodeTooltip(node)));
            nodeGroup.addEventListener('mouseleave', () => this.hideTooltip());

            this.nodeGroup.appendChild(nodeGroup);
        });
    }

    /**
     * 创建节点形状
     */
    createNodeShape(config) {
        const size = config.size || 10;
        const color = config.color || '#999';
        const shape = config.shape || 'circle';

        switch (shape) {
            case 'circle':
                return this.createSVGElement('circle', {
                    r: size,
                    fill: color,
                    stroke: '#fff',
                    'stroke-width': 2
                });

            case 'rect':
                return this.createSVGElement('rect', {
                    width: size * 2,
                    height: size * 2,
                    x: -size,
                    y: -size,
                    fill: color,
                    stroke: '#fff',
                    'stroke-width': 2
                });

            case 'diamond':
                const diamondPath = `M 0,${-size} L ${size},0 L 0,${size} L ${-size},0 Z`;
                return this.createSVGElement('path', {
                    d: diamondPath,
                    fill: color,
                    stroke: '#fff',
                    'stroke-width': 2
                });

            case 'triangle':
                const trianglePath = `M 0,${-size} L ${size * 0.866},${size * 0.5} L ${-size * 0.866},${size * 0.5} Z`;
                return this.createSVGElement('path', {
                    d: trianglePath,
                    fill: color,
                    stroke: '#fff',
                    'stroke-width': 2
                });

            case 'hexagon':
                const hexPath = this.createHexagonPath(size);
                return this.createSVGElement('path', {
                    d: hexPath,
                    fill: color,
                    stroke: '#fff',
                    'stroke-width': 2
                });

            default:
                return this.createSVGElement('circle', {
                    r: size,
                    fill: color,
                    stroke: '#fff',
                    'stroke-width': 2
                });
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
            this.nodes.forEach(node => {
                const label = this.createSVGElement('text', {
                    class: 'node-label',
                    x: node.x,
                    y: node.y + 20,
                    'text-anchor': 'middle',
                    'font-size': '12px',
                    fill: '#333',
                    style: 'pointer-events: none; user-select: none;'
                });
                label.textContent = node.name;
                this.labelGroup.appendChild(label);
            });
        }

        if (this.showEdgeLabels) {
            // 边标签
            this.edges.forEach(edge => {
                const source = this.getNodeById(edge.source);
                const target = this.getNodeById(edge.target);
                
                const label = this.createSVGElement('text', {
                    class: 'edge-label',
                    x: (source.x + target.x) / 2,
                    y: (source.y + target.y) / 2,
                    'text-anchor': 'middle',
                    'font-size': '10px',
                    fill: '#666',
                    style: 'pointer-events: none; user-select: none;'
                });
                label.textContent = edge.name;
                this.labelGroup.appendChild(label);
            });
        }
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
        this.nodeGroup.querySelectorAll('.node-group').forEach(el => {
            el.classList.remove('selected');
        });
        this.selectedEdge = null;

        // 选择新节点
        this.selectedNode = node;

        // 高亮选中的节点
        this.nodeGroup.querySelectorAll('.node-group').forEach(el => {
            const transform = el.getAttribute('transform');
            if (transform === `translate(${node.x}, ${node.y})`) {
                el.classList.add('selected');
            }
        });

        // 更新信息面板
        this.updateInfoPanel(node, 'node');
    }

    /**
     * 选择边
     */
    selectEdge(edge) {
        // 清除之前的选择
        this.edgeGroup.querySelectorAll('.edge').forEach(el => {
            el.classList.remove('selected');
        });
        this.selectedNode = null;

        // 选择新边
        this.selectedEdge = edge;

        // 高亮选中的边
        const source = this.getNodeById(edge.source);
        const target = this.getNodeById(edge.target);
        
        this.edgeGroup.querySelectorAll('.edge').forEach(el => {
            if (el.getAttribute('x1') == source.x && 
                el.getAttribute('y1') == source.y &&
                el.getAttribute('x2') == target.x && 
                el.getAttribute('y2') == target.y) {
                el.classList.add('selected');
            }
        });

        // 更新信息面板
        this.updateInfoPanel(edge, 'edge');
    }

    /**
     * 适应视图
     */
    fitToView() {
        if (!this.bounds || this.nodes.length === 0) return;

        const padding = 50;
        const scaleX = (this.width - padding * 2) / this.bounds.width;
        const scaleY = (this.height - padding * 2) / this.bounds.height;
        this.scale = Math.min(scaleX, scaleY, 1);

        const centerX = this.bounds.minX + this.bounds.width / 2;
        const centerY = this.bounds.minY + this.bounds.height / 2;

        this.translateX = this.width / 2 - centerX * this.scale;
        this.translateY = this.height / 2 - centerY * this.scale;

        this.updateTransform();
        this.updateZoomLevel(this.scale);
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
        this.scale = level / 100;
        this.updateTransform();
        this.updateZoomLevel(this.scale);
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
        this.labelGroup.querySelectorAll('.node-label').forEach(el => {
            el.style.display = show ? 'block' : 'none';
        });
    }

    /**
     * 切换边标签显示
     */
    toggleEdgeLabels(show) {
        this.showEdgeLabels = show;
        this.labelGroup.querySelectorAll('.edge-label').forEach(el => {
            el.style.display = show ? 'block' : 'none';
        });
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
        const svgData = new XMLSerializer().serializeToString(this.svg);
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
        const svgData = new XMLSerializer().serializeToString(this.svg);

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