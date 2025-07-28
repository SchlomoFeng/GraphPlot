/**
 * 主应用控制器
 * 协调数据解析器和图形可视化器，处理用户交互
 */

class PipelineVisualizationApp {
    constructor() {
        this.dataParser = new DataParser();
        this.visualizer = new GraphVisualizer('graph-svg');
        this.currentData = null;
        
        this.init();
    }

    /**
     * 初始化应用
     */
    init() {
        this.setupEventListeners();
        this.showLoadingState();
        
        // 尝试自动加载默认数据
        this.loadDefaultData();
    }

    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 文件加载按钮
        const loadFileBtn = document.getElementById('load-file-btn');
        const fileInput = document.getElementById('file-input');
        
        if (loadFileBtn && fileInput) {
            loadFileBtn.addEventListener('click', () => {
                fileInput.click();
            });
            
            fileInput.addEventListener('change', (event) => {
                this.handleFileLoad(event);
            });
        }

        // 加载默认数据按钮
        const loadDefaultBtn = document.getElementById('load-default-btn');
        if (loadDefaultBtn) {
            loadDefaultBtn.addEventListener('click', () => {
                this.loadDefaultData();
            });
        }

        // 重置视图按钮
        const resetViewBtn = document.getElementById('reset-view-btn');
        if (resetViewBtn) {
            resetViewBtn.addEventListener('click', () => {
                this.visualizer.resetView();
            });
        }

        // 导出按钮
        const exportPngBtn = document.getElementById('export-png-btn');
        const exportSvgBtn = document.getElementById('export-svg-btn');
        
        if (exportPngBtn) {
            exportPngBtn.addEventListener('click', () => {
                this.visualizer.exportPNG();
            });
        }
        
        if (exportSvgBtn) {
            exportSvgBtn.addEventListener('click', () => {
                this.visualizer.exportSVG();
            });
        }

        // 缩放控制
        const zoomLevel = document.getElementById('zoom-level');
        if (zoomLevel) {
            zoomLevel.addEventListener('input', (event) => {
                this.visualizer.setZoomLevel(parseInt(event.target.value));
            });
        }

        // 显示控制
        const showLabels = document.getElementById('show-labels');
        const showEdgeLabels = document.getElementById('show-edge-labels');
        
        if (showLabels) {
            showLabels.addEventListener('change', (event) => {
                this.visualizer.toggleLabels(event.target.checked);
            });
        }
        
        if (showEdgeLabels) {
            showEdgeLabels.addEventListener('change', (event) => {
                this.visualizer.toggleEdgeLabels(event.target.checked);
            });
        }

        // 拖拽文件支持
        this.setupDragAndDrop();
    }

    /**
     * 设置拖拽文件功能
     */
    setupDragAndDrop() {
        const dropZone = document.querySelector('.visualization-area');
        if (!dropZone) return;

        dropZone.addEventListener('dragover', (event) => {
            event.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (event) => {
            event.preventDefault();
            dropZone.classList.remove('drag-over');
            
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                this.loadFile(files[0]);
            }
        });
    }

    /**
     * 处理文件加载
     */
    handleFileLoad(event) {
        const file = event.target.files[0];
        if (file) {
            this.loadFile(file);
        }
    }

    /**
     * 加载文件
     */
    loadFile(file) {
        this.showLoadingState('正在读取文件...');
        
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const content = event.target.result;
                this.processData(content);
            } catch (error) {
                this.showError('文件读取失败: ' + error.message);
            }
        };
        
        reader.onerror = () => {
            this.showError('文件读取失败');
        };
        
        reader.readAsText(file, 'utf-8');
    }

    /**
     * 加载默认数据
     */
    loadDefaultData() {
        this.showLoadingState('正在加载默认数据...');
        
        // 使用fetch加载默认的JSON文件
        fetch('0708烟台_updated.txt')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.text();
            })
            .then(content => {
                this.processData(content);
            })
            .catch(error => {
                console.error('加载默认数据失败:', error);
                this.showError('无法加载默认数据文件。请手动选择数据文件。');
            });
    }

    /**
     * 处理数据
     */
    processData(content) {
        try {
            this.showLoadingState('正在解析数据...');
            
            // 解析数据
            const parsedData = this.dataParser.parseData(content);
            
            // 获取节点类型配置
            const nodeTypeConfig = this.dataParser.getNodeTypeConfig();
            
            // 准备可视化数据
            this.currentData = {
                ...parsedData,
                nodeTypeConfig: nodeTypeConfig
            };
            
            this.showLoadingState('正在渲染图形...');
            
            // 延迟一点让UI更新
            setTimeout(() => {
                // 传递给可视化器
                this.visualizer.setData(this.currentData);
                
                // 更新统计信息
                this.updateStats();
                
                // 更新图例
                this.updateLegend();
                
                // 隐藏加载状态
                this.hideLoadingState();
                
                console.log('数据加载和可视化完成');
            }, 100);
            
        } catch (error) {
            console.error('处理数据时出错:', error);
            this.showError('数据处理失败: ' + error.message);
        }
    }

    /**
     * 更新统计信息
     */
    updateStats() {
        if (!this.currentData) return;
        
        const nodeCount = document.getElementById('node-count');
        const edgeCount = document.getElementById('edge-count');
        const typeCount = document.getElementById('type-count');
        
        if (nodeCount) nodeCount.textContent = this.currentData.nodes.length;
        if (edgeCount) edgeCount.textContent = this.currentData.edges.length;
        if (typeCount) typeCount.textContent = this.currentData.nodeTypes.length;
    }

    /**
     * 更新图例
     */
    updateLegend() {
        if (!this.currentData) return;
        
        const legendContainer = document.getElementById('legend-container');
        if (!legendContainer) return;
        
        legendContainer.innerHTML = '';
        
        // 为每种节点类型创建图例项
        this.currentData.nodeTypes.forEach(type => {
            const config = this.currentData.nodeTypeConfig[type];
            if (!config) return;
            
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            
            const symbol = document.createElement('div');
            symbol.className = 'legend-symbol';
            symbol.style.backgroundColor = config.color;
            
            // 根据形状设置样式
            switch (config.shape) {
                case 'circle':
                    symbol.style.borderRadius = '50%';
                    break;
                case 'diamond':
                    symbol.style.transform = 'rotate(45deg)';
                    break;
                case 'triangle':
                    symbol.style.clipPath = 'polygon(50% 0%, 0% 100%, 100% 100%)';
                    symbol.style.backgroundColor = config.color;
                    break;
            }
            
            const text = document.createElement('span');
            text.className = 'legend-text';
            text.textContent = `${config.label} (${this.getTypeCount(type)})`;
            
            legendItem.appendChild(symbol);
            legendItem.appendChild(text);
            legendContainer.appendChild(legendItem);
            
            // 添加点击事件来高亮相同类型的节点
            legendItem.addEventListener('click', () => {
                this.highlightNodeType(type);
            });
        });
    }

    /**
     * 获取指定类型的节点数量
     */
    getTypeCount(type) {
        return this.currentData.nodes.filter(node => node.type === type).length;
    }

    /**
     * 高亮指定类型的节点
     */
    highlightNodeType(type) {
        // 这里可以实现节点类型的高亮功能
        console.log(`高亮节点类型: ${type}`);
        // TODO: 在可视化器中实现这个功能
    }

    /**
     * 显示加载状态
     */
    showLoadingState(message = '正在加载...') {
        const visualizationArea = document.querySelector('.visualization-area');
        if (!visualizationArea) return;
        
        // 移除现有的加载指示器
        this.hideLoadingState();
        
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading';
        loadingDiv.textContent = message;
        loadingDiv.id = 'loading-indicator';
        
        visualizationArea.appendChild(loadingDiv);
    }

    /**
     * 隐藏加载状态
     */
    hideLoadingState() {
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }

    /**
     * 显示错误信息
     */
    showError(message) {
        this.hideLoadingState();
        
        const visualizationArea = document.querySelector('.visualization-area');
        if (!visualizationArea) return;
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 1rem;
            max-width: 400px;
            text-align: center;
            z-index: 1000;
        `;
        
        errorDiv.innerHTML = `
            <h4>错误</h4>
            <p>${message}</p>
            <button onclick="this.parentElement.remove()" 
                    style="margin-top: 10px; padding: 5px 15px; background: #721c24; color: white; border: none; border-radius: 3px; cursor: pointer;">
                关闭
            </button>
        `;
        
        visualizationArea.appendChild(errorDiv);
        
        // 5秒后自动移除错误信息
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 5000);
    }

    /**
     * 获取当前数据（供其他组件使用）
     */
    getCurrentData() {
        return this.currentData;
    }
}

// 在页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PipelineVisualizationApp();
});

// 导出供其他模块使用
window.PipelineVisualizationApp = PipelineVisualizationApp;