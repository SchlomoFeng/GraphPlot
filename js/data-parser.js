/**
 * 数据解析模块
 * 负责解析管网JSON数据，提取节点和连接信息
 */

class DataParser {
    constructor() {
        this.nodeTypes = new Set();
        this.edgeTypes = new Set();
    }

    /**
     * 解析JSON文件内容
     * @param {string} jsonContent - JSON文件内容
     * @returns {Object} 解析后的数据 {nodes, edges, bounds}
     */
    parseData(jsonContent) {
        try {
            const data = JSON.parse(jsonContent);
            
            if (!data.nodelist || !data.linklist) {
                throw new Error('数据格式错误：缺少nodelist或linklist字段');
            }

            const nodes = this.parseNodes(data.nodelist);
            const edges = this.parseEdges(data.linklist);
            const bounds = this.calculateBounds(nodes);

            console.log(`解析完成: ${nodes.length} 个节点, ${edges.length} 条连接`);
            console.log(`节点类型: ${Array.from(this.nodeTypes).join(', ')}`);
            console.log(`边类型: ${Array.from(this.edgeTypes).join(', ')}`);

            return {
                nodes,
                edges,
                bounds,
                nodeTypes: Array.from(this.nodeTypes),
                edgeTypes: Array.from(this.edgeTypes)
            };
        } catch (error) {
            console.error('解析数据时出错:', error);
            throw error;
        }
    }

    /**
     * 解析节点数据
     * @param {Array} nodelist - 原始节点列表
     * @returns {Array} 解析后的节点数组
     */
    parseNodes(nodelist) {
        const nodes = [];

        nodelist.forEach((node, index) => {
            try {
                // 解析parameter字段
                let parameter = {};
                if (node.parameter) {
                    if (typeof node.parameter === 'string') {
                        parameter = JSON.parse(node.parameter);
                    } else {
                        parameter = node.parameter;
                    }
                }

                // 提取位置信息
                let x = 0, y = 0;
                if (parameter.styles && parameter.styles.position) {
                    x = parameter.styles.position.x || 0;
                    y = parameter.styles.position.y || 0;
                }

                // 确定节点类型
                const type = parameter.type || 'Unknown';
                this.nodeTypes.add(type);

                // 创建节点对象
                const nodeObj = {
                    id: node.id,
                    name: node.name || `节点_${index}`,
                    type: type,
                    x: x,
                    y: y,
                    originalData: node,
                    parameter: parameter,
                    // 添加一些常用的参数便于快速访问
                    properties: this.extractNodeProperties(parameter)
                };

                nodes.push(nodeObj);
            } catch (error) {
                console.warn(`解析节点 ${node.id} 时出错:`, error);
                // 创建一个默认节点
                nodes.push({
                    id: node.id,
                    name: node.name || `节点_${index}`,
                    type: 'Error',
                    x: 0,
                    y: 0,
                    originalData: node,
                    parameter: {},
                    properties: {}
                });
            }
        });

        return nodes;
    }

    /**
     * 解析连接数据
     * @param {Array} linklist - 原始连接列表
     * @returns {Array} 解析后的连接数组
     */
    parseEdges(linklist) {
        const edges = [];

        linklist.forEach((link, index) => {
            try {
                // 解析parameter字段
                let parameter = {};
                if (link.parameter) {
                    if (typeof link.parameter === 'string') {
                        parameter = JSON.parse(link.parameter);
                    } else {
                        parameter = link.parameter;
                    }
                }

                // 确定连接类型
                const type = parameter.type || 'Unknown';
                this.edgeTypes.add(type);

                // 创建连接对象
                const edgeObj = {
                    id: link.id,
                    name: link.name || `连接_${index}`,
                    type: type,
                    source: link.sourceid,
                    target: link.targetid,
                    originalData: link,
                    parameter: parameter,
                    // 添加一些常用的参数便于快速访问
                    properties: this.extractEdgeProperties(parameter)
                };

                edges.push(edgeObj);
            } catch (error) {
                console.warn(`解析连接 ${link.id} 时出错:`, error);
                // 创建一个默认连接
                edges.push({
                    id: link.id,
                    name: link.name || `连接_${index}`,
                    type: 'Error',
                    source: link.sourceid,
                    target: link.targetid,
                    originalData: link,
                    parameter: {},
                    properties: {}
                });
            }
        });

        return edges;
    }

    /**
     * 提取节点的关键属性
     * @param {Object} parameter - 节点参数
     * @returns {Object} 关键属性对象
     */
    extractNodeProperties(parameter) {
        const properties = {};

        // 基本信息
        if (parameter.type) properties['类型'] = parameter.type;
        if (parameter.name) properties['名称'] = parameter.name;
        if (parameter.groupId) properties['组ID'] = parameter.groupId;

        // 参数信息
        if (parameter.parameter) {
            const params = parameter.parameter;
            
            // 常见的参数
            if (params.FluidPackage) properties['流体包'] = params.FluidPackage;
            if (params.Valve_Diameter) properties['阀门直径'] = `${params.Valve_Diameter} mm`;
            if (params.Valve_Opening) properties['阀门开度'] = params.Valve_Opening;
            if (params.Temperature) properties['温度'] = `${params.Temperature} °C`;
            if (params.Pressure) properties['压力'] = `${params.Pressure} Pa`;
            if (params.Mass_Flow) properties['质量流量'] = params.Mass_Flow;
        }

        // 位置信息
        if (parameter.styles && parameter.styles.position) {
            const pos = parameter.styles.position;
            properties['位置'] = `(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)})`;
        }

        return properties;
    }

    /**
     * 提取连接的关键属性
     * @param {Object} parameter - 连接参数
     * @returns {Object} 关键属性对象
     */
    extractEdgeProperties(parameter) {
        const properties = {};

        // 基本信息
        if (parameter.type) properties['类型'] = parameter.type;
        if (parameter.name) properties['名称'] = parameter.name;

        // 管道参数
        if (parameter.parameter) {
            const params = parameter.parameter;
            
            if (params.Length) properties['长度'] = `${params.Length} m`;
            if (params.Inner_Diameter) properties['内径'] = `${params.Inner_Diameter} mm`;
            if (params.Outer_Diameter) properties['外径'] = `${params.Outer_Diameter} mm`;
            if (params.Thickness) properties['壁厚'] = `${params.Thickness} mm`;
            if (params.Material) properties['材料'] = params.Material;
            if (params.Roughness) properties['粗糙度'] = params.Roughness;
            if (params.ElevationChange) properties['高程变化'] = `${params.ElevationChange} m`;
            if (params.PressureDrop) properties['压降'] = `${params.PressureDrop} Pa`;
            if (params.Velocity) properties['流速'] = `${params.Velocity} m/s`;
        }

        // 样式信息
        if (parameter.styles) {
            const styles = parameter.styles;
            if (styles.color) properties['颜色'] = styles.color;
            if (styles.width) properties['线宽'] = styles.width;
        }

        return properties;
    }

    /**
     * 计算节点的边界框
     * @param {Array} nodes - 节点数组
     * @returns {Object} 边界框 {minX, maxX, minY, maxY, width, height}
     */
    calculateBounds(nodes) {
        if (nodes.length === 0) {
            return { minX: 0, maxX: 100, minY: 0, maxY: 100, width: 100, height: 100 };
        }

        const xs = nodes.map(n => n.x);
        const ys = nodes.map(n => n.y);

        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);

        return {
            minX,
            maxX,
            minY,
            maxY,
            width: maxX - minX,
            height: maxY - minY
        };
    }

    /**
     * 获取节点类型的配置
     * @returns {Object} 节点类型配置
     */
    getNodeTypeConfig() {
        const configs = {
            'VavlePro': { 
                shape: 'diamond', 
                color: '#3498db', 
                size: 12,
                label: '阀门'
            },
            'Pipe': { 
                shape: 'rect', 
                color: '#2ecc71', 
                size: 10,
                label: '管道'
            },
            'Pump': { 
                shape: 'triangle', 
                color: '#e74c3c', 
                size: 14,
                label: '泵'
            },
            'Tank': { 
                shape: 'rect', 
                color: '#f39c12', 
                size: 16,
                label: '储罐'
            },
            'HeatExchanger': { 
                shape: 'hexagon', 
                color: '#9b59b6', 
                size: 14,
                label: '换热器'
            },
            'Compressor': { 
                shape: 'triangle', 
                color: '#e67e22', 
                size: 14,
                label: '压缩机'
            },
            'Separator': { 
                shape: 'circle', 
                color: '#1abc9c', 
                size: 12,
                label: '分离器'
            },
            'Unknown': { 
                shape: 'circle', 
                color: '#95a5a6', 
                size: 8,
                label: '未知'
            },
            'Error': { 
                shape: 'cross', 
                color: '#e74c3c', 
                size: 10,
                label: '错误'
            }
        };

        // 为动态发现的类型添加默认配置
        this.nodeTypes.forEach(type => {
            if (!configs[type]) {
                const colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#e67e22', '#1abc9c'];
                const shapes = ['circle', 'rect', 'diamond', 'triangle'];
                configs[type] = {
                    shape: shapes[type.length % shapes.length],
                    color: colors[type.length % colors.length],
                    size: 10,
                    label: type
                };
            }
        });

        return configs;
    }
}

// 导出供其他模块使用
window.DataParser = DataParser;