import { app } from '../../scripts/app.js'

class Visualizer {
    constructor(node, nodeData, container, visualSrc) {
        let that = this
        this.node = node
        this.iframe = document.createElement('iframe')
        Object.assign(this.iframe, {
            scrolling: 'no',
            overflow: 'hidden'
        })
        this.iframe.src = '/extensions/' + nodeData.python_module.split('.')[1] + '/html/' + visualSrc + '.html'
        container.appendChild(this.iframe)
        // 监听 iframe 加载完成
        this.iframe.onload = () => {
            // 获取 iframe 中的 document 对象
            const iframeDocument = this.iframe.contentWindow.document;
            // 修改 iframe 中的 CSS 文件路径
            const links = iframeDocument.querySelectorAll('link[rel="stylesheet"]');
            links.forEach(link => {
                const oldHref = link.href;
                const newHref = this.adjustUrlPath(oldHref);
                link.href = newHref;
            });
            // 修改 iframe 中的 JS 文件路径
            const scripts = iframeDocument.querySelectorAll('script[src]');
            scripts.forEach(script => {
                const oldSrc = script.src;
                const newSrc = this.adjustUrlPath(oldSrc);
                script.src = newSrc;
            });
            const iframeDoc = that.iframe.contentDocument || that.iframe.contentWindow.document;
            const previewHtml = that.iframe.contentDocument.documentElement.innerHTML
            iframeDoc.open();
            iframeDoc.write(previewHtml);
            iframeDoc.close();
        };
        // 函数：根据需要调整路径
        this.adjustUrlPath = (url) => {
            return url.replace('comfyui_path', nodeData.python_module.split('.')[1]);
        };
    }

    updateVisual(params) {
        const iframeDocument = this.iframe.contentWindow.document
        const previewScript = iframeDocument.getElementById('visualizer')
        previewScript.setAttribute('filepath', params + Math.random())
    }

    remove() {
        this.container.remove()
    }
}

function createVisualizer(node, inputName, typeName, nodeData, app) {
    function checkLength() {
        if (node.widgets != undefined && node.widgets.length > 0) {
            clearInterval(interval);
            // 执行函数
            node.name = inputName
            const widget = {
                type: typeName,
                name: 'preview3d',
                callback: () => { },
                draw: function (ctx, node, widgetWidth, widgetY, widgetHeight) {
                    const margin = 10
                    const top_offset = 22
                    const top_label_offset = 8
                    const visible = app.canvas.ds.scale > 0.5 && this.type === typeName
                    const w = widgetWidth - margin * 4
                    const clientRectBound = ctx.canvas.getBoundingClientRect()
                    const transform = new DOMMatrix()
                        .scaleSelf(
                            clientRectBound.width / ctx.canvas.width,
                            clientRectBound.height / ctx.canvas.height
                        )
                        .multiplySelf(ctx.getTransform())
                        .translateSelf(margin, margin + widgetY)

                    Object.assign(this.visualizer.style, {
                        left: `${margin + transform.e}px`,
                        top: `${margin +
                            transform.f +
                            (node.widgets.length - 1) * top_label_offset +
                            top_offset
                            }px`,
                        width: `${w * transform.a}px`,
                        height: `${w * transform.d - widgetHeight - margin * 15 * transform.d
                            }px`,
                        position: 'absolute',
                        overflow: 'hidden',
                        zIndex: app.graph._nodes.indexOf(node)
                    })

                    Object.assign(this.visualizer.children[0].style, {
                        transformOrigin: '50% 50%',
                        width: '100%',
                        height: '100%',
                        border: '0 none'
                    })

                    this.visualizer.hidden = !visible
                }
            }

            const container = document.createElement('div')
            container.id = `Comfy3D_${inputName}`

            node.visualizer = new Visualizer(node, nodeData, container, typeName)
            widget.visualizer = container
            widget.parent = node

            document.body.appendChild(widget.visualizer)

            node.addCustomWidget(widget)

            node.updateParameters = params => {
                node.visualizer.updateVisual(params)
            }

            // Events for drawing backgound
            node.onDrawBackground = function (ctx) {
                if (!this.flags.collapsed) {
                    node.visualizer.iframe.hidden = false
                } else {
                    node.visualizer.iframe.hidden = true
                }
            }

            // Make sure visualization iframe is always inside the node when resize the node
            node.onResize = function () {
                let [w, h] = this.size
                if (w <= 600) w = 600
                if (h <= 500) h = 500

                if (w > 600) {
                    h = w - 100
                }

                this.size = [w, h]
            }

            // Events for remove nodes
            node.onRemoved = () => {
                for (let w in node.widgets) {
                    if (node.widgets[w].visualizer) {
                        node.widgets[w].visualizer.remove()
                    }
                }
            }

            return {
                widget: widget
            }

        } else {
            console.log('waiting for input', node)
        }
    }
    let interval = setInterval(checkLength, 500);
}

app.registerExtension({
    name: 'THREE_Visualizer',
    async init(app) { },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 如果存导入存在该控件则增加两个事件响应
        if (nodeData.display_name == 'EX_TriMeshViewer') {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = async function () {
                // 控件创建 增加控件 重复构件 再次运行
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
                let Preview3DNode = app.graph._nodes.filter(
                    wi => wi.type == 'EX_TriMeshViewer'
                )
                let nodeName = `Preview3DNode_${Preview3DNode.length}`
                const result = await createVisualizer.apply(this, [
                    this,
                    nodeName,
                    "threeVisualizer",
                    nodeData,
                    app
                ])
                this.setSize([600, 500])
                return r
            }
            nodeType.prototype.onExecuted = async function (message) {
                // 控件结束 运行查询
                this.updateParameters("running");
            }
        }
    }
})
