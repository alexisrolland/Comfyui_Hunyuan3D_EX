[English](README.md) | [简体中文](README_zh_CN.md)

# ComfyUI_Hunyuan3D_EX
该插件主要是做了一些代码上的优化：
1. 优化了混元3D的内部代码，完全删除了Pytorch3d相关代码，优化成Trimesh，这个库安装很简单。`注：删除了导出动图和贴图的功能，这部分我觉得对3D建模人员来说很简单`
2. 显示部分，考虑到ComfyUI-3D-Pack这种库非常难装，我参考ComfyUI-Flowty-TripoSR的三维显示节点优化了一版，应该解决了大部分问题，但是我只在我的环境中测试过，如果有问题请反馈。
3. 关于腾讯的混元3D的ComfyUI运行节点，我将ComfyUI_Huyuan3D作者的运行代码进行了拆分与优化。

#### 具体节点：

- **EX_Hunyuan3DNode**
1. **sixImages**：六视图图像，尺寸固定为1024*1536，RGB格式，腾讯混元使用的六视图位置有些不同，如果想用其他模型的六视图需要注意转换。
2. **originalImages**：原图图像，尺寸固定为512*512，RGB格式，对模型好像没有影响，查源码是必须要增加的，测试过程中不知道有什么用，无论图片如何好像都是一样的效果。
3. **mesh**: triMesh格式的3D模型。
4. **other**: 其他参数，都是字面意思，开启简化模型，最大面数才有用。

- **EX_GenerateSixViews**
1. **rgba**：一张RGBA格式的图片，尺寸最好是矩形的，否则会变形，尺寸最好都在1024以上。`Comfyui本身有一个图像遮罩到RGBA图片的功能，可以用这个功能生成RGBA图片。`
2. **sixImages**：六视图图像，输出，一张图片，尺寸固定为1024*1536，RGB格式。
3. **originalImages**：原图图像，输出，一张图片，尺寸固定为512*512，RGB格式。

- **EX_TriMeshViewer**
1. **mesh**: triMesh格式的3D模型。
2. **other**: 其他参数，都是字面意思，保存在output文件夹中。

- **EX_RemoveBackground**
一个很简单的背景去除节点，输入一张图片，输出去除背景后的图片，可以自定义模型，具体模型可依据Rembg作者提供的模型下载地址。

- **EX_SquareImage**
ComfyUI_Huyuan3D作者制作的节点，未调整。

## 参考资料
- [Tencent/Hunyuan3D-1](https://github.com/Tencent/Hunyuan3D-1) - A Unified Framework for Text-to-3D and Image-to-3D Generation
- [ComfyUI_Huyuan3D](https://github.com/TTPlanetPig/Comfyui_Hunyuan3D) - This is a custom node to support hunyuan3D in comfyui.
- [ComfyUI-Flowty-TripoSR](https://github.com/flowtyone/ComfyUI-Flowty-TripoSR) - This is a custom node that lets you use TripoSR right from ComfyUI.

