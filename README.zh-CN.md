# IS3D - 偶像高斯点云 3D 生成器（Idol Splat 3D Generator）

[English](README.md) | **简体中文**

把**一张参考图**变成高斯点云、带纹理的高模，或自动展 UV 并烘焙贴图的低模 ——
全部在显卡上完成，直接用 NVIDIA 驱动推理。

不需要 PyTorch，不需要 ONNX Runtime，不需要 CUDA Toolkit，不需要 `pip install`
任何东西。推理引擎是一个 1.2 MB 的 DLL，只调用系统里已有的 `nvcuda.dll`，加载时
把内嵌的 PTX 即时编译成当前显卡的机器码。

```
一张参考图
  -> BiRefNet 抠图（可选）
  -> DINOv3 ViT-H/16+ 图像 token + FLUX.2 VAE latent
  -> 50 步 rectified flow 采样（fp16，CFG）
  -> 八叉树锚点 -> 每个锚点 32 个高斯点
  -> 高斯 PLY / 高模 / 低模 + 烘焙贴图
  -> 追加到 Blender 场景的 3D 游标处
```

## 特性

- **一张图，三种产物**：`高斯点云`（GPUSplat 点云）、`高模`（高斯颜色迁移的稠密
  网格）、`低模`（自动展 UV 并烘焙基础色 + 法线贴图）。
- **内置抠图**：BiRefNet Swin-L + 可变形 ASPP 已移植进原生引擎，由「抠除背景」
  复选框控制。alpha 与 PyTorch 参考实现余弦相似度 0.99997。
- **显存占用低**：开了抠图峰值 1.99 GiB，不开 1.59 GiB。每个阶段结束立刻把权重和
  中间张量从显存卸载，3.5 GB 的权重文件不会整块驻留显存。
- **没有 Python 依赖栈**：不用 pip，不用 CUDA Toolkit，不用装运行库。机器上只要
  有 NVIDIA 驱动就能跑。
- **界面不卡**：生成跑在独立的 headless Blender 子进程里，视图仍然可操作，随时
  可以取消。
- **面板里的资产库**：每次生成都会留一张 384 px 缩略图，可以直接拖回任意场景；
  资产可以自己命名，留空则按编号命名（`is3d-0001`、`is3d-0002`……）。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 系统 | Windows 10/11 x64 |
| 显卡 | NVIDIA RTX 30/40/50 系列（计算能力 >= 8.6） |
| 驱动 | 能 JIT 运行 PTX `compute_86` 的驱动（2021 年之后） |
| Blender | 5.3 或更新（Steam 版可用） |
| 显存 | 实测约 2 GB，建议 4 GB，硬上限 6 GB |
| 磁盘 | 权重约 4 GB |

## 安装

```powershell
git clone https://github.com/MARK42IRPC/Idol-Splat-3dGen_BlenderScript
cd Idol-Splat-3dGen_BlenderScript
powershell -ExecutionPolicy Bypass -File install/install.ps1
```

脚本会自动找到 Blender（Steam 库和官方安装包都支持），把插件装进
`scripts/addons/is3d`，从 GitHub Release 下载引擎 DLL 和权重分片，逐个校验
SHA-256，再拼回 `idol_weights.bin`。之后在 `编辑 > 偏好设置 > 插件` 里搜索
**IS3D** 启用。

| 参数 | 作用 |
| --- | --- |
| `-AssetBaseUrl <url>` | 换一个下载地址（内网共享、自建镜像） |
| `-BlenderRoot <dir>` | 指定某个 Blender 安装目录 |
| `-AddonsDir <dir>` | 直接写到指定的 `scripts/addons` 目录 |
| `-Version 0.4.0` | 安装其它版本 |
| `-SkipAssets` | 只装代码（`is3d/native/` 已有资源时用） |
| `-Source` | 从当前仓库拷贝 `is3d/`，不下载发布 zip |
| `-FromZip <path>` | 安装手头已有的插件 zip |
| `-KeepParts` | 保留 `%TEMP%` 里的权重分片 |
| `-Force` | 强制重新下载并覆盖 |

## 使用

在 3D 视图按 `N` 打开侧栏，切到 **IS3D** 标签页。

| 控件 | 说明 |
| --- | --- |
| 参考图 | 输入图片 |
| 抠除背景 | 先用内置 BiRefNet 抠图；如果图片本来就是干净剪影可以关掉（省约 0.4 GiB 显存和 1 秒） |
| 产出 | `高斯点云`、`高模`（顶点色）、`低模`（UV + 烘焙贴图） |
| 高斯点数量 / 三角面数 | 输出预算 |
| 输出目录 | 任务目录、PLY、报告的落盘位置 |

生成结果会作为 `IS3D 生成资产` 集合追加到场景中的 3D 游标处，可以整体移动或删除。

## 原理

阶段划分、算子清单和显存策略见 `docs/ARCHITECTURE.md`。简单说：

1. **预处理**：缩放、可选 BiRefNet 抠图、主体包围盒裁切。
2. **编码**：DINOv3 ViT-H/16+ 出图像 token，FLUX.2 VAE 出 latent。
3. **去噪**：50 步 rectified flow 采样，fp16，带 CFG。
4. **锚点 / 高斯**：在预测出的八叉树上布锚点，每个锚点展开 32 个高斯点。
5. **导出**：高斯 PLY + 相机 JSON；高模/低模在 Blender 侧用同一套高斯颜色重建。

引擎全程只保留一个 CUDA 上下文，每个阶段一块显存池，阶段结束立即卸载权重。

## 模型文件

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `idol_engine.dll` | 1.2 MB | 原生推理引擎（内嵌 PTX） |
| `idol_weights.bin` | 3.30 GiB | 全部阶段权重，`IDOLW1` 容器，2349 个张量 |

GitHub 发布附件单个上限 2 GiB，所以权重按 `idol_weights.bin.partNN` 切片发布，
安装脚本负责拼接和再校验。格式与下载顺序见 `docs/ASSETS.md`。

仓库里不放权重。这份权重是对下面几个上游模型的**纯推理重打包**，不想下发布包的
话可以自己用 `native/engine/pack_weights.py` 重新生成：

| 上游模型 | 在流程中的角色 |
| --- | --- |
| [speridlabs/idol-sdk](https://github.com/speridlabs/idol-sdk) | 图像 -> latent -> 高斯锚点，splat 解码 |
| [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2) | FLUX.2 klein VAE 编码器 |
| [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | DINOv3 ViT-H/16+ 条件特征 |
| [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) | 抠图 |

## 从源码构建

`docs/BUILD.md` 写了 MSVC + CUDA 12.6 工具链、用 NVRTC 生成 PTX（不需要
`nvcc`）、如何把上游 checkpoint 打包成 `idol_weights.bin`，以及怎么生成发布产物。

## 验证

`validation/` 是开发期间用的校验脚本：

- `check_birefnet.py`：原生 BiRefNet 与 PyTorch 参考逐层比对（alpha 余弦
  0.99997，最差层 0.975）。
- `run_matte.py`：引擎端到端抠图（与参考 IoU 0.9887）。
- `blender_package_check.py`：在 headless Blender 里启用打包插件、生成资产、导入并
  渲染两个视角，断言覆盖率与颜色方差。

## 许可

代码 MIT（见 LICENSE）。模型权重见 NOTICE.md，遵循上游模型各自的许可，仅供研究
与个人使用。
