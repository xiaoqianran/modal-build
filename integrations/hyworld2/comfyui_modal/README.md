# ComfyUI_HYWorld2 on Modal

这个目录会把 `https://github.com/xiaoqianran/ComfyUI_HYWorld2` 自动安装到 ComfyUI，并在 Modal GPU 容器中启动 Web UI。

## 部署

```bash
cd /workspace/hyworld2-modal
modal deploy modal_app.py --stream-logs
```

默认 GPU 是 `H100`。需要换 GPU 时：

```bash
MODAL_GPU=L40S modal deploy modal_app.py --stream-logs
```

GPU 同时用于构建 HYWorld2 的 CUDA 原生扩展和运行 ComfyUI。更换 GPU 后应重新部署，让扩展针对新的 GPU 重新构建。

## 持久化

Modal Volume 名为 `comfyui-hyworld2-data`，用于持久化：

- `/data/models`
- `/data/input`
- `/data/output`
- `/data/user`
- `/data/huggingface`
- `/data/torch`
- `/data/cache`

因此 HYWorld2 / WorldStereo / Qwen 等节点首次下载模型后，后续容器可以复用。

## 停止

```bash
modal app stop comfyui-hyworld2
```
