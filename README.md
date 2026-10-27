# 显示器屏幕状态识别

上传图片后，Python 后端调用 SiliconFlow 多模态模型，仅检测显示器屏幕，并返回带唯一 ID 的标准 JSON。前端会自动绘制识别框、显示状态与置信度，并支持下载标注图片。

## 支持状态

- `正常`
- `蓝屏`
- `黑屏`
- `贴纸`：屏幕上存在纸张、贴纸或类似 A4 纸的遮挡物

## 返回格式

```json
{
  "detections": [
    {
      "id": "screen-1",
      "bbox_2d": [0, 0, 1000, 1000],
      "label": "显示器屏",
      "state": "正常",
      "confidence": 0.95
    }
  ]
}
```

`bbox_2d` 使用 Qwen 0～1000 归一化坐标。后端会过滤无效目标、规范状态并重新生成 `screen-1`、`screen-2` 等唯一 ID。

## 启动

视频事件分析依赖 OpenCV、NumPy 和 Requests。优先使用项目内已有的 `.conda` 环境，或在其他虚拟环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

建议先在 SiliconFlow 控制台作废项目资料中曾经明文保存的旧密钥，并创建新密钥。

```bash
cd /Volumes/code/projects/screen-recognition
cp .env.example .env
```

编辑 `.env`，为当前模型配置对应的 API Key 设置环境变量（例如
`SILICONFLOW_API_KEY` 或 `HUGGING_FACE_API_KEY`），然后启动：

```bash
python3 server.py
```

浏览器访问：<http://127.0.0.1:8000>

也可以使用环境变量启动，不创建 `.env`：

```bash
SILICONFLOW_API_KEY='your_new_api_key_here' python3 server.py
```

若要允许局域网其他设备访问：

```bash
python3 server.py --host 0.0.0.0 --port 8000
```

## 接口

### 健康检查

```text
GET /api/health
```

### 图片识别

```text
POST /api/recognize
Content-Type: application/json
```

请求体：

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

支持 JPG、PNG、WebP，图片上限为 32 MB。API Key 仅由后端读取，不会发送到浏览器。

图片与视频的模型请求超时时间均为 1 小时，图片模式浏览器请求超时也为 1 小时。

### 视频事件分析

页面中的“视频事件分析”页签会上传视频，并在独立后台进程中复用
`vlm_video_demo.py` 的时间轴抽帧和滑动窗口分析。前端展示每个窗口的
`summary` 与实际输入图片；视频播放结束不会停止任务，点击“重置”才会终止任务并清理临时文件。
