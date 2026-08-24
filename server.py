#!/usr/bin/env python3
"""Screen-state recognition web service using only Python's standard library."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import mimetypes
import os
import re
import sys
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = PROJECT_DIR / "public"
#DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
#API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_MODEL = "qwen3"
API_URL = "http://192.168.2.185:18082/v1/chat/completions"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 15 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_STATES = ("正常", "蓝屏", "黑屏", "贴纸")


SYSTEM_PROMPT = """你是一个严格的显示器屏幕检测器。分析用户上传的图片，只识别显示器或自助设备上的实际显示屏区域，不要识别机柜、按钮、键盘、插槽、指示灯、纸张本身或其他物体。

对每个可见显示屏判断且只能使用以下状态之一：
- 正常：屏幕正常显示业务界面、桌面、图像或其他有效内容。
- 蓝屏：屏幕整体呈现系统蓝屏或明显的蓝色故障画面。
- 黑屏：屏幕关闭、无显示内容或整体近似黑色。
- 贴纸：屏幕表面被纸张、贴纸或类似 A4 纸的物体覆盖或遮挡。

坐标必须是相对于整张原图的 0 到 1000 归一化坐标，格式为 [x1,y1,x2,y2]，左上角为原点。confidence 必须是 0 到 1 之间的数字。即使没有检测到显示屏，也必须返回 detections 空数组。

只输出一个 JSON 对象，不要输出 Markdown、解释或额外文字。结构必须为：
{"detections":[{"id":"screen-1","bbox_2d":[x1,y1,x2,y2],"label":"显示器屏","state":"正常","confidence":0.95}]}"""


class ServiceError(Exception):
    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = int(status)


def load_dotenv(path: Path) -> None:
    """Load a small, conventional .env file without overriding shell variables."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            os.environ.setdefault(key, value)


def parse_image_data(value: Any) -> tuple[str, bytes, str]:
    if not isinstance(value, str):
        raise ServiceError("缺少图片数据。")

    match = re.fullmatch(
        r"data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\r\n]+)", value
    )
    if not match:
        raise ServiceError("图片数据格式无效，请重新选择图片。")

    mime_type = match.group(1).lower()
    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise ServiceError("仅支持 JPG、PNG 或 WebP 图片。")

    try:
        raw = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ServiceError("图片 Base64 数据无效。") from exc

    if not raw:
        raise ServiceError("图片内容为空。")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ServiceError("图片不能超过 10 MB。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

    canonical_data_url = f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"
    return mime_type, raw, canonical_data_url


def call_model(image_data_url: str) -> tuple[Any, dict[str, Any]]:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        raise ServiceError(
            "服务端尚未配置 SILICONFLOW_API_KEY。",
            HTTPStatus.SERVICE_UNAVAILABLE,
        )

    model = os.environ.get("SILICONFLOW_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url, "detail": "high"},
                    },
                    {
                        "type": "text",
                        "text": "检测这张图片中的所有显示器屏幕，并严格按指定 JSON 结构返回。",
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False,
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            upstream = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_text = exc.read(4096).decode("utf-8", errors="replace")
        if exc.code in (401, 403):
            message = "模型服务鉴权失败，请检查服务端 API Key。"
        elif exc.code == 429:
            message = "模型服务请求过于频繁，请稍后重试。"
        elif exc.code in (503, 504):
            message = "模型服务暂时繁忙，请稍后重试。"
        else:
            message = f"模型服务调用失败（HTTP {exc.code}）。"
        print(f"SiliconFlow error {exc.code}: {error_text}", file=sys.stderr)
        raise ServiceError(message, HTTPStatus.BAD_GATEWAY) from exc
    except urllib.error.URLError as exc:
        print(f"SiliconFlow connection error: {exc}", file=sys.stderr)
        raise ServiceError("无法连接模型服务，请检查网络后重试。", HTTPStatus.BAD_GATEWAY) from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        print(f"SiliconFlow response error: {exc}", file=sys.stderr)
        raise ServiceError("模型服务响应超时或格式异常。", HTTPStatus.BAD_GATEWAY) from exc

    try:
        content = upstream["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ServiceError("模型响应缺少识别结果。", HTTPStatus.BAD_GATEWAY) from exc

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise ServiceError("模型返回了空识别结果。", HTTPStatus.BAD_GATEWAY)

    parsed = parse_model_json(content)
    meta = {
        "model": upstream.get("model", model),
        "usage": upstream.get("usage", {}),
    }
    return parsed, meta


def parse_model_json(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ServiceError("模型未返回有效 JSON。", HTTPStatus.BAD_GATEWAY)


def normalize_state(value: Any) -> str | None:
    text = str(value or "").strip()
    if text in ALLOWED_STATES:
        return text
    aliases = (
        ("贴", "贴纸"),
        ("遮挡", "贴纸"),
        ("纸", "贴纸"),
        ("蓝屏", "蓝屏"),
        ("黑屏", "黑屏"),
        ("关闭", "黑屏"),
        ("无显示", "黑屏"),
        ("正常", "正常"),
    )
    for keyword, normalized in aliases:
        if keyword in text:
            return normalized
    return None


def normalize_detections(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        items = value.get("detections", [])
    elif isinstance(value, list):
        items = value
    else:
        items = []
    if not isinstance(items, list):
        raise ServiceError("模型结果中的 detections 不是数组。", HTTPStatus.BAD_GATEWAY)

    normalized: list[dict[str, Any]] = []
    seen_boxes: set[tuple[float, float, float, float]] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_2d", item.get("bbox"))
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            coords = [float(number) for number in bbox[:4]]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(number) for number in coords):
            continue

        x1, y1, x2, y2 = coords
        left, right = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        top, bottom = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        if right - left < 1 or bottom - top < 1:
            continue

        state = normalize_state(item.get("state", item.get("status")))
        if state is None:
            continue

        try:
            confidence = float(item.get("confidence", item.get("score", 0)))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence > 1 and confidence <= 100:
            confidence /= 100
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        clean_box = tuple(round(number, 2) for number in (left, top, right, bottom))
        if clean_box in seen_boxes:
            continue
        seen_boxes.add(clean_box)

        normalized.append(
            {
                "id": f"screen-{len(normalized) + 1}",
                "bbox_2d": list(clean_box),
                "label": "显示器屏",
                "state": state,
                "confidence": confidence,
            }
        )

    return normalized


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ScreenRecognition/1.0"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "model": os.environ.get("SILICONFLOW_MODEL", DEFAULT_MODEL),
                    "apiKeyConfigured": bool(os.environ.get("SILICONFLOW_API_KEY", "").strip()),
                }
            )
            return

        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (PUBLIC_DIR / relative).resolve()
        try:
            target.relative_to(PUBLIC_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/recognize":
            self.send_json({"error": "接口不存在。"}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ServiceError("请求内容为空。")
            if content_length > MAX_REQUEST_BYTES:
                raise ServiceError("请求不能超过 15 MB。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            if "application/json" not in self.headers.get("Content-Type", ""):
                raise ServiceError("请求必须使用 application/json。")

            try:
                body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ServiceError("请求 JSON 无效。") from exc
            if not isinstance(body, dict):
                raise ServiceError("请求 JSON 必须是对象。")

            _, _, image_data_url = parse_image_data(body.get("image"))
            model_result, meta = call_model(image_data_url)
            detections = normalize_detections(model_result)
            self.send_json({"detections": detections, "meta": meta})
        except ServiceError as exc:
            self.send_json({"error": str(exc)}, exc.status)
        except Exception as exc:  # Keep unexpected internals out of the browser.
            print(f"Unexpected server error: {exc!r}", file=sys.stderr)
            self.send_json({"error": "服务端发生未预期错误。"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="显示器屏幕状态识别服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Screen recognition service: http://{args.host}:{args.port}")
    print(f"Model: {os.environ.get('SILICONFLOW_MODEL', DEFAULT_MODEL)}")
    print(
        "API key: configured" if os.environ.get("SILICONFLOW_API_KEY", "").strip() else "API key: NOT configured"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping service.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
