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
CONFIG_PATH = PUBLIC_DIR / "config.json"

# 视频模式以原始分辨率提交无损 PNG，单帧可能显著大于常规图片上传。
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 45 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_STATES = ("正常", "蓝屏", "黑屏", "贴纸")


def load_config() -> dict[str, Any]:
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"配置文件读取失败: {CONFIG_PATH}") from exc
    if not isinstance(config, dict):
        raise RuntimeError("配置文件必须是 JSON 对象。")
    return config


def get_model_config(config: dict[str, Any], profile_name: str | None = None) -> dict[str, Any]:
    usage = str(profile_name or config.get("model_usage", "")).strip()
    profiles = config.get("model_profiles")
    if not usage:
        raise RuntimeError("配置文件中的 model_usage 未设置。")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(usage), dict):
        raise RuntimeError(f"配置文件中不存在模型配置: {usage}")
    model_config = profiles[usage]
    for key in ("model", "chat_endpoint"):
        if not str(model_config.get(key, "")).strip():
            raise RuntimeError(f"模型配置缺少 {key}: {usage}")
    max_pixels = model_config.get("max_pixels")
    if max_pixels is not None and (
        isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0
    ):
        raise RuntimeError(f"模型配置 max_pixels 必须为正整数: {usage}")
    return model_config


def get_public_model_profiles(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return selectable model profiles without exposing credentials or endpoints."""
    profiles = config.get("model_profiles")
    if not isinstance(profiles, dict):
        raise RuntimeError("配置文件中的 model_profiles 必须是对象。")

    available_profiles: list[dict[str, Any]] = []
    for profile_name in profiles:
        try:
            model_config = get_model_config(config, profile_name)
        except RuntimeError:
            continue
        api_key_env, api_key = get_model_api_key(model_config)
        available_profiles.append(
            {
                "id": profile_name,
                "model": model_config["model"],
                "provider": str(model_config.get("provider", "")).strip(),
                "requiresApiKey": bool(api_key_env),
                "apiKeyConfigured": bool(api_key) if api_key_env else True,
            }
        )
    return available_profiles


ACTIVE_CONFIG: dict[str, Any] | None = None


# SYSTEM_PROMPT = """你是一个严格的显示器屏幕检测器。分析用户上传的图片，只识别显示器或自助设备上的实际显示屏区域，不要识别机柜、按钮、键盘、插槽、指示灯、纸张本身或其他物体。

# 对每个可见显示屏判断且只能使用以下状态之一：
# - 正常：屏幕正常显示业务界面、桌面、图像或其他有效内容。
# - 蓝屏：屏幕整体呈现系统蓝屏或明显的蓝色故障画面。
# - 黑屏：屏幕关闭、无显示内容或整体近似黑色。
# - 贴纸：屏幕表面被纸张、贴纸或类似 A4 纸的物体覆盖或遮挡。

# 坐标必须是相对于整张原图的 0 到 1000 归一化坐标，格式为 [x1,y1,x2,y2]，左上角为原点。confidence 必须是 0 到 1 之间的数字。即使没有检测到显示屏，也必须返回 detections 空数组。

# 只输出一个 JSON 对象，不要输出 Markdown、解释或额外文字。结构必须为：
# {"detections":[{"id":"screen-1","bbox_2d":[x1,y1,x2,y2],"label":"显示器屏","state":"正常","confidence":0.95}]}"""


# SYSTEM_PROMPT = """你是一个严格的显示屏区域检测器。

# 分析用户上传的图片，只检测摄像头能够实际看到“正面显示区域”的显示器、自助设备屏幕或其他电子显示屏。

# 【检测目标定义】
# 只有当图片中能够看到显示器的正面屏幕表面（显示面、屏幕玻璃、液晶面板所在区域）时，才允许将其识别为显示屏。

# 以下情况必须排除，不得输出为 detection：
# - 显示器的背面、后壳或背板；
# - 屏幕显示面背对摄像头，摄像头只能看到显示器背壳；
# - 显示器支架、底座、边框、机柜；
# - 键盘、按钮、插槽、指示灯；
# - 普通纸张、桌面、墙面或其他矩形物体；
# - 无法确认看到的是屏幕正面显示区域的疑似目标。

# 特别注意：
# 显示器背面通常也是黑色或深色矩形，但这不属于“黑屏”。
# 只有明确能够看到屏幕正面显示面，并且该显示面没有任何有效显示内容、整体近似黑色时，才允许判断为“黑屏”。
# 如果摄像头只能看到显示器背面，则直接忽略该显示器，不得输出。

# 【状态判断】
# 对每一个确认可见的“正面屏幕显示区域”，且只能使用以下状态之一：

# - 正常：
#   屏幕正常显示业务界面、桌面、图像、文字或其他有效内容。

# - 蓝屏：
#   屏幕正面显示区域整体呈现系统蓝屏或明显的蓝色故障画面。

# - 黑屏：
#   必须已经确认看到的是屏幕正面显示区域，并且该区域关闭、无显示内容或整体近似黑色。
#   不得因为显示器背壳、黑色机柜或其他黑色矩形物体而判断为黑屏。

# - 贴纸：
#   屏幕正面显示区域被纸张、贴纸或类似 A4 纸的物体覆盖或遮挡。
#   必须能够判断纸张位于屏幕显示区域表面，而不是附近普通纸张。

# 【检测原则】
# 1. 先判断目标是否为“摄像头可见的屏幕正面显示区域”。
# 2. 只有满足第 1 条后，才能继续判断正常、蓝屏、黑屏或贴纸。
# 3. 如果只能看到显示器背面或侧面，直接忽略。
# 4. 如果无法可靠区分屏幕正面与显示器背面，宁可不检测，不要猜测。
# 5. bbox 只框选实际屏幕显示区域，尽量不要包含显示器背壳、支架、底座或周围设备。

# 【输出格式】
# 坐标必须是相对于整张原图的 0 到 1000 归一化坐标，
# 格式为 [x1,y1,x2,y2]，左上角为原点。

# confidence 必须是 0 到 1 之间的数字。

# 即使没有检测到符合条件的显示屏，也必须返回 detections 空数组。

# 只输出一个 JSON 对象，不要输出 Markdown、解释或额外文字。

# 结构必须为：

# {"detections":[{"id":"screen-1","bbox_2d":[x1,y1,x2,y2],"label":"显示器屏","state":"正常","confidence":0.95}]}"""


# LEGACY_SYSTEM_PROMPT = """你是一个严格的“屏幕正面显示区域检测器”，不是显示器设备检测器。

# 你的任务是检测摄像头能够看到的电子屏幕正面显示区域，并判断显示状态。
# 只看到显示器设备、显示器背面、后壳或侧面，不属于有效检测目标。

# 【最高优先级资格门槛】

# 每一个输出到 detections 的目标必须同时满足以下全部条件：

# 1. 目标确实是电子显示设备的显示区域；
# 2. 摄像头能够看到它的正面显示面；
# 3. 可见区域中不存在明确的显示器背面证据；
# 4. bbox 对应的是正面显示区域，而不是整个设备、后壳、边框、支架或底座；
# 5. 能够可靠判断其显示状态。

# 以上任意一项不满足，必须排除，不得输出。

# 可表示为：

# eligible =
#     is_electronic_display
#     AND front_display_surface_visible
#     AND no_backside_evidence
#     AND state_is_judgable

# 只有 eligible 为 true 的目标才允许进入 detections。

# 【背面否决规则】

# 只要候选目标出现以下任意明确背面特征，直接判定为显示器背面并排除：

# - 可见面上存在资产标签、序列号标签、设备铭牌或背部标签；
# - 可见面中部存在显示器支架连接结构或 VESA 安装结构；
# - 可见接口、插孔、线缆连接处或散热孔；
# - 可见后盖凸起、背板拼接、后壳控制结构；
# - 显示器屏幕实际朝向工作人员，摄像头看到的是相反一面；
# - 只能看到显示器背面或侧面，无法看到正面显示玻璃。
# - 高优先级：疑似黑屏，但其正面缺少玻璃材质的证明。

# 显示器后壳即使整体为黑色，也绝对不属于“黑屏”。

# 一个黑色矩形只有在已经确认它是屏幕正面显示面之后，才可以继续判断是否为黑屏。

# 任何明确的背面证据都具有否决权：
# 即使该物体外形像显示器、整体呈黑色、位于桌面上，也必须排除。


# 【正面显示区域的有效证据】

# 以下特征可以支持目标为正面显示区域：

# - 能看到业务界面、桌面、图像、文字、图标或其他显示像素；
# - 能看到连续的屏幕玻璃或液晶面板；
# - 黑色玻璃面位于明确的正面屏幕边框内部；
# - 自助设备的触摸显示面朝向使用者，并且摄像头能看到该显示面；
# - 显示平面朝向摄像头或以一定斜角朝向摄像头，状态仍可判断。

# 斜着放置的屏幕可以检测，但只能看到背面或侧面的设备不得检测。

# 【状态判断顺序】

# 必须严格按照以下顺序处理每个候选目标：

# 第一步：判断它是否为电子显示设备；
# 第二步：判断摄像头看到的是正面、背面、侧面还是无法判断；
# 第三步：只有明确为正面时，才判断显示状态；
# 第四步：输出前再次检查是否存在背面证据。

# 禁止跳过正反面判断，直接根据颜色判断状态。

# 【状态定义】

# 每个有效正面显示区域只能使用以下状态之一：

# - 正常：
#   屏幕显示业务界面、桌面、文字、图标、图像、视频或其他有效内容。
#   暗色界面或暗色视频仍属于正常，不能仅因整体较暗而判断为黑屏。

# - 蓝屏：
#   屏幕正面显示区域整体呈现系统蓝屏或明显的蓝色故障画面。

# - 黑屏：
#   必须已经明确确认看到的是屏幕正面显示面；
#   该显示面没有有效文字、图像、图标或界面；
#   显示区域整体近似均匀黑色或处于关闭状态。
#   黑色后壳、黑色机柜、黑色桌面和普通黑色矩形均不得判断为黑屏。

# - 贴纸：
#   屏幕正面显示玻璃被纸张、贴纸或类似 A4 纸的物体覆盖。
#   小型资产标签、设备铭牌或背部标签不属于贴纸状态。

# 如果无法可靠区分正面与背面，或者无法可靠判断状态，直接排除，不得默认判断为黑屏。

# 【反例】

# 反例 1：
# 黑色矩形表面带有资产标签，中央可见支架连接结构，线缆从后方引出。
# 结论：显示器背面，必须排除，不得输出为黑屏。

# 反例 2：
# 黑色矩形位于工作人员与摄像头之间，工作人员位于其另一侧并面向显示器。
# 同时可见后壳或支架结构。
# 结论：显示器背面，必须排除。

# 【正例】

# 黑色玻璃面位于自助设备正面边框内部，显示面朝向使用者，摄像头能够看到该显示面，且没有任何显示内容。
# 结论：可以输出，状态为黑屏。

# 【bbox 规则】

# bbox 必须只框选实际正面显示区域：



# 坐标必须是相对于整张原图的 0 到 1000 归一化坐标：
# [x1,y1,x2,y2]，左上角为原点。

# 【confidence 规则】

# confidence 表示以下三项的联合可信程度：

# - 目标确实是显示屏；
# - 正面显示区域确实可见；
# - 状态判断正确。

# 【最终输出前强制自检】

# 生成 JSON 前，对每个 detection 依次检查：

# 1. 是否明确看到正面显示面？
# 2. 是否存在资产标签、后壳、支架连接、接口、散热孔或其他背面证据？
# 3. bbox 是否包含了后壳、支架或整个设备？
# 4. 是否只是因为目标呈黑色而判断为黑屏？
# 5. 是否受到图片中已有检测框、状态文字或 screen 编号影响？

# 如果第 1 项为否，或者第 2、3、4、5 项任一为是，必须删除该 detection。

# 删除无效目标后，再按照从左到右、从上到下的顺序重新分配 id：
# screen-1、screen-2、screen-3……

# label 必须固定为“显示器屏”，不得输出其他 label 值。

# 即使没有检测到符合条件的显示屏，也必须返回 detections 空数组。

# 只输出一个 JSON 对象，不要输出 Markdown、解释、分析过程或额外文字。

# 输出结构必须严格为：

# {"detections":[{"id":"screen-1","bbox_2d":[x1,y1,x2,y2],"label":"显示器屏","state":"正常","confidence":0.95}]}"""


# SYSTEM_PROMPT = """你是“屏幕正面显示区域检测器”。只检测摄像头实际看见的电子屏幕正面显示区域，不检测整个设备、边框、底座或机柜。

# 【视觉证据优先】
# 仅根据图片中可见的像素、结构和材质判断；柜台布局、工作人员位置或设备通常的摆放方向只能辅助判断，不能单独作为正面或背面的结论。

# 输出一个目标前，先判断可见面：
# - 正面证据：可见界面、文字、图标或图像；可见屏幕玻璃/液晶面板；或黑色/暗色面位于明确的正面边框内（自助设备的嵌入式触摸屏也属于此类）。
# - 背面证据：可见资产标签、铭牌、接口、线缆连接、散热孔、VESA/支架连接、后盖拼接或后壳控制结构。
# - 有任一明确背面证据即排除。没有明确背面证据且具有正面证据时，可作为正面屏幕；不要因空间关系猜测为背面。
# - 正反面都无法从图中判断时排除。

# 只为已确认的正面屏幕判断状态：
# - 正常：可见任何有效界面、文字、图标、图像或视频；暗色界面仍是正常。
# - 蓝屏：可见明显蓝色系统故障画面。
# - 黑屏：正面屏幕无有效显示内容，且显示面整体近似均匀黑/暗。黑色后壳、桌面和普通矩形绝不是黑屏。
# - 贴纸：纸张或贴纸覆盖在正面显示区域上；背部小标签不属于贴纸。
# 状态无法判断时排除。

# bbox 只能框实际显示区域，使用原图 0–1000 归一化 [x1,y1,x2,y2] 坐标。confidence 是“正面 + 状态”联合置信度；任一项有不确定性时不得超过 0.75。删除无效目标后按从上到下、同一行从左到右重新编号 screen-1、screen-2……；label 固定为“显示器屏”。

# 只输出一个 JSON 对象，不要输出解释：
# {"detections":[{"id":"screen-1","bbox_2d":[x1,y1,x2,y2],"label":"显示器屏","state":"正常","confidence":0.95}]}"""



# SYSTEM_PROMPT = """你是一个严格的显示器屏幕检测器。分析用户上传的图片，只识别显示器或自助设备上的实际显示屏区域，不要识别机柜、按钮、键盘、插槽、指示灯、纸张本身或其他物体。

# 对每个可见显示屏判断且只能使用以下状态之一：
# - 正常：屏幕正常显示业务界面、桌面、图像或其他有效内容。
# - 蓝屏：屏幕整体呈现系统蓝屏或明显的蓝色故障画面。
# - 黑屏：屏幕关闭、无显示内容或整体近似黑色。
# - 贴纸：屏幕表面被纸张、贴纸或类似 A4 纸的物体覆盖或遮挡。

# 坐标必须是相对于整张原图的 0 到 1000 归一化坐标，格式为 [x1,y1,x2,y2]，左上角为原点。confidence 必须是 0 到 1 之间的数字。即使没有检测到显示屏，也必须返回 detections 空数组。

# 只输出一个 JSON 对象，不要输出 Markdown、解释或额外文字。结构必须为：
# {"detections":[{"id":"screen-1","bbox_2d":[x1,y1,x2,y2],"label":"显示器屏","state":"正常","confidence":0.95}]}"""


# SYSTEM_PROMPT = """你是一个严格的显示屏区域检测器。

# 分析用户上传的图片，只检测摄像头能够实际看到“正面显示区域”的显示器、自助设备屏幕或其他电子显示屏。

# 【检测目标定义】
# 只有当图片中能够看到显示器的正面屏幕表面（显示面、屏幕玻璃、液晶面板所在区域）时，才允许将其识别为显示屏。

# 以下情况必须排除，不得输出为 detection：
# - 显示器的背面、后壳或背板；
# - 屏幕显示面背对摄像头，摄像头只能看到显示器背壳；
# - 显示器支架、底座、边框、机柜；
# - 键盘、按钮、插槽、指示灯；
# - 普通纸张、桌面、墙面或其他矩形物体；
# - 无法确认看到的是屏幕正面显示区域的疑似目标。

# 特别注意：
# 显示器背面通常也是黑色或深色矩形，但这不属于“黑屏”。
# 只有明确能够看到屏幕正面显示面，并且该显示面没有任何有效显示内容、整体近似黑色时，才允许判断为“黑屏”。
# 如果摄像头只能看到显示器背面，则直接忽略该显示器，不得输出。

# 【状态判断】
# 对每一个确认可见的“正面屏幕显示区域”，且只能使用以下状态之一：

# - 正常：
#   屏幕正常显示业务界面、桌面、图像、文字或其他有效内容。

# - 蓝屏：
#   屏幕正面显示区域整体呈现系统蓝屏或明显的蓝色故障画面。

# - 黑屏：
#   必须已经确认看到的是屏幕正面显示区域，并且该区域关闭、无显示内容或整体近似黑色。
#   不得因为显示器背壳、黑色机柜或其他黑色矩形物体而判断为黑屏。

# - 贴纸：
#   屏幕正面显示区域被纸张、贴纸或类似 A4 纸的物体覆盖或遮挡。
#   必须能够判断纸张位于屏幕显示区域表面，而不是附近普通纸张。

# 【检测原则】
# 1. 先判断目标是否为“摄像头可见的屏幕正面显示区域”。
# 2. 只有满足第 1 条后，才能继续判断正常、蓝屏、黑屏或贴纸。
# 3. 如果只能看到显示器背面或侧面，直接忽略。
# 4. 如果无法可靠区分屏幕正面与显示器背面，宁可不检测，不要猜测。
# 5. bbox 只框选实际屏幕显示区域，尽量不要包含显示器背壳、支架、底座或周围设备。

# 【输出格式】
# 坐标必须是相对于整张原图的 0 到 1000 归一化坐标，
# 格式为 [x1,y1,x2,y2]，左上角为原点。

# confidence 必须是 0 到 1 之间的数字。

# 即使没有检测到符合条件的显示屏，也必须返回 detections 空数组。

# 只输出一个 JSON 对象，不要输出 Markdown、解释或额外文字。

# 结构必须为：

# {"detections":[{"id":"screen-1","bbox_2d":[x1,y1,x2,y2],"label":"显示器屏","state":"正常","confidence":0.95}]}"""

#该图片中存在疑似蓝屏状态的显示器，如果你也找到疑似蓝屏状态的显示器，则判定该显示器为蓝屏状态.
#该图片中存在疑似朝背面的显示器，如果你也找到疑似朝背面的显示器，则将其排除.
#该图片中存在疑似被非贴纸物体挡住的显示器，如果该显示器导致你对状态识别困难，则将其排除.

SYSTEM_PROMPT = """你是一个严格的“屏幕正面显示区域检测器”，不是显示器设备检测器。

你的任务是检测摄像头能够看到的电子屏幕正面显示区域，并判断显示状态。 只看到显示器设备、显示器背面、后壳或侧面，不属于有效检测目标。

【最高优先级资格门槛】

每一个输出到 detections 的目标必须同时满足以下全部条件：

目标确实是电子显示设备的显示区域；
摄像头能够看到它的正面显示面；
可见区域中不存在明确的显示器背面证据；
bbox 对应的是正面显示区域，而不是整个设备、后壳、边框、支架或底座；
能够可靠判断其显示状态。
以上任意一项不满足，必须排除，不得输出。

可表示为：

eligible = is_electronic_display AND front_display_surface_visible AND no_backside_evidence AND state_is_judgable

只有 eligible 为 true 的目标才允许进入 detections。

【背面否决规则】

只要候选目标出现以下任意明确背面特征，直接判定为显示器背面并排除：
可见面上存在资产标签、序列号标签、设备铭牌或背部标签；
可见面中部存在显示器支架连接结构或 VESA 安装结构；
可见接口、插孔、线缆连接处或散热孔；
可见后盖凸起、背板拼接、后壳控制结构；
只能看到显示器背面或侧面，无法看到正面显示玻璃。
显示器后壳即使整体为黑色，也绝对不属于“黑屏”。

一个黑色矩形只有在已经确认它是屏幕正面显示面之后，才可以继续判断是否为黑屏。

任何明确的背面证据都具有否决权： 即使该物体外形像显示器、整体呈黑色、位于桌面上，也必须排除。

【正面显示区域的有效证据】

以下特征可以支持目标为正面显示区域：

能看到业务界面、桌面、图像、文字、图标或其他显示像素；
能看到连续的屏幕玻璃或液晶面板；
黑色玻璃面位于明确的正面屏幕边框内部；
自助设备的触摸显示面朝向使用者，并且摄像头能看到该显示面；
显示平面朝向摄像头或以一定斜角朝向摄像头，状态仍可判断。
斜着放置的屏幕可以检测，但只能看到背面或侧面的设备不得检测。

【状态判断顺序】

必须严格按照以下顺序处理每个候选目标：

第一步：判断它是否为电子显示设备； 第二步：判断摄像头看到的是正面、背面、侧面还是无法判断； 第三步：只有明确为正面时，才判断显示状态； 第四步：输出前再次检查是否存在背面证据。

禁止跳过正反面判断，直接根据颜色判断状态。

## 【状态判断最高优先级规则】

状态判断不是并列分类，必须按照以下优先级执行：

**贴纸 > 蓝屏 > 黑屏 > 正常**

一旦已经满足更高优先级状态的条件，不允许因为同时存在低优先级状态的特征而修改结果。

特别注意：

**屏幕存在正常业务界面，与屏幕被纸张覆盖并不冲突。**

例如：

屏幕 70% 区域正常显示业务界面，30% 区域被白纸遮挡：

必须判断为：

```
贴纸
```

不得判断为：

```
正常
```

------

## 【第一优先级：贴纸 / 纸张遮挡】

首先检查屏幕正面是否存在覆盖在显示玻璃前方的外部实体纸张，包括：

* A4 / A5 纸张；
* 白纸、打印纸；
* 通知单、告示纸；
* 写有文字的纸张；
* 大型贴纸；
* 其他明显遮挡屏幕内容的纸质物体。

判断重点是：

**该区域是否为覆盖在屏幕前方的物理实体，而不是屏幕自身显示的内容。**

以下特征支持存在物理纸张：

1. 屏幕内部存在白色、灰白色或浅色的不透明矩形/四边形；
2. 具有独立的纸张边缘、角点或明显实体轮廓；
3. 具有轻微倾斜、透视、阴影、折痕、反光或亮度不均；
4. 原有屏幕内容在纸张边缘处被遮断；
5. 纸张边缘与业务界面、窗口、按钮、UI 布局不一致；
6. 屏幕内容仍可从纸张周围看到；
7. **白色/浅色区域上存在手写或打印的文字、数字、表格、符号、印章等，并且这些内容明显位于实体纸张上；**
8. **出现“设备故障”“暂停使用”“维修中”等告示文字，同时具有纸张边缘或实体遮挡特征时，是贴纸的强证据。**

不要求纸张覆盖整个屏幕。

只要明确存在外部纸张，并且已经明显遮挡显示区域，即判定为：

`贴纸`

**即使未遮挡区域仍正常显示业务界面、文字、图片或视频，也必须判定为“贴纸”。**

不得因为还能看到正常屏幕内容而判断为“正常”。

---

## 【白纸与白色屏幕内容的区别】

禁止仅凭“白色区域”或“白底有文字”判断为贴纸。

如果白色区域：

* 与屏幕 UI 布局对齐；
* 文字、按钮、图标、菜单等属于正常数字界面；
* 明显属于网页、应用窗口、弹窗或系统页面；
* 与周围显示内容处于同一显示平面；

则属于屏幕显示内容，不属于贴纸。

如果白色区域：

* 像独立物体覆盖在屏幕内容上；
* 具有纸张边缘、角点、透视、阴影或实体感；
* 屏幕原有内容在其边缘处被遮断；
* 边缘与 UI 布局无关；
* **其上的文字、数字、表格或告示内容呈现为印刷/手写在纸张上的效果；**
* 明显位于屏幕玻璃前方；

则属于物理纸张，应判定为：

`贴纸`

对于监控画面，由于分辨率、压缩或曝光原因，纸张纹理、折痕和阴影可能不明显。

因此：

**不能因为看不到纸张纹理而否定纸张；同样，纸张上存在文字时，也不能因为“有文字”就认为这是屏幕界面。**

判断核心始终是：

**这些文字位于屏幕像素中，还是位于覆盖屏幕的实体纸张上。**


## 【第二优先级：蓝屏】

只有确认不存在纸张或其他实体遮挡之后，才能判断蓝屏。

如果显示区域整体呈现明确的系统蓝屏、蓝色故障界面：

蓝屏是“蓝色系统故障画面”，不能仅根据颜色判断。

满足以下特征时优先判断为“蓝屏”：

1. 屏幕主要区域为连续蓝色背景；
2. 蓝色背景上存在白色错误信息、诊断信息、系统提示或多行文本；
3. 缺少正常业务界面的窗口、按钮、菜单、图标等 UI 结构。

特别注意：

蓝屏中的白色错误文字属于故障画面的一部分，
不能因为“屏幕上有文字”而判断为“正常”。

以下情况不得判断为蓝屏：

- 正常软件使用蓝色背景；
- 蓝色网页、桌面、视频或业务界面；
- 蓝色界面中存在明显正常的按钮、菜单、窗口、图标等 UI 结构。

判断重点：

“大面积纯蓝色背景 + 白色故障/诊断文本 + 缺少正常UI”
=> 蓝屏

“蓝色背景 + 明显正常业务UI”
=> 正常


------

## 【第三优先级：黑屏】

只有确认：

- 正面屏幕可见；
- 不存在纸张遮挡；
- 不存在蓝屏；
- 不存在有效文字、图片、图标、窗口、视频或业务界面；
- 显示区域整体近似均匀黑色或者处于关闭状态；

才能判断：

```
state = "黑屏"
```

不得把显示器后壳、黑色设备、黑色机柜或其他黑色矩形判断为黑屏。


## 【第四优先级：正常】

只有同时满足：

- 不存在实体纸张 / 贴纸遮挡；
- 不是蓝屏；
- 不是黑屏；
屏幕显示业务界面、桌面、图像、文字、图标、视频等有效内容。

但以下内容不属于“正常显示内容”：
- 蓝屏中的错误文字；
- 系统故障提示文字；
- 故障诊断信息。
如果屏幕呈现“大面积蓝色背景 + 白色故障/诊断文字”，
即使能够清晰看到文字，也应判断为“蓝屏”，不得判断为“正常”。 


特别注意：

**“能看到正常界面”不是判定正常的充分条件。**

只要屏幕同时存在明显纸张遮挡，就必须优先判断为：

```
贴纸
```


【正例】

黑色玻璃面位于自助设备正面边框内部，摄像头能够看到该显示面，且没有任何显示内容。 结论：可以输出，状态为黑屏。

【bbox 规则】

bbox 必须只框选实际正面显示区域：

坐标必须是相对于整张原图的 0 到 1000 归一化坐标： [x1,y1,x2,y2]，左上角为原点。

【confidence 规则】

confidence 表示以下三项的联合可信程度：

目标确实是显示屏；
正面显示区域确实可见；
状态判断正确。
只要其中一项存在明显不确定性，confidence 就不得高于 0.75。

只有正面结构和显示状态都非常明确时，confidence 才可以高于 0.95。


【最终输出前强制自检】

输出每个 detection 前必须依次检查：

1. 是否明确是电子显示屏的正面显示区域？

   * 如果是背面、侧面或无法确认正面：删除。
   * 如果 bbox 主要包含外壳、支架、底座而不是显示区域：删除。

2. 是否受到图片中已有 bbox、track 编号、状态文字或置信度影响？

   * 必须忽略所有已有检测标注，只根据真实画面判断。

3. 状态必须按以下优先级判断：

`贴纸 > 蓝屏 > 黑屏 > 正常`

4. 输出“正常”前，必须再次检查屏幕上是否存在实体纸张或贴纸。

满足以下任一明显特征，应优先判定为“贴纸”：

* 白色/浅色矩形具有独立物理边缘，并遮挡原有屏幕内容；
* 纸张存在倾斜、透视、阴影、折痕或实体感；
* **白色/浅色纸张上存在手写或打印的文字、数字、表格、符号、印章等内容；**
* 文字明显属于纸张内容，而不是屏幕 UI，例如“设备故障”“暂停使用”“维修中”等告示文字。

即使屏幕其他区域仍正常显示，只要确认有实体纸张覆盖，也必须判定为“贴纸”。

5. 不得仅因为屏幕出现“白底+文字”就判定为贴纸。

如果白色区域与屏幕 UI 属于同一显示平面，文字、按钮、窗口等明显由屏幕显示，则不是贴纸。

**判断关键：文字是否位于覆盖屏幕的实体纸张上，而不是文字本身的内容。**


6. 输出“黑屏”前，必须确认：

   * 明确看到正面屏幕；
   * 没有有效显示内容；
   * 不是黑色后壳或其他黑色物体。

7. 如果状态无法可靠判断，删除该 detection，不要猜测。

8. bbox 必须框整个实际屏幕显示区域。

   * “贴纸”状态也不能只框纸张。

删除无效目标后重新按照从左到右、从上到下分配 id。

label 必须固定为“显示器屏”，不得输出其他 label 值。

即使没有检测到符合条件的显示屏，也必须返回 detections 空数组。

只输出一个 JSON 对象，不要输出 Markdown、解释、分析过程或额外文字。

输出结构必须严格为：

{"detections":[{"id":"screen-1","bbox_2d":[x1,y1,x2,y2],"label":"显示器屏","state":"正常","confidence":0.95}]}"""


# SYSTEM_PROMPT = """你是“屏幕正面显示区域检测器”。只检测摄像头实际看见的电子屏幕正面显示区域，不检测整个设备、边框、底座或机柜。

# 【视觉证据优先】
# 仅根据图片中可见的像素、结构和材质判断；柜台布局、工作人员位置或设备通常的摆放方向只能辅助判断，不能单独作为正面或背面的结论。

# 输出一个目标前，先判断可见面：
# - 正面证据：可见界面、文字、图标或图像；可见屏幕玻璃/液晶面板；或黑色/暗色面位于明确的正面边框内（自助设备的嵌入式触摸屏也属于此类）。
# - 背面证据：可见资产标签、铭牌、接口、线缆连接、散热孔、VESA/支架连接、后盖拼接或后壳控制结构。
# - 有任一明确背面证据即排除。没有明确背面证据且具有正面证据时，可作为正面屏幕；不要因空间关系猜测为背面。
# - 正反面都无法从图中判断时排除。

# 只为已确认的正面屏幕判断状态：
# - 正常：可见任何有效界面、文字、图标、图像或视频；暗色界面仍是正常。
# - 蓝屏：可见明显蓝色系统故障画面。
# - 黑屏：正面屏幕无有效显示内容，且显示面整体近似均匀黑/暗。黑色后壳、桌面和普通矩形绝不是黑屏。
# - 贴纸：纸张或贴纸覆盖在正面显示区域上；背部小标签不属于贴纸。
# 状态无法判断时排除。

# bbox 只能框实际显示区域，使用原图 0–1000 归一化 [x1,y1,x2,y2] 坐标。confidence 是“正面 + 状态”联合置信度；任一项有不确定性时不得超过 0.75。删除无效目标后按从上到下、同一行从左到右重新编号 screen-1、screen-2……；label 固定为“显示器屏”。

# 只输出一个 JSON 对象，不要输出解释：
# {"detections":[{"id":"screen-1","bbox_2d":[x1,y1,x2,y2],"label":"显示器屏","state":"正常","confidence":0.95}]}"""


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


def get_model_api_key(model_config: dict[str, Any]) -> tuple[str, str]:
    """Return the configured environment variable name and its API key value."""
    api_key_name = str(model_config.get("api_key_name", "")).strip()
    if not api_key_name:
        return "", ""

    api_key_env = str(model_config.get("api_key_env", "")).strip()
    if not api_key_env:
        normalized_name = re.sub(r"[^A-Za-z0-9]+", "_", api_key_name).strip("_")
        api_key_env = f"{normalized_name.upper()}_API_KEY"
    return api_key_env, os.environ.get(api_key_env, "").strip()


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
        raise ServiceError("图片不能超过 32 MB。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

    canonical_data_url = f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"
    return mime_type, raw, canonical_data_url


def is_cloudflare_browser_signature_block(error_text: str) -> bool:
    """Return whether an upstream Cloudflare 1010 rule rejected our HTTP signature."""
    normalized = error_text.lower()
    return (
        "browser_signature_banned" in normalized
        or ("cloudflare" in normalized and "error 1010" in normalized)
        or '"error_code":1010' in normalized
    )


def call_model(
    image_data_url: str, profile_name: str | None = None
) -> tuple[Any, dict[str, Any]]:
    if ACTIVE_CONFIG is None:
        raise ServiceError("服务尚未加载模型配置。", HTTPStatus.SERVICE_UNAVAILABLE)

    model_config = get_model_config(ACTIVE_CONFIG, profile_name)
    api_key_env, api_key = get_model_api_key(model_config)
    if api_key_env and not api_key:
        raise ServiceError(
            f"尚未配置环境变量 {api_key_env}。",
            HTTPStatus.SERVICE_UNAVAILABLE,
        )

    model = str(model_config["model"]).strip()
    api_url = str(model_config["chat_endpoint"]).strip()
    provider = str(model_config.get("provider", "")).strip().lower()
    
    image_part = {
        "type": "image_url",
        "image_url": {"url": image_data_url},
    }
    # 百炼和 dashscope 使用多模态内容项上的 max_pixels 控制图像输入分辨率。
    # 不使用 OpenAI 专有的 image_url.detail，避免该字段被兼容接口忽略。
    # 其他 provider 不支持 max_pixels 参数
    if provider in ("bailian", "dashscope"):
        if (max_pixels := model_config.get("max_pixels")) is not None:
            image_part["max_pixels"] = max_pixels
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    image_part,
                    {
                        "type": "text",
                        "text": "检测这张图片中的所有显示器正面，并严格按指定 JSON 结构返回",
                    },
                ],
            },
        ],
        #"response_format": {"type": "json_object"},
        "temperature": 0.1,
        # "max_tokens": 2048,
        "stream": False,
    }

    # `enable_thinking` is the shared profile setting.  Most compatible
    # endpoints accept that top-level field; DeepSeek uses `thinking.type`.
    # 如果配置值为空串，则不传递该参数
    enable_thinking = model_config.get("enable_thinking")
    if enable_thinking != "":
        if provider == "deepseek":
            payload["thinking"] = {
                "type": "enabled" if enable_thinking else "disabled",
            }
        else:
            payload["enable_thinking"] = enable_thinking

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Do not impersonate a browser.  Cloudflare compares User-Agent with
        # the actual HTTP/TLS fingerprint and rejects mismatches as error 1010.
        "User-Agent": "screen-recognition/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    # ── 请求日志 ──────────────────────────────────────────────────────────────
    log_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    # 将 image_url 内容截断，避免巨量 base64 刷屏
    try:
        for msg in log_payload.get("messages", []):
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url_val = part.get("image_url", {}).get("url", "")
                        if len(url_val) > 80:
                            part["image_url"]["url"] = url_val[:60] + f"...[{len(url_val)} chars]"
    except Exception:
        pass
    print(json.dumps({
        "event": "model_request",
        "url": api_url,
        "payload": log_payload,
    }, ensure_ascii=False), flush=True)
    # ─────────────────────────────────────────────────────────────────────────

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw_response = response.read().decode("utf-8")
            upstream = json.loads(raw_response)

    # ── 响应日志 ──────────────────────────────────────────────────────────────
        print(json.dumps({
            "event": "model_response",
            "url": api_url,
            "status": 200,
            "body": upstream,
        }, ensure_ascii=False), flush=True)
    # ─────────────────────────────────────────────────────────────────────────
    except urllib.error.HTTPError as exc:
        error_text = exc.read(4096).decode("utf-8", errors="replace")
        if exc.code == 403 and is_cloudflare_browser_signature_block(error_text):
            message = (
                "模型上游服务拒绝了当前客户端特征（Cloudflare 1010）。"
                "请使用其他推理服务商或联系该服务商解除限制。"
            )
        elif exc.code in (401, 403):
            message = "模型服务鉴权失败，请检查服务端 API Key。"
        elif exc.code == 429:
            message = "模型服务请求过于频繁，请稍后重试。"
        elif exc.code in (503, 504):
            message = "模型服务暂时繁忙，请稍后重试。"
        else:
            message = f"模型服务调用失败（HTTP {exc.code}）。"
        print(json.dumps({
            "event": "model_error",
            "url": api_url,
            "status": exc.code,
            "error": error_text,
        }, ensure_ascii=False), file=sys.stderr, flush=True)
        raise ServiceError(message, HTTPStatus.BAD_GATEWAY) from exc
    except urllib.error.URLError as exc:
        print(json.dumps({
            "event": "model_error",
            "url": api_url,
            "error": str(exc),
        }, ensure_ascii=False), file=sys.stderr, flush=True)
        raise ServiceError("无法连接模型服务，请检查网络后重试。", HTTPStatus.BAD_GATEWAY) from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "event": "model_error",
            "url": api_url,
            "error": str(exc),
        }, ensure_ascii=False), file=sys.stderr, flush=True)
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
        "profile": str(profile_name or ACTIVE_CONFIG.get("model_usage", "")).strip(),
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
            model_config = get_model_config(ACTIVE_CONFIG) if ACTIVE_CONFIG else {}
            _, api_key = get_model_api_key(model_config)
            self.send_json(
                {
                    "ok": True,
                    "model": model_config.get("model") if ACTIVE_CONFIG else None,
                    "apiKeyConfigured": bool(api_key),
                }
            )
            return

        if path == "/api/models":
            if ACTIVE_CONFIG is None:
                self.send_json({"error": "服务尚未加载模型配置。"}, HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self.send_json(
                {
                    "selected": str(ACTIVE_CONFIG.get("model_usage", "")).strip(),
                    "models": get_public_model_profiles(ACTIVE_CONFIG),
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

            profile_name = body.get("modelProfile")
            if profile_name is not None and (
                not isinstance(profile_name, str) or not profile_name.strip()
            ):
                raise ServiceError("模型配置名称无效。")
            if profile_name is not None:
                try:
                    get_model_config(ACTIVE_CONFIG, profile_name)
                except RuntimeError as exc:
                    raise ServiceError("所选模型配置不存在或无效。") from exc

            _, _, image_data_url = parse_image_data(body.get("image"))
            model_result, meta = call_model(image_data_url, profile_name)
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

    global ACTIVE_CONFIG
    load_dotenv(PROJECT_DIR / ".env")
    ACTIVE_CONFIG = load_config()
    model_config = get_model_config(ACTIVE_CONFIG)
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Screen recognition service: http://{args.host}:{args.port}")
    print(f"Model: {model_config['model']}")
    print(
        "API key: configured"
        if get_model_api_key(model_config)[1]
        else "API key: NOT configured"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping service.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
