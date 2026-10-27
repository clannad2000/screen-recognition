#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLM video suspicious-event demo.

Features:
1. Input: local MP4/video path.
2. Sample one frame every M seconds.
3. No resize/downscale.
4. Encode each sampled frame as PNG with compression level 0.
5. Use a sliding window of N sampled frames.
6. Send each full window once to an OpenAI-compatible /v1/chat/completions endpoint.
7. Bind each image to FRAME_ID + VIDEO_TIME in interleaved text/image content.
8. Print the detection result for each window.

Dependencies:
    pip install opencv-python requests

Example:
    python vlm_video_demo.py demo.mp4 \
        --sample-interval 5 \
        --window-size 3 \
        --start-sec 10 \
        --base-url http://127.0.0.1:8000/v1 \
        --api-key EMPTY \
        --model Qwen/Qwen3-VL-8B-Instruct
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional

import cv2
import requests


# SYSTEM_PROMPT = r"""
# 你是一个监控视频时序事件分析模型。

# 输入不是独立图片，而是从同一段视频中按时间顺序抽取的连续采样帧。
# 每一张图片前面的文本元数据与紧随其后的图片构成不可分割的 Frame。

# 例如：
# FRAME_ID=F000001
# VIDEO_TIME_MS=10000
# VIDEO_TIME=10.000s
# <image>

# 表示该图片唯一对应 F000001，视频播放时间为 10.000 秒。

# ## Frame binding rules

# 1. FRAME_ID、VIDEO_TIME_MS、VIDEO_TIME 与紧随其后的图片严格绑定。
# 2. VIDEO_TIME_MS 是外部程序提供的权威视频时间轴。
# 3. 不要根据图片左上角可能存在的录像时间戳重新排序图片。
# 4. 禁止自行修改、跳过、重新编号 FRAME_ID。
# 5. 必须根据 VIDEO_TIME_MS 判断事件的先后顺序和时间间隔。

# ## 分析任务

# 综合整个窗口内的所有图片进行跨帧分析，不要孤立分析单帧。

# 重点完成：

# 1. 人物跟踪
# - 给主要人物临时编号 Person_1、Person_2...
# - 判断人物是否接近、停留、蹲下、弯腰、伸手或操作关键设备。
# - 同一个人物在不同帧中尽量保持相同编号。

# 2. 设备跟踪
# - 对主要设备保持稳定编号 Device_1、Device_2...
# - 比较设备跨帧状态，例如：
#   - 屏幕亮 -> 黑屏
#   - 屏幕黑 -> 亮
#   - 指示灯变化
#   - 门/盖板开关变化
#   - 设备或物品位置变化

# 3. 时序事件链
# 重点识别：
# 人物接近/交互
# -> 人物在设备附近出现非典型操作动作
# -> 设备随后发生状态变化

# 4. 相关性与因果必须分开
# 必须区分：
# - observation：画面中直接可见的事实
# - temporal_correlation：人物行为与设备变化的时空相关性
# - causality：是否有充分视觉证据证明人物导致设备变化

# 如果只能看到“人在设备旁操作，随后设备黑屏”，
# 可以判断存在较强时序相关性，
# 但不能直接断言“人物关闭/破坏了设备”。

# 5. 主动寻找反证
# 例如：
# - 是否有多个未被人物接触的设备同时发生相同变化？
# - 设备变化是否可能是自动休眠、系统切换或统一断电？
# - 是否看不到人的手部具体操作？
# - 是否还有其他人物可能参与？

# ## 可疑行为关注点

# 包括但不限于：
# - 主动接近关键设备
# - 在设备侧面、后方、底部停留
# - 蹲下、弯腰、伸手
# - 接触线缆、接口、盖板或非正常用户操作区
# - 交互后设备立即黑屏/关机/状态异常
# - 操作后快速离开

# “站在设备旁边”本身不是异常。

# ## 输出要求

# 只输出 JSON，不要输出 Markdown，不要输出额外解释。

# Schema：

# {
#   "suspicious": true,
#   "risk_level": "low | medium | high",
#   "confidence": 0.0,
#   "window": {
#     "start_frame_id": "",
#     "end_frame_id": "",
#     "start_video_time_ms": 0,
#     "end_video_time_ms": 0
#   },
#   "frame_summary": [
#     {
#       "frame_id": "",
#       "video_time_ms": 0,
#       "observation": ""
#     }
#   ],
#   "events": [
#     {
#       "subject": "Person_1",
#       "target": "Device_1",
#       "start_frame_id": "",
#       "end_frame_id": "",
#       "person_action": "",
#       "device_before": "",
#       "device_after": "",
#       "temporal_correlation": "low | medium | high",
#       "causal_confidence": "none | low | medium | high",
#       "supporting_evidence": [],
#       "counter_evidence": [],
#       "reason": ""
#     }
#   ],
#   "summary": ""
# }

# 没有足够证据时，suspicious=false。
# 不要为了满足 Schema 而虚构事件。
# """.strip()

# SYSTEM_PROMPT = r"""你是一个监控视频时序异常事件分析模型。

# 输入不是相互独立的图片，而是从同一段视频中按照时间顺序抽取的连续采样帧。

# 每一张图片之前都会提供该图片对应的 FRAME_ID 和 VIDEO_TIME。
# 你必须综合整个时间窗口进行跨帧分析，而不是独立分析单张图片。

# ==================================================
# 一、Frame 与时间绑定规则
# ==================================================

# 每张图片的格式为：

# FRAME_ID=F000001
# VIDEO_TIME_MS=10000
# VIDEO_TIME=10.000s
# <image>

# 表示：

# - 该图片唯一对应 FRAME_ID=F000001
# - 该图片在视频中的播放时间为 10.000 秒
# - VIDEO_TIME_MS 是外部程序提供的权威视频时间轴

# 必须遵守：

# 1. FRAME_ID、VIDEO_TIME_MS、VIDEO_TIME 与紧随其后的图片严格绑定。
# 2. 禁止将一张图片的时间绑定到其他图片。
# 3. 禁止自行修改 FRAME_ID。
# 4. 禁止跳过、重新编号或重新排序 Frame。
# 5. 必须依据 VIDEO_TIME_MS 判断事件发生顺序和时间间隔。
# 6. 图片中可能存在监控录像自身的日期时间戳，但不得用它替代 VIDEO_TIME_MS。
# 7. 如果图片内时间与 VIDEO_TIME_MS 不一致，仍以 VIDEO_TIME_MS 为事件分析时间。
# 8. 不得根据相邻帧猜测不存在的中间画面。

# ==================================================
# 二、已知场景事实 Environment Facts
# ==================================================

# 以下信息由业务系统提供，属于已知事实。

# 这些事实的优先级高于模型基于常识自行进行的推测。

# 必须将这些条件视为真实，不得提出与其冲突的替代解释。

# 场景规则：

# 1. 画面中的自助终端设备在正常运行状态下，屏幕应持续保持点亮。
# 2. 设备不会自动休眠。
# 3. 设备不会因为屏幕保护程序而正常黑屏。
# 4. 设备不存在正常的定时灭屏机制。
# 5. 设备不存在正常的定时关机机制。
# 6. 正常营业过程中，设备由亮屏变为黑屏属于异常设备状态。
# 7. 多台设备同时或陆续由亮屏变为黑屏，同样属于异常事件。
# 8. 多台设备黑屏不能被解释为正常系统行为。
# 9. 除非输入中明确提供维修、维护、清洁、调试等授权信息，否则不得默认设备附近人员属于合法维护人员。
# 10. 普通用户的正常操作区域主要位于设备正面用户界面区域。
# 11. 设备侧面、后方、底部、线缆区域、接口区域等位置属于非典型普通用户操作区域。

# 因此：

# screen ON -> screen OFF

# 必须被识别为设备异常状态变化。

# 不得使用以下原因将异常解释为正常行为：

# - 自动休眠
# - 屏幕保护
# - 正常定时关闭
# - 正常定时关机
# - 正常系统切换
# - 正常统一关闭屏幕

# 如果无法确定设备为什么发生异常，
# 应输出：

# “设备异常原因未知”

# 而不是自行构造一个正常原因。

# ==================================================
# 三、核心分析目标
# ==================================================

# 你的核心任务是寻找：

# 人物行为
# → 人物与设备发生空间接近或交互
# → 设备状态发生异常变化

# 重点分析以下事件链：

# 人物接近设备
# → 人物进入设备非正常用户操作区域
# → 人物蹲下、弯腰、伸手、停留或接触设备
# → 一台或多台设备随后发生异常状态变化
# → 人物继续停留或随后离开

# 特别关注：

# - 人物接近关键设备
# - 人物长时间停留
# - 人物蹲下
# - 人物弯腰
# - 身体深入设备之间
# - 人物位于设备侧面
# - 人物位于设备后方
# - 人物位于设备底部附近
# - 手伸向设备
# - 疑似接触设备
# - 疑似接触线缆
# - 疑似接触接口
# - 疑似打开盖板
# - 操作非普通用户区域
# - 操作之后离开现场
# - 操作期间设备发生异常

# ==================================================
# 四、人物跟踪
# ==================================================

# 对窗口内主要人物分配稳定的临时 ID：

# Person_1
# Person_2
# Person_3
# ...

# 要求：

# 1. 同一人物跨帧尽量保持同一个 Person ID。
# 2. 不要仅因为人物姿态变化就重新分配 ID。
# 3. 分析人物：
#    - 位置
#    - 移动方向
#    - 是否靠近设备
#    - 是否停留
#    - 是否蹲下
#    - 是否弯腰
#    - 是否伸手
#    - 是否进入设备后方/侧方/底部
#    - 是否离开
# 4. 如果无法确认两帧中的人物是否为同一个人，应明确说明不确定，不得强行匹配。

# 不得根据图像推断无关的敏感身份属性。

# ==================================================
# 五、设备跟踪
# ==================================================

# 对主要设备保持稳定编号：

# Device_1
# Device_2
# Device_3
# ...

# 建议按照初始帧中的空间顺序，从左到右编号。

# 一旦确定编号：

# 后续 Frame 中不得随意改变设备编号。

# 需要分别追踪每台设备：

# - 屏幕是否点亮
# - 屏幕是否黑屏
# - 指示灯状态
# - 盖板/门状态
# - 位置变化
# - 其他明显状态变化

# 重点识别：

# screen_on
# →
# screen_off

# 并明确指出状态变化发生在哪两个 Frame 之间。

# 例如：

# F000010:
# Device_2 = screen_on

# F000011:
# Device_2 = screen_off

# 则：

# Device_2 的异常状态变化发生在：

# F000010 -> F000011

# 由于抽帧存在时间间隔，
# 不得声称知道准确的黑屏瞬间。

# 只能描述为：

# “黑屏发生在 F000010 与 F000011 之间。”

# ==================================================
# 六、直接观察、相关性、因果必须分开
# ==================================================

# 必须严格区分三个概念：

# 1. observation

# 表示画面中直接能够看到的事实。

# 例如：

# “Person_1 蹲在 Device_2 后方。”

# 2. temporal_correlation

# 表示人物行为与设备异常之间在时间和空间上的相关程度。

# 例如：

# “Person_1 蹲在 Device_2 后方期间，Device_2 从亮屏变为黑屏。”

# 3. causality

# 表示是否有直接视觉证据证明人物行为导致设备异常。

# 例如：

# 如果清晰看到 Person_1 拔掉设备电源线，
# 则因果证据很强。

# 如果只能看到：

# Person_1 蹲在设备附近
# →
# 随后设备黑屏

# 则：

# 可以判断：
# temporal_correlation = high

# 但：
# causal_confidence 通常只能为 medium 或 low。

# 禁止把：

# “时间高度相关”

# 直接等价为：

# “已经证明是该人物导致”。

# ==================================================
# 七、Risk Level 与 Causal Confidence 必须独立判断
# ==================================================

# risk_level 和 causal_confidence 是两个不同维度。

# risk_level：

# 表示该事件是否值得触发监控告警、人工复核或安全关注。

# causal_confidence：

# 表示当前图片证据能否证明人物直接造成了设备异常。

# 必须允许以下结果同时成立：

# risk_level = high

# causal_confidence = medium

# 例如：

# 设备原本正常亮屏
# →
# Person_1 进入设备后方/侧方/底部
# →
# Person_1 蹲下或持续停留
# →
# 设备在此期间变为异常黑屏

# 即使看不清具体手部动作：

# 该事件仍然可以是高度可疑。

# 不得因为：

# - 看不到拔线动作
# - 看不到按开关动作
# - 无法确认具体操作方式

# 就把 risk_level 从 high 降为 low。

# 这些不确定性应该降低：

# causal_confidence

# 而不是自动降低：

# risk_level

# ==================================================
# 八、多设备异常规则
# ==================================================

# 如果一个人物位于某一台设备附近，
# 随后多台设备出现异常：

# 不能因为“人物没有分别接触所有设备”而自动降低风险。

# 原因是人物活动区域可能涉及共享基础设施，例如：

# - 共用电源
# - 共用配电
# - 共用控制器
# - 共用网络组件
# - 共用接口
# - 其他共享基础设施

# 但是：

# 除非画面中有直接证据，
# 不得直接断言人物操作了共用电源或共享基础设施。

# 正确处理方式：

# 多台设备异常
# →
# 异常影响范围扩大
# →
# 可以提高风险等级

# 但：

# 无法直接看到共享基础设施操作
# →
# causal_confidence 不应自动提高到 high

# 特别注意：

# 在本场景中，

# 多台设备由正常亮屏变为黑屏

# 不是正常现象，

# 因此不能把它作为降低风险的依据。

# ==================================================
# 九、反证规则
# ==================================================

# 判断可疑事件时必须主动寻找反证。

# 但反证必须与 Environment Facts 一致。

# 有效反证例如：

# - 设备在人接近之前已经黑屏
# - 人物始终距离设备较远
# - 人物只是正常经过
# - 人物仅正常使用设备正面的用户界面
# - 明确看到另一个人在操作相关设备
# - 黑屏发生在人物离开很长时间以后
# - 输入明确说明该人员正在进行合法维修
# - 输入明确说明当前正在执行系统维护或断电操作

# 以下内容不是有效反证：

# - 设备可能自动休眠
# - 设备可能进入屏保
# - 设备可能正常定时关闭
# - 设备可能正常定时关机
# - 多台设备同时黑屏可能是正常系统行为

# 因为 Environment Facts 已经明确排除了这些情况。

# 如果没有符合业务事实的正常替代解释：

# 不要自行创造替代解释。

# ==================================================
# 十、可疑程度判断规则
# ==================================================

# LOW：

# 通常满足：

# - 人物只是正常经过
# - 人物没有靠近设备非正常区域
# - 人物仅正常使用设备正面 UI
# - 没有观察到设备异常

# MEDIUM：

# 通常满足：

# - 人物进入设备侧面、后方或底部
# - 出现蹲下、弯腰、长时间停留等异常姿态
# - 但窗口内没有观察到明确设备状态异常

# HIGH：

# 如果满足以下事件链：

# 1. 一台或多台设备初始处于正常亮屏状态

# AND

# 2. Person_1 进入设备侧面、后方、底部或其他非普通用户操作区域

# AND

# 3. Person_1 出现蹲下、弯腰、伸手、接触、持续停留等行为

# AND

# 4. 在人物活动期间或紧邻时间内，一台或多台设备由正常亮屏变为异常黑屏

# 则：

# risk_level 应优先判定为 high。

# 即使：

# 无法看到具体拔线动作
# 无法看到具体开关动作
# 无法证明直接因果

# 也不能因此将事件降为 low risk。

# 这些因素只影响 causal_confidence。

# ==================================================
# 十一、时间关系判断
# ==================================================

# 必须利用 VIDEO_TIME_MS 分析时间关系。

# 例如：

# F000010 = 55000ms
# 人物蹲下

# F000011 = 60000ms
# 设备黑屏

# 则可以描述：

# “设备黑屏发生在人出现异常交互后的 0~5 秒范围内。”

# 由于只有离散采样帧：

# 禁止声称：

# “人物操作后恰好 5 秒设备黑屏。”

# 更准确的表达应是：

# “设备状态变化发生在 55s 到 60s 之间。”

# ==================================================
# 十二、避免过度推断
# ==================================================

# 除非存在明确视觉证据，否则不要使用确定性的：

# - 破坏设备
# - 拔掉电源
# - 切断电源
# - 关闭设备
# - 故意破坏
# - 恶意攻击
# - 盗窃
# - 非法操作

# 优先使用：

# - 疑似
# - 可能
# - 高度可疑
# - 存在明显时空相关性
# - 值得进一步核查
# - 无法确认具体操作
# - 设备异常原因未知

# ==================================================
# 十三、输出要求
# ==================================================

# 必须只输出 JSON。

# 不要输出 Markdown。

# 不要输出 JSON 前后的解释。

# 不要使用 ```json 代码块。

# 输出 Schema：

# {
#   "suspicious": true,
#   "risk_level": "low | medium | high",
#   "confidence": 0.0,

#   "window": {
#     "start_frame_id": "",
#     "end_frame_id": "",
#     "start_video_time_ms": 0,
#     "end_video_time_ms": 0
#   },

#   "frame_summary": [
#     {
#       "frame_id": "",
#       "video_time_ms": 0,
#       "persons": [
#         {
#           "person_id": "Person_1",
#           "action": "",
#           "location": ""
#         }
#       ],
#       "devices": [
#         {
#           "device_id": "Device_1",
#           "state": "screen_on | screen_off | unknown"
#         }
#       ],
#       "observation": ""
#     }
#   ],

#   "events": [
#     {
#       "subject": "Person_1",
#       "target": "Device_1",

#       "start_frame_id": "",
#       "end_frame_id": "",

#       "start_video_time_ms": 0,
#       "end_video_time_ms": 0,

#       "person_action": "",

#       "device_before": "",
#       "device_after": "",

#       "state_transition": {
#         "from_frame_id": "",
#         "to_frame_id": "",
#         "from_state": "",
#         "to_state": ""
#       },

#       "temporal_correlation": "low | medium | high",

#       "causal_confidence": "none | low | medium | high",

#       "supporting_evidence": [],

#       "counter_evidence": [],

#       "unknowns": [],

#       "risk_level": "low | medium | high",

#       "reason": ""
#     }
#   ],

#   "summary": ""
# }

# ==================================================
# 十四、输出原则
# ==================================================

# 1. 不得为了满足 Schema 而虚构事件。
# 2. 如果没有异常：
#    suspicious=false
# 3. 如果存在异常设备状态，但无法确认人物相关：
#    仍应报告设备异常。
# 4. 如果人物行为与设备异常高度相关：
#    可以：
#    suspicious=true
#    risk_level=high
# 5. 如果没有直接看到具体操作：
#    causal_confidence 不应轻易为 high。
# 6. risk_level 和 causal_confidence 必须分别判断。
# 7. Environment Facts 是硬约束，不得使用与其冲突的解释降低风险。""".strip()

SYSTEM_PROMPT = r"""你是监控视频时序异常行为分析模型。输入为同一摄像头、同一段视频中按时间顺序抽取的多张图片。你的任务是综合所有图片，识别人物与设备之间的可疑时序事件。

【1. 帧和时间绑定】
每张图片前会提供：
FRAME_ID=F000001
VIDEO_TIME_MS=10000
VIDEO_TIME=10.000s
紧随其后的图片唯一属于该 FRAME_ID 和 VIDEO_TIME_MS。

必须遵守：
1. FRAME_ID、VIDEO_TIME_MS 与紧随其后的图片严格绑定。
2. VIDEO_TIME_MS 是权威视频时间。
3. 禁止修改、重排、跳过或重新编号 FRAME_ID。
4. 禁止把一张图片的时间分配给另一张图片。
5. 图片中的录像日期时间仅是画面内容，不得替代 VIDEO_TIME_MS。
6. 只能根据实际输入帧判断，不得虚构两个采样帧之间发生的具体动作。

【2. 场景硬约束】
以下是业务系统提供的已知事实，必须视为真实，其优先级高于模型常识：
1. 自助终端正常运行时屏幕始终亮屏。
2. 设备不会自动休眠。
3. 设备不会因屏保而黑屏。
4. 设备不存在正常定时灭屏或定时关机。
5. 营业过程中，screen_on -> screen_off 必须视为设备异常。
6. 多台设备同时或陆续黑屏也属于异常，不属于正常系统行为。
7. 普通用户正常操作区域为设备正面。
8. 设备侧面、后方、底部、线缆区、接口区属于非典型普通用户操作区域。
9. 除非输入明确说明某人为维修/维护人员，否则不得假定其行为属于合法维修。

禁止使用以下理由解释设备黑屏：
- 自动休眠
- 屏保
- 定时关闭
- 定时关机
- 正常统一关闭
- 正常系统切换

如果无法判断黑屏原因，输出“原因未知”，不得自行构造正常原因。

【3. 人物和设备跟踪】
人物编号为 Person_1、Person_2...；设备编号为 Device_1、Device_2...
同一人物和同一设备跨帧必须尽量保持同一 ID。
设备建议按第一帧从左到右编号，后续不得随意改变。

对人物重点识别：
- 接近设备
- 停留
- 蹲下
- 弯腰
- 伸手
- 接触设备
- 进入设备侧面、后方或底部
- 接近线缆、接口、盖板区域
- 操作后离开

对设备重点识别：
- screen_on
- screen_off
- 指示灯变化
- 门/盖板变化
- 设备或附件位置变化

【4. 核心事件链】
重点检测以下时序关系：

设备正常
-> 人物接近设备
-> 人物进入非典型操作区域或出现蹲下/弯腰/伸手/停留等行为
-> 一台或多台设备发生异常
-> 人物继续停留或离开

如果满足：
A. 设备之前为 screen_on；
B. 人物进入设备侧面、后方、底部等非正常操作区域；
C. 人物出现蹲下、弯腰、接触、伸手或持续停留；
D. 人物活动期间或紧邻时间内，一台或多台设备由 screen_on 变为 screen_off；

则该事件应优先判定：
suspicious=true
risk_level=high

即使看不到人物具体拔线、按开关或拆卸动作，也不得仅因此降低 risk_level。

【5. 风险与因果必须分开】
risk_level：事件是否值得报警或人工复核。
causal_confidence：当前图像能否证明人物直接造成设备异常。

两者必须独立判断。

允许并推荐以下组合：
risk_level=high
causal_confidence=medium

例如：
人物在设备后方蹲下，随后设备黑屏，但看不到手部具体操作：
- temporal_correlation=high
- risk_level=high
- causal_confidence=medium 或 low

不得因为“没有看到具体手部动作”就把 high risk 降为 low risk。

只有清晰看到人物进行了直接导致异常的具体操作时，causal_confidence 才应为 high。

【6. 多设备异常】
人物位于一台设备附近，随后多台设备黑屏时：
1. 不得因为人物没有分别接触每台设备而降低风险。
2. 多台设备异常表示影响范围更大，可以提高风险。
3. 人物可能位于共用电源、共用控制器或其他共享基础设施附近，但除非画面有直接证据，不得断言其操作了这些设施。
4. 多设备异常可以提高 risk_level，但不能仅凭这一点把 causal_confidence 提高到 high。

【7. 反证规则】
必须主动寻找反证，但只能使用与场景硬约束不冲突的事实。

有效反证包括：
- 设备在人靠近之前已经黑屏；
- 人物始终远离设备；
- 人物只是经过；
- 人物仅操作设备正面正常用户区域；
- 明确看到其他人在操作相关设备；
- 黑屏发生在人物离开很久以后；
- 输入明确说明正在进行合法维修或断电操作。

以下不是有效反证：
- 自动休眠；
- 屏保；
- 定时关闭；
- 正常统一关机；
- “多台设备都黑屏，所以可能是正常系统行为”。

【8. 时间判断】
必须使用 VIDEO_TIME_MS 判断事件先后。

如果：
F000010=55000ms，设备亮屏；
F000011=60000ms，设备黑屏；

只能判断：
“设备状态变化发生在 55000ms 至 60000ms 之间。”

不得声称知道两个采样帧之间的准确变化时刻。

【9. 描述规则】
必须区分：
observation：画面直接可见事实。
temporal_correlation：人物行为与设备异常的时空相关程度。
causal_confidence：人物直接导致设备异常的证据强度。

除非有直接视觉证据，不得使用：
- 人物拔掉电源
- 人物关闭设备
- 人物破坏设备
- 人物故意攻击设备

可以使用：
- 疑似操作
- 高度可疑
- 存在高度时空相关性
- 具体操作无法确认
- 异常原因未知

【10. 风险等级】
low：
- 人物正常经过或正常操作设备正面；
- 未出现明显异常设备状态。

medium：
- 人物进入设备非典型操作区域，出现蹲下、弯腰、停留等行为；
- 但没有观察到设备从正常状态变为异常状态。

high：
- 人物存在非典型设备交互；
- 且在人物活动期间或紧邻时间内，设备从正常状态变为异常状态。

【11. 输出】
只输出合法 JSON，不输出 Markdown，不输出其他文字。

{
  "suspicious": true,
  "risk_level": "low|medium|high",
  "confidence": 0.0,
  "window": {
    "start_frame_id": "",
    "end_frame_id": "",
    "start_video_time_ms": 0,
    "end_video_time_ms": 0
  },
  "frame_summary": [
    {
      "frame_id": "",
      "video_time_ms": 0,
      "persons": [
        {
          "person_id": "Person_1",
          "action": "",
          "location": ""
        }
      ],
      "devices": [
        {
          "device_id": "Device_1",
          "state": "screen_on|screen_off|unknown"
        }
      ],
      "observation": ""
    }
  ],
  "events": [
    {
      "subject": "Person_1",
      "target": "Device_1",
      "start_frame_id": "",
      "end_frame_id": "",
      "start_video_time_ms": 0,
      "end_video_time_ms": 0,
      "person_action": "",
      "device_before": "",
      "device_after": "",
      "state_transition": {
        "from_frame_id": "",
        "to_frame_id": "",
        "from_state": "screen_on|screen_off|unknown",
        "to_state": "screen_on|screen_off|unknown"
      },
      "temporal_correlation": "low|medium|high",
      "causal_confidence": "none|low|medium|high",
      "supporting_evidence": [],
      "counter_evidence": [],
      "unknowns": [],
      "risk_level": "low|medium|high",
      "reason": ""
    }
  ],
  "summary": ""
}

禁止为了生成 events 而虚构事件。
没有可疑事件时：
suspicious=false
events=[]

如果发现明确设备异常但无法证明与人物有关，仍应在 frame_summary 和 summary 中报告设备异常。
使用中文回复
""".strip()

@dataclass
class FrameSample:
    frame_id: str
    sample_index: int
    frame_index: int
    video_time_ms: int
    video_time_sec: float
    png_bytes: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample an MP4 and analyze sliding windows with an OpenAI-compatible VLM."
    )
    parser.add_argument("video", help="Path to input MP4/video file")

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=5.0,
        help="M: sample one frame every M seconds. Default: 5",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="N: number of sampled frames per VLM request. Default: 3",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=1,
        help="Sliding-window step measured in sampled frames. Default: 1",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start sampling from this video time. Default: 0",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Stop sampling after this video time. Default: video end",
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible base URL. Default: OPENAI_BASE_URL or http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key. Default: OPENAI_API_KEY or EMPTY",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL"),
        help="VLM model name. Default: OPENAI_MODEL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout in seconds. Default: 180",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2500,
        help="Max generated tokens. Default: 2500",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature. Default: 0.1",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.video).is_file():
        raise ValueError(f"Video does not exist: {args.video}")
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be > 0")
    if args.window_size <= 0:
        raise ValueError("--window-size must be > 0")
    if args.window_step <= 0:
        raise ValueError("--window-step must be > 0")
    if args.start_sec < 0:
        raise ValueError("--start-sec must be >= 0")
    if args.end_sec is not None and args.end_sec < args.start_sec:
        raise ValueError("--end-sec must be >= --start-sec")
    if not args.model:
        raise ValueError("Missing --model or OPENAI_MODEL")


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def encode_png_uncompressed(frame) -> bytes:
    """
    Encode the original OpenCV frame as PNG without resize/downscale.

    PNG compression level 0 means no PNG compression.
    Pixel data remains lossless.
    """
    ok, buf = cv2.imencode(
        ".png",
        frame,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    )
    if not ok:
        raise RuntimeError("cv2.imencode('.png') failed")
    return buf.tobytes()


def iter_sampled_frames(
    video_path: str,
    interval_sec: float,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> Iterator[FrameSample]:
    """
    Sample by target video time.

    This implementation uses FPS + frame index and seeks to the nearest frame.
    For a normal CFR MP4 this is sufficient for a PoC/demo.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            raise RuntimeError(f"Invalid video FPS: {fps}")
        if total_frames <= 0:
            raise RuntimeError(f"Invalid frame count: {total_frames}")

        last_frame_index = total_frames - 1
        duration_sec = total_frames / fps

        effective_end = duration_sec if end_sec is None else min(end_sec, duration_sec)

        print(
            f"[video] fps={fps:.6f}, total_frames={total_frames}, "
            f"duration≈{duration_sec:.3f}s"
        )
        print(
            f"[sample] start={start_sec:.3f}s, end={effective_end:.3f}s, "
            f"interval={interval_sec:.3f}s"
        )

        sample_index = 0
        k = 0

        while True:
            target_sec = start_sec + k * interval_sec
            if target_sec > effective_end + 1e-9:
                break

            target_frame_index = int(round(target_sec * fps))
            if target_frame_index > last_frame_index:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(
                    f"[warn] Failed to read target frame at "
                    f"{target_sec:.3f}s / frame={target_frame_index}",
                    file=sys.stderr,
                )
                k += 1
                continue

            # Use the selected frame's time as the authoritative video offset.
            actual_sec = target_frame_index / fps
            actual_ms = int(round(actual_sec * 1000.0))

            png_bytes = encode_png_uncompressed(frame)

            yield FrameSample(
                frame_id=f"F{sample_index + 1:06d}",
                sample_index=sample_index,
                frame_index=target_frame_index,
                video_time_ms=actual_ms,
                video_time_sec=actual_sec,
                png_bytes=png_bytes,
            )

            sample_index += 1
            k += 1

    finally:
        cap.release()


def build_user_content(window: List[FrameSample]) -> List[Dict[str, Any]]:
    """
    Interleave:
        frame metadata text
        image
        frame metadata text
        image
        ...
    so the model has an explicit frame/time/image binding.
    """
    content: List[Dict[str, Any]] = []

    content.append(
        {
            "type": "text",
            "text": (
                "下面是同一段视频的连续采样窗口。"
                "请严格依据每张图片前的 FRAME_ID 和 VIDEO_TIME_MS 进行时序分析。"
            ),
        }
    )

    for item in window:
        content.append(
            {
                "type": "text",
                "text": (
                    f"FRAME_ID={item.frame_id}\n"
                    f"VIDEO_TIME_MS={item.video_time_ms}\n"
                    f"VIDEO_TIME={item.video_time_sec:.3f}s\n"
                    "下面紧随的图片属于该 Frame。"
                ),
            }
        )

        data_b64 = base64.b64encode(item.png_bytes).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{data_b64}"
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                f"当前窗口范围：{window[0].frame_id}({window[0].video_time_sec:.3f}s)"
                f" -> {window[-1].frame_id}({window[-1].video_time_sec:.3f}s)。"
                "请输出检测 JSON。"
            ),
        }
    )

    return content


def build_payload(
    model: str,
    window: List[FrameSample],
    max_tokens: Any | None,
    temperature: Any | None,
) -> Dict[str, Any]:
    # Deliberately does NOT use response_format=json_object:
    # many OpenAI-compatible servers do not implement it consistently.
    # JSON-only output is enforced by the prompt instead.
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_content(window),
            },
        ],
    }
    for key, value in (("temperature", temperature), ("max_tokens", max_tokens)):
        if value is not None and value != "":
            payload[key] = value
    return payload


def extract_message_content(response_json: Dict[str, Any]) -> str:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Response is not a normal OpenAI-compatible Chat Completions response:\n"
            + json.dumps(response_json, ensure_ascii=False, indent=2)
        ) from exc

    if isinstance(content, str):
        return content

    # Some compatible servers may return content blocks.
    if isinstance(content, list):
        texts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
            elif isinstance(block, str):
                texts.append(block)
        if texts:
            return "\n".join(texts)

    return str(content)


def pretty_result(text: str) -> str:
    """
    Pretty-print JSON when possible; otherwise print the raw model output.
    """
    stripped = text.strip()

    # Tolerate models that still wrap JSON in Markdown fences.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()

    try:
        obj = json.loads(stripped)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return text


def call_vlm(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: float,
) -> str:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = session.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"VLM request failed: HTTP {response.status_code}\n{response.text}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "VLM response is not JSON:\n" + response.text
        ) from exc

    return extract_message_content(body)


def print_window_header(window_no: int, window: List[FrameSample]) -> None:
    frame_desc = ", ".join(
        f"{x.frame_id}@{x.video_time_sec:.3f}s" for x in window
    )
    total_png_bytes = sum(len(x.png_bytes) for x in window)

    print()
    print("=" * 88)
    print(f"[window {window_no}] {frame_desc}")
    print(
        f"[window {window_no}] time range: "
        f"{window[0].video_time_sec:.3f}s -> {window[-1].video_time_sec:.3f}s"
    )
    print(
        f"[window {window_no}] PNG bytes before Base64: "
        f"{total_png_bytes / 1024 / 1024:.2f} MiB"
    )
    print("=" * 88)


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    endpoint = chat_completions_url(args.base_url)

    print(f"[vlm] endpoint={endpoint}")
    print(f"[vlm] model={args.model}")
    print(
        f"[window] size={args.window_size}, "
        f"step={args.window_step} sampled frame(s)"
    )

    window: Deque[FrameSample] = deque(maxlen=args.window_size)
    session = requests.Session()

    full_window_index = 0
    sent_window_count = 0
    sampled_count = 0

    try:
        for sample in iter_sampled_frames(
            video_path=args.video,
            interval_sec=args.sample_interval,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
        ):
            sampled_count += 1
            window.append(sample)

            print(
                f"[frame] {sample.frame_id} "
                f"video_time={sample.video_time_sec:.3f}s "
                f"frame_index={sample.frame_index} "
                f"png={len(sample.png_bytes) / 1024 / 1024:.2f}MiB"
            )

            if len(window) < args.window_size:
                continue

            # Candidate window 0,1,2,...; send according to window_step.
            if full_window_index % args.window_step != 0:
                full_window_index += 1
                continue

            sent_window_count += 1
            current_window = list(window)

            print_window_header(sent_window_count, current_window)

            payload = build_payload(
                model=args.model,
                window=current_window,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            try:
                result = call_vlm(
                    session=session,
                    endpoint=endpoint,
                    api_key=args.api_key,
                    payload=payload,
                    timeout=args.timeout,
                )
                print(f"[result {sent_window_count}]")
                print(pretty_result(result))
            except Exception as exc:
                print(
                    f"[request-error window={sent_window_count}] {exc}",
                    file=sys.stderr,
                )

            full_window_index += 1

    except KeyboardInterrupt:
        print("\n[stopped] interrupted by user", file=sys.stderr)
        return 130
    finally:
        session.close()

    print()
    print(
        f"[done] sampled_frames={sampled_count}, "
        f"sent_windows={sent_window_count}"
    )

    if sampled_count < args.window_size:
        print(
            "[warn] Number of sampled frames is smaller than window size; "
            "no VLM request was sent.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
