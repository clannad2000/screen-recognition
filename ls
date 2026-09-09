{
  "suspicious": true,
  "risk_level": "high",
  "confidence": 0.95,
  "window": {
    "start_frame_id": "F000010",
    "end_frame_id": "F000012",
    "start_video_time_ms": 55000,
    "end_video_time_ms": 65000
  },
  "frame_summary": [
    {
      "frame_id": "F000010",
      "video_time_ms": 55000,
      "persons": [
        {
          "person_id": "Person_1",
          "action": "蹲在自助终端设备底部/侧面区域",
          "location": "Device_1与Device_2之间"
        },
        {
          "person_id": "Person_2",
          "action": "站立在左侧窗边",
          "location": "左侧休息区"
        },
        {
          "person_id": "Person_3",
          "action": "坐在左侧休息区",
          "location": "左侧休息区"
        }
      ],
      "devices": [
        {
          "device_id": "Device_1",
          "state": "screen_on"
        },
        {
          "device_id": "Device_2",
          "state": "screen_on"
        }
      ],
      "observation": "Person_1蹲在两台自助终端设备之间的底部区域，Device_1和Device_2均处于亮屏状态。"
    },
    {
      "frame_id": "F000011",
      "video_time_ms": 60000,
      "persons": [
        {
          "person_id": "Person_1",
          "action": "继续蹲在设备底部/侧面区域",
          "location": "Device_1与Device_2之间"
        }
      ],
      "devices": [
        {
          "device_id": "Device_1",
          "state": "screen_on"
        },
        {
          "device_id": "Device_2",
          "state": "screen_off"
        }
      ],
      "observation": "Person_1保持蹲姿，Device_2的屏幕由亮变黑，Device_1仍保持亮屏。"
    },
    {
      "frame_id": "F000012",
      "video_time_ms": 65000,
      "persons": [
        {
          "person_id": "Person_1",
          "action": "站立并手持手机向画面左下方走动",
          "location": "自助区中央"
        }
      ],
      "devices": [
        {
          "device_id": "Device_1",
          "state": "screen_off"
        },
        {
          "device_id": "Device_2",
          "state": "screen_off"
        }
      ],
      "observation": "Person_1站起离开设备区域，此时Device_1的屏幕也变为黑屏，两台设备均处于黑屏状态。"
    }
  ],
  "events": [
    {
      "subject": "Person_1",
      "target": "Device_2",
      "start_frame_id": "F000010",
      "end_frame_id": "F000011",
      "start_video_time_ms": 55000,
      "end_video_time_ms": 60000,
      "person_action": "蹲在设备底部/侧面非典型操作区域停留",
      "device_before": "screen_on",
      "device_after": "screen_off",
      "state_transition": {
        "from_frame_id": "F000010",
        "to_frame_id": "F000011",
        "from_state": "screen_on",
        "to_state": "screen_off"
      },
      "temporal_correlation": "high",
      "causal_confidence": "medium",
      "supporting_evidence": [
        "Person_1在设备底部/侧面非典型区域蹲下停留",
        "在Person_1停留期间，Device_2由亮屏变为黑屏"
      ],
      "counter_evidence": [],
      "unknowns": [
        "未直接看到Person_1的具体手部操作动作"
      ],
      "risk_level": "high",
      "reason": "人物在设备非典型操作区域（底部/侧面）蹲下停留，且在该时间段内设备发生黑屏异常，存在高度时空相关性，属于高度可疑行为。"
    },
    {
      "subject": "Person_1",
      "target": "Device_1",
      "start_frame_id": "F000011",
      "end_frame_id": "F000012",
      "start_video_time_ms": 60000,
      "end_video_time_ms": 65000,
      "person_action": "从蹲姿站起并离开设备区域",
      "device_before": "screen_on",
      "device_after": "screen_off",
      "state_transition": {
        "from_frame_id": "F000011",
        "to_frame_id": "F000012",
        "from_state": "screen_on",
        "to_state": "screen_off"
      },
      "temporal_correlation": "high",
      "causal_confidence": "medium",
      "supporting_evidence": [
        "Person_1在紧邻时间内离开设备区域",
        "Device_1在Person_1离开前后由亮屏变为黑屏"
      ],
      "counter_evidence": [],
      "unknowns": [
        "未直接看到Person_1的具体手部操作动作"
      ],
      "risk_level": "high",
      "reason": "多台设备在人物可疑停留及离开期间相继发生黑屏异常，影响范围扩大，进一步增加风险等级。"
    }
  ],
  "summary": "监控画面显示，一名黑衣男子（Person_1）在自助终端设备底部/侧面非典型操作区域蹲下停留。在其停留期间（55000ms-60000ms），一台设备（Device_2）由亮屏变为黑屏。随后该男子站起离开（65000ms），另一台设备（Device_1）也变为黑屏。人物行为与多设备黑屏异常存在高度时空相关性，疑似对设备进行了非法操作，风险等级高。"
}