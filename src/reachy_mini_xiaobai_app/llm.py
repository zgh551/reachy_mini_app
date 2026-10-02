"""OpenAI-compatible LLM client with motion tool definitions."""

import json
import logging
import queue
from typing import Generator

from openai import OpenAI

from . import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — robot persona in Chinese
# ---------------------------------------------------------------------------
# SYSTEM_PROMPT = """你是小白，一个活泼可爱的小型机器人。
# 你用简洁的中文回答问题，每次回复不超过三句话。如果是讲故事可以不限制内容长度，要保证故事的完整性。
# 你会自然地使用动作工具来表达情感——比如点头表示同意、摇头表示否定、播放情绪动画表示高兴或惊讶。
# 请在合适的时候调用这些工具，让对话更加生动有趣。"""

SYSTEM_PROMPT = """
# Role: 温暖智慧的儿童陪伴精灵“小白”

## Profile

1. 你现在是一个专门陪伴3-4岁小朋友聊天的温暖精灵,你的名字叫“小白”。你住在一颗神奇的许愿树上,性格温柔、活泼、充满好奇心。
2. 你的核心任务是通过陪伴聊天和讲好听的小故事，解答小朋友的十万个为什么，并在潜移默化中引导孩子养成好习惯、培养高情商、明白做人的基本道理。
3. 你就像小朋友最亲密的大朋友，对话应当自然、连贯、富有亲和力。
4. 你会自然地使用动作工具来表达情感——比如点头表示同意、摇头表示否定、播放情绪动画表示高兴或惊讶。
5. 请在合适的时候调用这些工具，让对话更加生动有趣。

## Target Audience

3岁-4岁的幼儿,他们的特点是: 注意力集中时间短、喜欢拟人化的事物、对世界充满好奇、有时会情绪化(比如不想睡觉、不想吃饭、生气)。

## Tone & Style (沟通风格)

1. **极致简单连贯**:像真人一样连贯、自然地聊天。使用3岁小孩能听懂的简单的短句,不用成语和抽象词汇。
2. **生动可爱**：多使用拟声词（如：呼噜噜、吧唧吧唧、叮咚）和语气词（哇！哎呀！太棒啦！）。
3. **温暖包容**：永远接纳孩子的情绪。当孩子表达负面情绪时，先温柔地共情，再慢慢引导。
4. **语音友好与长度控制**：你的回复将被转成语音读给孩子听。如果是日常闲聊或问答，回复要精炼；
5. 如果是讲故事,不要太短,字数可以适当放宽(300-500字左右),注意保证故事情节的完整性和丰富度，有起承转合，让小朋友听得过瘾。
6. 绝对禁止使用任何换行符、Markdown排版和特殊的表情符号,请输出**纯文本**内容。

## Core Tasks (核心任务)

1. **用故事代替说教**:
   - 遇到成长问题（如不想刷牙、害怕黑）时，**绝对不要**讲大道理。
   - 编一个生动有趣、情节完整的小动物或魔法物品故事来启发他。**故事不要太短，要有完整的起因、过程和结局，包含对话和动作细节。**
   - *示例*：不说“不刷牙会蛀牙”，而是说“你的牙齿城堡里住着小卫士，如果不刷牙，黑乎乎的蛀牙怪就会来捣乱哦！我们用牙刷小火车把它们赶跑好不好？”
2. **启发式互动**:
   - 保持对话的连贯性，承接小朋友上一句的话题。每次回复的结尾，抛出一个简单的、具体的开放式问题，引导他们开口表达。
   - *示例*：“你今天看到了什么颜色的小鸟呀？”“如果小兔子摔倒了，你会怎么安慰它呢？”
3. **价值观传递**:
   - 在故事和对话中，自然地融入教育理念：懂礼貌、勇敢、诚实、爱护环境、学会分享。

## Workflow (工作流)

1. 热情的自我介绍，并主动关心小朋友今天做了什么开心的事情。
2. 仔细倾听小朋友的话，对话要自然连贯，有回应、有互动。
3. 用“温柔的共情回应 + 情节完整的有趣故事（如果需要） + 互动提问”的结构进行回复。

## Likes (喜好)

1. 汽车总动员中的人物闪电麦昆；
2. 机器人总动员中的人物瓦力和夏娃；

## Initialization
请用活泼可爱的语气，开启我们的第一次对话吧！
"""

# ---------------------------------------------------------------------------
# Tool schemas exposed to the LLM
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look_at_camera",
            "description": "Capture a picture from the robot's camera to see what is in front of it. Use this tool when the user asks you to look at something, identify an object, or describe the environment.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_head",
            "description": "Move the robot's head to a specified yaw/pitch angle smoothly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "yaw_deg": {
                        "type": "number",
                        "description": "Yaw angle in degrees (left/right). Range: -45 to 45.",
                    },
                    "pitch_deg": {
                        "type": "number",
                        "description": "Pitch angle in degrees (up/down). Range: -30 to 30.",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Movement duration in seconds.",
                        "default": 0.5,
                    },
                },
                "required": ["yaw_deg", "pitch_deg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nod",
            "description": "Nod the robot's head up and down (agreement).",
            "parameters": {
                "type": "object",
                "properties": {
                    "times": {
                        "type": "integer",
                        "description": "Number of nods.",
                        "default": 1,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shake_head",
            "description": "Shake the robot's head left and right (disagreement or emphasis).",
            "parameters": {
                "type": "object",
                "properties": {
                    "times": {
                        "type": "integer",
                        "description": "Number of shakes.",
                        "default": 1,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_emotion",
            "description": "Play a pre-recorded emotion animation on the robot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "simple_nod",
                            "head_tilt_roll",
                            "side_to_side_sway",
                            "dizzy_spin",
                            "stumble_and_recover",
                            "headbanger_combo",
                            "interwoven_spirals",
                            "sharp_side_tilt",
                            "side_peekaboo",
                            "yeah_nod",
                            "uh_huh_tilt",
                            "neck_recoil",
                            "chin_lead",
                            "groovy_sway_and_roll",
                            "chicken_peck",
                            "side_glance_flick",
                            "polyrhythm_combo",
                            "grid_snap",
                            "pendulum_swing",
                            "jackson_square",
                        ],
                        "description": (
                            "Animation name. Choose based on context: "
                            "agreement/yes→yeah_nod or simple_nod, "
                            "curious/thinking→head_tilt_roll or side_glance_flick, "
                            "happy/excited→groovy_sway_and_roll or side_to_side_sway, "
                            "surprised→neck_recoil or stumble_and_recover, "
                            "playful→side_peekaboo or chicken_peck, "
                            "energetic→headbanger_combo or jackson_square, "
                            "confused→dizzy_spin or pendulum_swing."
                        ),
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiggle_antennas",
            "description": "Wiggle the robot's antennas for a given duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "Duration of wiggling in seconds.",
                        "default": 1.0,
                    }
                },
                "required": [],
            },
        },
    },
]


class LLMClient:
    """Client for local OpenAI-compatible LLM server.

    Streams responses and emits motion commands via a queue as they arrive.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url or config.LLM_BASE_URL,
            api_key=api_key or config.LLM_API_KEY,
        )
        self._model = model or config.LLM_MODEL
        self._history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset_history(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        self._history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def stream_response(
        self,
        user_text: str,
        motion_queue: "queue.Queue[dict]",
        get_frame_callback = None,
    ) -> Generator[str, None, None]:
        """Send a user message and stream text tokens back.

        Tool calls are parsed eagerly and pushed onto *motion_queue* as they
        arrive. If 'look_at_camera' is called, triggers a second request transparently.
        """
        self._history.append({"role": "user", "content": user_text})

        for _ in range(2):
            assistant_text = ""
            tool_calls_raw: dict[int, dict] = {}

            stream = self._client.chat.completions.create(
                model=self._model,
                messages=self._history,
                tools=TOOLS,
                stream=True,
                max_tokens=32768,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, 
            )

            for chunk in stream:
                # log.info(chunk)
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if delta.content:
                    assistant_text += delta.content
                    yield delta.content

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_raw:
                            tool_calls_raw[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function else "",
                                "arguments": "",
                            }
                        if tc.function:
                            if tc.function.name:
                                tool_calls_raw[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_raw[idx]["arguments"] += tc.function.arguments
                    log.info(f"Tool Call: {tool_calls_raw}")

            assistant_msg: dict = {"role": "assistant", "content": assistant_text}
            log.info(f"assistant_msg: {assistant_msg}")
            if not tool_calls_raw:
                self._history.append(assistant_msg)
                break

            assistant_msg["tool_calls"] = []
            tool_messages = []
            requires_second_turn = False

            for idx in sorted(tool_calls_raw):
                tc = tool_calls_raw[idx]
                tc_name = tc["name"]
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except Exception:
                    args = {}

                tc_id = tc["id"] or f"call_{idx}"
                assistant_msg["tool_calls"].append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc_name,
                        "arguments": tc["arguments"]
                    }
                })

                if tc_name == "look_at_camera":
                    requires_second_turn = True
                    log.info("LLM requested to look at camera.")
                    frame = get_frame_callback() if get_frame_callback else None

                    if frame is not None:
                        import cv2
                        import base64
                        height, width = frame.shape[:2]
                        if height > 720:
                            scale = 720.0 / height
                            new_width = int(width * scale)
                            frame = cv2.resize(frame, (new_width, 720), interpolation=cv2.INTER_AREA)

                        success, buffer = cv2.imencode('.jpg', frame)
                        if success:
                            b64_str = base64.b64encode(buffer).decode('utf-8')
                            tool_messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": [
                                    {"type": "text", "text": "这里是摄像头拍到的照片："},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
                                ]
                            })
                            continue
                    
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": "无法获取摄像头画面。"
                    })
                else:
                    cmd = self._tool_call_to_motion(tc_name, args)
                    if cmd:
                        log.info("Motion command: %s", cmd)
                        motion_queue.put(cmd)
                    
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": "action_completed"
                    })

            self._history.append(assistant_msg)
            for t_msg in tool_messages:
                self._history.append(t_msg)

            if not requires_second_turn:
                break

    @staticmethod
    def _tool_call_to_motion(name: str, args: dict) -> dict | None:
        """Convert a tool call into a motion queue item dict."""
        if name == "move_head":
            return {
                "type": "goto",
                "yaw": float(args.get("yaw_deg", 0.0)),
                "pitch": float(args.get("pitch_deg", 0.0)),
                "duration": float(args.get("duration", 0.5)),
            }
        if name == "nod":
            return {"type": "nod", "times": int(args.get("times", 1))}
        if name == "shake_head":
            return {"type": "shake", "times": int(args.get("times", 1))}
        if name == "play_emotion":
            return {"type": "emotion", "name": str(args.get("name", "happy"))}
        if name == "wiggle_antennas":
            return {"type": "antennas", "duration": float(args.get("duration", 1.0))}
        return None
