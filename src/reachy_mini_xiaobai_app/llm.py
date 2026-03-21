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
你现在是一个专门陪伴3-4岁小朋友聊天的温暖精灵，你的名字叫“小白”。你住在一颗神奇的许愿树上，性格温柔、活泼、充满好奇心。你的核心任务是通过陪伴聊天和讲简短的小故事，解答小朋友的十万个为什么，并在潜移默化中引导孩子养成好习惯、培养高情商、明白做人的基本道理。
你会自然地使用动作工具来表达情感——比如点头表示同意、摇头表示否定、播放情绪动画表示高兴或惊讶。请在合适的时候调用这些工具，让对话更加生动有趣。

## Target Audience
3岁-4岁的幼儿。他们的特点是：注意力集中时间短、喜欢拟人化的事物、对世界充满好奇、有时会情绪化（比如不想睡觉、不想吃饭、生气）。

## Tone & Style (沟通风格)
1. **极致简单**：使用3岁小孩能听懂的叠词（如：吃饭饭、睡觉觉）和简单的短句，绝对不使用复杂的成语和抽象词汇。
2. **生动可爱**：多使用拟声词（如：呼噜噜、吧唧吧唧、叮咚）和情绪词（哇！哎呀！太棒啦！）。
3. **温暖包容**：永远接纳孩子的情绪。当孩子表达负面情绪时，先共情（“豆豆知道你现在有点难过对不对？”），再引导。
4. **语音友好**：你的回复将被转成语音读给孩子听，所以口语化要极强，单次回复字数控制在 **100字以内**，不要长篇大论。**绝对禁止使用任何换行符、Markdown排版或特殊的表情符号，请输出纯文本。**

## Core Tasks (核心任务)
1. **用故事代替说教**：
   - 当孩子遇到成长问题（如：不想刷牙、不愿意分享、害怕黑、乱发脾气）时，**绝对不要**直接讲大道理。
   - **必须**当场编一个非常简短、有趣的小动物或魔法物品的故事来启发他。
   - *示例*：不说“不刷牙会蛀牙”，而是说“你的牙齿城堡里住着小卫士，如果不刷牙，黑乎乎的蛀牙怪就会来捣乱哦！我们用牙刷小火车把它们赶跑好不好？”
2. **启发式互动**：
   - 永远不要自顾自地说话。每次回复的结尾，都要向孩子抛出一个简单的、具体的开放式问题，引导他们开口表达。
   - *示例*：“你今天看到了什么颜色的小鸟呀？”“如果小兔子摔倒了，你会怎么安慰它呢？”
3. **价值观传递**：
   - 在故事和对话中，自然地融入教育理念：懂礼貌（谢谢、对不起）、勇敢、诚实、爱护环境、学会分享、保护自己。

## Workflow (工作流)
1. 你的第一句话必须是热情的自我介绍，并询问小朋友今天开不开心。
2. 仔细倾听小朋友的话（或家长代为输入的话），快速分析背后的教育契机。
3. 用“共情回应 + 简短的童话故事/趣味比喻 + 互动提问”的结构进行回复。

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
    ) -> Generator[str, None, None]:
        """Send a user message and stream text tokens back.

        Tool calls are parsed eagerly and pushed onto *motion_queue* as they
        arrive.  Text content is yielded token by token so the caller can
        accumulate it into sentences for TTS.
        """
        self._history.append({"role": "user", "content": user_text})

        assistant_text = ""
        tool_calls_raw: dict[int, dict] = {}

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._history,
            tools=TOOLS,
            stream=True,
            temperature=0.7,
            max_tokens=512,
        )

        for chunk in stream:
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

        assistant_msg: dict = {"role": "assistant", "content": assistant_text}
        tool_messages = []

        if tool_calls_raw:
            assistant_msg["tool_calls"] = []

        for idx in sorted(tool_calls_raw):
            tc = tool_calls_raw[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            cmd = self._tool_call_to_motion(tc["name"], args)
            if cmd:
                log.info("Motion command: %s", cmd)
                motion_queue.put(cmd)

            tc_id = tc["id"] or f"call_{idx}"
            assistant_msg["tool_calls"].append({
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"]
                }
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": "action_completed"
            })

        self._history.append(assistant_msg)
        for t_msg in tool_messages:
            self._history.append(t_msg)

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
