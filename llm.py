"""OpenAI-compatible LLM client with motion tool definitions."""

import json
import queue
from typing import Generator, Iterator

from openai import OpenAI

# ---------------------------------------------------------------------------
# System prompt — robot persona in Chinese
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是小白，一个活泼可爱的小型机器人。
你用简洁的中文回答问题，每次回复不超过三句话。
你会自然地使用动作工具来表达情感——比如点头表示同意、摇头表示否定、播放情绪动画表示高兴或惊讶。
请在合适的时候调用这些工具，让对话更加生动有趣。"""

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
                        "description": (
                            "Emotion name. Available options depend on the robot's "
                            "emotions library (e.g. 'happy', 'sad', 'surprised', "
                            "'curious', 'bored', 'excited')."
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
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "local",
        model: str = "local-model",
    ) -> None:
        """Initialise the LLM client.

        Args:
            base_url: Base URL of the OpenAI-compatible server.
            api_key: API key (any non-empty string works for local servers).
            model: Model name string to send in requests.
        """
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
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

        Args:
            user_text: Transcribed user utterance.
            motion_queue: Thread-safe queue for motion command dicts.

        Yields:
            Text tokens from the assistant response.
        """
        self._history.append({"role": "user", "content": user_text})

        # Accumulate assistant reply for history
        assistant_text = ""
        tool_calls_raw: dict[int, dict] = {}  # index → partial tool call

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

            # --- text content ---
            if delta.content:
                assistant_text += delta.content
                yield delta.content

            # --- tool calls (streamed incrementally) ---
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

        # Flush completed tool calls into motion queue
        for idx in sorted(tool_calls_raw):
            tc = tool_calls_raw[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            cmd = self._tool_call_to_motion(tc["name"], args)
            if cmd:
                motion_queue.put(cmd)

        # Store assistant turn in history
        self._history.append({"role": "assistant", "content": assistant_text})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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