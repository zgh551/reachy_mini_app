from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.200.252:30000/v1",
    api_key="not-needed",
)

SYSTEM_PROMPT = """你是小白，一个活泼可爱的小型机器人。
你用简洁的中文回答问题，每次回复不超过三句话。
你会自然地使用动作工具来表达情感——比如点头表示同意、摇头表示否定、播放情绪动画表示高兴或惊讶。
请在合适的时候调用这些工具，让对话更加生动有趣。"""

_history = [{"role": "system", "content": SYSTEM_PROMPT}]
_history.append({"role": "user", "content": "讲个小白兔的故事"})

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

stream = client.chat.completions.create(
    model="./Qwen3-ASR-1.7B/",
    messages=_history,
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

is_thinking_done = False
for chunk in stream:
    # print(chunk)
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta is None:
        continue

    # 2. 打印正式回答 (content)
    if delta.content:
        # 当第一次遇到 content 时，说明思考结束，换行打印正式回答
        if not is_thinking_done:
            print("\n\n【正式回答】:", end="")
            is_thinking_done = True
            
        print(delta.content, end="", flush=True)
    # --- text content ---
    # if delta.content:
    #     assistant_text += delta.content
    #     yield delta.content

    # # --- tool calls (streamed incrementally) ---
    # if delta.tool_calls:
    #     for tc in delta.tool_calls:
    #         idx = tc.index
    #         if idx not in tool_calls_raw:
    #             tool_calls_raw[idx] = {
    #                 "id": tc.id or "",
    #                 "name": tc.function.name if tc.function else "",
    #                 "arguments": "",
    #             }
    #         if tc.function:
    #             if tc.function.name:
    #                 tool_calls_raw[idx]["name"] = tc.function.name
    #             if tc.function.arguments:
    #                 tool_calls_raw[idx]["arguments"] += tc.function.arguments