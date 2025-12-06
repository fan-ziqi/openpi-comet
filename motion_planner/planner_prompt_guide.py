"""
VLM will be trained with 4 types of data:
- video (2FPS, 64 frames, 352 px) + single question + subtask answer
- video (2FPS, 64 frames, 352 px) + single question + subtask answer detailed
- video (2FPS, 64 frames, 352 px) + history of subtask + subtask answer
- video (2FPS, 64 frames, 352 px) + history of subtask + subtask answer detailed
"""

import base64
import json
import re
import sys
import time
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image


def load_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def encode_file_to_data_url(path: str) -> tuple[str, str]:
    data = load_bytes(path)
    # Heuristically decide MIME from extension
    lower = path.lower()
    if lower.endswith(".mp4"):
        mime = "video/mp4"
        key = "video_url"
    elif lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        mime = "image/png" if lower.endswith(".png") else "image/jpeg"
        key = "image_url"
    else:
        raise ValueError("Unsupported file type. Use .mp4 or common image formats.")
    b64 = base64.b64encode(data).decode("utf-8")
    return key, f"data:{mime};base64,{b64}"


def run_inference(
    endpoint_base_url: str,
    api_key: str,
    model: str,
    user_text: str,
    media_paths: list[str] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    seed: int | None = None,
    timeout_seconds: int = 240,
    retries: int = 5,
):
    client = OpenAI(base_url=endpoint_base_url, api_key=api_key)
    client.timeout = timeout_seconds

    content = []
    if media_paths:
        for path in media_paths:
            key, data_url = encode_file_to_data_url(path)
            content.append({"type": key, key: {"url": data_url}})
    content.append({"type": "text", "text": user_text})

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": content,
        }
    ]
    messages_typed: list[ChatCompletionMessageParam] = cast(
        list[ChatCompletionMessageParam], messages
    )

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages_typed,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            text = resp.choices[0].message.content
            return text
        except Exception as e:
            last_error = e
            if attempt == retries:
                break
            wait_seconds = min(2 ** (attempt - 1), 30)
            print(
                f"Request failed (attempt {attempt}/{retries}): {e}. Retrying in {wait_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Inference failed after {retries} attempts: {last_error}")


def strip_think_tags(text: str) -> str:
    """
    Remove content within <think></think> tags from the text.
    Handles multiple scenarios:
    1. Proper <think>...</think> pairs - removes content between tags
    2. Only </think> tag - removes everything up to and including </think>
    3. No think tags - returns text as-is

    Args:
        text: Input text that may contain <think></think> tags

    Returns:
        Text with <think></think> content removed and cleaned up
    """
    # Check if there's a proper <think>...</think> pair
    if "<think>" in text.lower() and "</think>" in text.lower():
        # Remove everything between <think> and </think> tags (including the tags)
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    elif "</think>" in text.lower():
        # If there's only </think> without <think>, remove everything up to and including </think>
        cleaned_text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    else:
        # No think tags at all, use the text as is
        cleaned_text = text

    # Clean up extra whitespace
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text.strip())
    return cleaned_text


def generate_stylized_plan(
    list_of_plans: list[str], current_plan: str, current_plan_completed: bool = False
) -> str:
    """
    Generate a stylized high-level plan list with status markers.

    Rules:
    - All plans before the current plan are marked as [o].
    - If current_plan_completed is False:
        - Current plan is marked as [-].
        - All plans after the current plan are marked as [x].
    - If current_plan_completed is True:
        - Current plan is marked as [o].
        - The plan immediately after the current plan is marked as [-] (if it exists).
        - All remaining plans after that are marked as [x].

    Args:
        list_of_plans: Ordered list of plan step strings.
        current_plan: The plan string that is the current focus.
        current_plan_completed: Whether the current plan has been completed.

    Returns:
        A single string with one formatted plan per line.

    Raises:
        ValueError: If current_plan is not found in list_of_plans.
    """
    normalized_plans = [p.strip() for p in list_of_plans]
    target = current_plan.strip()
    try:
        current_index = next(i for i, p in enumerate(normalized_plans) if p == target)
    except StopIteration:
        raise ValueError(f"current_plan not found in list_of_plans: '{current_plan}'")

    lines: list[str] = []
    for i, plan in enumerate(list_of_plans):
        if i < current_index:
            prefix = "[o]"
        elif i == current_index:
            prefix = "[o]" if current_plan_completed else "[-]"
        elif current_plan_completed and i == current_index + 1:
            prefix = "[-]"
        else:
            prefix = "[x]"
        lines.append(f"{prefix} {plan}")
    return "\n".join(lines)


def generate_plan(task_description: str, image: Image.Image) -> str:
    user_prompt = f"""Given the task '{task_description}', break it down into several concrete high-level steps."""

    # Run VLM
    # NOTE: this takes image as input
    response = run_inference(endpoint_base_url, api_key, model, user_prompt, [image])

    # sample response
    response = """<think>The robot needs to first retrieve the pesticide atomizer from the floor before applying it to each tree. Each tree must receive a complete spraying cycle to ensure full trunk coverage.</think>

[
  "Retrieve the pesticide atomizer from the floor",
  "Apply pesticide to the first tree until trunk is fully covered",
  "Apply pesticide to the second tree until trunk is fully covered"
]"""
    plans = strip_think_tags(response)
    plans = json.loads(plans)
    current_plan = plans[0]
    plan_status = generate_stylized_plan(plans, current_plan, current_plan_completed=False)
    return plan_status


def generate_subtask(
    task_description: str, plan_status: str, subtask_history: list[str], video
) -> str:
    # NOTE: this is single round
    user_prompt = f"""You are given the task of '{task_description}'. The status of the plans are:
{plan_status}
Note that [-] indicates in progress. [o] indicates completed. [x] indicates not started.
The last high-level objective given to the robot was '{subtask_history[-1]}'.Based on your analysis, what should be the next high-level objective the robot should achieve?"""

    # Run VLM
    # NOTE: this takes video as input
    response = run_inference(endpoint_base_url, api_key, model, user_prompt, [video])

    # sample response
    response = """<think>The pesticide atomizer is currently not visible in the robot's field of view, so the robot must explore the garden floor to locate it before any movement toward it can occur. The exploration base motion action type confirms the immediate goal is to find the object's position.</think>
Move to the pesticide atomizer on the floor."""
    subtask = strip_think_tags(response)

    subtask_history.append(subtask)
    return subtask


def generate_action():
    # TODO: VLA
    return


def plan_critique(
    task_description: str, plan_status: str, subtask_history: list[str], video
) -> str:
    # NOTE: this is single round
    user_prompt = f"""You are given the task of '{task_description}'. The status of the plans are:
{plan_status}
Note that [-] indicates in progress. [o] indicates completed. [x] indicates not started.
The last high-level objective given to the robot was '{subtask_history[-1]}'. Based on your analysis, what are the updated status of the plans?"""

    # Run VLM
    # NOTE: this takes video as input
    response = run_inference(endpoint_base_url, api_key, model, user_prompt, [video])

    # sample response
    response = """<think>The robot is currently moving toward the pesticide atomizer on the ground, indicating it has not yet retrieved the device. Since the atomizer is still on the floor and no spraying has begun, the next logical step is to secure the atomizer before proceeding to treat the trees.</think>
[-] Retrieve the pesticide atomizer from the floor
[x] Apply pesticide to the first tree until trunk is fully covered
[x] Apply pesticide to the second tree until trunk is fully covered"""
    updated_plan_status = strip_think_tags(response)
    return updated_plan_status


def main():
    """
    Pseudo code for planning
    """
    task_description = "In the garden, pick up the pesticide atomizer on the floor and spray pesticide onto both trees until each tree trunk is fully covered."
    image = get_from_env()
    plan_status = generate_plan(task_description, image)

    subtask_history = []
    action_history = []
    while not get_termination_from_env():
        video = get_from_env()  # 0, t, 100

        # plan1:  xxxx,frame, text
        # plan2:  [xxxx     xxxxxxx      xxxxxxx      xxxxx     xxxxx] + text
        # plan3:  text + [x  x   x   x   x   x] 64
        # subtask -> nature language   skill: [skill1, skill2, ...]
        subtask = generate_subtask(task_description, plan_status, subtask_history, video)
        action = generate_action(subtask, action_history, video)
        updated_plan_status = plan_critique(task_description, plan_status, subtask_history, video)

        # reset subtask history if plan status is updated
        if updated_plan_status != plan_status:
            subtask_history = []
            action_history = []

        plan_status = updated_plan_status


if __name__ == "__main__":
    main()
