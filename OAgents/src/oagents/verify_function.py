#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI


PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(PARENT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

KEY = os.getenv("OPENAI_API_KEY")
URL = os.getenv("OPENAI_BASE_URL")
# Step-evaluation judge model; override with VERIFY_MODEL to align it with the
# experiment model instead of the default.
MODEL = "gpt-4.1"


def _judge_model() -> str:
    return os.getenv("VERIFY_MODEL") or MODEL


def _judge_request_kwargs(model: str) -> dict:
    # Reasoning models reject `temperature` and require `max_completion_tokens`.
    if any(tag in model for tag in ("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": 2048}
    return {"max_tokens": 2048, "temperature": 0.0}

client = None
async_client = None

if KEY:
    client = OpenAI(api_key=KEY, base_url=URL, timeout=600.0, max_retries=3)
    async_client = AsyncOpenAI(api_key=KEY, base_url=URL, timeout=600.0)


def extract_json_objects(text: str) -> list:
    matches = []
    stack = []
    start_index = None

    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start_index = i
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start_index is not None:
                    matches.append(text[start_index : i + 1])
                    start_index = None
    return matches


def clean_and_parse_json(json_str: str) -> Optional[Dict]:
    try:
        json_str = re.sub(r",\s*\}", "}", json_str)
        json_str = re.sub(r'(?<=\{)\s*([^":]+?)\s*:', r'"\1":', json_str)
        json_str = re.sub(r"\}\s*$", "}", json_str)
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def parse_first_valid_json(text: str) -> dict:
    for match in extract_json_objects(text):
        cleaned = clean_and_parse_json(match)
        if cleaned:
            return cleaned
    return {}


async def async_evaluate_answer(text: str, system_prompt: str, mode: str = "PRM") -> Any:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    model = _judge_model()

    try:
        if async_client is None:
            raise RuntimeError("OPENAI_API_KEY is not set; step-evaluation judge is unavailable.")
        response = await async_client.chat.completions.create(
            model=model,
            messages=messages,
            **_judge_request_kwargs(model),
        )
        content = response.choices[0].message.content

        if mode in ("ORM", "PRM"):
            result = parse_first_valid_json(content)
            return result.get("score", 0), result.get("analysis", ""), result.get("should_replan", False)

        elif mode in ("ORM-list-wise", "PRM-list-wise"):
            result = parse_first_valid_json(content)
            return result.get("analysis", ""), result.get("index", 0)

        elif mode == "reflection":
            result = parse_first_valid_json(content)
            return result

        else:
            return content

    except Exception as e:
        print(f"Async Error processing item: {e}")
        return None


def evaluate_answer(text: str, system_prompt: str, mode: str = "PRM") -> Any:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    model = _judge_model()

    try:
        if client is None:
            raise RuntimeError("OPENAI_API_KEY is not set; step-evaluation judge is unavailable.")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **_judge_request_kwargs(model),
        )
        content = response.choices[0].message.content

        if mode in ("ORM", "PRM"):
            result = parse_first_valid_json(content)
            return result.get("score", 0), result.get("analysis", ""), result.get("should_replan", False)
        elif mode in ("ORM-list-wise", "PRM-list-wise"):
            result = parse_first_valid_json(content)
            return result.get("analysis", ""), result.get("index", 0)
        elif mode == "reflection":
            result = parse_first_valid_json(content)
            return result

        else:
            return content

    except Exception as e:
        # A silent no-op judge would invisibly disable adaptive re-planning, so shout.
        print(f"[verify_function] evaluate_answer failed (model={model}, mode={mode}): {e}")
        return None
