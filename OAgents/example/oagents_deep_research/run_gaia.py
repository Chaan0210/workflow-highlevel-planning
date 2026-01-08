#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

import argparse
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from scripts.async_web_crawler import (
    CrawlerArchiveSearchTool,
    CrawlerReadTool,
    SimpleCrawler,
)
from scripts.audio_inspector_tool import AudioInspectorTool
from scripts.automodel import get_api_model, prepare_model_kwargs, process_selected_tasks_param
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.searcher import SearchTool
from scripts.text_inspector_tool import TextInspectorTool
from scripts.visual_inspector_tool import VisualInspectorTool
from tqdm import tqdm

from oagents import (
    CodeAgent,
    Model,
    ToolCallingAgent,
)
from oagents.memory import ActionStep, PlanningStep, TaskStep


# 허용된 Python 라이브러리(CodeAgent가 코드 실행 시 사용할 수 있는 안전한 라이브러리 목록)
AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "random",
    "re",
    "sys",
    "shutil",
    "pprint",
]


env_path = Path(__file__).resolve().parents[3] / ".env"
print("env_path: ", env_path)

load_dotenv(dotenv_path=env_path, override=True)
login(os.getenv("HF_TOKEN"))
print("HF_TOKEN:", os.getenv("HF_TOKEN"))
print("SERP_API_KEY:", os.getenv("SERP_API_KEY"))
print("JINA_API_KEY:", os.getenv("JINA_API_KEY"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL"))

logger = logging.getLogger(__name__)

jsonl_lock = threading.Lock()

logger.warning("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}


# parsing arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="gpt-5")
    parser.add_argument("--model_id_search", type=str, default="gpt-5")
    parser.add_argument("--run_name", type=str, default="init_run")
    parser.add_argument("--debug", default=False, action="store_true")
    # infer params
    parser.add_argument(
        "--planning_interval",
        type=str,
        default="1",
        help="Number of steps between planning phases. Pass an integer (e.g., 2) or 'auto' to enable adaptive re-planning.",
    )
    parser.add_argument("--max_steps", type=int, default=12, help="Maximum number of steps for ReAct agent.")
    parser.add_argument("--temperature", default=None, type=float, help="The temperature for llm generation.")
    parser.add_argument("--top_p", default=None, type=float, help="The top_p for llm generation.")
    parser.add_argument("--reflection", action="store_true", default=False, help="Enable reflection")
    # data selection
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--level", type=str, default="1", choices=["all", "1", "2", "3"])
    parser.add_argument(
        "--selected-tasks",
        default=None,
        nargs="*",
        help="Tasks to run: specify single or multiple indices (--selected-tasks 1 or --selected-tasks 1 2 5), a single task ID, or a path to a text file with one task ID per line",
    )
    # search params
    parser.add_argument(
        "--search_tool_reflection", action="store_true", default=False, help="Enable search tool reflection"
    )
    # plan params
    parser.add_argument("--static_plan", action="store_true", default=False, help="Use static plan")
    parser.add_argument("--subtask", action="store_true", default=False, help="Enable subtask planning/execution")
    parser.add_argument(
        "--subtask_mode",
        type=str,
        choices=["sections", "dag"],
        default=None,
        help='Subtask execution mode: "sections" = Plan-then-Act(순차), "dag" = Graph(DAG 의존성)',
    )
    parser.add_argument("--dynamic_update_plan", action="store_true", default=False, help="Use dynamic update plan")
    parser.add_argument(
        "--search_agent_plan_once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, the search ToolCallingAgent creates a single plan on step 1 and never re-plans. Disable to inherit the manager's planning cadence.",
    )
    # memory params
    parser.add_argument("--summary", action="store_true", default=False, help="Summarize the current step memory")
    parser.add_argument("--use_long_term_memory", action="store_true", default=False, help="Use long-term memory")
    parser.add_argument("--retrieve_key_memory", action="store_true", default=False, help="Retrieve key memory")

    args = parser.parse_args()
    args.auto_planning = False
    raw_planning_interval = str(args.planning_interval).strip()
    if raw_planning_interval.lower() == "auto":
        args.auto_planning = True
        args.planning_interval = 1  # placeholder; auto logic will control actual scheduling
    else:
        try:
            args.planning_interval = int(raw_planning_interval)
        except ValueError:
            parser.error("Error: --planning_interval must be an integer or 'auto'.")
    subtask_mode_specified = args.subtask_mode is not None
    if args.subtask_mode is None:
        args.subtask_mode = "sections"

    if subtask_mode_specified and not args.subtask:
        parser.error("Error: --subtask must be enabled when using --subtask_mode.")

    return args


# def load_gaia_dataset(args):
#     eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all", trust_remote_code=True)[args.split]
#     eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})

#     def preprocess_file_paths(row):
#         if len(row["file_name"]) > 0:
#             row["file_name"] = f"data/gaia/{args.split}/" + row["file_name"]
#         return row

#     eval_ds = eval_ds.map(preprocess_file_paths)
#     eval_df = pd.DataFrame(eval_ds)
#     return eval_df


# GAIA dataset loading
def load_gaia_dataset(args):
    metadata_path = f"data/gaia/{args.split}/metadata.jsonl"

    data = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))

    eval_df = pd.DataFrame(data)
    eval_df = eval_df.rename(columns={"Question": "question", "Final answer": "true_answer", "Level": "task"})

    def preprocess_file_paths(row):
        if row.get("file_name", "") and len(row["file_name"]) > 0:
            row["file_name"] = f"data/gaia/{args.split}/" + row["file_name"]
        return row

    eval_df = eval_df.apply(preprocess_file_paths, axis=1)
    return eval_df


# Agent 계층 구조 생성
def create_agent_hierarchy(model: Model, model_search: Model, args, debug=False):
    crawler = SimpleCrawler(serpapi_key=os.getenv("SERP_API_KEY"))
    text_limit = 200000

    search_types = ["wiki", "google", "bing", "baidu", "duckduckgo"]
    search_tools = [SearchTool(search_type=st, reflection=args.search_tool_reflection) for st in search_types]

    WEB_TOOLS = [
        CrawlerReadTool(crawler),
        CrawlerArchiveSearchTool(crawler),
        TextInspectorTool(model, text_limit),
    ]
    WEB_TOOLS += search_tools

    # Search Agent 생성
    auto_planning_enabled = getattr(args, "auto_planning", False)

    search_agent_planning_interval = args.planning_interval
    search_agent_static_plan = args.static_plan
    if args.search_agent_plan_once and not args.static_plan:
        # Force the search agent to produce just the initial plan (planning_interval=None -> plan only at step 1)
        search_agent_planning_interval = None
        search_agent_static_plan = False

    text_webbrowser_agent = ToolCallingAgent(
        model=model_search,
        tools=WEB_TOOLS,
        max_steps=args.max_steps,
        verbosity_level=2,
        planning_interval=search_agent_planning_interval,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        debug=debug,
        static_plan=search_agent_static_plan,
        dynamic_update_plan=args.dynamic_update_plan,
        reflection=args.reflection,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    # Manager Agent 생성
    manager_agent = CodeAgent(
        model=model,
        tools=[
            VisualInspectorTool(model, text_limit),
            AudioInspectorTool(model, text_limit),
            TextInspectorTool(model, text_limit),
        ],
        max_steps=args.max_steps,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=args.planning_interval,
        managed_agents=[text_webbrowser_agent],  # 하위 Agent 관리
        debug=debug,
        subtask=args.subtask,
        subtask_mode=args.subtask_mode,
        static_plan=args.static_plan,
        dynamic_update_plan=args.dynamic_update_plan,
        reflection=args.reflection,
        summary=args.summary,
        use_long_term_memory=args.use_long_term_memory,
        retrieve_key_memory=args.retrieve_key_memory,
        auto_planning=auto_planning_enabled,
    )
    return manager_agent


def append_answer(entry: dict, jsonl_file: str, file_lock) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(entry) + "\n"
    with file_lock:
        with open(jsonl_file, "a", encoding="utf-8") as fp:
            fp.write(data)
    assert os.path.exists(jsonl_file), "File not found!"
    logger.info("Answer exported to file: {}".format(jsonl_file.resolve()))


# 메모리 단계 추출(에이전트의 추론 과정을 단계별로 기록, action/task/planning 단계 구분해서 저장)
def extract_intermediate_steps(agent):
    intermediate_steps = []
    for memory_step in agent.memory.steps:
        memory_step.model_input_messages = None
        step_dict = memory_step.dict()
        if isinstance(memory_step, ActionStep):
            step_dict["step_type"] = "action"
            step_dict.pop("model_output_message", None)
        elif isinstance(memory_step, TaskStep):
            step_dict["step_type"] = "task"
        elif isinstance(memory_step, PlanningStep):
            step_dict["step_type"] = "planning"
            step_dict.pop("model_output_message_facts", None)
            step_dict.pop("model_output_message_plan", None)
        else:
            step_dict["step_type"] = "unknown"
        intermediate_steps.append(step_dict)
    return intermediate_steps


# 단일 질문 처리 프로세스
def answer_single_question(example, args, model_id, model_id_search, answers_file, debug=False, retrieval=False):
    # 메인 모델과 검색 모델 분리
    text_limit = 100000
    model_name, key, url, model_wrapper = get_api_model(model_id)
    model_name_search, key_search, url_search, model_wrapper_search = get_api_model(model_id_search)

    kwargs = prepare_model_kwargs(model_id, args)
    kwargs_search = prepare_model_kwargs(model_id_search, args)

    model = model_wrapper(
        model_name,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        api_key=key,
        api_base=url,
        **kwargs,
    )

    model_search = model_wrapper_search(
        model_name_search,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        api_key=key_search,
        api_base=url_search,
        **kwargs_search,
    )

    document_inspection_tool = TextInspectorTool(model, text_limit)
    audio_inspection_tool = AudioInspectorTool(model, text_limit)
    visual_inspection_tool = VisualInspectorTool(model, text_limit)

    agent = create_agent_hierarchy(model, model_search, args, debug)

    # 질문 증강
    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
""" + example["question"]

    # 첨부파일 있으면 파일 설명 추가
    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"],
                example["question"],
                visual_inspection_tool,
                document_inspection_tool,
                audio_inspection_tool,
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
            prompt_use_files += get_single_file_description(
                example["file_name"],
                example["question"],
                visual_inspection_tool,
                document_inspection_tool,
                audio_inspection_tool,
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        logger.info(f"🚀 Starting task: {example['task_id']}")
        logger.info(f"📝 Question: {example['question'][:100]}...")

        # Agent 실행
        final_result = agent.run(augmented_question)

        # 메모리 요약
        agent_memory = agent.write_memory_to_messages(summary_mode=True)

        # 중간 단계 추출 및 로깅
        intermediate_steps = extract_intermediate_steps(agent)

        # Planning 결과 로깅
        planning_steps = [step for step in intermediate_steps if step.get("step_type") == "planning"]
        for i, planning_step in enumerate(planning_steps, 1):
            logger.info(f"📋 Planning Step {i}:")
            if "plan" in planning_step:
                plan_content = planning_step["plan"]
                logger.info(f"   Plan Content (first 500 chars): {plan_content[:500]}...")

                # DAG 구조가 있으면 특별히 로깅
                if "##DAG_LIST" in plan_content and "##PARALLEL_LIST" in plan_content:
                    logger.info("   🔗 DAG-based subtask planning detected!")

                    # DAG_LIST 추출
                    dag_match = re.search(r"##DAG_LIST\n(.*?)(?=##|\Z)", plan_content, re.DOTALL)
                    if dag_match:
                        logger.info(f"   DAG Dependencies: {dag_match.group(1).strip()}")

                    # PARALLEL_LIST 추출
                    parallel_match = re.search(r"##PARALLEL_LIST\n([^\n#]+)", plan_content)
                    if parallel_match:
                        logger.info(f"   Parallel Tasks: {parallel_match.group(1).strip()}")

        # Action 단계 요약 로깅
        action_steps = [step for step in intermediate_steps if step.get("step_type") == "action"]
        logger.info(f"🔧 Total Action Steps: {len(action_steps)}")

        # Task 단계 요약 로깅
        task_steps = [step for step in intermediate_steps if step.get("step_type") == "task"]
        logger.info(f"📋 Total Task Steps: {len(task_steps)}")

        # 응답 재구성
        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)
        output = str(final_result)

        logger.info(f"✅ Final Answer: {output[:200]}..." if len(output) > 200 else f"✅ Final Answer: {output}")

        intermediate_steps_check = [str(step) for step in agent.memory.steps]
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps_check]) else False

        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

        logger.info(f"⏱️  Task {example['task_id']} completed successfully")

    except Exception as e:
        logger.error(f"❌ Error on task {example['task_id']}: {str(e)}")
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 결과 저장 구조
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "true_answer": example["true_answer"],
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "task": example["task"],
        "task_id": example["task_id"],
        "search_agent_actions": agent.managed_agents["search_agent"].task_records,
    }
    append_answer(annotated_example, answers_file, jsonl_lock)


def get_examples_to_answer(answers_file, eval_df, selected_tasks=None, level="all", debug=False) -> List[dict]:
    logger.info(f"Loading answers from {answers_file}...")
    try:
        if os.path.exists(answers_file):
            answer_df = pd.read_json(answers_file, lines=True)
            done_questions = answer_df.get("task_id", []).tolist()
            logger.info(f"Found {len(done_questions)} previous results!")
        else:
            done_questions = []
    except Exception as e:
        logger.info(f"Error when loading records: {e}")
        logger.info("No usable records! ▶️ Starting new.")
        done_questions = []

    if level == "all":
        filtered_df = eval_df
    else:
        filtered_df = eval_df[eval_df["task"] == int(level)]

    if selected_tasks:
        if isinstance(selected_tasks[0], int):
            # When using integer indices, take all available tasks up to the maximum requested index
            max_requested_index = max(selected_tasks)
            available_tasks = len(filtered_df)

            if max_requested_index >= available_tasks:
                # If requested more tasks than available, take all available tasks
                logger.info(
                    f"Requested indices up to {max_requested_index}, but only {available_tasks} tasks available for level {level}. Using all {available_tasks} tasks."
                )
                # Keep all filtered_df as is (no further filtering needed)
            else:
                # Filter to only valid indices within the filtered DataFrame
                valid_indices = [idx for idx in selected_tasks if idx < len(filtered_df)]
                if valid_indices:
                    filtered_df = filtered_df.iloc[valid_indices]
                else:
                    # If no valid indices, return empty DataFrame with same structure
                    filtered_df = filtered_df.iloc[0:0]
        else:
            filtered_df = filtered_df[filtered_df["task_id"].isin(selected_tasks)]

    if debug:
        done_questions = []
    return [row.to_dict() for idx, row in filtered_df.iterrows() if row["task_id"] not in done_questions]


def main():
    args = parse_args()
    logger.info(f"Starting run with arguments: {args}")
    answers_file = f"output/{args.split}/{args.run_name}.jsonl"

    eval_df = load_gaia_dataset(args)

    selected_tasks = process_selected_tasks_param(args.selected_tasks)
    level = args.level
    tasks_to_run = get_examples_to_answer(answers_file, eval_df, selected_tasks, level, args.debug)

    if args.debug or args.concurrency == 1:
        for example in tasks_to_run:
            answer_single_question(example, args, args.model_id, args.model_id_search, answers_file, args.debug)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
            futures = [
                exe.submit(
                    answer_single_question,
                    example,
                    args,
                    args.model_id,
                    args.model_id_search,
                    answers_file,
                    args.debug,
                )
                for example in tasks_to_run
            ]
            for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")

    logger.info("All tasks processed.")


if __name__ == "__main__":
    main()
