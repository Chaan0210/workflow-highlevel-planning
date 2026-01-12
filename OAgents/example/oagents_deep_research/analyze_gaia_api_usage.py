#!/usr/bin/env python
"""
Estimate GPT-5 API call usage from GAIA JSONL outputs.

Notes on how calls are counted (based on agent implementation):
1) PlanningStep uses two model calls (facts + plan).
2) ActionStep uses one model call in normal mode.
   - When the plan contains subtasks (##STx), CodeAgent executes each subtask with
     an additional model call. These show up as multiple tool_calls in one ActionStep.
   - Therefore, action model calls are counted as max(1, len(tool_calls)).
3) search_agent_actions store only non-final tool calls plus the final answer.
   - Minimum search-agent calls = tool_calls + (1 if answer exists).
   - Planning calls for the search agent are not persisted, so totals are lower bounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class UsageStats:
    path: Path
    tasks: int = 0
    manager_planning_steps: int = 0
    manager_action_steps: int = 0
    manager_action_tool_calls: int = 0
    manager_action_calls: int = 0
    manager_action_steps_no_tools: int = 0
    search_tasks: int = 0
    search_tool_calls: int = 0
    search_answers: int = 0

    @property
    def manager_planning_calls(self) -> int:
        return self.manager_planning_steps * 2

    @property
    def manager_total_calls(self) -> int:
        return self.manager_planning_calls + self.manager_action_calls

    @property
    def search_total_calls_min(self) -> int:
        return self.search_tool_calls + self.search_answers

    @property
    def total_calls_min(self) -> int:
        return self.manager_total_calls + self.search_total_calls_min


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] {path.name}:{line_no} invalid JSON ({exc})", file=sys.stderr)


def count_action_calls(step: Dict[str, Any]) -> int:
    tool_calls = step.get("tool_calls")
    if isinstance(tool_calls, list):
        return max(1, len(tool_calls))
    return 1


def count_action_tool_calls(step: Dict[str, Any]) -> int:
    tool_calls = step.get("tool_calls")
    if isinstance(tool_calls, list):
        return len(tool_calls)
    return 0


def count_search_calls(actions: Dict[str, Any]) -> tuple[int, int, int]:
    if not isinstance(actions, dict):
        return 0, 0, 0
    search_tasks = len(actions)
    tool_calls = 0
    answers = 0
    for entry in actions.values():
        if not isinstance(entry, dict):
            continue
        entry_tool_calls = entry.get("tool_calls")
        if isinstance(entry_tool_calls, list):
            tool_calls += len(entry_tool_calls)
        if entry.get("answer") is not None:
            answers += 1
    return search_tasks, tool_calls, answers


def analyze_file(path: Path) -> UsageStats:
    stats = UsageStats(path=path)
    for record in iter_jsonl(path):
        stats.tasks += 1
        steps = record.get("intermediate_steps") or []
        if not isinstance(steps, list):
            steps = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_type = step.get("step_type")
            if step_type == "planning":
                stats.manager_planning_steps += 1
            elif step_type == "action":
                stats.manager_action_steps += 1
                stats.manager_action_calls += count_action_calls(step)
                tool_calls = count_action_tool_calls(step)
                stats.manager_action_tool_calls += tool_calls
                if tool_calls == 0:
                    stats.manager_action_steps_no_tools += 1
        search_tasks, search_tool_calls, search_answers = count_search_calls(record.get("search_agent_actions") or {})
        stats.search_tasks += search_tasks
        stats.search_tool_calls += search_tool_calls
        stats.search_answers += search_answers
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate GPT-5 API call usage from GAIA JSONL results.")
    parser.add_argument(
        "--input",
        action="append",
        help="Path to a JSONL file. Repeat for multiple files.",
    )
    parser.add_argument(
        "--glob",
        default="output/validation/gpt5-*.jsonl",
        help="Glob to load when --input is not provided.",
    )
    return parser.parse_args()


def format_stats(stats: UsageStats) -> List[str]:
    manager_calls_per_task = stats.manager_total_calls / stats.tasks if stats.tasks else 0.0
    total_calls_min_per_task = stats.total_calls_min / stats.tasks if stats.tasks else 0.0
    search_calls_min_per_task = stats.search_total_calls_min / stats.tasks if stats.tasks else 0.0
    return [
        f"File: {stats.path}",
        f"  Tasks: {stats.tasks}",
        f"  Manager calls: {stats.manager_total_calls} (planning={stats.manager_planning_calls}, action={stats.manager_action_calls})",
        f"  Manager action steps: {stats.manager_action_steps} (tool_calls={stats.manager_action_tool_calls}, no_tools={stats.manager_action_steps_no_tools})",
        f"  Search calls (min): {stats.search_total_calls_min} (tasks={stats.search_tasks}, tool_calls={stats.search_tool_calls}, answers={stats.search_answers})",
        f"  Calls per task: manager={manager_calls_per_task:.2f}, search_min={search_calls_min_per_task:.2f}, total_min={total_calls_min_per_task:.2f}",
    ]


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.input:
        paths = [Path(p).expanduser().resolve() for p in args.input]
    else:
        paths = sorted(script_dir.glob(args.glob))
    if not paths:
        print("No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    totals = UsageStats(path=Path("<all>"))
    for path in paths:
        stats = analyze_file(path)
        for line in format_stats(stats):
            print(line)
        print()

        totals.tasks += stats.tasks
        totals.manager_planning_steps += stats.manager_planning_steps
        totals.manager_action_steps += stats.manager_action_steps
        totals.manager_action_tool_calls += stats.manager_action_tool_calls
        totals.manager_action_calls += stats.manager_action_calls
        totals.manager_action_steps_no_tools += stats.manager_action_steps_no_tools
        totals.search_tasks += stats.search_tasks
        totals.search_tool_calls += stats.search_tool_calls
        totals.search_answers += stats.search_answers

    if len(paths) > 1:
        print("Overall:")
        for line in format_stats(totals):
            print(line)


if __name__ == "__main__":
    main()
