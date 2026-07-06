#!/usr/bin/env python
"""
Analyze GAIA benchmark runs produced by run_gaia.py.

The script ingests one or more JSONL result files, computes detailed per-task metrics
such as tool usage, planning cadence, search agent behavior, error categories, and
accuracy, then prints comparative summaries across different planning strategies.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import textwrap
from collections import Counter
from itertools import combinations
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from scripts.gaia_scorer import check_close_call, question_scorer  # noqa: E402


def parse_run_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Run spec must look like label=path/to/file.jsonl")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty.")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Result file not found: {path}")
    return label, path


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    # Append-mode resume can write several rows per task; keep the latest attempt.
    deduped: Dict[Any, dict] = {}
    for index, record in enumerate(records):
        deduped[record.get("task_id", f"__row_{index}")] = record
    if len(deduped) != len(records):
        print(f"  (dropped {len(records) - len(deduped)} duplicate task_id rows, keeping the latest)")
    return list(deduped.values())


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on discordant pair counts (binomial, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def parse_timestamp(timestamp: Any) -> datetime | None:
    if not timestamp:
        return None
    if isinstance(timestamp, datetime):
        return timestamp
    ts = str(timestamp).strip()
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
    return None


def shorten(text: str | None, width: int = 120) -> str:
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    if not collapsed:
        return ""
    return textwrap.shorten(collapsed, width=width, placeholder="…")


def normalize_level(level: Any) -> Any:
    try:
        if isinstance(level, float) and level.is_integer():
            return int(level)
        return int(str(level))
    except (ValueError, TypeError):
        return level


def detect_plan_mode(planning_steps: List[dict]) -> Tuple[str, bool, bool, bool]:
    if not planning_steps:
        return "reactive", False, False, False
    joined_plan = "\n".join(
        step.get("plan", "") for step in planning_steps if isinstance(step, dict) and step.get("plan")
    )
    joined_plan = joined_plan or ""
    has_dag = "##DAG_LIST" in joined_plan or "##PARALLEL_LIST" in joined_plan and "##DAG_LIST" in joined_plan
    has_parallel = "##PARALLEL_LIST" in joined_plan
    has_sections = "##ST" in joined_plan or "##SECTION" in joined_plan
    if has_dag:
        mode = "dag"
    elif has_sections:
        mode = "sections"
    else:
        mode = "linear"
    return mode, has_dag, has_parallel, has_sections


def extract_dag_edges(planning_steps: List[dict]) -> List[Tuple[str, str]]:
    """Parse ##DAG_LIST edges from the latest plan text that carries one."""
    edges: List[Tuple[str, str]] = []
    for step in planning_steps:
        plan = step.get("plan") or "" if isinstance(step, dict) else ""
        match = re.search(r"##DAG_LIST\n(.*?)(?=##|\Z)", plan, flags=re.DOTALL)
        if not match:
            continue
        block = re.sub(r"(?<!['\"])(ST\d+)(?!['\"])", r"'\1'", match.group(1).strip())
        try:
            parsed = ast.literal_eval(block)
            edges = [(str(a).strip(), str(b).strip()) for (a, b) in parsed]
        except Exception:
            continue
    return edges


def dag_is_chain(edges: List[Tuple[str, str]]) -> bool | None:
    """True when the dependency graph is a pure chain (every node has at most one
    predecessor and one successor) — i.e. behaviourally equivalent to sections."""
    if not edges:
        return None
    indegree: Counter = Counter()
    outdegree: Counter = Counter()
    for a, b in edges:
        outdegree[a] += 1
        indegree[b] += 1
    return max(outdegree.values()) <= 1 and max(indegree.values()) <= 1


def extract_tool_name(call: dict) -> str:
    if not isinstance(call, dict):
        return ""
    if "function" in call and isinstance(call["function"], dict):
        return str(call["function"].get("name") or "")
    if "name" in call:
        return str(call.get("name") or "")
    return ""


def analyze_record(record: dict, run_label: str) -> dict:
    steps = record.get("intermediate_steps") or []
    if not isinstance(steps, list):
        steps = []
    planning_steps = [step for step in steps if isinstance(step, dict) and step.get("step_type") == "planning"]
    action_steps = [step for step in steps if isinstance(step, dict) and step.get("step_type") == "action"]
    plan_mode, has_dag, has_parallel, has_sections = detect_plan_mode(planning_steps)

    manager_counter = Counter()
    search_counter = Counter()
    action_duration = 0.0
    manager_error_steps = 0
    subtask_failures = 0
    subtask_success = 0

    for step in action_steps:
        for call in step.get("tool_calls") or []:
            tool_name = extract_tool_name(call)
            if tool_name:
                manager_counter[tool_name] += 1
        duration = step.get("duration")
        if isinstance(duration, (int, float)):
            action_duration += float(duration)
        if step.get("error"):
            manager_error_steps += 1
        action_output = step.get("action_output")
        if isinstance(action_output, list):
            for entry in action_output:
                if not isinstance(entry, dict):
                    continue
                status = entry.get("status")
                if status == "failed":
                    subtask_failures += 1
                elif status == "succeeded":
                    subtask_success += 1

    search_actions = record.get("search_agent_actions") or {}
    if not isinstance(search_actions, dict):
        search_actions = {}
    search_tasks_invoked = len(search_actions)
    search_answers = 0
    for entry in search_actions.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("answer"):
            search_answers += 1
        for call in entry.get("tool_calls") or []:
            name = extract_tool_name(call) or call.get("name")
            if name:
                search_counter[str(name)] += 1

    prediction = record.get("prediction")
    true_answer = record.get("true_answer", "")
    if prediction is None:
        correct = False
        close_call = False
        prediction_text = None
    else:
        prediction_text = str(prediction)
        true_text = str(true_answer)
        try:
            correct = bool(question_scorer(prediction_text, true_text))
        except Exception:
            correct = False
        close_call = bool(check_close_call(prediction_text, true_text, correct) and not correct)

    level = normalize_level(record.get("task"))
    duration_seconds = None
    start_dt = parse_timestamp(record.get("start_time"))
    end_dt = parse_timestamp(record.get("end_time"))
    if start_dt and end_dt:
        duration_seconds = max((end_dt - start_dt).total_seconds(), 0.0)
    elif action_duration:
        duration_seconds = action_duration

    planning_steps_count = len(planning_steps)
    replan_count = max(planning_steps_count - 1, 0)
    tool_calls_total = sum(manager_counter.values())
    search_tool_calls_total = sum(search_counter.values())
    subtask_attempts = subtask_failures + subtask_success

    parsing_error = bool(record.get("parsing_error"))
    iteration_limit = bool(record.get("iteration_limit_exceeded"))
    agent_error = bool(record.get("agent_error"))

    # Fields recorded by run_gaia.py since the fidelity fix (absent in old files).
    token_usage = record.get("token_usage") or {}

    def token_count(agent_key: str, kind: str):
        value = (token_usage.get(agent_key) or {}).get(kind)
        return value if isinstance(value, (int, float)) else None

    token_parts = [
        token_count("manager", "input"),
        token_count("manager", "output"),
        token_count("search_agent", "input"),
        token_count("search_agent", "output"),
    ]
    total_tokens = sum(part for part in token_parts if part is not None) if any(
        part is not None for part in token_parts
    ) else None

    subtask_recs = [entry for entry in (record.get("subtask_records") or []) if isinstance(entry, dict)]
    subtask_rec_failed = sum(1 for entry in subtask_recs if entry.get("status") == "failed")
    parallel_batches = {
        tuple(entry.get("parallel_batch") or [])
        for entry in subtask_recs
        if len(entry.get("parallel_batch") or []) > 1
    }
    dag_edges = extract_dag_edges(planning_steps)

    if correct:
        error_category = "correct"
    elif agent_error:
        error_category = "agent_error"
    elif iteration_limit:
        error_category = "iteration_limit"
    elif parsing_error:
        error_category = "parsing_error"
    elif prediction is None:
        error_category = "no_prediction"
    else:
        error_category = "wrong_answer"

    return {
        "run_label": run_label,
        "task_id": record.get("task_id"),
        "level": level,
        "question": record.get("question"),
        "question_short": shorten(record.get("question")),
        "prediction": prediction,
        "prediction_short": shorten(prediction_text) if prediction_text else None,
        "true_answer": true_answer,
        "agent_name": record.get("agent_name"),
        "start_time": record.get("start_time"),
        "end_time": record.get("end_time"),
        "duration_seconds": duration_seconds,
        "action_duration_seconds": action_duration if action_duration else None,
        "action_steps": len(action_steps),
        "planning_steps": planning_steps_count,
        "replan_count": replan_count,
        "used_replanning": replan_count > 0,
        "plan_mode": plan_mode,
        "plan_contains_dag": has_dag,
        "plan_contains_parallel": has_parallel,
        "plan_contains_sections": has_sections,
        "tool_calls_total": tool_calls_total,
        "tool_calls_per_step": (tool_calls_total / len(action_steps)) if action_steps else 0.0,
        "tool_counter": manager_counter,
        "search_tool_counter": search_counter,
        "search_tool_calls_total": search_tool_calls_total,
        "search_tasks_invoked": search_tasks_invoked,
        "search_answers": search_answers,
        "search_calls_per_task": (search_tool_calls_total / search_tasks_invoked) if search_tasks_invoked else 0.0,
        "used_search_agent": search_tasks_invoked > 0,
        "subtask_failure_count": subtask_failures,
        "subtask_success_count": subtask_success,
        "subtask_attempts": subtask_attempts,
        "subtask_failure_rate": (subtask_failures / subtask_attempts) if subtask_attempts else 0.0,
        "action_error_steps": manager_error_steps,
        "manager_error_rate": (manager_error_steps / len(action_steps)) if action_steps else 0.0,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit,
        "agent_error": agent_error,
        "agent_error_message": record.get("agent_error"),
        "error_category": error_category,
        "correct": correct,
        "close_call": close_call,
        "recorded_planning_mode": record.get("planning_mode"),
        "recorded_replan_count": record.get("replan_count"),
        "plan_parse_failures": record.get("plan_parse_failures"),
        "manager_input_tokens": token_count("manager", "input"),
        "manager_output_tokens": token_count("manager", "output"),
        "search_input_tokens": token_count("search_agent", "input"),
        "search_output_tokens": token_count("search_agent", "output"),
        "total_tokens": total_tokens,
        "subtask_rec_total": len(subtask_recs),
        "subtask_rec_failed": subtask_rec_failed,
        "used_parallel_batch": bool(parallel_batches),
        "parallel_batch_count": len(parallel_batches),
        "dag_edge_count": len(dag_edges),
        "dag_is_chain": dag_is_chain(dag_edges),
        "git_commit": record.get("git_commit"),
    }


def build_run_summary(rows: List[dict], run_label: str, run_path: Path) -> Tuple[dict, pd.DataFrame]:
    total_tool_counter = Counter()
    total_search_counter = Counter()
    for row in rows:
        total_tool_counter.update(row["tool_counter"])
        total_search_counter.update(row["search_tool_counter"])
        row["tool_calls_breakdown"] = dict(row["tool_counter"])
        row["search_tool_calls_breakdown"] = dict(row["search_tool_counter"])
        row.pop("tool_counter", None)
        row.pop("search_tool_counter", None)

    df = pd.DataFrame(rows)
    duration_series = pd.to_numeric(df.get("duration_seconds"), errors="coerce")
    level_breakdown: Dict[Any, dict] = {}
    if not df.empty and "level" in df:
        for level_value, group in df.groupby("level"):
            level_breakdown[level_value] = {
                "tasks": int(len(group)),
                "accuracy": float(group["correct"].mean()) if not group.empty else 0.0,
                "avg_steps": float(group["action_steps"].mean()) if not group.empty else 0.0,
                "avg_replans": float(group["replan_count"].mean()) if not group.empty else 0.0,
            }

    failure_breakdown = df["error_category"].value_counts().to_dict() if "error_category" in df else {}

    def numeric_series(column: str) -> pd.Series:
        if column not in df:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[column], errors="coerce").dropna()

    recorded_replans = numeric_series("recorded_replan_count")
    parse_failures = numeric_series("plan_parse_failures")
    total_tokens = numeric_series("total_tokens")
    dag_rows = df[df["dag_edge_count"] > 0] if "dag_edge_count" in df else pd.DataFrame()
    dag_chain_flags = dag_rows["dag_is_chain"].dropna() if not dag_rows.empty else pd.Series(dtype=object)

    summary = {
        "run_label": run_label,
        "file": str(run_path),
        "num_tasks": int(len(df)),
        "levels": sorted({lvl for lvl in df.get("level", []) if pd.notna(lvl)}),
        "accuracy": float(df["correct"].mean()) if not df.empty else 0.0,
        "close_call_rate": float(df["close_call"].mean()) if not df.empty else 0.0,
        "avg_duration": float(duration_series.mean()) if not duration_series.empty else 0.0,
        "median_duration": float(duration_series.median()) if not duration_series.empty else 0.0,
        "avg_action_steps": float(df["action_steps"].mean()) if not df.empty else 0.0,
        "avg_planning_steps": float(df["planning_steps"].mean()) if not df.empty else 0.0,
        "avg_replans": float(df["replan_count"].mean()) if not df.empty else 0.0,
        "replanned_pct": float(df["used_replanning"].mean()) if not df.empty else 0.0,
        "avg_tool_calls": float(df["tool_calls_total"].mean()) if not df.empty else 0.0,
        "avg_tool_density": float(df["tool_calls_per_step"].mean()) if not df.empty else 0.0,
        "avg_search_calls": float(df["search_tool_calls_total"].mean()) if not df.empty else 0.0,
        "avg_search_tasks": float(df["search_tasks_invoked"].mean()) if not df.empty else 0.0,
        "search_usage_rate": float(df["used_search_agent"].mean()) if not df.empty else 0.0,
        "manager_error_rate": float(df["manager_error_rate"].mean()) if not df.empty else 0.0,
        "subtask_usage_rate": float((df["subtask_attempts"] > 0).mean()) if not df.empty else 0.0,
        "subtask_failure_rate": float(
            df.loc[df["subtask_attempts"] > 0, "subtask_failure_rate"].mean()
        )
        if not df.loc[df["subtask_attempts"] > 0, "subtask_failure_rate"].empty
        else 0.0,
        "tool_call_counter": total_tool_counter,
        "search_tool_call_counter": total_search_counter,
        "level_breakdown": level_breakdown,
        "failure_breakdown": failure_breakdown,
        "wrong_task_ids": df.loc[~df["correct"], "task_id"].dropna().tolist(),
        "auto_plan_share": float((df["plan_mode"] == "dag").mean()) if "plan_mode" in df else 0.0,
        # Manipulation-check aggregates (None when the fields are absent, e.g. old files).
        "recorded_modes": (
            df["recorded_planning_mode"].dropna().value_counts().to_dict()
            if "recorded_planning_mode" in df
            else {}
        ),
        "git_commits": sorted(df["git_commit"].dropna().unique().tolist()) if "git_commit" in df else [],
        "avg_recorded_replans": float(recorded_replans.mean()) if not recorded_replans.empty else None,
        "recorded_replanned_pct": float((recorded_replans > 0).mean()) if not recorded_replans.empty else None,
        "plan_parse_failure_rate": float((parse_failures > 0).mean()) if not parse_failures.empty else None,
        "avg_total_tokens": float(total_tokens.mean()) if not total_tokens.empty else None,
        "parallel_batch_rate": float(df["used_parallel_batch"].mean()) if "used_parallel_batch" in df else None,
        "dag_planned_tasks": int(len(dag_rows)),
        "dag_chain_rate": float(dag_chain_flags.astype(bool).mean()) if not dag_chain_flags.empty else None,
    }
    return summary, df


def format_counter(counter: Counter, top_n: int) -> str:
    if not counter:
        return "n/a"
    return ", ".join(f"{name}={count}" for name, count in counter.most_common(top_n))


def print_incorrect_tasks(df: pd.DataFrame, limit: int) -> None:
    incorrect = df[~df["correct"]].copy()
    if incorrect.empty:
        print("No incorrect tasks for this run.")
        return
    incorrect = incorrect.sort_values(by=["level", "task_id"])
    incorrect = incorrect.assign(duration_min=incorrect["duration_seconds"] / 60 if "duration_seconds" in incorrect else 0)
    cols = [
        "task_id",
        "level",
        "question_short",
        "true_answer",
        "prediction_short",
        "error_category",
        "replan_count",
        "tool_calls_total",
        "duration_min",
    ]
    existing_cols = [col for col in cols if col in incorrect.columns]
    print(incorrect[existing_cols].head(limit).to_string(index=False))


def print_run_summary(summary: dict, df: pd.DataFrame, top_tools: int, incorrect_limit: int) -> None:
    print("=" * 100)
    print(f"Run: {summary['run_label']} ({summary['num_tasks']} tasks)")
    print(f"File: {summary['file']}")
    print(
        f"Accuracy: {summary['accuracy']*100:.2f}% | Close-call rate: {summary['close_call_rate']*100:.2f}% "
        f"| Avg duration: {summary['avg_duration']/60:.2f} min | Median duration: {summary['median_duration']/60:.2f} min"
    )
    print(
        f"Avg steps: {summary['avg_action_steps']:.2f} | Avg planning steps: {summary['avg_planning_steps']:.2f} "
        f"| Avg re-plans: {summary['avg_replans']:.2f} (triggered on {summary['replanned_pct']*100:.1f}% of tasks)"
    )
    print(
        f"Avg tool calls: {summary['avg_tool_calls']:.2f} ({summary['avg_tool_density']:.2f} per step) "
        f"| Avg search tool calls: {summary['avg_search_calls']:.2f} "
        f"(search used in {summary['search_usage_rate']*100:.1f}% of tasks)"
    )
    if summary["level_breakdown"]:
        level_rows = []
        for lvl, stats in summary["level_breakdown"].items():
            level_rows.append(
                {
                    "Level": lvl,
                    "Tasks": stats["tasks"],
                    "Accuracy (%)": round(stats["accuracy"] * 100, 2),
                    "Avg Steps": round(stats["avg_steps"], 2),
                    "Avg Replans": round(stats["avg_replans"], 2),
                }
            )
        print("\nLevel breakdown:")
        print(pd.DataFrame(level_rows).to_string(index=False))
    def fmt(value, pct=False):
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}%" if pct else f"{value:.2f}"

    print("\nManipulation checks:")
    commits = summary.get("git_commits") or []
    print(
        f" - recorded mode(s): {summary.get('recorded_modes') or 'n/a'} "
        f"| commit(s): {[c[:8] for c in commits] or 'n/a'}"
    )
    if len(commits) > 1:
        print(" - ⚠️  MIXED COMMITS inside one result file — rows are not comparable!")
    print(
        f" - recorded re-plans: avg {fmt(summary['avg_recorded_replans'])} "
        f"(fired on {fmt(summary['recorded_replanned_pct'], pct=True)} of tasks) "
        f"| plan parse failures on {fmt(summary['plan_parse_failure_rate'], pct=True)} of tasks"
    )
    if summary["dag_planned_tasks"]:
        print(
            f" - dag structure: {summary['dag_planned_tasks']} task(s) with edges, "
            f"chain-shaped {fmt(summary['dag_chain_rate'], pct=True)} "
            f"(a high chain rate means dag ≈ sections in practice) "
            f"| parallel batches used on {fmt(summary['parallel_batch_rate'], pct=True)} of tasks"
        )
    tokens = summary["avg_total_tokens"]
    print(f" - avg tokens/task (manager+search, in+out): {'n/a' if tokens is None else format(tokens, ',.0f')}")

    print("\nTop manager tools:", format_counter(summary["tool_call_counter"], top_tools))
    print("Top search tools:", format_counter(summary["search_tool_call_counter"], top_tools))
    if summary["failure_breakdown"]:
        print("\nFailure breakdown:")
        for reason, count in summary["failure_breakdown"].items():
            print(f" - {reason}: {count}")
    print("\nIncorrect tasks (sample):")
    print_incorrect_tasks(df, incorrect_limit)
    print()


def build_summary_table(summaries: List[dict]) -> None:
    rows = []
    for summary in summaries:
        correct = int(round(summary["accuracy"] * summary["num_tasks"]))
        low, high = wilson_ci(correct, summary["num_tasks"])
        rows.append(
            {
                "Run": summary["run_label"],
                "Tasks": summary["num_tasks"],
                "Levels": ",".join(str(lvl) for lvl in summary["levels"]) or "-",
                "Accuracy (%)": round(summary["accuracy"] * 100, 2),
                "95% CI (%)": f"[{low * 100:.1f}, {high * 100:.1f}]",
                "Avg Steps": round(summary["avg_action_steps"], 2),
                "Avg Replans": round(summary["avg_replans"], 2),
                "Avg Tool Calls": round(summary["avg_tool_calls"], 2),
                "Search Usage (%)": round(summary["search_usage_rate"] * 100, 1),
            }
        )
    print("Overall summary:")
    print(pd.DataFrame(rows).to_string(index=False))
    print()


def print_significance_tests(all_df: pd.DataFrame, run_labels: List[str]) -> None:
    """Paired McNemar exact test for every pair of runs, on their shared tasks."""
    if len(run_labels) < 2 or all_df.empty:
        return
    pivot = all_df.pivot_table(index="task_id", columns="run_label", values="correct", aggfunc="first")
    rows = []
    for label_a, label_b in combinations([lbl for lbl in run_labels if lbl in pivot.columns], 2):
        paired = pivot[[label_a, label_b]].dropna()
        if paired.empty:
            continue
        a = paired[label_a].astype(bool)
        b = paired[label_b].astype(bool)
        b_only_a = int((a & ~b).sum())
        b_only_b = int((~a & b).sum())
        p_value = mcnemar_exact_p(b_only_a, b_only_b)
        rows.append(
            {
                "Run A": label_a,
                "Run B": label_b,
                "Shared": len(paired),
                "A only correct": b_only_a,
                "B only correct": b_only_b,
                "Acc A (%)": round(a.mean() * 100, 1),
                "Acc B (%)": round(b.mean() * 100, 1),
                "McNemar p": round(p_value, 4),
                "Significant(α=.05)": "yes" if p_value < 0.05 else "no",
            }
        )
    if not rows:
        return
    print("=" * 100)
    print("Paired significance (exact McNemar on shared tasks):")
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "Note: with ~30 shared tasks only large accuracy gaps reach significance; "
        "treat non-significant differences as noise."
    )
    print()


def print_comparison(all_df: pd.DataFrame, run_labels: List[str], disagreement_limit: int) -> None:
    if len(run_labels) < 2 or all_df.empty:
        return
    pivot = all_df.pivot_table(
        index=["task_id", "level", "question_short", "true_answer"],
        columns="run_label",
        values="correct",
        aggfunc="first",
    )
    pivot = pivot.dropna(how="all")
    disagreement_mask = pivot.apply(lambda row: row.dropna().nunique() > 1, axis=1)
    disagreements = pivot[disagreement_mask]
    if not disagreements.empty:
        print("=" * 100)
        print("Tasks with differing outcomes across runs:")
        print(disagreements.head(disagreement_limit).to_string())
    incorrect_mask = pivot.apply(lambda row: (row.dropna() == False).any(), axis=1)
    incorrect_rows = pivot[incorrect_mask]
    if not incorrect_rows.empty:
        print("\nTasks missed by at least one run:")
        print(incorrect_rows.head(disagreement_limit).to_string())
    print()


def export_results(all_df: pd.DataFrame, path: Path) -> None:
    export_df = all_df.copy()
    if "tool_calls_breakdown" in export_df:
        export_df["tool_calls_breakdown"] = export_df["tool_calls_breakdown"].apply(
            lambda data: json.dumps(data, ensure_ascii=False)
        )
    if "search_tool_calls_breakdown" in export_df:
        export_df["search_tool_calls_breakdown"] = export_df["search_tool_calls_breakdown"].apply(
            lambda data: json.dumps(data, ensure_ascii=False)
        )
    export_df.to_csv(path, index=False)
    print(f"Per-task metrics exported to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GAIA JSONL results and compare planning strategies.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form label=/path/to/results.jsonl. Use multiple --run entries to compare runs.",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        help="Optional GAIA levels to include (e.g., --levels 1 2). Defaults to all levels present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of tasks per run (useful for quick sanity checks).",
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        help="Optional path to export the combined per-task metrics as CSV.",
    )
    parser.add_argument(
        "--top_tools",
        type=int,
        default=5,
        help="Number of top tools to display per run.",
    )
    parser.add_argument(
        "--show_incorrect",
        type=int,
        default=10,
        help="Number of incorrect tasks to display per run.",
    )
    parser.add_argument(
        "--show_comparison",
        type=int,
        default=20,
        help="Maximum number of comparison rows to display for disagreements.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_specs = [parse_run_spec(spec) for spec in args.run]
    summaries = []
    per_run_dfs: List[pd.DataFrame] = []

    for label, path in run_specs:
        print(f"Loading {label} from {path} ...")
        records = load_jsonl(path)
        if args.limit:
            records = records[: args.limit]
        rows = []
        for record in records:
            level = normalize_level(record.get("task"))
            if args.levels and level not in args.levels:
                continue
            rows.append(analyze_record(record, label))
        if not rows:
            print(f"No tasks left after filtering for run {label}.")
            continue
        summary, df = build_run_summary(rows, label, path)
        summaries.append(summary)
        per_run_dfs.append(df)

    if not summaries:
        print("No runs to analyze. Double-check the inputs.")
        return

    build_summary_table(summaries)
    for summary, df in zip(summaries, per_run_dfs, strict=False):
        print_run_summary(summary, df, args.top_tools, args.show_incorrect)

    combined_df = pd.concat(per_run_dfs, ignore_index=True)
    print_significance_tests(combined_df, [label for label, _ in run_specs])
    print_comparison(combined_df, [label for label, _ in run_specs], args.show_comparison)

    if args.export_csv:
        export_results(combined_df, Path(args.export_csv).expanduser().resolve())


if __name__ == "__main__":
    main()
