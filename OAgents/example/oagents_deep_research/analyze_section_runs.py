#!/usr/bin/env python3
"""
Analyze GAIA run outputs to verify subtask (section) execution order and synchronization.

The script consumes the JSONL artifacts produced by `run_gaia.py` and inspects the planning
steps plus action traces to ensure:
  * Subtasks were planned (##ST blocks) and parsed correctly.
  * Every planned subtask executed at least once.
  * Execution order respects the sections plan or DAG dependencies.
  * Each section returned a non-empty output and no unexpected duplicates occurred.

Example:
    python analyze_section_runs.py output/validation/gpt-5-gaia-level1-30-sections.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass
class SubtaskSpec:
    code: str
    title: str
    steps: Sequence[str]


@dataclass
class SubtaskRun:
    code: str
    output: str | None
    raw_output: str | None
    tool_name: str
    raw_step_index: int
    title: str
    context: str
    context_refs: list[str]
    context_parsed: bool
    is_final_answer: bool
    context_payload: list[str] | None = None


PLAN_SECTION_RE = re.compile(r"##(ST\d+)([^\n]*?)\n([\s\S]*?)(?=\n##ST|\Z)")
PARALLEL_LIST_RE = re.compile(r"##PARALLEL_LIST\s*\n([^\n#]+)")
DAG_LIST_RE = re.compile(r"##DAG_LIST\s*\n(.*?)(?=##|\Z)", re.DOTALL)
NUMBERED_STEP_RE = re.compile(r"^\s*\d+\.\s.*$", re.MULTILINE)
PROMPT_BLOCK_RE = re.compile(
    r"ANSWER THE SUBTASK:(?P<body>.*?)Produce code to execute this subtask\. If this subtask yields a final solution, call final_answer\(\).",
    re.DOTALL,
)


def load_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_plan(plan_text: str) -> tuple[list[str], dict[str, SubtaskSpec], list[tuple[str, str]]]:
    parallel_candidates: list[str] = []
    parallel_match = PARALLEL_LIST_RE.search(plan_text)
    if parallel_match:
        for candidate in re.split(r"[,\s]+", parallel_match.group(1).strip()):
            if candidate.startswith("ST"):
                parallel_candidates.append(candidate)

    subtasks: dict[str, SubtaskSpec] = {}
    for match in PLAN_SECTION_RE.finditer(plan_text):
        code, header_trailer, body = match.groups()
        title_fragment = header_trailer.strip()
        if title_fragment.startswith(":"):
            title_fragment = title_fragment[1:].strip()
        elif title_fragment.startswith("[") and title_fragment.endswith("]"):
            title_fragment = title_fragment[1:-1].strip()
        steps = NUMBERED_STEP_RE.findall(body)
        subtasks[code] = SubtaskSpec(code=code, title=title_fragment or code, steps=steps or [])

    dag_edges: list[tuple[str, str]] = []
    dag_match = DAG_LIST_RE.search(plan_text)
    if dag_match:
        raw_edges = dag_match.group(1).strip()
        if raw_edges:
            try:
                parsed = ast.literal_eval(raw_edges)
                if isinstance(parsed, (list, tuple)):
                    for edge in parsed:
                        if isinstance(edge, (list, tuple)) and len(edge) == 2:
                            a, b = (str(edge[0]).strip(), str(edge[1]).strip())
                            if a.startswith("ST") and b.startswith("ST"):
                                dag_edges.append((a, b))
            except (ValueError, SyntaxError):
                # Ignore malformed DAG definitions; handled downstream.
                pass

    return parallel_candidates, subtasks, dag_edges


def derive_batches(
    subtasks: dict[str, SubtaskSpec],
    parallel_list: list[str],
    dag_edges: list[tuple[str, str]],
    preferred_mode: str,
) -> tuple[list[list[str]], str]:
    all_nodes = sorted(subtasks, key=lambda code: int(code[2:]) if code[2:].isdigit() else code)
    mode = preferred_mode
    if preferred_mode == "auto":
        mode = "dag" if dag_edges else "sections"

    if not all_nodes:
        return [], mode

    if mode == "sections" or not dag_edges:
        sequence = parallel_list or all_nodes
        batches = [[code] for code in sequence if code in subtasks]
        missing = [code for code in all_nodes if code not in sequence]
        batches.extend([[code] for code in missing])
        return batches, "sections"

    # DAG scheduling using Kahn's algorithm; batches correspond to topological levels.
    indegree = {node: 0 for node in all_nodes}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for src, dst in dag_edges:
        if src in subtasks and dst in subtasks:
            adjacency[src].append(dst)
            indegree[dst] += 1

    frontier = [node for node, deg in indegree.items() if deg == 0]

    # Respect PARALLEL_LIST ordering hints when available.
    def priority(node: str) -> tuple[int, int]:
        base_idx = parallel_list.index(node) if node in parallel_list else len(parallel_list) + 1
        numeric = int(node[2:]) if node[2:].isdigit() else len(parallel_list) + 1
        return base_idx, numeric

    batches: list[list[str]] = []
    processed = 0
    while frontier:
        frontier.sort(key=priority)
        level = frontier.copy()
        batches.append(level)
        frontier.clear()
        for node in level:
            processed += 1
            for neighbour in adjacency[node]:
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    frontier.append(neighbour)

    if processed != len(all_nodes):
        # Fallback: append any remaining nodes based on numeric order to surface the cycle.
        leftovers = [node for node, deg in indegree.items() if deg > 0]
        leftovers.sort(key=lambda code: int(code[2:]) if code[2:].isdigit() else code)
        for node in leftovers:
            batches.append([node])

    return batches, "dag"


def ensure_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def extract_prompt_blocks(raw_input: str | None) -> list[str]:
    if not raw_input:
        return []
    return [match.group("body") for match in PROMPT_BLOCK_RE.finditer(raw_input)]


def parse_prompt_block(block: str) -> tuple[str, str, bool]:
    """
    Returns (title, context, parsed_ok).
    """
    if not block:
        return "", "", False
    first = block.find("```")
    if first == -1:
        return "", block.strip(), False
    after_first = block[first + 3 :]
    second = after_first.find("```")
    if second == -1:
        return "", after_first.strip(), False
    inner = after_first[:second]
    rest = after_first[second + 3 :]
    title = ""
    for line in inner.strip().splitlines():
        if line.strip().startswith("subtask:"):
            title = line.split("subtask:", 1)[1].strip()
            break
    context = rest.strip()
    return title, context, True


def collect_subtask_runs(intermediate_steps: Iterable[dict]) -> list[SubtaskRun]:
    executions: list[SubtaskRun] = []
    for index, step in enumerate(intermediate_steps):
        if step.get("step_type") != "action":
            continue
        tool_calls = step.get("tool_calls") or []
        tool_call_lookup = {
            call.get("id"): (call.get("name", ""), call) for call in tool_calls if isinstance(call, dict)
        }
        if not tool_calls:
            tool_calls = []

        action_output = step.get("action_output")
        structured_runs = False
        if isinstance(action_output, list):
            for entry in action_output:
                if isinstance(entry, dict) and entry.get("subtask"):
                    structured_runs = True
                    code = entry.get("subtask")
                    if not code:
                        continue
                    call_id = entry.get("tool_call_id")
                    tool_name = entry.get("tool_name") or tool_call_lookup.get(call_id, ("", None))[0]
                    context = entry.get("context") or ""
                    context_sources = entry.get("context_sources") or entry.get("context_refs") or []
                    if isinstance(context_sources, str):
                        context_sources = [context_sources]
                    context_payload = entry.get("context_payload")
                    effective_output = entry.get("effective_output", entry.get("output"))
                    raw_output = entry.get("raw_output", effective_output)
                    executions.append(
                        SubtaskRun(
                            code=str(code),
                            output=str(effective_output) if effective_output is not None else None,
                            raw_output=str(raw_output) if raw_output is not None else None,
                            tool_name=tool_name,
                            raw_step_index=index,
                            title=str(entry.get("title") or code),
                            context=context,
                            context_refs=[str(ref) for ref in context_sources],
                            context_parsed=True,
                            is_final_answer=bool(entry.get("is_final_answer")),
                            context_payload=[str(item) for item in context_payload] if context_payload else None,
                        )
                    )
            if structured_runs:
                continue

        outputs = ensure_list(action_output)
        if len(outputs) < len(tool_calls):
            outputs.extend([None for _ in range(len(tool_calls) - len(outputs))])
        prompt_blocks = extract_prompt_blocks(step.get("model_input_messages"))
        parsed_prompts = [parse_prompt_block(block) for block in prompt_blocks]
        subtask_calls = [
            (idx, call) for idx, call in enumerate(tool_calls) if (call.get("id") or "").startswith("call_sub_")
        ]
        if not subtask_calls:
            continue
        for order, (call_index, call) in enumerate(subtask_calls):
            call_id = call.get("id", "")
            code = call_id.split("_")[-1]
            title = ""
            context = ""
            parsed_ok = False
            if order < len(parsed_prompts):
                title, context, parsed_ok = parsed_prompts[order]
            context_refs = re.findall(r"-\s*(ST\d+)", context) if context else []
            executions.append(
                SubtaskRun(
                    code=code,
                    output=str(outputs[call_index]) if call_index < len(outputs) else None,
                    raw_output=str(outputs[call_index]) if call_index < len(outputs) else None,
                    tool_name=call.get("name", ""),
                    raw_step_index=index,
                    title=title or code,
                    context=context,
                    context_refs=context_refs,
                    context_parsed=parsed_ok,
                    is_final_answer=False,
                )
            )
    return executions


def first_subtask_plan(intermediate_steps: Sequence[dict]) -> str | None:
    for step in reversed(intermediate_steps):
        if step.get("step_type") != "planning":
            continue
        plan_text = step.get("plan") or ""
        if "##ST" in plan_text:
            return plan_text
    return None


def dependencies_from_edges(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    deps: dict[str, set[str]] = defaultdict(set)
    for src, dst in edges:
        deps[dst].add(src)
    return deps


def evaluate_synchronization(
    planned_batches: list[list[str]],
    executed: list[SubtaskRun],
    deps: dict[str, set[str]],
    mode: str,
) -> dict:
    planned_linear = [code for batch in planned_batches for code in batch]
    executed_codes = [run.code for run in executed]

    missing = [code for code in planned_linear if code not in executed_codes]
    extra = [code for code in executed_codes if code not in planned_linear]

    duplicate_codes = {code for code in executed_codes if executed_codes.count(code) > 1}

    order_issue = None
    if mode == "sections":
        expected = planned_linear[: len(executed_codes)]
        if executed_codes != expected:
            for idx, (expected_code, actual_code) in enumerate(zip(expected, executed_codes, strict=False)):
                if expected_code != actual_code:
                    order_issue = f"Expected {expected_code} at position {idx}, saw {actual_code}"
                    break
            else:
                if len(executed_codes) != len(expected):
                    order_issue = "Executed code count mismatched planned count"
    else:
        seen: set[str] = set()
        violation = None
        for idx, code in enumerate(executed_codes):
            blockers = deps.get(code, set()) - seen
            if blockers:
                violation = f"{code} started before {', '.join(sorted(blockers))}"
                order_issue = violation
                break
            seen.add(code)

    empty_outputs = [run.code for run in executed if run.output in ("", "None", "null", None)]

    sync_ok = not (missing or extra or duplicate_codes or order_issue or empty_outputs)
    return {
        "planned_order": planned_linear,
        "executed_order": executed_codes,
        "missing": missing,
        "extra": extra,
        "duplicates": sorted(duplicate_codes),
        "order_issue": order_issue,
        "empty_outputs": empty_outputs,
        "sync_ok": sync_ok,
    }


def evaluate_context_flow(executed: list[SubtaskRun]) -> dict:
    missing_context: list[str] = []
    partial_context: list[tuple[str, list[str]]] = []
    bad_references: list[tuple[str, list[str]]] = []
    unparsed_prompts: list[str] = []

    for idx, run in enumerate(executed):
        if not run.context_parsed:
            unparsed_prompts.append(run.code)
        prev_codes = [prior.code for prior in executed[:idx]]
        if not prev_codes:
            continue

        if not run.context.strip():
            missing_context.append(run.code)
            continue

        missing_refs = [code for code in prev_codes if code not in run.context_refs]
        if missing_refs:
            partial_context.append((run.code, missing_refs))

        bad_refs = [ref for ref in run.context_refs if ref not in prev_codes]
        if bad_refs:
            bad_references.append((run.code, bad_refs))

    context_ok = not (missing_context or partial_context or bad_references or unparsed_prompts)
    return {
        "missing_context": missing_context,
        "partial_context": partial_context,
        "bad_references": bad_references,
        "unparsed": unparsed_prompts,
        "context_ok": context_ok,
    }


def analyze_record(record: dict, preferred_mode: str) -> dict:
    steps = record.get("intermediate_steps") or []
    plan_text = first_subtask_plan(steps)
    if not plan_text:
        return {
            "task_id": record.get("task_id"),
            "status": "no_subtasks",
            "details": "No ##ST subtasks detected in planning steps.",
        }

    parallel_list, subtask_specs, dag_edges = parse_plan(plan_text)
    batches, mode = derive_batches(subtask_specs, parallel_list, dag_edges, preferred_mode)
    executions = collect_subtask_runs(steps)
    deps = dependencies_from_edges(dag_edges)
    sync_summary = evaluate_synchronization(batches, executions, deps, mode)
    context_summary = evaluate_context_flow(executions)

    return {
        "task_id": record.get("task_id"),
        "task_level": record.get("task"),
        "question": record.get("question"),
        "mode": mode,
        "subtask_specs": subtask_specs,
        "batches": batches,
        "executions": executions,
        "sync": sync_summary,
        "context": context_summary,
        "parsing_error": record.get("parsing_error"),
        "iteration_limit": record.get("iteration_limit_exceeded"),
        "agent_error": record.get("agent_error"),
        "prediction_present": record.get("prediction") is not None,
    }


def render_report(result: dict, verbose: bool) -> None:
    task_id = result.get("task_id")
    status = result.get("status")
    if status == "no_subtasks":
        print(f"[{task_id}] Skipped: {result.get('details')}")
        return

    sync = result["sync"]
    mode = result["mode"]
    summary = "PASS" if sync["sync_ok"] else "FAIL"
    headline = f"[{task_id}] {summary} (mode={mode}, planned={len(result['subtask_specs'])}, executed={len(result['executions'])})"
    print(headline)

    if sync["missing"]:
        print(f"  - Missing subtasks: {', '.join(sync['missing'])}")
    if sync["extra"]:
        print(f"  - Unexpected subtasks: {', '.join(sync['extra'])}")
    if sync["duplicates"]:
        print(f"  - Duplicate executions: {', '.join(sync['duplicates'])}")
    if sync["order_issue"]:
        print(f"  - Order violation: {sync['order_issue']}")
    if sync["empty_outputs"]:
        print(f"  - Empty outputs: {', '.join(sync['empty_outputs'])}")
    if result.get("parsing_error"):
        print("  - Parsing error flagged by agent.")
    if result.get("iteration_limit"):
        print("  - Iteration limit exceeded.")
    if result.get("agent_error"):
        print(f"  - Agent error: {result['agent_error']}")

    context_summary = result.get("context", {})
    if context_summary and not context_summary.get("context_ok", True):
        missing_context = context_summary.get("missing_context") or []
        if missing_context:
            print(f"  - Missing context bridging: {', '.join(missing_context)}")
        partial = context_summary.get("partial_context") or []
        for code, missing_refs in partial:
            print(f"  - Partial context for {code}: missing refs {', '.join(missing_refs)}")
        bad_refs = context_summary.get("bad_references") or []
        for code, refs in bad_refs:
            print(f"  - Bad context references for {code}: {', '.join(refs)} (not yet executed)")
        unparsed = context_summary.get("unparsed") or []
        if unparsed:
            print(f"  - Could not parse prompts for: {', '.join(unparsed)}")

    if verbose:
        spec_map: dict[str, SubtaskSpec] = result["subtask_specs"]
        execution_map = {run.code: run for run in result["executions"]}
        for batch_index, batch in enumerate(result["batches"], start=1):
            print(f"    Batch {batch_index}: {', '.join(batch)}")
            for code in batch:
                spec = spec_map.get(code)
                run = execution_map.get(code)
                status_marker = "✓" if run else "✗"
                title = run.title if run else (spec.title if spec else code)
                output_preview = ""
                if run and run.output is not None:
                    out = run.output.replace("\n", " ").strip()
                    output_preview = f" | output: {out[:80]}{'…' if len(out) > 80 else ''}"
                    if run.raw_output is not None and run.raw_output != run.output:
                        raw = run.raw_output.replace("\n", " ").strip()
                        output_preview += f" | raw: {raw[:40]}{'…' if len(raw) > 40 else ''}"
                context_info = ""
                if run and run.context:
                    ctx = run.context.replace("\n", " ").strip()
                    context_info = f" | context: {ctx[:80]}{'…' if len(ctx) > 80 else ''}"
                if run and run.context_payload:
                    payload = "; ".join(run.context_payload)
                    payload = payload.replace("\n", " ").strip()
                    payload_preview = payload[:80] + ("…" if len(payload) > 80 else "")
                    context_info += f" | ctx_sources: {payload_preview}"
                print(f"      {status_marker} {code} - {title}{output_preview}{context_info}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate subtask execution order in GAIA run outputs.")
    parser.add_argument("jsonl_path", type=Path, help="Path to the run output JSONL file.")
    parser.add_argument(
        "--subtask-mode",
        choices=["sections", "dag", "auto"],
        default="auto",
        help="Interpretation of the plan structure. Default: auto-detect via DAG definitions.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-subtask details.")
    args = parser.parse_args()

    if not args.jsonl_path.exists():
        raise SystemExit(f"File not found: {args.jsonl_path}")

    results = [analyze_record(record, args.subtask_mode) for record in load_records(args.jsonl_path)]
    total = len(results)
    sync_pass = sum(1 for result in results if result.get("status") != "no_subtasks" and result["sync"]["sync_ok"])
    sync_fail = sum(1 for result in results if result.get("status") != "no_subtasks" and not result["sync"]["sync_ok"])
    context_fail = sum(
        1
        for result in results
        if result.get("status") != "no_subtasks" and result.get("context") and not result["context"]["context_ok"]
    )
    skipped = sum(1 for result in results if result.get("status") == "no_subtasks")

    for result in results:
        render_report(result, args.verbose)

    print("\nSummary:")
    print(f"  Records processed : {total}")
    print(f"  Sync pass         : {sync_pass}")
    print(f"  Sync fail         : {sync_fail}")
    print(f"  Context issues    : {context_fail}")
    print(f"  No subtask plan   : {skipped}")


if __name__ == "__main__":
    main()
