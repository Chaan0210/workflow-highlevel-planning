import os
import re

os.environ.setdefault("OPENAI_API_KEY", "test-key-offline")

import oagents.agents as agents_module
from oagents.agents import CodeAgent
from oagents.memory import ActionStep, PlanningStep
from oagents.models import ChatMessage
from oagents.monitoring import LogLevel


SECTIONS_PLAN = """##PARALLEL_LIST
ST1, ST2
##ST1: gather alpha
1. find the alpha value
##ST2: conclude
1. combine results and answer
"""

DAG_PLAN = """##DAG_LIST
[('ST1', 'ST3'), ('ST2', 'ST3')]
##PARALLEL_LIST
ST1, ST2
##ST1: branch one
1. fetch part one
##ST2: branch two
1. fetch part two
##ST3: merge
1. merge both parts and answer
"""


class ScriptedModel:
    """Deterministic model: returns canned plans for planning prompts and canned
    code per subtask id, recording every prompt for hand-off assertions."""

    def __init__(self, plans, subtask_codes):
        self.plans = list(plans)
        self.subtask_codes = {st: list(codes) for st, codes in subtask_codes.items()}
        self.plan_calls = 0
        self.fact_calls = 0
        self.react_calls = 0
        self.plan_prompts = []
        self.react_prompts = []
        self.subtask_prompts = {}

    def __call__(self, messages, stop_sequences=None, **kwargs):
        text = str(messages)
        if stop_sequences == ["<end_plan>"]:
            self.plan_prompts.append(text)
            plan = self.plans[min(self.plan_calls, len(self.plans) - 1)]
            self.plan_calls += 1
            return ChatMessage(role="assistant", content=plan)

        if stop_sequences == ["<end_code>", "Observation:"]:
            marker = text.find("[CURRENT SUBTASK]")
            match = re.search(r"subtask: (ST\d+)", text[marker:]) if marker != -1 else None
            if match:
                st = match.group(1)
                self.subtask_prompts.setdefault(st, []).append(text)
                codes = self.subtask_codes[st]
                code = codes.pop(0) if len(codes) > 1 else codes[0]
                return ChatMessage(role="assistant", content=code)
            self.react_calls += 1
            self.react_prompts.append(text)
            return ChatMessage(role="assistant", content="```python\nfinal_answer('react-fallback')\n```")

        self.fact_calls += 1
        return ChatMessage(role="assistant", content="facts")


def make_agent(
    model, mode, auto_planning=False, reflection=False, max_steps=8, parallel_subtasks=False, plan_as_prompt=False
):
    return CodeAgent(
        tools=[],
        model=model,
        max_steps=max_steps,
        subtask=True,
        subtask_mode=mode,
        planning_interval=0,
        auto_planning=auto_planning,
        reflection=reflection,
        parallel_subtasks=parallel_subtasks,
        plan_as_prompt=plan_as_prompt,
        verbosity_level=LogLevel.ERROR,
    )


def test_sections_execute_one_subtask_per_step_with_full_handoff():
    model = ScriptedModel(
        plans=[SECTIONS_PLAN],
        subtask_codes={
            "ST1": ["```python\nprint('RESULT_ST1 alpha=42')\n```"],
            "ST2": ["```python\nfinal_answer('42')\n```"],
        },
    )
    agent = make_agent(model, "sections")
    result = agent.run("find alpha")

    assert str(result) == "42"
    assert [entry["subtask"] for entry in agent.subtask_records] == ["ST1", "ST2"]
    assert all(entry["status"] == "succeeded" for entry in agent.subtask_records)

    # One subtask per ActionStep — the plan is NOT consumed inside a single step.
    subtask_steps = [
        step
        for step in agent.memory.steps
        if isinstance(step, ActionStep) and step.tool_calls and str(step.tool_calls[0].id).startswith("call_sub_")
    ]
    assert len(subtask_steps) == 2

    # Sections hand-off: the second section sees the first section's result.
    assert "RESULT_ST1" in model.subtask_prompts["ST2"][0]
    # And the executor is shown the plan itself.
    assert "##ST1" in model.subtask_prompts["ST1"][0]


def test_dag_executes_in_dependency_order_with_gated_context():
    model = ScriptedModel(
        plans=[DAG_PLAN],
        subtask_codes={
            "ST1": ["```python\nprint('RESULT_ST1 part-one')\n```"],
            "ST2": ["```python\nprint('RESULT_ST2 part-two')\n```"],
            "ST3": ["```python\nfinal_answer('merged')\n```"],
        },
    )
    agent = make_agent(model, "dag")
    result = agent.run("merge two branches")

    assert str(result) == "merged"
    assert [entry["subtask"] for entry in agent.subtask_records] == ["ST1", "ST2", "ST3"]

    # Dependency-gated hand-off: ST2 has no edge from ST1, so it must NOT see
    # ST1's result (this is the behavioural difference vs sections mode)...
    assert "RESULT_ST1" not in model.subtask_prompts["ST2"][0]
    # ...while ST3 depends on both branches and must see both results.
    assert "RESULT_ST1" in model.subtask_prompts["ST3"][0]
    assert "RESULT_ST2" in model.subtask_prompts["ST3"][0]


def test_sections_pass_prior_results_where_dag_would_not():
    # Same plan shape as the DAG test but in sections mode: ST2 now DOES see ST1.
    model = ScriptedModel(
        plans=[SECTIONS_PLAN],
        subtask_codes={
            "ST1": ["```python\nprint('RESULT_ST1 alpha=42')\n```"],
            "ST2": ["```python\nfinal_answer('42')\n```"],
        },
    )
    agent = make_agent(model, "sections")
    agent.run("find alpha")
    assert "RESULT_ST1" in model.subtask_prompts["ST2"][0]


def test_failed_subtask_is_isolated_and_triggers_auto_replan():
    plan_v2 = """##PARALLEL_LIST
ST1
##ST1: retry differently
1. use the fallback source and answer
"""
    model = ScriptedModel(
        plans=[SECTIONS_PLAN, plan_v2],
        subtask_codes={
            "ST1": [
                "```python\nraise ValueError('boom')\n```",  # plan v1: fails
                "```python\nfinal_answer('recovered')\n```",  # plan v2: succeeds
            ],
            "ST2": ["```python\nfinal_answer('unused')\n```"],
        },
    )
    agent = make_agent(model, "sections", auto_planning=True)
    result = agent.run("recover from a failed subtask")

    assert str(result) == "recovered"
    assert agent.replan_count == 1
    assert agent._subtask_state["plan_version"] == 2
    statuses = [(entry["plan_version"], entry["subtask"], entry["status"]) for entry in agent.subtask_records]
    assert statuses[0] == (1, "ST1", "failed")
    assert statuses[-1] == (2, "ST1", "succeeded")
    # The failed attempt's outcome is carried into the new plan's execution context.
    assert "plan1:ST1" in model.subtask_prompts["ST1"][1]

    plan_steps = [step for step in agent.memory.steps if isinstance(step, PlanningStep)]
    assert len(plan_steps) == 2


def test_judge_should_replan_triggers_reflection_replan(monkeypatch):
    plan_v2 = """##PARALLEL_LIST
ST1
##ST1: act on judge feedback
1. produce the final answer
"""
    model = ScriptedModel(
        plans=[SECTIONS_PLAN, plan_v2],
        subtask_codes={
            "ST1": [
                "```python\nprint('RESULT_ST1 alpha=42')\n```",  # plan v1: fine but judge objects
                "```python\nfinal_answer('42')\n```",  # plan v2
            ],
            "ST2": ["```python\nfinal_answer('unused')\n```"],
        },
    )
    # Judge verdict: low score, wants a re-plan. Patched so no API call is made.
    monkeypatch.setattr(agents_module, "evaluate_answer", lambda *args, **kwargs: (2.0, "wrong direction", True))

    agent = make_agent(model, "sections", auto_planning=True, reflection=True)
    result = agent.run("follow the judge")

    assert str(result) == "42"
    assert agent.replan_count >= 1
    # The re-plan went through the reflection path and was told what already ran.
    assert any("ALREADY COMPLETED SUBTASKS" in prompt for prompt in model.plan_prompts[1:])


def test_partial_parallel_list_still_schedules_every_subtask():
    partial_plan = """##PARALLEL_LIST
ST2
##ST1: first
1. do first thing
##ST2: second
1. do second thing
##ST3: third
1. answer
"""
    model = ScriptedModel(
        plans=[partial_plan],
        subtask_codes={
            "ST1": ["```python\nprint('RESULT_ST1')\n```"],
            "ST2": ["```python\nprint('RESULT_ST2')\n```"],
            "ST3": ["```python\nfinal_answer('all-done')\n```"],
        },
    )
    agent = make_agent(model, "sections")
    result = agent.run("partial parallel list")

    assert str(result) == "all-done"
    # ST2 is ordered first (hint), but ST1/ST3 must still run instead of being dropped.
    assert [entry["subtask"] for entry in agent.subtask_records] == ["ST2", "ST1", "ST3"]


def test_dag_parallel_batch_executes_ready_set_concurrently_in_one_step():
    import threading

    # Both ready branches must be in flight at the same time to pass this barrier;
    # sequential execution would deadlock (broken barrier) and fail the test.
    barrier = threading.Barrier(2, timeout=10)

    class BarrierModel(ScriptedModel):
        def __call__(self, messages, stop_sequences=None, **kwargs):
            text = str(messages)
            if stop_sequences == ["<end_code>", "Observation:"] and (
                "subtask: ST1" in text.split("[CURRENT SUBTASK]")[-1]
                or "subtask: ST2" in text.split("[CURRENT SUBTASK]")[-1]
            ):
                barrier.wait()
            return super().__call__(messages, stop_sequences=stop_sequences, **kwargs)

    model = BarrierModel(
        plans=[DAG_PLAN],
        subtask_codes={
            "ST1": ["```python\nprint('RESULT_ST1 part-one')\n```"],
            "ST2": ["```python\nprint('RESULT_ST2 part-two')\n```"],
            "ST3": ["```python\nfinal_answer('merged')\n```"],
        },
    )
    agent = make_agent(model, "dag", parallel_subtasks=True)
    result = agent.run("merge two branches in parallel")

    assert str(result) == "merged"
    # ST1+ST2 form one parallel batch; results merge deterministically in priority order.
    assert [entry["subtask"] for entry in agent.subtask_records] == ["ST1", "ST2", "ST3"]
    assert agent.subtask_records[0]["parallel_batch"] == ["ST1", "ST2"]
    assert agent.subtask_records[1]["parallel_batch"] == ["ST1", "ST2"]
    assert agent.subtask_records[2]["parallel_batch"] == ["ST3"]

    # The whole batch lives in ONE ActionStep (two entries), ST3 in another.
    batch_steps = [
        step
        for step in agent.memory.steps
        if isinstance(step, ActionStep) and isinstance(step.action_output, list) and step.action_output
    ]
    assert [len(step.action_output) for step in batch_steps] == [2, 1]

    # Dependency-gated hand-off still holds in the parallel path.
    assert "RESULT_ST1" not in model.subtask_prompts["ST2"][0]
    assert "RESULT_ST1" in model.subtask_prompts["ST3"][0]
    assert "RESULT_ST2" in model.subtask_prompts["ST3"][0]


def test_plan_as_prompt_keeps_plain_react_execution():
    model = ScriptedModel(plans=[SECTIONS_PLAN], subtask_codes={})
    agent = make_agent(model, "sections", plan_as_prompt=True)
    result = agent.run("plan as guidance only")

    assert str(result) == "react-fallback"
    # No subtask executor involvement whatsoever...
    assert agent.subtask_records == []
    assert agent._subtask_state is None
    assert agent.plan_parse_failures == 0
    # ...but the ReAct step DOES see the ##ST plan as [PLAN] guidance in memory.
    assert model.react_calls >= 1
    assert "##ST1" in model.react_prompts[0]


def test_unparseable_plan_falls_back_loudly_to_react():
    # Plan text with no ##ST blocks at all -> retry once (same bad plan) -> ReAct fallback.
    model = ScriptedModel(plans=["1. just a flat plan\n2. no subtasks here"], subtask_codes={})
    agent = make_agent(model, "sections")
    result = agent.run("fallback check")

    assert str(result) == "react-fallback"
    assert agent.plan_parse_failures >= 1
    assert model.react_calls >= 1
