import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-offline")

from oagents.agents import CodeAgent
from oagents.memory import PlanningStep
from oagents.models import ChatMessage
from oagents.monitoring import LogLevel


class DummyModel:
    """
    Deterministic model used to simulate a first-step failure followed by a successful retry.
    It returns:
      * simple strings for fact gathering
      * canned plan content for planning prompts
      * executable python code for action steps
    """

    def __init__(self, action_scripts=None):
        self.plan_calls = 0
        self.action_calls = 0
        self.fact_calls = 0
        self.action_scripts = action_scripts or [
            "```python\nraise ValueError('boom')\n```",
            "```python\nfinal_answer('done')\n```",
        ]

    def __call__(self, messages, stop_sequences=None, **kwargs):
        if stop_sequences == ["<end_plan>"]:
            self.plan_calls += 1
            return ChatMessage(role="assistant", content=f"Plan #{self.plan_calls}")

        if stop_sequences == ["<end_code>", "Observation:"]:
            script = self.action_scripts[min(self.action_calls, len(self.action_scripts) - 1)]
            self.action_calls += 1
            return ChatMessage(role="assistant", content=script)

        self.fact_calls += 1
        return ChatMessage(role="assistant", content=f"Facts #{self.fact_calls}")


def test_auto_replanning_triggers_on_failed_action():
    # planning_interval must be None here: with interval=1 the agent plans every
    # step anyway and the test would pass even if auto_planning were a no-op.
    model = DummyModel()
    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=3,
        auto_planning=True,
        planning_interval=None,
        verbosity_level=LogLevel.ERROR,
    )

    final_answer = agent.run("test auto re-plan")
    assert final_answer

    plan_steps = [step for step in agent.memory.steps if isinstance(step, PlanningStep)]
    assert len(plan_steps) == 2, "Initial plan plus one re-plan should be recorded"
    assert "Plan #2" in plan_steps[-1].plan
    assert model.plan_calls == 2
    assert model.action_calls == 2
    assert agent.replan_count == 1


def test_no_replanning_when_actions_succeed():
    model = DummyModel(action_scripts=["```python\nfinal_answer('done')\n```"])
    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=3,
        auto_planning=True,
        planning_interval=None,
        verbosity_level=LogLevel.ERROR,
    )

    final_answer = agent.run("test no re-plan on success")
    assert final_answer

    plan_steps = [step for step in agent.memory.steps if isinstance(step, PlanningStep)]
    assert len(plan_steps) == 1, "Only the initial plan should be recorded"
    assert agent.replan_count == 0


def test_interval_none_without_auto_planning_plans_once():
    # Discriminating control: with auto_planning off the failure must NOT re-plan.
    model = DummyModel()
    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=3,
        auto_planning=False,
        planning_interval=None,
        verbosity_level=LogLevel.ERROR,
    )

    final_answer = agent.run("test plan-once baseline")
    assert final_answer

    plan_steps = [step for step in agent.memory.steps if isinstance(step, PlanningStep)]
    assert len(plan_steps) == 1
    assert model.plan_calls == 1
    assert model.action_calls == 2


def test_static_plan_never_plans():
    model = DummyModel(action_scripts=["```python\nfinal_answer('done')\n```"])
    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=3,
        static_plan=True,
        planning_interval=1,
        verbosity_level=LogLevel.ERROR,
    )

    final_answer = agent.run("test reactive baseline")
    assert final_answer

    plan_steps = [step for step in agent.memory.steps if isinstance(step, PlanningStep)]
    assert len(plan_steps) == 0, "static_plan must suppress every planning step"
    assert model.plan_calls == 0
