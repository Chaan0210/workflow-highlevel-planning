import pytest

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

    def __init__(self):
        self.plan_calls = 0
        self.action_calls = 0
        self.fact_calls = 0

    def __call__(self, messages, stop_sequences=None, **kwargs):
        if stop_sequences == ["<end_plan>"]:
            self.plan_calls += 1
            return ChatMessage(role="assistant", content=f"Plan #{self.plan_calls}")

        if stop_sequences == ["<end_code>", "Observation:"]:
            self.action_calls += 1
            if self.action_calls == 1:
                return ChatMessage(role="assistant", content="```python\nraise ValueError('boom')\n```")
            return ChatMessage(role="assistant", content="```python\nfinal_answer('done')\n```")

        self.fact_calls += 1
        return ChatMessage(role="assistant", content=f"Facts #{self.fact_calls}")


def test_auto_replanning_triggers_on_failed_action():
    model = DummyModel()
    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=3,
        auto_planning=True,
        planning_interval=1,
        verbosity_level=LogLevel.ERROR,
    )

    final_answer = agent.run("test auto re-plan")
    assert final_answer

    plan_steps = [step for step in agent.memory.steps if isinstance(step, PlanningStep)]
    assert len(plan_steps) == 2, "Initial plan plus one re-plan should be recorded"
    assert "Plan #2" in plan_steps[-1].plan
    assert model.plan_calls == 2
    assert model.action_calls == 2
