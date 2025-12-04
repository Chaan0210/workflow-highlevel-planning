### GAIA benchmark 실험

```
cd ./OAgents/example/oagents_deep_research

# 1. ----------------- Reactive -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt5 --model_id_search gpt-5 \
  --run_name gpt-5-gaia-l1-30-react-2025-12-02 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --static_plan \
  --max_steps 20 \
  --search_tool_reflection \
  --search_agent_plan_once

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt5 --model_id_search gpt-5 \
  --run_name gpt-5-gaia-l2-30-react-2025-12-02 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --static_plan \
  --max_steps 20 \
  --search_tool_reflection \
  --search_agent_plan_once

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt5 --model_id_search gpt-5 \
  --run_name gpt-5-gaia-l3-30-react-2025-12-02 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --static_plan \
  --max_steps 20 \
  --search_tool_reflection \
  --search_agent_plan_once


# 2. ----------------- Sequential Planning -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-pta-2025-12-02 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval 0 \
  --search_agent_plan_once

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-pta-2025-12-02 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval 0 \
  --search_agent_plan_once

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-pta-2025-12-02 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval 0 \
  --search_agent_plan_once


# 3. ----------------- Dependency Graph Planning -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-dag-2025-12-02 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval 0 \
  --search_agent_plan_once

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-dag-2025-12-02 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval 0 \
  --search_agent_plan_once

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-dag-2025-12-02 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval 0 \
  --search_agent_plan_once


# 4. ----------------- Sequential Planning + Re-planning -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-pta-rp-2025-12-02 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval auto \
  --search_agent_plan_once

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-pta-rp-2025-12-02 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval auto \
  --search_agent_plan_once

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-pta-rp-2025-12-02 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval auto \
  --search_agent_plan_once


# 5. ----------------- Dependency Graph Planning + Re-planning -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-dag-rp-2025-12-02 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval auto \
  --search_agent_plan_once

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-dag-rp-2025-12-02 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval auto \
  --search_agent_plan_once

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-dag-rp-2025-12-02 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval auto \
  --search_agent_plan_once
```

### GAIA benchmark evaluating

```
python evaluate_gaia_results.py output/validation/***.jsonl
```

### 결과 저장 구조

```
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
    "search_agent_actions": agent.managed_agents['search_agent'].task_records,
}
```
