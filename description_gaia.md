### GAIA benchmark 실험

```
cd ./OAgents/example/oagents_deep_research

# tmux 사용법
tmux ls
tmux new -s {session_name}
tmux attach -t {session_name}
tmux kill-session -t {session_name}
# Detach: Ctrl+b & d

# run_small.sh
bash run_small.sh

# run_full.sh
bash run_full.sh

# 1. ----------------- Reactive -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-react-2026-01-08 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --static_plan \
  --max_steps 20 \
  --search_tool_reflection

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-react-2026-01-08 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --static_plan \
  --max_steps 20 \
  --search_tool_reflection

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-react-2026-01-08 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --static_plan \
  --max_steps 20 \
  --search_tool_reflection


# 2. ----------------- Sequential Planning -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-pta-2026-01-08 \
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
  --run_name gpt5-gaia-l2-30-pta-2026-01-08 \
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
  --run_name gpt5-gaia-l3-30-pta-2026-01-08 \
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
  --run_name gpt5-gaia-l1-30-dag-2026-01-08 \
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
  --run_name gpt5-gaia-l2-30-dag-2026-01-08 \
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
  --run_name gpt5-gaia-l3-30-dag-2026-01-08 \
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
  --run_name gpt5-gaia-l1-30-pta-rp-2026-01-08 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval auto \
  --search_agent_plan_once \
  --reflection

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-pta-rp-2026-01-08 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval auto \
  --search_agent_plan_once \
  --reflection

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-pta-rp-2026-01-08 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval auto \
  --search_agent_plan_once \
  --reflection


# 5. ----------------- Dependency Graph Planning + Re-planning -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-dag-rp-2026-01-08 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval auto \
  --search_agent_plan_once \
  --reflection

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-dag-rp-2026-01-08 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval auto \
  --search_agent_plan_once \
  --reflection

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-dag-rp-2026-01-08 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval auto \
  --search_agent_plan_once \
  --reflection
```

### GAIA benchmark evaluating

```
python evaluate_gaia_results.py output/validation/***.jsonl
```

```
python OAgents/example/oagents_deep_research/analyze_gaia_results.py \
  --run react=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l1-30-react-2025-11-20.jsonl \
  --run pta=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l1-30-pta-2025-12-02.jsonl \
  --run dag=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l1-30-dag-2025-12-02.jsonl \
  --run pta_rp=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l1-30-pta-rp-2025-12-06.jsonl \
  --run dag_rp=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l1-30-dag-rp-2025-12-06.jsonl \
  --show_incorrect 50 --show_comparison 50

python OAgents/example/oagents_deep_research/analyze_gaia_results.py \
  --run react=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l2-30-react-2025-11-20.jsonl \
  --run pta=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l2-30-pta-2025-12-02.jsonl \
  --run dag=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l2-30-dag-2025-12-02.jsonl \
  --run pta_rp=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l2-30-pta-rp-2025-12-07.jsonl \
  --run dag_rp=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l2-30-dag-rp-2025-12-07.jsonl \
  --show_incorrect 50 --show_comparison 50

python OAgents/example/oagents_deep_research/analyze_gaia_results.py \
  --run react=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l3-30-react-2025-11-20.jsonl \
  --run pta=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l3-30-pta-2025-12-02.jsonl \
  --run dag=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l3-30-dag-2025-12-02.jsonl \
  --run pta_rp=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l3-30-pta-rp-2025-12-07.jsonl \
  --run dag_rp=OAgents/example/oagents_deep_research/output/validation/gpt5-gaia-l3-30-dag-rp-2025-12-07.jsonl \
  --show_incorrect 50 --show_comparison 50
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
