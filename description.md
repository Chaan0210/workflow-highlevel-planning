### GAIA benchmark 실험

```
cd ./OAgents/example/oagents_deep_research

# 1. ----------------- Static plan -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt-5-gaia-l1-30-react-2025-11-09 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --static_plan \
  --max_steps 20 \
  --search_tool_reflection

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt-5-gaia-l2-30-react-2025-11-09 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --static_plan \
  --max_steps 20 \
  --search_tool_reflection

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt-5-gaia-l3-30-react-2025-11-09 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --static_plan \
  --max_steps 20 \
  --search_tool_reflection


# 2. ----------------- Plan-then-Act -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-pta-2025-10-23 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval 0

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-pta-2025-10-23 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval 0

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-pta-2025-10-23 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval 0


# 3. ----------------- Dependency Graph -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-dag-2025-10-23 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --dynamic_update_plan \
  --planning_interval 0

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-dag-2025-10-23 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --dynamic_update_plan \
  --planning_interval 0

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-dag-2025-10-23 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --dynamic_update_plan \
  --planning_interval 0


# 4. ----------------- Plan-then-Act + Re-planning -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-pta-rp-2025-10-23 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval auto

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-pta-rp-2025-10-23 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval auto

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-pta-rp-2025-10-23 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval auto


# 5. ----------------- Dependency Graph + Re-planning -----------------
# Level 1
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l1-30-dag-rp-2025-10-23 \
  --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --dynamic_update_plan \
  --planning_interval auto

# Level 2
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l2-30-dag-rp-2025-10-23 \
  --level 2 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --dynamic_update_plan \
  --planning_interval auto

# Level 3
python run_gaia.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-gaia-l3-30-dag-rp-2025-10-23 \
  --level 3 --selected-tasks $(seq 0 29 | tr '\n' ' ') \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --dynamic_update_plan \
  --planning_interval auto
```

### GAIA benchmark evaluating

```
python evaluate_gaia_results.py output/validation/gpt-5-gaia-level1-30tasks.jsonl
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

### Basic analysis

```
python analyze_gaia_results.py
```

### With CSV export and plots

```
python analyze_gaia_results.py --export_csv --create_plots
```

### Specify custom directory

```
python analyze_gaia_results.py --output_dir /path/to/results --pattern "gpt-5\*.jsonl"
```

### Analyze subtask section mode

```
python analyze_section_runs.py <run>.jsonl --subtask-mode sections --verbose
```
