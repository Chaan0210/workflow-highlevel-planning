### SimpleQA benchmark 실험

```
cd ./OAgents/example/oagents_deep_research

# 1. ----------------- Reactive -----------------
python run_simpleqa.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt-5-simpleqa-l1-100-react-2025-12-06 \
  --level 1 --selected-tasks $(seq 0 99 | tr '\n' ' ') \
  --static_plan \
  --max_steps 20 \
  --search_tool_reflection \
  --search_agent_plan_once


# 2. ----------------- Sequential Planning -----------------
python run_simpleqa.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-simpleqa-l1-100-pta-2025-12-06 \
  --level 1 --selected-tasks $(seq 0 99 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval 0 \
  --search_agent_plan_once


# 3. ----------------- Dependency Graph Planning -----------------
python run_simpleqa.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-simpleqa-l1-100-dag-2025-12-06 \
  --level 1 --selected-tasks $(seq 0 99 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval 0 \
  --search_agent_plan_once


# 4. ----------------- Sequential Planning + Re-planning -----------------
python run_simpleqa.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-simpleqa-l1-100-pta-rp-2025-12-06 \
  --level 1 --selected-tasks $(seq 0 99 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode sections \
  --planning_interval auto \
  --search_agent_plan_once


# 5. ----------------- Dependency Graph Planning + Re-planning -----------------
python run_simpleqa.py \
  --concurrency 10 \
  --model_id gpt-5 --model_id_search gpt-5 \
  --run_name gpt5-simpleqa-l1-100-dag-rp-2025-12-06 \
  --level 1 --selected-tasks $(seq 0 99 | tr '\n' ' ') \
  --max_steps 20 \
  --search_tool_reflection \
  --subtask --subtask_mode dag \
  --planning_interval auto \
  --search_agent_plan_once
```

### SimpleQA benchmark evaluating

```
python evaluate_simpleqa_results.py --run_name <...> --judge_model_id gpt-4.1
```
