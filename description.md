### GAIA benchmark (30 tasks, level 1, gpt-5, OAgents) 실행

```
cd ./OAgents/example/oagents_deep_research

python run_gaia.py --model_id gpt-5 --model_id_search gpt-5 --run_name gpt-5-gaia-level1-30tasks --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ')

# max performance
python run_gaia.py --model_id gpt-5 --model_id_search gpt-5 --run_name gpt-5-gaia-level1-30tasks --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ') --max_steps 20 --planning_interval 2 --subtask --dynamic_update_plan --reflection --search_tool_reflection --summary --use_long_term_memory --retrieve_key_memory --debug
```

### GAIA benchmark evaluating

```
python evaluate_gaia_results.py output/validation/gpt-5-gaia-level1-30tasks.jsonl
```

### DAG visualizing

```
python visualize_dag.py --results_file output/validation/gpt-5-gaia-level1-30tasks.jsonl
```

### Bash Script

```
python -c "import sys; print('\n'.join(sys.path))"
```
