### GAIA benchmark (30 tasks, level 1, gpt-5, OAgents) 실행

```
cd ./OAgents/example/oagents_deep_research

python run_gaia.py --model_id gpt-5 --model_id_search gpt-5 --run_name gpt-5-gaia-level1-30tasks --level 1 --selected-tasks $(seq 0 29 | tr '\n' ' ')
```

### GAIA benchmark evaluating

```
python evaluate_gaia_results.py output/validation/gpt-5-gaia-level1-30tasks.jsonl
```
