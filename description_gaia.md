### 2026-07-03 알고리즘 충실도(fidelity) 수정 — 재실험 필수

`fix/hlp-algorithm-fidelity` 브랜치에서 5개 플래닝 알고리즘이 실제로 서로 다른 처치가 되도록 수정했다.
**이전 결과(2025-11-20 ~ 2025-12-07 jsonl)와 새 결과는 비교 불가**하므로 5개 구성 전부를 같은 커밋에서 재실행해야 한다.

바뀐 것 (실행 커맨드는 아래 기존 것 그대로 사용):

1. **플랜 지속성**: sections/dag 플랜이 스텝 1개 안에서 전부 소진되지 않고, **스텝마다 서브태스크 1개씩** 실행된다.
   플랜이 끝나면 결과 종합을 지시받는 ReAct 스텝으로 넘어간다.
2. **sections vs dag 차별화**: sections는 이전 섹션 결과 **전부**를, dag는 **직전 의존 노드 결과만** 컨텍스트로 받는다
   (dag 실행 순서는 Kahn ready-set 스케줄링). 컨텍스트 클리핑 1200자 → 4000자, 플랜 본문도 실행기에 노출.
3. **재플래닝**: 키워드 스캔 트리거 제거 → (a) 스텝 에러, (b) 서브태스크 실패, (c) judge의 `should_replan` 판정만 트리거.
   재플랜 시 완료된 서브태스크 결과를 컨텍스트로 승계하고 **남은 작업만** 재계획한다. judge 모델은 `VERIFY_MODEL` 환경변수로 변경 가능(기본 gpt-4.1).
4. **채점 경로**: 에이전트가 낸 final_answer를 앵커로 GAIA 포맷팅만 수행(전사에서 답을 재유도하지 않음). 플랜이 포함된 메모리 사용.
5. **처치 통제**: search agent는 5개 구성 모두 동일하게 1회 플랜(react 포함). `--reflection`은 매니저에만 적용.
6. **견고성**: 부분 PARALLEL_LIST 보정(섹션 누락 방지), 플랜 파싱 실패 시 1회 재플랜 후 요란한 폴백(`plan_parse_failures` 기록),
   서브태스크 단위 에러 격리, 결과 jsonl에 `planning_mode`/`replan_count`/`plan_parse_failures`/`subtask_records` 필드 추가.
7. **평가**: task_id 중복 제거(최신 우선), Wilson 95% CI, 페어드 McNemar 정확검정(`analyze_gaia_results.py`가 자동 출력).
   에러 태스크 재실행은 `--retry_errors` 플래그.

n=30/레벨이면 95% CI가 ±18%p까지 벌어진다. McNemar p가 유의하지 않은 차이는 노이즈로 취급하고,
가능하면 레벨당 문제 수를 늘리거나 반복 실행으로 페어드 비교를 강화할 것.

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
