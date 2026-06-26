# Live RL Experiments 2026-06-26

## 목적

Geometry Dash live RL agent가 단발성 clear가 아니라 greedy/evaluation 모드에서 반복적으로 clear할 수 있는지 확인하고, 실패 원인을 분리하기 위해 실험했다.

핵심 판단 기준:

- training attempt 중 clear 여부
- greedy evaluation clear rate
- best percent가 정책으로 보존되는지
- action collapse, 특히 idle/no-op collapse 여부

## 구현 변경 요약

이번 실험 전후로 다음 기능을 추가했다.

- `decision_stride`
  - policy decision을 여러 live tick에 걸쳐 유지한다.
  - 실험에서는 주로 `--decision-stride 2`를 사용했다. 240 FPS 환경에서 policy click grid를 약 120 Hz로 맞추기 위한 설정이다.

- greedy evaluation 및 best checkpoint
  - `--eval-attempts`
  - `--eval-interval-attempts`
  - `--best-checkpoint-path`
  - 학습 중 exploration 성과와 deterministic policy 성능을 분리해서 기록한다.

- DQN target 개선
  - Double DQN target selection
  - n-step return
  - 실험에서는 주로 `--n-step-return 8`, `--gamma 0.995`를 사용했다.

- action collapse diagnostics
  - selected action counts
  - greedy action counts
  - q margin stats
  - max selected/effective/intent run
  - collapse flags

- 실험용 repeat action penalty
  - `--repeat-action-penalty`
  - `--repeat-action-penalty-free-decisions`

- live observation actuator feature
  - latest input state
  - delayed/current input mismatch
  - pending humanized event count
  - visual delay ticks

검증:

- 전체 테스트 통과: `python -m pytest -q`

## 실험 결과

### 1. DQN stride2 gamma0.995 30 attempts

Artifact:

- `artifacts/live_rl_practice_20260626_131546/training_summary.json`
- clear attempt: `artifacts/live_rl_practice_20260626_131546/attempt_018/summary.json`

결과:

- attempts: 30
- training clear: 1
- best attempt: attempt 18
- best percent: 100.0
- attempt 18 clear: true
- decision count: 384
- executed event count: 133
- dropped event count: 0

해석:

- 이 실험은 실제 clear trajectory를 하나 얻었다는 점에서 중요하다.
- 다만 epsilon이 남아 있는 training attempt에서 나온 clear라서 greedy policy가 clear를 배운 증거는 아니다.
- 이 clear trajectory는 imitation/behavior cloning seed 후보로 보존할 가치가 있다.

### 2. DQN stride2 gamma0.995 400 attempts

Artifact:

- `artifacts/live_rl_practice_20260626_131716/training_summary.json`

결과:

- attempts: 400
- training clear: 0
- best attempt: attempt 4
- best percent: 62.678802490234375
- 이후 40-50%대 attempt는 가끔 있었지만 clear는 없었다.

상위 attempt:

| attempt | best percent | cleared |
|---:|---:|---|
| 4 | 62.6788 | false |
| 45 | 50.5852 | false |
| 87 | 46.9441 | false |
| 160 | 46.6840 | false |
| 23 | 46.6840 | false |

해석:

- 긴 학습에서도 clear trajectory가 policy로 안정적으로 보존되지 않았다.
- best progress가 초반 attempt에 몰려 있고, 이후 개선이 누적되지 않았다.
- replay에 좋은 경험이 들어가도 final greedy policy로 굳지 않는 문제가 있다.

### 3. 기존 checkpoint greedy 평가

Artifacts:

- `artifacts/live_rl_practice_20260626_133711/training_summary.json`
- `artifacts/live_rl_practice_20260626_133737/training_summary.json`

결과:

| checkpoint source | greedy eval | eval best percent | action pattern |
|---|---:|---:|---|
| 30-attempt final | 0/10 | 26.7880 | mostly idle |
| 400-attempt final | 0/10 | 26.7880 | complete idle/no-op |

해석:

- 가장 중요한 부정적 증거다.
- training 중 clear 또는 high-progress attempt가 있었지만 final greedy policy는 26-27% 근처에서 죽는다.
- 특히 400-attempt final은 거의 press를 하지 않는 idle policy로 collapse했다.

### 4. Double DQN + 8-step return + greedy eval

Artifact:

- `artifacts/live_rl_practice_20260626_134037/training_summary.json`

설정 요약:

- attempts: 120
- `--decision-stride 2`
- `--gamma 0.995`
- `--n-step-return 8`
- Double DQN enabled
- greedy eval: every 30 attempts, 5 attempts each

결과:

- training clear: 0
- best attempt: attempt 10
- best percent: 45.123538970947266
- greedy eval:
  - after 30: 0/5, best 26.7880
  - after 60: 0/5, best 26.7880
  - after 90: 0/5, best 26.7880
  - after 120: 0/5, best 26.7880

해석:

- Double DQN과 n-step return만으로는 collapse가 해결되지 않았다.
- training 중에는 exploration 때문에 40%대까지 가지만 greedy policy는 26-27% local policy로 돌아간다.

### 5. Repeat action penalty 실험

Artifact:

- `artifacts/live_rl_practice_repeat005_eval80/training_summary.json`

설정 요약:

- attempts: 80
- Double DQN enabled
- `--n-step-return 8`
- `--repeat-action-penalty 0.05`
- `--repeat-action-penalty-free-decisions 20`
- greedy eval: every 20 attempts, 5 attempts each

결과:

- training clear: 0
- best attempt: attempt 29
- best percent: 34.85045623779297
- greedy eval:
  - after 20: 0/5, best 28.3485
  - after 40: 0/5, best 26.7880
  - after 60: 0/5, best 26.7880
  - after 80: 0/5, best 26.7880

해석:

- 단순 반복 action penalty는 도움이 되지 않았다.
- best progress도 Double DQN + n-step baseline보다 낮아졌다.
- action collapse의 원인은 단순히 같은 action을 오래 반복하는 비용 부족이 아니라, state/reward/credit assignment 구조 문제에 가깝다.

### 6. Actuator/pending feature 실험

Artifact:

- `artifacts/live_rl_practice_actuator_eval80/training_summary.json`

설정 요약:

- attempts: 80
- Double DQN enabled
- `--n-step-return 8`
- repeat penalty 없음
- observation feature에 pending actuator state 추가
- greedy eval: every 20 attempts, 5 attempts each

결과:

- training clear: 0
- best attempt: attempt 41
- best percent: 38.28125
- greedy eval:
  - after 20: 0/5, best 27.34375
  - after 40: 0/5, best 26.8229
  - after 60: 0/5, best 26.8229
  - after 80: 0/5, best 26.8229

관찰:

- after 20 eval은 완전 idle collapse는 아니었다.
  - first eval action counts: hold 28, idle 75
- after 40부터 다시 idle/no-op collapse가 나타났다.
  - first eval action counts: hold 3, idle 100
  - later: hold 1-3, idle 100+

해석:

- actuator feature는 방향은 맞지만, 단독으로는 충분하지 않다.
- partial observability를 약간 줄였지만 Q-learning이 성공 trajectory를 안정적으로 보존하지 못하는 문제는 그대로 남았다.

### 7. Clear trajectory macro replay 시도

Input:

- `artifacts/live_rl_practice_20260626_131546/attempt_018/policy_intended_events.json`

Command:

```powershell
python scripts/run_geode_bridge_macro.py artifacts/live_rl_practice_20260626_131546/attempt_018/policy_intended_events.json artifacts/clear_attempt_018_policy_replay_trace.jsonl --timeout-seconds 15 --max-observations 1200
```

결과:

- trace 저장 중 실패
- error:

```text
gd_trace.trace_schema.TraceSchemaError: trace ticks must be strictly increasing at row 427
```

해석:

- macro replay의 성공/실패 여부를 아직 판단할 수 없다.
- bridge trace에 tick rewind 또는 duplicate/non-increasing tick이 들어온 것으로 보인다.
- replay 검증을 하려면 trace 저장 쪽에서 non-increasing tick diagnostics를 따로 처리하거나, reset/start guard를 붙인 replay runner가 필요하다.

## 종합 해석

현재 상태는 다음과 같다.

1. Agent는 exploration 중에 실제 clear trajectory를 한 번 찾았다.
2. 하지만 greedy policy가 반복 clear하는 상태는 아니다.
3. 긴 DQN 학습에서도 clear 경험이 final policy로 보존되지 않았다.
4. Double DQN과 n-step return은 필요한 개선이지만 충분하지 않았다.
5. 단순 repeat penalty는 오히려 성능을 낮췄다.
6. actuator/pending feature는 초반 collapse를 조금 늦췄지만, 이후 idle/no-op collapse를 막지 못했다.
7. 현재 가장 강한 자산은 attempt 18의 clear trajectory다.

가장 가능성 높은 실패 원인:

- delayed observation과 pending humanized input 때문에 같은 관측에 서로 다른 좋은 action이 섞인다.
- progress-heavy reward가 26-30%까지 살아남는 local policy를 강하게 보상한다.
- DQN이 sparse clear trajectory를 replay에서 충분히 보존하거나 일반화하지 못한다.
- greedy policy가 Q margin상 idle 쪽으로 조금씩 기울고, 이후 no-op/idle collapse로 고정된다.

## 다음 권장 방향

긴 DQN 실험을 더 돌리는 것보다 다음 순서가 유망하다.

1. Clear attempt 18 trajectory를 검증 가능한 macro/teacher trajectory로 정리한다.
2. 그 trajectory로 behavior cloning 또는 imitation pretraining을 한다.
3. pretrained policy를 greedy로 10-20회 평가한다.
4. greedy success가 일부라도 나오면, 그 정책에서 RL fine-tune을 시작한다.
5. replay buffer에 clear trajectory를 고정 비율로 섞는 demonstration replay를 추가한다.

즉 다음 실험 목표는:

```text
random/exploration DQN -> clear를 우연히 찾음
이제는 clear trajectory -> supervised seed -> greedy repeatability 확인
```

으로 전환하는 것이 좋다.

