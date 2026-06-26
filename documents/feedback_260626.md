# Geometry Dash Live DQN Feedback (2026-06-26)

## 0. Executive Summary

현재 결과는 **DQN target 수식 자체가 명백히 틀렸다는 증거는 아니다.** `gd_rl/live_learner.py`의 기본 TD target은 표준적인 vanilla 1-step DQN 형태다.

```text
Q(s, a) <- r + gamma * (1 - done) * max_a' Q_target(s', a')
```

다만 현재 live Geometry Dash 환경에서는 vanilla DQN의 불안정성뿐 아니라, 구현/실험 설계상 결과 해석을 흐릴 수 있는 문제가 있다. 특히 다음 항목은 다음 long live training 전에 우선적으로 고치는 것이 좋다.

1. `max_steps` timeout과 실제 terminal death/clear를 구분해야 한다.
2. live attempt 도중 backprop을 수행하면 control timing 자체가 달라질 수 있다.
3. action history에는 policy가 선택한 desired action과 adapter/actuator state를 분리해서 넣어야 한다.
4. delayed observation + pending humanized input 때문에 현재 문제는 POMDP에 가깝다.
5. final checkpoint만 저장/평가하면 안 되고, periodic greedy validation으로 best checkpoint를 선택해야 한다.
6. 다음 개선 순서는 `decision stride/action repeat -> n-step return -> Double DQN -> explicit actuator state -> recurrent model`이 적절하다.

---

## 1. 구현 버그 가능성이 큰 것

### 1.1 `max_steps` timeout을 true terminal처럼 처리하는 문제

현재 `LivePracticeEnv.step()`은 다음 중 하나라도 만족하면 `done=True`를 반환한다.

```text
death
clear
step_count >= max_steps
```

그런데 DQN replay에는 이 `done` 값이 그대로 저장된다. 그러면 `max_steps` 때문에 episode가 잘린 transition도 다음과 같이 target이 계산된다.

```text
target = r
```

즉 다음 상태의 Q-value를 bootstrap하지 않는다.

이것은 `max_steps=600` run에서 best 92%까지 갔지만 timeout된 사례에 특히 중요하다. 이 상태는 실제 death나 clear가 아니라 단순히 수집이 잘린 상태이므로, DQN 관점에서는 terminal로 처리하면 안 된다.

권장 수정:

```python
@dataclass(frozen=True)
class LiveStepResult:
    observation: LivePracticeObservation
    reward: float
    terminated: bool   # actual death or clear
    truncated: bool    # max_steps / time limit / controlled cutoff
    info: dict[str, Any]
```

DQN target은 다음처럼 계산한다.

```python
bootstrap_mask = 1.0 - terminated.float()
target = reward + bootstrap_mask * discount * next_q
```

구분 기준:

| 종료 원인 | terminated | truncated | bootstrap |
|---|---:|---:|---:|
| 실제 death | True | False | 안 함 |
| clear | True | False | 안 함 |
| max_steps | False | True | 함 |
| bridge fault / manual reset | 별도 abort | 별도 abort | replay 제외 권장 |

---

### 1.2 live attempt 도중 backprop이 control timing을 바꿀 수 있음

현재 DQN loop는 대략 다음 순서다.

```text
action 선택
-> env.step(intent)
-> replay append
-> _optimize_dqn() 호출
-> target sync 가능
-> 다음 action 선택
```

즉 warmup 이후에는 거의 매 live step마다 replay sampling, tensor 생성, forward, backward, optimizer step이 들어간다.

일반적인 Gym simulator라면 괜찮다. `env.step()` 이후 학습을 오래 해도 simulator 안의 시간은 멈춰 있기 때문이다.

하지만 Geometry Dash live 환경에서는 게임이 실시간으로 계속 진행된다. 따라서 backprop이 오래 걸리면 다음 decision이 실제 게임 tick 기준으로 늦게 나갈 수 있다.

예시:

```text
training:
  tick 100: hold 선택 -> press intent
  tick 101~?: Python이 DQN update 수행
  tick 108: 다음 observation 보고 idle 선택 -> release intent

evaluation:
  tick 100: hold 선택 -> press intent
  tick 101~102: 바로 다음 observation 보고 idle 선택 -> release intent
```

그러면 같은 Q-network라도 training 때와 evaluation 때의 release timing이 달라진다. Ship control은 몇 tick 차이로 궤적이 크게 달라질 수 있으므로, 이 차이는 중요하다.

특히 final greedy evaluation에서는 `--dqn-warmup-steps 100000`을 사용했다. 이 경우 `_optimize_dqn()`은 replay size가 100000에 도달하기 전까지 즉시 return하므로, evaluation 중에는 backprop/update 비용이 거의 없다.

따라서 현재 train/eval은 다음처럼 다를 수 있다.

```text
training:
  Human delay + motor delay + jitter + Python 학습 지연

evaluation:
  Human delay + motor delay + jitter
```

권장 수정:

```text
attempt 진행 중:
  frozen Q-network로 action inference만 수행
  transition은 replay에 저장만 함
  gradient update는 하지 않음

attempt 종료 후:
  replay buffer에서 여러 번 DQN update
  target network sync
  다음 attempt 시작
```

이렇게 하면 training play timing과 greedy evaluation timing이 훨씬 가까워진다. 또한 어떤 attempt가 clear했을 때 “하나의 고정 checkpoint가 만든 trajectory”라고 해석할 수 있다.

---

### 1.3 replay buffer sampling 비용

현재 replay sample은 매번 내부 deque를 list로 변환한 뒤 sample한다.

```python
random.sample(list(self._items), batch_size)
```

`replay_capacity=50000`이면 live step마다 50000개 reference를 새 list로 복사할 수 있다. 이는 작은 MLP의 GPU 계산보다 더 큰 overhead가 될 수 있고, live control 지연을 키울 수 있다.

권장 수정:

- replay buffer를 list/ring buffer 기반으로 구현한다.
- sampling 시 전체 복사를 피한다.
- live attempt 중 update를 끄면 이 문제의 control-timing 영향은 크게 줄어든다.

---

### 1.4 checkpoint load 시 config compatibility 검증 부족

Checkpoint에는 encoder config와 history length가 저장되지만, load 시 현재 실행 config와 compatible한지 강하게 검증하지 않는다.

확인해야 할 항목:

```text
algorithm
hidden_size
history_length
encoder max_tick / x_scale / y_scale / velocity_scale / rotation_scale
decision_stride
reward_config
DQN variant config
```

특히 `encoder.max_tick`은 `--max-steps`에 따라 달라질 수 있으므로, 다른 max_steps로 checkpoint를 평가하면 feature scale이 바뀔 수 있다.

---

## 2. Action History와 Actuator State 문제

현재 policy output은 두 가지 desired input state다.

```text
idle
hold
```

하지만 이것이 곧바로 게임 입력이 되는 것은 아니다.

```text
policy desired state
-> ButtonStateIntentAdapter
-> no_op / press / release intent
-> HumanizedAgent delay/drop/jitter
-> Geode bridge actual input
```

따라서 DQN state/history에는 다음 두 종류의 정보를 분리해서 넣어야 한다.

```text
1. policy가 무엇을 선택했는가?
   selected_desired_state = idle / hold

2. 그 결과 input system이 어떤 상태가 되었는가?
   commanded_state_after_adapter = idle / hold
   emitted_intent = no_op / press / release
   dwell_blocked = true / false
```

### 2.1 현재 문제가 생기는 예시

예를 들어:

```text
tick 10:
  adapter state = hold
  last transition = tick 10 press
  min_dwell_ticks = 4

tick 12:
  policy selected desired state = idle
  하지만 dwell 때문에 release 불가능
  adapter output = no_op
  adapter state = hold 유지
```

이 상황에서 중요한 정보는 다음이다.

```text
policy selected action: idle
emitted intent: no_op
commanded/adapter state: hold
dwell_blocked: true
```

하지만 현재 DQN loop에서는 history append 시 `decision.effective_input_state`와 `intent_kind`를 넣는다. 그러면 위 상황은 history에 다음처럼 남을 수 있다.

```text
desired_input_state = hold
intent_kind = no_op
```

즉 “policy는 idle을 원했지만 dwell에 막혔다”는 정보가 사라진다. Replay action은 `idle`인데 next-state history는 마치 `hold`를 원했던 것처럼 보일 수 있다.

권장 history entry:

```python
@dataclass
class LiveActionHistoryEntry:
    selected_desired_state: DesiredInputState       # policy output
    commanded_state_after_adapter: DesiredInputState # adapter state after gating
    emitted_intent_kind: ActionKind                 # no_op / press / release
    dwell_blocked: bool
    ticks_since_last_transition: int | None
```

추가로 current state feature에도 다음을 넣는 것이 좋다.

```text
adapter_commanded_state
can_transition_now
dwell_ticks_remaining
latest_tick - policy_observation.tick
pending_humanized_event_count
recent drop flag
observed delayed input_down
```

---

### 2.2 drop 이후 adapter desync 문제

Adapter는 intended state를 기준으로 press/release를 추적한다. 그런데 HumanizedAgent가 press를 drop하면 실제 게임 input은 바뀌지 않는다.

예시:

```text
tick 100:
  policy = hold
  adapter emits press
  adapter state = hold
  HumanizedAgent drops press
  actual game input = idle

tick 101:
  policy = hold
  adapter thinks already hold
  adapter emits no_op
  actual game input remains idle
```

즉 adapter state와 actual input state가 달라질 수 있다. 이 문제를 해결하려고 adapter를 observed `input_down`으로 매 tick 강제로 덮어쓰면 pending motor event가 있는 동안 duplicate press/release를 반복 발행할 수 있다.

더 좋은 구조:

```text
commanded adapter state
observed delayed input state
pending edge kind / due tick
last drop result
retry timeout
```

TopPlayer에서는 drop 확률이 낮지만 0은 아니며, ship에서는 한 번의 drop도 trajectory를 크게 망칠 수 있다.

---

## 3. Delayed Observation + Pending Actuator는 POMDP

POMDP라는 말은 다음 뜻이다.

```text
policy가 보는 observation만으로는
현재 실제 게임 상태와 입력 상태를 완전히 알 수 없다.
```

DQN은 보통 현재 state가 Markov하다고 가정한다.

```text
현재 state s_t와 action a_t만 알면
다음 state와 reward 분포가 정해진다.
```

하지만 현재 agent는 delayed compact observation을 본다. 즉 policy가 보는 것은 실제 현재 상태가 아니라 몇 tick 전 상태다.

또한 이전에 보낸 press/release가 HumanizedAgent 안에서 아직 pending 상태일 수 있다.

policy가 보는 것:

```text
몇 tick 전 percent
몇 tick 전 y / y_vel
몇 tick 전 input_down
몇 tick 전 mode / gravity
recent action history 일부
```

policy가 직접 모르는 것:

```text
현재 실제 y / y_vel
방금 보낸 press가 이미 적용됐는지
pending release가 곧 적용될지
이전 event가 drop됐는지
adapter가 hold라고 믿는지 idle이라고 믿는지
dwell timer가 얼마나 남았는지
Python/geode lag가 몇 tick인지
```

### 3.1 같은 observation, 다른 hidden state

예를 들어 policy가 보는 delayed observation이 둘 다 다음과 같다고 하자.

```text
5 tick 전:
  y 낮음
  y_vel 아래쪽
  input_down = false
```

하지만 실제 hidden state는 다를 수 있다.

Case A:

```text
이전에 press를 보냈고
HumanizedAgent에 pending press가 있음
곧 press가 실제 적용될 예정
```

Case B:

```text
이전에 press를 보냈지만 drop됨
pending press 없음
actual input도 여전히 idle
```

두 경우 모두 delayed observation은 같아 보일 수 있지만, 좋은 action은 다르다.

```text
Case A: 곧 press 효과가 오므로 idle/release가 맞을 수 있음
Case B: press가 drop됐으므로 hold/press가 맞을 수 있음
```

즉 DQN 입장에서는 같은 observation에 서로 다른 정답 action이 섞인다. 그러면 Q-value가 평균화되고, 한쪽 action으로 collapse하기 쉬워진다.

### 3.2 해결 방향

바로 recurrent model로 가기 전에, 먼저 다음 정보를 explicit feature로 넣는 것이 좋다.

```text
observation_age_ticks = latest.tick - policy_observation.tick
adapter_commanded_state
selected_desired_state history
emitted_intent history
pending_humanized_event_count
last_intended_edge_tick
last_commanded_transition_tick
dwell_ticks_remaining
recent drop count / last drop flag
```

그래도 부족하면 GRU/LSTM 기반 recurrent DQN을 고려한다. 단, recurrent model은 단순히 MLP를 GRU로 바꾸는 것으로 끝나지 않는다.

필요한 것:

```text
episode sequence replay
sequence sampling
hidden state burn-in
terminal mask
truncated mask
sufficient sequence length
```

---

## 4. Vanilla DQN의 예상 한계

### 4.1 epsilon-greedy clear와 greedy failure는 충분히 가능함

150-attempt training에서 clear가 2번 나왔지만 final deterministic greedy가 31.39%에서 5/5 death한 것은 이상한 현상이 아니다.

이유:

1. training 중에는 epsilon-greedy exploration이 있다.
2. epsilon=0.05라도 653-step episode에서는 non-greedy action이 여러 번 발생한다.
3. ship에서는 몇 번의 random release가 trajectory를 크게 바꿀 수 있다.
4. attempt 중에도 Q-network가 계속 update되므로, clear trajectory는 하나의 fixed policy가 만든 것이 아닐 수 있다.
5. final checkpoint는 clear 이후 몇 attempt 더 update된 결과라 clear-capable transient policy가 사라졌을 수 있다.

따라서 training clear는 “학습이 완전히 무의미하다”는 뜻은 아니지만, “final greedy policy가 clear할 수 있다”는 증거도 아니다.

---

### 4.2 gamma time scale 문제

현재 한 live step이 거의 1 tick이라면 `gamma=0.99`는 tick 단위 discount다. 그러면 reward half-life가 약 69 ticks 정도다.

```text
0.99^69 ≈ 0.5
```

653 tick 뒤 clear bonus는 초기 state에서 거의 사라진다.

```text
0.99^653 ≈ 0.0014
```

따라서 clear bonus는 1-step DQN으로 초반 action까지 전달되기 어렵다. Decision stride를 4 ticks로 두고 gamma를 decision 단위로 적용하면 effective horizon이 훨씬 길어진다.

권장:

```text
decision_stride = 4
gamma = 0.99 ~ 0.995 per decision
```

또는 tick 기반 discount를 유지하려면:

```text
discount = gamma_tick ** elapsed_ticks
gamma_tick은 0.999 근처부터 검토
```

---

### 4.3 vanilla max target의 overestimation

현재 DQN target은 다음 형태다.

```text
max_a Q_target(next_state, a)
```

이 방식은 function approximation error가 있을 때 Q-value overestimation을 만들 수 있다. Action이 idle/hold 두 개뿐이어도 문제가 사라지지는 않는다.

특히 hold가 약간 과대평가되면 전체 state-space에서 hold가 더 자주 선택되고, 다시 hold 경험이 더 많이 replay되면서 collapse가 강화될 수 있다.

권장:

```python
# Double DQN target
next_action = q_network(next_features).argmax(dim=1, keepdim=True)
next_q = target_network(next_features).gather(1, next_action).squeeze(1)
target = reward + discount * bootstrap_mask * next_q
```

---

## 5. Reward / Environment 설계 문제

### 5.1 progress reward는 거의 survival reward처럼 작동함

현재 step reward는 progress delta, best progress bonus, section survival bonus, clear bonus, death penalty 중심이다.

Geometry Dash에서는 살아 있기만 하면 percent가 자동으로 증가한다. 따라서 대부분의 nonterminal step에서 hold와 idle 모두 즉시 양의 progress reward를 받는다.

문제는 action의 진짜 효과가 늦게 나타난다는 점이다.

```text
현재 hold/idle 선택
-> 몇 tick 뒤 y/y_vel 변화
-> 더 나중에 death 또는 survival 차이
```

하지만 reward는 매 tick progress로 즉시 들어온다. 그러면 DQN은 다음과 같은 local policy를 높게 평가할 수 있다.

```text
계속 hold
-> 31%까지 안정적으로 progress reward 획득
-> 나중에 death
-> terminal death penalty는 상대적으로 작거나 너무 늦음
```

특히 death penalty가 percent가 높을수록 작아지는 구조라면, progress를 꽤 얻고 죽는 policy가 생각보다 나쁘지 않게 보일 수 있다.

---

### 5.2 input_rate_penalty는 조심해야 함

현재 문제는 rapid toggling보다 hold collapse에 가깝다. 이 상태에서 edge penalty를 키우면 press/release를 더 줄이고, “한 번 press 후 계속 hold/no_op”를 더 강화할 수 있다.

따라서 다음 long run에서는 input_rate_penalty를 크게 주는 것보다 decision stride와 dwell로 제어 빈도를 줄이는 것이 더 안전하다.

권장 reward 실험:

```text
progress shaping:
  progress_delta와 best_progress_bonus 중 하나만 사용

terminal:
  death = 일정한 -1 또는 -2
  clear = +5 또는 +10

section bonus:
  처음에는 제거하거나 아주 작게

input_rate_penalty:
  일단 0 유지
```

---

## 6. 개선 우선순위

### 6.1 가장 먼저: decision stride / action repeat

한 tick마다 desired state를 새로 결정할 필요가 없다. Ship control에서는 4 ticks 정도의 decision stride가 더 자연스럽다.

중요한 것은 press event를 반복 전송하는 것이 아니다.

```text
decision tick:
  policy가 desired hold 선택
  adapter가 필요하면 press intent 1개 생성

다음 3 ticks:
  desired state 유지
  새 edge intent 없음
  observation/reward만 누적
```

효과:

```text
240Hz decision -> 60Hz decision
Python/GPU overhead 감소
history 4개가 4 ticks가 아니라 16 ticks를 커버
min_dwell_ticks=4와 자연스럽게 정렬
gamma horizon 개선
rapid toggling 감소
```

---

### 6.2 다음: n-step return

Human visual delay 5 tick + motor delay 5 tick + ship inertia를 고려하면, 1-step target은 credit assignment가 너무 짧다.

권장 시작점:

```text
decision_stride = 4
n_step = 4 ~ 8 decisions
즉 16 ~ 32 ticks 정도의 return
```

n-step target:

```text
R_n = r_t + gamma r_{t+1} + ... + gamma^(n-1) r_{t+n-1}
target = R_n + gamma^n Q_target(s_{t+n}, a)
```

---

### 6.3 Double DQN

구현 비용이 작고 Q overestimation을 줄일 수 있으므로 n-step과 함께 넣는 것이 좋다.

우선순위:

```text
1. decision stride
2. n-step return
3. Double DQN
4. explicit actuator state features
5. recurrent model
6. dueling DQN
```

Dueling DQN은 현재 action이 두 개뿐이고 핵심 문제가 horizon/POMDP/live timing이므로 나중에 해도 된다.

---

## 7. Checkpoint / Evaluation Protocol

Final checkpoint만 저장하고 평가하는 것은 적절하지 않다. Off-policy neural RL에서 마지막 checkpoint가 best라는 보장은 없다.

또한 training 중 clear attempt는 epsilon exploration과 특정 human seed에 의존했을 수 있으므로, training best percent만으로 best checkpoint를 고르면 안 된다.

권장 protocol:

```text
training attempts 중 주기적으로 snapshot 저장
예: every 5 or 10 attempts

각 snapshot에 대해:
  learning off
  epsilon = 0
  replay에 eval transition 저장하지 않음
  fixed validation human seeds 5개 이상
  greedy deterministic evaluation

선택 기준:
  1. clear rate
  2. median best_percent
  3. 20th percentile best_percent
  4. median death_percent
  5. event count는 tie-breaker
```

최종 보고는 held-out human seeds로 한다.

```text
clear rate
mean/median best_percent
20th percentile best_percent
death_percent distribution
desired/commanded/executed action counts
Geode lag diagnostics
```

---

## 8. 다음 long live training 전 최소 실험

### 8.1 기존 artifact 분석

먼저 clear attempt 142/147과 greedy eval attempt들의 `geode_diagnostics.json`을 비교한다.

확인할 값:

```text
requested_to_received_lag_ticks p50 / p95 / max
requested_to_applied_lag_ticks p50 / p95 / max
received_to_applied_lag_ticks p50 / p95 / max
observation tick delta distribution
```

해석:

```text
training clear와 greedy eval의 lag가 다름:
  train/eval control timing mismatch 가능성 큼

lag가 거의 같음:
  policy/reward/credit assignment 문제 가능성 큼
```

---

### 8.2 final checkpoint epsilon sweep

같은 final checkpoint를 update 없이 평가한다.

```text
epsilon = 0.00
epsilon = 0.01
epsilon = 0.05
```

동일한 human seed panel로 비교한다.

해석:

```text
epsilon > 0에서만 좋아짐:
  clear가 exploration-dependent였을 가능성 큼

epsilon = 0에서도 특정 seed만 좋음:
  seed-specific trajectory 가능성 큼

모두 31% hold collapse:
  final Q-network 자체가 collapse
```

---

### 8.3 Q-margin 분석

Checkpoint를 기존 trace states에 replay해서 다음 값을 기록한다.

```text
Q_idle
Q_hold
Q_margin = Q_hold - Q_idle
```

binning 기준:

```text
percent
y
y_vel
delayed input_down
adapter commanded state
dwell remaining
```

보고 싶은 것:

```text
모든 state에서 Q_hold가 큼:
  명확한 hold collapse

Q 차이가 거의 0인데 hold 선택:
  tie-like bias / small approximation bias

특정 y/y_vel에서만 hold 과대:
  state/reward 설계 문제
```

---

## 9. 테스트 보강 체크리스트

다음 long run 전에 최소한 다음 unit test를 추가하는 것이 좋다.

```text
[ ] true terminal target은 r만 사용한다.
[ ] max_steps truncation target은 bootstrap한다.
[ ] Double DQN target은 online argmax + target evaluation을 사용한다.
[ ] n-step return이 terminal/truncation을 올바르게 처리한다.
[ ] target sync가 learner update count 기준으로 발생한다.
[ ] replay buffer가 capacity 초과 시 oldest transition을 제거한다.
[ ] dwell-blocked 상황에서 selected desired와 commanded state가 모두 history에 남는다.
[ ] dropped press 이후 adapter/observed input desync가 log/state에 반영된다.
[ ] equal tick observation은 death가 아니라 duplicate/stale로 처리한다.
[ ] checkpoint config mismatch 시 load를 거부한다.
```

---

## 10. 권장 next run 구성

150-attempt long run을 다시 돌리기 전에 20~30 attempts 정도의 짧은 실험을 권장한다.

```text
core:
  policy frozen during each attempt
  update only between attempts

control:
  decision_stride = 4
  min_dwell_ticks = 4
  history_length = 4 ~ 6 decisions

DQN:
  Double DQN = on
  n_step = 4 or 8 decisions
  gamma = 0.99 ~ 0.995 per decision
  learning_rate = 1e-4 or 3e-4

replay:
  warmup = at least 3~5 complete attempts
  replay update ratio = fixed, not tied to wall-clock live tick

reward:
  input_rate_penalty = 0 initially
  simplify progress shaping if hold collapse persists

evaluation:
  greedy validation every 5 attempts
  fixed validation human seeds
  save best_validation checkpoint
```

---

## 11. Final Diagnosis

현재 결과는 다음처럼 보는 것이 가장 타당하다.

```text
DQN target 수식 자체가 명백히 틀린 것은 아니다.

하지만 live training 중 clear는
  epsilon-greedy exploration,
  within-attempt network updates,
  live backprop-induced timing delay,
  noisy humanized execution,
  transient Q-values
의 조합으로 나온 것일 수 있다.

final greedy policy가 hold로 collapse한 것은
  vanilla 1-step DQN의 불안정성,
  progress-heavy delayed reward,
  partial observability,
  missing actuator state,
  final-only checkpoint selection
때문에 충분히 가능한 현상이다.
```

따라서 다음 작업은 “DQN target을 고치는 것”보다 다음이 우선이다.

```text
1. terminal/truncation 분리
2. live control 중 backprop 제거
3. action history/state 정리
4. decision stride 도입
5. n-step + Double DQN
6. periodic greedy validation checkpoint
7. Geode lag diagnostics 기반 train/eval timing 검증
```
---

## 12. Live DQN run log - 2026-06-26

Result log only. Do not treat this section as root-cause analysis.

### Code/config facts before the run

- DQN epsilon default was changed to PickleGawd-style schedule:
  `epsilon = max(0.01, 1.0 * 0.995 ** (attempt_index - 1))`.
- PickleGawd reward style was available and used for the live runs:
  default reward `0.01`, jump punishment `-0.2`, death penalty `-10`,
  clear bonus `100`.
- Clear delay was set to `8.0` seconds.
- `--max-steps 0` was added to disable live-env truncation.
- Related tests passed before the no-cap run:
  `python -m pytest tests/test_live_practice_env.py::test_live_env_allows_zero_max_steps_to_disable_truncation tests/test_live_practice_env.py::test_live_env_reports_max_steps_as_truncation tests/test_run_live_rl_practice_geode.py::test_build_live_rl_configs_keep_smoke_defaults -q`.

### Artifacts

- 20-attempt smoke:
  `artifacts/live_dqn_picklegawd_epsilon_smoke_260626`
  - Used PickleGawd reward + PickleGawd epsilon.
  - This smoke accidentally used runner defaults for some DQN knobs:
    `decision_stride=1`, `n_step_return=1`, `gamma=0.99`,
    `batch_size=32`, `warmup_steps=32`, `replay_capacity=2048`.
  - No clears.
  - Training best overall: `41.276039123535156`.
  - Eval after 20 attempts collapsed to no-input:
    3/3 eval attempts at `26.82291603088379`, action counts all idle.

- First 300 attempt try:
  `artifacts/live_dqn_picklegawd_epsilon_300attempt_260626`
  - Intended 300 attempts, but stopped/exited at saved attempt count `36`.
  - No clears.
  - Attempt 34 reached `78.25520324707031%` and was truncated by
    `max_steps=600`; it was not recorded as clear.
  - `runner_stderr.log`: `error: bridge communication failed: cannot read from timed out object`.

- Second 300 attempt try:
  `artifacts/live_dqn_picklegawd_epsilon_300attempt_260626_r2`
  - Started with `max_steps=1000`.
  - Stopped immediately after user rejected fixed max-step cap.

- Main no-cap run:
  `artifacts/live_dqn_picklegawd_epsilon_300attempt_260626_r3`
  - Command used: PickleGawd reward, PickleGawd epsilon,
    `decision_stride=4`, `n_step_return=8`, `gamma=0.995`,
    Double DQN on, `learning_rate=0.0003`, `batch_size=64`,
    `replay_capacity=10000`, `warmup_steps=256`,
    `target_update_interval=250`, `eval_attempts=3`,
    `eval_interval_attempts=50`, `post_terminal_delay_seconds=8`,
    `timeout_seconds=15`, `max_steps=0`, `fps=144`.
  - Process was manually stopped after the user said to leave analysis to
    the next chat.
  - `training_summary.json` may be partially written because of manual stop.
    Use per-attempt `summary.json` files for the stable result log.
  - Saved per-attempt summary dirs counted: `235`.
  - First saved attempt index: `1`.
  - Last saved attempt dir index: `312` because eval attempts share the
    live attempt numbering space.

### Main no-cap run results

- Clear attempt dirs:
  `49`, `55`, `83`, `96`, `119`, `162`.
- All six clear attempt diagnostics recorded `live_post_terminal_delay`
  with `delay_seconds=8.0`.
- Clear delay ticks:
  - attempt 49: tick `768`
  - attempt 55: tick `767`
  - attempt 83: tick `767`
  - attempt 96: tick `770`
  - attempt 119: tick `768`
  - attempt 162: tick `767`

Window stats from saved per-attempt summaries:

```text
window 1-50:
  clears=1
  best=100.000000
  mean_best=32.468919
  mean_reward=-26.020600
  mean_events=29.180000

window 51-100:
  clears=3
  best=100.000000
  mean_best=35.848326
  mean_reward=-20.098800
  mean_events=29.480000

window 101-150:
  clears=1
  best=100.000000
  mean_best=30.433705
  mean_reward=-17.575400
  mean_events=20.260000

window 151-200:
  clears=1
  best=100.000000
  mean_best=29.081327
  mean_reward=-14.831400
  mean_events=17.080000

window 201-235:
  clears=0
  best=35.019455
  mean_best=27.032169
  mean_reward=-12.934571
  mean_events=9.885714
```

Eval summaries observed before stop:

```text
after_attempt=50:
  clear_count=0
  best_percent_overall=26.82291603088379
  mean_best_percent=26.82291603088379

after_attempt=100:
  clear_count=0
  best_percent_overall=26.753246307373047
  mean_best_percent=26.753246307373047

after_attempt=150:
  clear_count=0
  best_percent_overall=26.82291603088379
  mean_best_percent=26.82291603088379

after_attempt=200:
  clear_count=0
  best_percent_overall=26.718547821044922
  mean_best_percent=26.718547821044922
```

Latest 10 saved attempt summaries before stop:

```text
attempt 303: best=26.82291603088379, executed_events=0, cleared=false
attempt 304: best=26.753246307373047, executed_events=0, cleared=false
attempt 305: best=26.753246307373047, executed_events=0, cleared=false
attempt 306: best=26.753246307373047, executed_events=0, cleared=false
attempt 307: best=26.82291603088379, executed_events=0, cleared=false
attempt 308: best=26.82291603088379, executed_events=0, cleared=false
attempt 309: best=26.82291603088379, executed_events=0, cleared=false
attempt 310: best=26.718547821044922, executed_events=0, cleared=false
attempt 311: best=26.718547821044922, executed_events=0, cleared=false
attempt 312: best=26.718547821044922, executed_events=0, cleared=false
```
