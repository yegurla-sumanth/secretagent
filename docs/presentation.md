# Ptool Orchestrator: Auto-Composing LLM Pipelines

## Presentation Notes for Professor Meeting

---

## 1. The Problem

**Current state:** In secretagent, pipelines are hand-coded Python functions. A human decides which ptools to use and how to wire them together.

```python
# Someone manually wrote this workflow
def answer_question_workflow(narrative, question, choices):
    evidence = extract_suspects_and_evidence(narrative)
    verified = verify_alibis(narrative, evidence)
    text = deduce_murderer(narrative, verified, question, choices)
    return extract_index(text, choices)
```

**The question:** Can an LLM automatically discover this composition given only the ptool signatures and a task description?

---

## 2. Our Approach: The Ptool Orchestrator

**Key insight:** Separate the *composition problem* (which ptools to use, in what order, with what data flow) from the *execution problem* (how each ptool works internally).

The orchestrator:
- Takes a set of ptool **signatures + docstrings** (no implementation details)
- Takes a **task description** in natural language
- Uses a powerful LLM to generate a **Python workflow function**
- Validates the generated code with a **smoke test + retry** (pass@k)

**What it does NOT see:** Any existing hand-coded workflows, implementation details, or test data.

---

## 3. Architecture

```
User provides: ptools + task description + test case
         |
    [PtoolCatalog]  -- collects signatures/docstrings
         |
    [compose()]     -- single LLM call (Qwen 3.5 9B or 397B)
         |
    [ruff --fix]    -- deterministic code cleanup
         |
    [Pipeline]      -- exec() with ptools in namespace
         |
    [smoke test]    -- validate on first example
         |           -- retry up to max_retries on failure
    [Result]        -- callable bound to Interface
```

**Critical design decision:** The generated code calls `Interface` objects, not raw functions. This means the entire existing infrastructure (recording, cost tracking, caching, implementation swapping) works automatically.

---

## 4. Implementation Details

### Module structure (4 files, ~400 lines total)

| Module | Lines | Purpose |
|--------|-------|---------|
| `catalog.py` | ~90 | Collects ptool metadata for LLM prompt |
| `composer.py` | ~130 | LLM call + code extraction + ruff fix + retry |
| `pipeline.py` | ~80 | Compiles generated code into callable |
| `__init__.py` | ~100 | OrchestrateFactory + public API |

### Key technical choices

1. **Single prompt, not multi-step** — One well-crafted prompt suffices. The old AgentProject used 3 prompts (Analyze -> Compose -> Refine) which was over-engineered.

2. **Python code output, not JSON spec** — Simplest thing that works. The generated code is a regular Python function. No intermediate representation, no renderer.

3. **pass@k with error feedback** — On failure, the error message is appended to the prompt. Each retry is *informed*, not random. Configurable via `orchestrate.max_retries`.

4. **ruff --fix cleanup** — Deterministic post-processing catches formatting issues and simple mistakes before compilation.

### Thinking model compatibility fix

Discovered that Qwen 3.5 (a thinking model) splits output into `reasoning_content` and `content` fields. The `<answer>` tags that SimulateFactory needs can land in either field. Fix: prefer `content` if it has `<answer>` tags, fall back to `reasoning_content` if not.

This fix alone took accuracy from **0% to 80-90%** on both models.

---

## 5. Results

### Experiment: Math Word Problem Pipeline (10 cases)

**Setup:**
- 4 ptools: `parse_numbers`, `identify_operation`, `compute_result`, `format_answer`
- All implemented via `simulate` (LLM-backed, not hardcoded)
- Model: Qwen 3.5 9B for both ptools AND orchestration
- 10 test cases (addition, subtraction, multiplication, division)

| Metric | Hand-coded | Orchestrated |
|--------|:---:|:---:|
| **Accuracy** | **80%** (8/10) | **90%** (9/10) |
| Errors | 0 | 0 |
| Pass@k | N/A | 2 |

**The orchestrated pipeline matched or exceeded hand-coded accuracy.**

The 1-case difference (case 2) is due to LLM non-determinism in the underlying ptool calls — both pipelines have identical code structure.

### Generated code (by the 9B model)

```python
def solve_math_problem(problem: str) -> str:
    numbers = parse_numbers(problem)
    operation = identify_operation(problem)
    result = compute_result(numbers, operation)
    answer = format_answer(problem, result)
    return answer
```

This is **structurally identical** to the hand-coded pipeline — discovered purely from signatures and docstrings.

### Integration test results (5 tests, 397B orchestrator)

All 5 integration tests pass:
- 2-step pipeline (double -> add_one): pass@1
- 3-step pipeline (double -> add_one -> negate): pass@1
- Retry with smoke test: pass@1
- Config-driven `implement_via('orchestrate')`: pass

### Unit test results

22 unit tests, all passing (no API needed):
- Catalog construction, filtering, rendering
- Code extraction, def-line stripping
- Pipeline compilation, tool dispatch
- Build pipeline from interfaces

---

## 6. What This Enables (Long-term Vision)

### Phase 1 (DONE): Auto-Composition
LLM generates pipeline code from ptool signatures.

### Phase 2: Metric Collection
Aggregate per-ptool metrics from evaluation runs:
- Cost (USD per call)
- Latency (seconds per call)
- Success rate (when used vs. unused)
- Lift = success_rate - unused_success_rate

### Phase 3: Optimizer-Based Selection
Given ptool metrics + cost budget, an off-the-shelf optimizer selects the best subset:
- Greedy by lift (simple baseline)
- Knapsack solver (budget-constrained)
- Bayesian optimization (explores interactions)

### Phase 4: Pipeline Bank
Store, retrieve, and version generated pipelines:
- Embed task descriptions for similarity search
- Adapt existing pipelines for new tasks
- Track pipeline evolution as metrics change

### Phase 5+: Self-Correction, A/B Testing, Distillation
- Re-generate pipelines when accuracy drops
- Compare orchestrated vs hand-coded statistically
- Distill successful pipelines into learned Python implementations

---

## 7. Connection to TRACE Architecture

Our design maps to the TRACE (Tool-Routed Architecture for Controlled Execution) pattern:

| TRACE Concept | Our Implementation |
|---------------|-------------------|
| Deterministic tools | Ptools (Interface + Implementation) |
| Orchestrator (LLM) | compose() with Qwen 3.5 |
| Audit trail | record.py (already exists) |
| Self-correction | compose_with_retry (pass@k) |
| Cost tracking | llm_util stats + evaluate.py |

**Key difference from TRACE:** Our "tools" are themselves LLM-backed (via simulate), not purely deterministic. But the *composition* is what the orchestrator controls — the structure is deterministic even if the tool internals are probabilistic.

---

## 8. Relationship to Old Codebase (AgentProject)

| Aspect | AgentProject | secretagent |
|--------|-------------|-------------|
| Pipeline spec | JSON PipelineSpec + StageSpec | Python code (direct) |
| Composition | 3-prompt chain (Analyze->Compose->Refine) | Single prompt |
| Code generation | CodeRenderer (spec -> Python) | Direct code output |
| Domain adaptation | DomainAdapter ABC | Generic (no adapter needed) |
| Cost tracking | CostTable + LLMS.json | litellm completion_cost |
| Complexity | ~2000 lines across 6 files | ~400 lines across 4 files |

The new orchestrator achieves the same core functionality in **5x less code** by leveraging the cleaner secretagent architecture.

---

## 9. Technical Contributions

1. **Thinking model compatibility** — Discovered and fixed a systematic issue where thinking models (Qwen 3.5) split `<answer>` tags between `reasoning_content` and `content` fields. Prefer-content-with-fallback strategy.

2. **Pass@k for pipeline generation** — Framing pipeline generation reliability as pass@k (standard in code generation research). The orchestrator achieves pass@2 with the 9B model.

3. **Streaming support** — Added `llm.stream` config for real-time output visibility during long-running experiments.

4. **Config-driven orchestration** — `method: orchestrate` in YAML config, fully integrated with existing eval/recording infrastructure.

---

## 10. Next Steps

1. **Run on MUSR benchmark** — Murder mysteries, object placement, team allocation. Compare orchestrated vs hand-coded on the real benchmarks.

2. **Build metric aggregator** — Read results.jsonl to compute per-ptool success/cost/latency metrics.

3. **Implement ptool selector** — Greedy selection by lift as the first optimizer.

4. **Pipeline bank** — Save/load generated pipelines for reuse.

5. **Multi-dataset evaluation** — Test orchestrator generalization across different task types.
