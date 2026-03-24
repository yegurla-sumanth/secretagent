# Secretagent: Codebase for Dummies

*Accurate as of commit `10c9bd9` (2026-03-23). Refresh periodically as the codebase evolves.*

A complete zero-to-hero guide to the secretagent framework. No prior knowledge assumed.

---

## Table of Contents

- [Part 0: Background Concepts](#part-0-background-concepts)
- [Part 1: The Core Idea](#part-1-the-core-idea)
- [Part 2: How Interfaces Work](#part-2-how-interfaces-work)
- [Part 3: Factories — How Implementations Get Built](#part-3-factories--how-implementations-get-built)
- [Part 4: Configuration System](#part-4-configuration-system)
- [Part 5: LLM Calls and Caching](#part-5-llm-calls-and-caching)
- [Part 6: Recording System](#part-6-recording-system)
- [Part 7: Evaluation System](#part-7-evaluation-system)
- [Part 8: The Learning System](#part-8-the-learning-system--closing-the-loop)
- [Part 9: The Ptools Pattern](#part-9-the-ptools-pattern)
- [Part 10: End-to-End Walkthrough](#part-10-end-to-end-walkthrough--sports-understanding-benchmark)
- [Part 11: Architecture Diagram](#part-11-architecture-diagram)
- [Part 12: CLI Tools Reference](#part-12-cli-tools-reference)
- [Summary: Key Design Principles](#summary-key-design-principles)

---

## Part 0: Background Concepts

Before diving in, here are the libraries and concepts you'll encounter throughout the codebase.

### Python Decorators

A decorator wraps a function to modify its behavior. The `@` syntax is shorthand:

```python
@my_decorator
def my_function():
    pass

# This is exactly equivalent to:
def my_function():
    pass
my_function = my_decorator(my_function)
```

In secretagent, `@interface` is a decorator that takes your function and turns it into a special `Interface` object.

### Pydantic

A Python library for defining data structures with **automatic type validation**. You declare fields with types, and Pydantic ensures the data always matches:

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="Alice", age=30)        # Works fine
user = User(name="Alice", age="hello")   # Raises validation error!
```

Secretagent uses Pydantic for its internal data models (`Interface`, `Implementation`, `Case`, `Dataset`) and optionally for structured LLM outputs.

### LLM (Large Language Model)

A neural network (like GPT-4, Claude, etc.) that takes text in and produces text out. You send it a "prompt" (instructions) and it responds with generated text. In secretagent, LLMs are used to simulate what Python functions would do, or to generate code.

### litellm

A Python library that provides a **unified interface** to many LLM providers (OpenAI, Anthropic, Google, Together AI, etc.). Instead of learning each provider's API, you just call:

```python
litellm.completion(model="gpt-4o", messages=[...])
litellm.completion(model="claude-haiku-4-5-20251001", messages=[...])
litellm.completion(model="together_ai/deepseek-ai/DeepSeek-V3.1", messages=[...])
```

Same function, different model name.

### OmegaConf

A configuration library that lets you work with nested dictionaries using dot notation. Instead of `config['llm']['model']`, you write `config.get('llm.model')`. It can load from YAML files and merge multiple configs together.

### cachier

A caching library that stores function results on disk. If you call `expensive_function("same input")` twice, the second call reads from disk instead of recomputing. In secretagent, this means LLM calls are cached — run an experiment twice and the second time is free (no API costs).

### smolagents

A library from HuggingFace that provides, among other things, a **sandboxed Python executor**. It can run LLM-generated Python code safely, preventing it from doing dangerous things like deleting files or accessing the network.

### pydantic-ai

A framework built on Pydantic for creating AI "agents" that can use tools. An agent can:
1. Receive a prompt
2. Think about it
3. Call tools (Python functions you provide)
4. Get results back
5. Think more
6. Produce a final structured output (validated by Pydantic)

This think-act-observe loop is called **ReAct** (Reasoning + Acting).

---

## Part 1: The Core Idea

### The Problem

In ML/AI research, you often want to:
- Try different ways of solving sub-problems (LLM vs code vs lookup table)
- Run experiments comparing approaches
- Cache expensive LLM calls so you don't pay twice
- Track costs and performance
- Learn from past runs to make future runs cheaper

Secretagent solves all of this with one key insight: **make everything look like a Python function call**.

### The Solution: Interfaces

An **Interface** is a Python function that has:
- A name
- A type signature (what it takes in and returns)
- A docstring (describing what it does)
- **No implementation** (or an implementation that can be swapped at runtime)

```python
from secretagent.core import interface

@interface
def sport_for(x: str) -> str:
    """Return the name of the sport associated with a player, action, or event."""
```

This doesn't compute anything yet. It's a **specification** — "I need a function that takes a string and returns a sport name."

Later, you **bind** it to an implementation:

```python
sport_for.implement_via('simulate')  # "use an LLM to predict the output"
```

Now `sport_for("LeBron James")` works — it asks an LLM: "Given this function description and input, what would it return?" The LLM says `"basketball"`.

The magic: you can swap the implementation without changing any calling code:

```python
sport_for.implement_via('learned', learner='rote')  # use a lookup table instead
```

Same call, same result, zero API cost.

---

## Part 2: How Interfaces Work

**File:** `src/secretagent/core.py`

### What `@interface` Does Step by Step

When you write `@interface` on a function, Python calls the `interface()` function with your function as input. Here's what happens (`core.py:78-99`):

```python
def interface(func):
    full_src = inspect.getsource(func)  # grab the source code as a string
    trimmed_src = full_src[full_src.find('\ndef')+1:]  # remove the @interface line
    result = Interface(
        func=func,            # the Python function object
        name=func.__name__,   # e.g., "sport_for"
        doc=func.__doc__,     # the docstring
        src=trimmed_src,      # source code (used in LLM prompts later!)
        annotations=func.__annotations__,  # e.g., {'x': str, 'return': str}
    )
    _INTERFACES.append(result)  # register in a global list
    return result  # `sport_for` now points to an Interface object, not a function
```

The variable `sport_for` is no longer a regular function — it's an `Interface` object. But it's callable (it has a `__call__` method), so you can still use it like a function.

### What Happens When You Call an Interface

`sport_for("LeBron James")` hits `Interface.__call__` (`core.py:44-48`):

```python
def __call__(self, *args, **kw):
    if self.implementation is None:
        raise NotImplementedError(...)  # hasn't been bound yet!
    return self.implementation.implementing_fn(*args, **kw)
```

It simply delegates to whatever callable function was bound as the implementation.

### Binding an Implementation

`sport_for.implement_via('simulate')` hits (`core.py:50-54`):

```python
def implement_via(self, method: str, **kwargs):
    factory = _FACTORIES[method]  # look up 'simulate' in the factory registry
    self.implementation = factory.build_implementation(self, **kwargs)
```

It finds the factory named `'simulate'`, asks it to build a callable function, and stores it.

### Three Ways to Bind

```python
# Way 1: Create first, bind later
@interface
def sport_for(x: str) -> str: ...
sport_for.implement_via('simulate')

# Way 2: Create and bind at the same time
@implement_via('simulate')
def sport_for(x: str) -> str: ...

# Way 3: Bind everything from a config dict (most common in experiments)
implement_via_config(ptools_module, {'sport_for': {'method': 'simulate'}, ...})
```

Way 3 is what experiment files use. One YAML config file controls all implementations.

---

## Part 3: Factories — How Implementations Get Built

### What is a Factory?

A Factory is a class that knows how to create a working function from an Interface. Think of it as a "strategy" for implementation. Each factory is registered by name in a global dictionary:

```python
_FACTORIES = {
    'direct':             DirectFactory(),          # use real Python code
    'simulate':           SimulateFactory(),         # LLM predicts output
    'prompt_llm':         PromptLLMFactory(),        # LLM with custom prompt
    'program_of_thought': PoTFactory(),              # LLM writes & runs code
    'simulate_pydantic':  SimulatePydanticFactory(), # pydantic-ai agent
    'learned':            LearnedFunctionFactory(),  # lookup table from past runs
    'orchestrate':        OrchestrateFactory(),      # auto-generate workflow from ptools
}
```

### The Factory Base Class (`core.py:146-169`)

Every factory must implement one method:

```python
class Factory(BaseModel):
    def build_fn(self, interface, **kwargs) -> Callable:
        """Return a function that implements this interface."""
        ...
```

The parent class wraps this with `build_implementation()` which adds metadata tracking (which factory created it, what kwargs were used) — this is important for the learning system later.

---

### Factory 1: DirectFactory — "Just Run Python Code"

**File:** `src/secretagent/implement_core.py:76-87`

The simplest factory. Uses an actual Python function as the implementation.

```python
# Mode 1: The stub's own body IS the implementation
@implement_via('direct')
def add(a: int, b: int) -> int:
    return a + b  # this code runs when you call add(2, 3)

# Mode 2: Point to a different function
are_sports_consistent.implement_via('direct', fn=sports_understanding_workflow)

# Mode 3: Dotted name string (useful in config files)
# In YAML:  method: direct
#           fn: ptools.sports_understanding_workflow
```

**Why this matters:** This is how **workflows** work. A workflow is a plain Python function that calls other Interfaces:

```python
def sports_understanding_workflow(sentence):
    player, action, event = analyze_sentence(sentence)  # calls LLM
    player_sport = sport_for(player)                     # calls LLM
    action_sport = sport_for(action)                     # calls LLM
    result = consistent_sports(player_sport, action_sport)  # calls LLM
    return result
```

The workflow is bound via `direct` — it's just Python orchestration. But `analyze_sentence`, `sport_for`, and `consistent_sports` are separate Interfaces that can each use different factories (LLM, lookup table, etc.).

---

### Factory 2: SimulateFactory — "LLM, Pretend You're This Function"

**File:** `src/secretagent/implement_core.py:89-144`

This is the workhorse factory. Let's trace exactly what happens when you call `sport_for("LeBron James")`:

**Step A: Bind time** — `sport_for.implement_via('simulate')`

The factory creates a "closure" — a function that remembers the interface it was created for:

```python
def result_fn(*args, **kw):
    prompt = create_prompt(interface, *args, **kw)        # build the prompt
    llm_output, stats = llm_util.llm(prompt, model)       # call the LLM
    answer = parse_output(return_type, llm_output)         # extract answer
    record.record(func='sport_for', args=args, output=answer, stats=stats)
    return answer
```

**Step B: Call time** — `sport_for("LeBron James")`

First, `create_prompt()` loads the template file `src/secretagent/prompt_templates/simulate.txt` and substitutes values. The resulting prompt looks like:

```
Consider the following documentation stub for a Python function. Note
that this is documentation, not a full implementation.

    def sport_for(x: str)-> str:
      """Return the name of the sport associated with a player, action, or event.

      Examples:
      >>> sport_for('Bam Adebayo')
      'basketball'
      >>> sport_for('scored a reverse layup')
      'basketball'
      """

Imagine that this function was fully implemented as suggested by the
documentation stub, and that function were called with these arguments:

x = 'LeBron James'

Propose a possible output of the function for these inputs that is
consistent with the documentation.

<answer>
FINAL ANSWER
</answer>
```

**Step C: The LLM responds** with something like:

```
Based on the documentation, LeBron James is a well-known basketball player.

<answer>
basketball
</answer>
```

**Step D: parse_output()** (`implement_core.py:130-144`)

1. Regex finds text between `<answer>` and `</answer>` -> `"basketball"`
2. Looks at the return type annotation (`str`)
3. Calls `str("basketball")` -> `"basketball"`

If the return type were:
- `int` -> `int("42")` -> `42`
- `float` -> `float("3.14")` -> `3.14`
- `tuple[str, str]` -> `ast.literal_eval("('foo', 'bar')")` -> `('foo', 'bar')`

(`ast.literal_eval` safely evaluates Python literal expressions from strings — it's a safe way to parse structured data.)

**Step E: Caching and recording**
- The LLM call was cached to disk by `cache_util` — same input next time costs nothing
- `record.record()` logs the call for evaluation tracking

**Optional features:**
- **Few-shot examples:** `implement_via('simulate', example_file='examples.json')` adds example input/output pairs to the prompt, helping the LLM understand the expected format
- **Chain of thought:** If `llm.thinking` is `True` in config, adds `<thought>` scaffolding so the LLM can reason before answering
- **Per-call config:** `implement_via('simulate', llm={'model': 'gpt-4o'})` temporarily overrides the model for this specific interface

---

### Factory 3: PromptLLMFactory — "Use Your Own Prompt"

**File:** `src/secretagent/implement_core.py:147-205`

When the default simulate prompt doesn't work well for your task, you write your own. Uses Python's `string.Template` with `$variable` placeholders that match the function's argument names.

```python
@implement_via('prompt_llm',
    prompt_template_str='Translate $english_sentence to French. Return only the translation.',
    llm={'model': 'claude-haiku-4-5-20251001'})
def translate(english_sentence: str) -> str:
    """Translate English to French."""
```

When `translate("What's for lunch?")` is called:
1. Template substitution: `"Translate What's for lunch? to French. Return only the translation."`
2. Send to LLM
3. Extract answer based on `answer_pattern`

**The `answer_pattern` parameter:**
- Default: `<answer>(.*)</answer>` — extract between tags (same as simulate)
- `None` with `str` return type: use the entire LLM response as-is
- Custom regex: any pattern with one capture group

A real example from the benchmarks (`benchmarks/sports_understanding/prompt_templates/zeroshot.txt`):

```
Determine plausibility of a sports-related sentence.
Your answer should be either 'True' or 'False'. Use the following format:
<answer>
YOUR ANSWER
</answer>
The sentence is: $sentence
```

You can also load templates from files:

```python
my_func.implement_via('prompt_llm', prompt_template_file='my_prompt.txt')
```

---

### Factory 4: PoTFactory (Program of Thought) — "LLM Writes Code"

**File:** `src/secretagent/implement_core.py:207-294`

Instead of asking the LLM "what's the answer?", it asks "write Python code that computes the answer." The code runs in a sandbox.

**Why is this useful?** For complex multi-step reasoning, code is more reliable than natural language. The LLM is better at writing `if event: ...` than at doing conditional reasoning in prose.

**Detailed walkthrough:**

**Step A: Bind time** (`implement_core.py:210-236`)

```python
are_sports_consistent.implement_via('program_of_thought')
```

1. **Resolve tools** — which other Interfaces can the generated code call?
   - `tools='__all__'` (default): every other implemented Interface
   - `tools=None`: no tools, just built-in Python
   - `tools=[analyze_sentence, sport_for]`: specific list

2. **Create sandbox** — `LocalPythonExecutor` from smolagents
   - Tool functions are injected so the code can call them
   - `final_answer` function is added to capture the return value

**Step B: Call time** — `are_sports_consistent("Tim Duncan scored from inside the paint.")`

`create_prompt()` builds a prompt from `src/secretagent/prompt_templates/program_of_thought.txt`:

```
Consider the following documentation stub for a Python function.

    def are_sports_in_sentence_consistent(sentence: str) -> bool:
      """Determine plausibility of a sports-related sentence."""

The unseen implementation might call any of the following 'tool' methods.

    def analyze_sentence(sentence: str) -> tuple[str, str, str]:
      """Extract a names of a player, and action, and an optional event."""

    def sport_for(x: str) -> str:
      """Return the name of the sport associated with a player, action, or event."""

    def consistent_sports(sport1: str, sport2: str) -> bool:
      """Compare two descriptions of sports, and determine if they are consistent."""

Imagine that all of these functions were implemented... and that function
are_sports_in_sentence_consistent was called with these arguments:

sentence = 'Tim Duncan scored from inside the paint.'

Output Python code that would compute the output by calling the tools,
FOR THESE INPUTS. The code MUST call final_answer(result) as the last line.

Note it is NOT necessary for the code to be a completely general
implementation, it only needs to work for these specific inputs.
```

**Step C: LLM generates code:**

```python
sentence = 'Tim Duncan scored from inside the paint.'
player, action, event = analyze_sentence(sentence)
player_sport = sport_for(player)
action_sport = sport_for(action)
result = consistent_sports(player_sport, action_sport)
if event:
    event_sport = sport_for(event)
    result = result and consistent_sports(player_sport, event_sport)
final_answer(result)
```

**Step D: Code is extracted** from the LLM response using regex.

**Step E: Code is executed in the sandbox**

The sandbox has `analyze_sentence`, `sport_for`, `consistent_sports`, and `final_answer` available as callable functions. When the generated code calls `analyze_sentence(sentence)`, it goes through the *actual* `analyze_sentence` implementation (which might itself be an LLM call via `simulate`).

The `final_answer(result)` call captures the return value.

**Step F: Result is recorded** with the generated code saved in `step_info`:

```python
record.record(
    func='are_sports_in_sentence_consistent',
    args=('Tim Duncan scored from inside the paint.',),
    output=True,
    stats=stats,
    step_info={'generated_code': '...the code above...'}
)
```

**The `tools` parameter in detail:**

```python
# All other implemented Interfaces available as tools (default)
my_func.implement_via('program_of_thought', tools='__all__')

# No tools — generated code can only use built-in Python
my_func.implement_via('program_of_thought', tools=None)

# Specific tools only
my_func.implement_via('program_of_thought', tools=[analyze_sentence, sport_for])
```

---

### Factory 5: SimulatePydanticFactory — "Structured Output + Agent Tool Use"

**File:** `src/secretagent/implement_pydantic.py`

This uses pydantic-ai's `Agent` framework instead of raw litellm calls. It's more powerful but heavier.

**What makes it different from SimulateFactory?**

| Feature | SimulateFactory | SimulatePydanticFactory |
|---------|----------------|------------------------|
| How it calls LLM | `litellm.completion()` directly | pydantic-ai `Agent.run_sync()` |
| Output parsing | Regex: `<answer>...</answer>` | Automatic via Pydantic validation |
| Return types | str/int/float/literal_eval | Any Pydantic model |
| Tool use | No | Yes — agent autonomously calls tools |
| Caching | Via `cache_util.cached()` | Custom hash function |

**What is pydantic-ai's Agent?**

An Agent is an AI that can:
1. Receive a prompt
2. Think about it
3. Decide to call a "tool" (a Python function you provide)
4. Get the tool's result back
5. Think more
6. Produce a final answer in a structured format (validated by Pydantic)

This think-act-observe loop can repeat multiple times. The agent decides when it has enough information to answer.

**Example with structured output:**

```python
from pydantic import BaseModel

class StructuredSportsSentence(BaseModel):
    player: str
    action: str
    event: str | None

@implement_via('simulate_pydantic')
def analyze_sentence(sentence: str) -> StructuredSportsSentence:
    """Extract player, action, and event."""
```

The return type is a Pydantic model — pydantic-ai ensures the LLM's output matches this structure exactly. No regex parsing needed.

**Example with tool use:**

```python
@implement_via('simulate_pydantic', tools=[analyze_sentence, sport_for])
def is_consistent(sentence: str) -> bool: ...
```

When called, the agent might:
1. Think: "I need to extract the player and action"
2. Call `analyze_sentence("Tim Duncan scored...")` -> `StructuredSportsSentence(...)`
3. Think: "Now I need to find the sports"
4. Call `sport_for("Tim Duncan")` -> `"basketball"`
5. Call `sport_for("scored from inside the paint")` -> `"basketball"`
6. Think: "Both basketball, so consistent"
7. Return: `True`

**Important limitation:** Some models (like `together_ai/Qwen/Qwen3.5-9B` and `together_ai/google/gemma-3n-E4B-it`) don't support tool use. These models can't be used with `simulate_pydantic` when tools are involved.

**Custom caching** (`implement_pydantic.py:23-33`):

The cachier library needs a unique "key" for each call. But Agent objects aren't hashable. So `_run_agent_hashkey()` creates a SHA-256 hash from:

```python
(interface_name, model_name, str(return_type), prompt_text, tuple(tool_names))
```

---

### Factory 6: LearnedFunctionFactory — "Use Past Experience"

**File:** `src/secretagent/learn/implement_learn.py`

Loads a Python file generated by the learning system (covered in detail in Part 8).

```yaml
ptools:
  sport_for:
    method: learned
    learner: rote
    backoff: true
```

This:
1. Finds the most recent `training_data/*sport_for__rote/learned.py`
2. Imports the `sport_for` function from it
3. If `backoff=true`, wraps it: try the learned function first, fall back to the original LLM for unseen inputs

---

### Factory 7: OrchestrateFactory — "Auto-Generate the Workflow"

**Files:** `src/secretagent/orchestrate/`

With the factories above, you still need to hand-write workflow functions (like `sports_understanding_workflow`). The orchestrator **automates that step** — it asks a powerful LLM to compose available ptools into a pipeline, given just a task description.

**How it works:**

```yaml
ptools:
  analyze_sentence:
    method: simulate
  sport_for:
    method: simulate
  consistent_sports:
    method: simulate
  # Listed LAST so the above ptools are already implemented
  are_sports_in_sentence_consistent:
    method: orchestrate
    task_description: "Determine if sports references in a sentence are consistent."
```

**The full flow:**

1. **PtoolCatalog** (`catalog.py`) collects all implemented Interfaces (except the orchestrated one), extracting their signatures and docstrings

2. **compose()** (`composer.py`) sends a prompt to a powerful LLM with:
   - All tool stubs (signatures + docstrings)
   - The task description
   - The function signature the code should fit into
   - Instructions to write the function body using only the available tools

3. **Code extraction**: finds the ```` ```python ``` ```` block in the LLM response

4. **ruff --fix**: runs the code through the ruff linter for deterministic cleanup

5. **Pipeline** (`pipeline.py`) compiles the generated code via `exec()` in a namespace containing the ptool Interfaces, creating a callable function

6. The function is bound to the Interface via `OrchestrateFactory`

**The prompt** (`orchestrate/prompt_templates/compose.txt`) looks like:

```
You are an expert Python programmer. Your task is to write the body of a
workflow function that solves a task by composing calls to available tools.

## Available tools
```python
def analyze_sentence(sentence: str) -> tuple[str, str, str]:
    """Extract player, action, and event..."""

def sport_for(x: str) -> str:
    """Return the sport associated with..."""
```

## Task
Determine if sports references in a sentence are consistent.

## Function signature
```python
def are_sports_in_sentence_consistent(sentence: str) -> bool:
    # YOUR CODE HERE
```

Write ONLY the function body. Return your code in a ```python block.
```

The LLM generates something like:

```python
player, action, event = analyze_sentence(sentence)
player_sport = sport_for(player)
action_sport = sport_for(action)
result = consistent_sports(player_sport, action_sport)
if event:
    event_sport = sport_for(event)
    result = result and consistent_sports(player_sport, event_sport)
return result
```

**Retry mechanism**: `compose_with_retry()` adds automatic error recovery:
1. Generate code
2. Run a smoke test (build Pipeline, call with test input)
3. If it fails, append the error message to the prompt
4. LLM sees both the failed code and the error, and generates a fix
5. Repeat up to `max_retries` times (default: 3)

**Important ordering rule:** The orchestrated interface must be listed **last** in the `ptools:` config section. The orchestrator builds its catalog from currently-implemented interfaces, so other ptools must be implemented first.

**Config keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `orchestrate.model` | (required) | LLM for pipeline composition (use a powerful model) |
| `orchestrate.max_retries` | 3 | Max retry attempts |
| `echo.orchestrate` | False | Print generated pipeline code |
| `echo.orchestrate_llm` | False | Print raw LLM output |

**Why this matters:** The orchestrator replaces hand-coded workflows with LLM-generated ones. Combined with the learning system, you get a full automation loop: define stubs -> LLM generates workflow -> LLM implements steps -> learn from results -> deploy cheap implementations.

---

### Factory Comparison

The factories form a spectrum from cheap/rigid to expensive/flexible:

| Factory | Cost | Flexibility | Best For |
|---------|------|-------------|----------|
| `learned` | Free (lookup) | Only known inputs | Repeated sub-problems |
| `direct` | Free (Python) | Requires code | Workflows, known algorithms |
| `simulate` | 1 LLM call | Anything describable | Simple predictions |
| `prompt_llm` | 1 LLM call | Custom prompts | Task-specific prompting |
| `program_of_thought` | 1+ LLM calls | Multi-step reasoning | Code generation + execution |
| `simulate_pydantic` | 1+ LLM calls | Structured output + tools | Complex structured outputs |
| `orchestrate` | 1 LLM call (at bind time) | Auto-generates workflows | Replacing hand-coded workflows |

The learning system lets you start on the expensive end and progressively move to the cheap end.

---

## Part 4: Configuration System

**File:** `src/secretagent/config.py`

All behavior in secretagent is controlled by configuration. The config is a nested dictionary managed by OmegaConf.

### Loading Config

```python
from secretagent import config

# From a YAML file
config.configure(yaml_file='conf.yaml')

# From Python dicts
config.configure(llm={'model': 'gpt-4o'})

# From command-line style strings ("dotlist")
config.configure(dotlist=['llm.model=gpt-4o', 'cachier.enable_caching=false'])
```

All calls **merge** into the existing config (they don't replace it). This lets you layer configs:

```python
config.configure(yaml_file='base_config.yaml')    # base settings
config.configure(yaml_file='experiment.yaml')       # experiment overrides
config.configure(dotlist=['llm.model=gpt-4o'])     # CLI overrides
```

### Reading Config

```python
config.get('llm.model')              # returns None if not set
config.get('llm.model', 'gpt-4o')   # returns default if not set
config.require('llm.model')          # CRASHES if not set (fail-early pattern)
```

`require()` is the "fail-early" pattern — crash immediately when a required setting is missing, rather than getting a confusing error three functions deep.

### Temporary Overrides

```python
with config.configuration(cachier={'enable_caching': False}):
    # caching is OFF inside this block
    result = my_interface("input")
# caching is automatically RESTORED here
```

This is a "context manager" — the `with` block saves the current config, applies changes, and restores the original when the block exits. Essential for testing and debugging.

### Config Namespaces Explained

```yaml
llm:
  model: claude-haiku-4-5-20251001  # Which LLM to use (any litellm model name)
  thinking: true                     # Add <thought> scaffolding for chain-of-thought

cachier:
  enable_caching: true    # Whether to cache LLM calls to disk
  cache_dir: llm_cache    # Directory for the cache files

echo:                     # Debugging output controls
  model: true             # Print "calling model X" before each LLM call
  llm_input: true         # Print the full prompt in a pretty box
  llm_output: true        # Print the LLM response in a pretty box
  code_eval_output: true  # Print result of executing generated code (PoT)
  service: true           # Print service information
  call: true              # Print function call signatures

evaluate:
  expt_name: my_experiment   # Tag for this run (appears in directory names)
  result_dir: results        # Base directory for saving results
  record_details: true       # Include full rollouts in results.jsonl (needed for learning!)

ptools:                      # Maps each Interface to its implementation
  analyze_sentence:
    method: simulate
  sport_for:
    method: learned
    learner: rote
    backoff: true

dataset:
  split: valid         # Which data split to use
  shuffle_seed: 137    # Random seed for reproducibility
  n: 20                # How many examples to run (omit for all)

learn:
  train_dir: training  # Where learned implementations are stored

orchestrate:
  model: together_ai/Qwen/Qwen3.5-397B-A17B  # Powerful LLM for workflow generation
  max_retries: 3                                # Retry attempts for compose_with_retry
```

### Available Models

The `llm.model` value is passed to litellm, so any model litellm supports works. Here are some commonly used options:

| Model | Provider | Cost (input/output per 1M tokens) | Notes |
|-------|----------|-----------------------------------|-------|
| `claude-haiku-4-5-20251001` | Anthropic | Cheap, stable | Needs Anthropic API key |
| `together_ai/deepseek-ai/DeepSeek-V3.1` | Together AI | Cheap | Strong reasoning |
| `together_ai/openai/gpt-oss-20b` | Together AI | $0.05 / $0.20 | Very cheap |
| `together_ai/openai/gpt-oss-120b` | Together AI | $0.15 / $0.60 | Good value, larger |
| `together_ai/Qwen/Qwen3.5-9B` | Together AI | $0.10 / $0.15 | Good value, **no tool use** |
| `together_ai/google/gemma-3n-E4B-it` | Together AI | $0.02 / $0.04 | Ultra-cheap, **no tool use** |
| `together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct` | Together AI | $0.15 / $1.50 | MoE architecture |

**Important:** Models marked "no tool use" cannot be used with the `simulate_pydantic` factory when tools are involved.

---

## Part 5: LLM Calls and Caching

### `llm_util.py` — The Single Gateway

**File:** `src/secretagent/llm_util.py`

All LLM calls in the framework go through one function: `llm(prompt, model)`.

```python
def _llm_impl(prompt, model):
    messages = [dict(role='user', content=prompt)]
    response = litellm.completion(model=model, messages=messages)
    model_output = response.choices[0].message.content
    stats = dict(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency=time.time() - start_time,
        cost=completion_cost(completion_response=response),
    )
    return model_output, stats

def llm(prompt, model):
    return cached(_llm_impl)(prompt, model)  # wraps with caching
```

The `stats` dict tracks everything about each call — tokens used, time taken, cost in USD. These stats flow through the recording system and are aggregated by the evaluator.

### `cache_util.py` — The Lazy Caching Pattern

**File:** `src/secretagent/cache_util.py`

**The problem:** Python's `@cachier` decorator normally needs parameters at **import time** (when the file is first loaded). But secretagent's cache directory comes from config, which is loaded **later** at runtime.

**The solution:** `cached(fn)` reads cache config at **call time**:

```python
def cached(fn, **cachier_kw):
    cachier_cfg = dict(config.get('cachier', {}) or {})
    enable = cachier_cfg.pop('enable_caching', True)

    if not enable:
        return fn  # no caching, return the function as-is

    merged = {**cachier_cfg, **cachier_kw}  # merge config with explicit kwargs
    cache_key = (fn, str(sorted(merged.items())))

    if cache_key not in _DECORATED:
        _DECORATED[cache_key] = cachier_decorator(**merged)(fn)

    return _DECORATED[cache_key]
```

**What this means in practice:**
1. First call: `llm("What sport does LeBron play?", "gpt-4o")` -> actual LLM call, result saved to disk
2. Second call with same args -> reads from disk, zero API cost
3. Change `cachier.cache_dir` -> different cache directory
4. Set `cachier.enable_caching=False` -> no caching at all

---

## Part 6: Recording System

**File:** `src/secretagent/record.py`

### What It Does

Tracks every Interface call during execution — which function was called, with what arguments, what it returned, and the LLM usage stats.

### How to Use It

```python
from secretagent import record

with record.recorder() as rollout:
    result = sport_for("LeBron James")

print(rollout)
# [{'func': 'sport_for',
#   'args': ('LeBron James',),
#   'kw': {},
#   'output': 'basketball',
#   'stats': {'input_tokens': 150, 'output_tokens': 5, 'latency': 0.8, 'cost': 0.0003}}]
```

### How It Works Internally

Simple global state:

1. `recorder()` sets a global flag `RECORDING = True` and creates an empty list `RECORD = []`
2. Inside the context, factories call `record.record(func=..., args=..., output=..., stats=...)`
3. `record()` appends to `RECORD` if `RECORDING` is True
4. When the `with` block exits, `RECORDING` is set to False
5. You still have the data through the `rollout` variable (which pointed to the same list)

The recording system is used by:
- **Evaluation** — to track LLM usage per test case
- **Learning** — to extract (input, output) pairs from past runs

---

## Part 7: Evaluation System

**Files:** `src/secretagent/evaluate.py`, `src/secretagent/dataset.py`

### Dataset and Case

A `Case` is a single test example:

```python
from secretagent.dataset import Case, Dataset

case = Case(
    name="test.001",
    input_args=("DeMar DeRozan was called for the goal tend.",),
    expected_output=True  # the sentence IS plausible
)
```

A `Dataset` is a list of Cases with metadata:

```python
dataset = Dataset(name="sports", split="valid", cases=[case1, case2, ...])
dataset = dataset.configure(shuffle_seed=137, n=20)  # shuffle, then take first 20
```

### Evaluator

You subclass `Evaluator` and implement one method — how to compare predicted vs. expected output:

```python
from secretagent.evaluate import Evaluator

class MyEvaluator(Evaluator):
    def compare_predictions(self, predicted, expected):
        return {'correct': float(predicted == expected)}
```

### What `evaluate()` Does Step by Step

```python
evaluator = MyEvaluator()
csv_path = evaluator.evaluate(dataset, my_interface)
```

1. Creates a timestamped directory: `results/20260323.143022.my_experiment/`
2. For each test case (with a progress bar):
   a. Wraps the call in `record.recorder()` to track all sub-calls
   b. Calls the Interface: `predicted = my_interface(*case.input_args)`
   c. If an exception occurs, captures it as a string instead of crashing
   d. Compares: `metrics = compare_predictions(predicted, case.expected_output)`
   e. Sums LLM stats across all sub-calls (total tokens, cost, latency)
   f. Writes the result as a JSONL line (streaming — you can monitor progress)
3. Saves:
   - `results.csv` — metrics in tabular form
   - `results.jsonl` — detailed results (with full rollouts if `record_details=True`)
   - `config.yaml` — snapshot of the exact config used (for reproducibility)

---

## Part 8: The Learning System — Closing the Loop

**Files:** `src/secretagent/learn/base.py`, `src/secretagent/learn/baselines.py`, `src/secretagent/learn/implement_learn.py`

### The Big Picture

The learning system **distills** expensive LLM behavior into cheap implementations:

```
Run with LLM -> Record all calls -> Learn patterns -> Deploy as lookup table -> Re-run cheaper
```

### Step 1: Run Experiments with Recording

```bash
uv run python expt.py run evaluate.record_details=true evaluate.expt_name=training_run
```

The `record_details=true` flag is critical — it saves the **full rollout** (every Interface call) in `results.jsonl`. Without it, you only get the final prediction and metrics.

A JSONL line with rollout looks like:

```json
{
  "case_name": "valid.001",
  "predicted_output": true,
  "expected_output": true,
  "correct": 1.0,
  "rollout": [
    {"func": "analyze_sentence", "args": ["Tim Duncan scored..."], "output": ["Tim Duncan", "scored", ""]},
    {"func": "sport_for", "args": ["Tim Duncan"], "output": "basketball"},
    {"func": "sport_for", "args": ["scored from inside the paint"], "output": "basketball"},
    {"func": "consistent_sports", "args": ["basketball", "basketball"], "output": true}
  ]
}
```

### Step 2: The Learner Collects Data (`learn/base.py`)

The `Learner` base class provides the data collection pipeline:

**`learn(dirs, latest, check)`** — top-level entry point:

```python
def learn(self, dirs, latest=1, check=None):
    self.collect_distillation_data(dirs, latest, check)
    print(f'collected {len(self.dataset.cases)} examples')
    self.fit()
    output_file = self.save_code()
    print(self.report())
```

**`collect_distillation_data(dirs)`** — find and filter result directories:

```python
filtered_dirs = savefile.filter_paths(dirs, latest=latest, dotlist=check or [])
self.dataset = self._collect_and_store_data(filtered_dirs)
```

`filter_paths` can:
- Keep only the `latest` k directories per experiment tag
- Filter by config constraints (e.g., `check=['llm.model=gpt-4o']` only keeps runs using gpt-4o)

**`_extract_cases_from_record(record)`** — extract Cases for the target Interface from a rollout:

```python
for step in record.get('rollout', []):
    if step['func'] == self.interface_name:  # e.g., 'sport_for'
        yield Case(
            input_args=step.get('args'),       # ('LeBron James',)
            expected_output=step.get('output')  # 'basketball'
        )
```

**Provenance tracking** — the learner saves:
- `data.json` — the collected Dataset (all input/output pairs)
- `sources.txt` — which result directories the data came from
- `source_configs/*.yaml` — copies of each source's config.yaml (needed for backoff)

### Step 3: RoteLearner Fits the Data (`learn/baselines.py`)

The simplest possible learner — a majority-vote lookup table.

**`fit()`** (`baselines.py:34-52`):

```python
def fit(self):
    counts = defaultdict(Counter)  # {input_key: {output_key: count}}
    for case in self.dataset.cases:
        # Make inputs hashable (lists->tuples, dicts->sorted tuple pairs)
        args_key = _make_hashable(case.input_args or [])
        kw_key = _make_hashable(case.input_kw or {})
        input_key = (args_key, kw_key)
        output_key = _make_hashable(case.expected_output)
        counts[input_key][output_key] += 1

    # For each unique input, pick the most common output
    self._most_common_output = {}
    for input_key, counter in counts.items():
        best_output, _ = counter.most_common(1)[0]
        self._most_common_output[input_key] = original_output[best_output]
```

**Why `_make_hashable()`?** Python dictionary keys must be hashable (immutable). JSON data has lists and dicts, which aren't hashable. This function recursively converts:
- `['a', 'b']` -> `('a', 'b')`  (list to tuple)
- `{'x': 1}` -> `(('x', 1),)`    (dict to sorted tuple of pairs)

### Step 4: RoteLearner Generates Code

`save_code()` writes a `learned.py` file:

```python
"""Auto-generated rote-learned implementation for sport_for."""

def _make_hashable(obj):
    if isinstance(obj, list):
        return tuple(_make_hashable(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    return obj

_MOST_COMMON_OUTPUT = {
    (('LeBron James',), ()): 'basketball',
    (('Tim Duncan',), ()): 'basketball',
    (('scored a touchdown',), ()): 'American football',
    # ... hundreds more entries from training data ...
}

def sport_for(*args, **kw):
    args_key = _make_hashable(list(args))
    kw_key = _make_hashable(kw)
    return _MOST_COMMON_OUTPUT.get((args_key, kw_key))  # returns None if not found
```

### Step 5: LearnedFunctionFactory Loads the Code (`learn/implement_learn.py`)

At runtime, with this config:

```yaml
ptools:
  sport_for:
    method: learned
    learner: rote
    backoff: true
learn:
  train_dir: training
```

**`build_fn()`** does:

1. **Find the file**: looks for `training/*sport_for__rote/learned.py` and picks the most recent (by timestamp in directory name)
2. **Import it**: uses `importlib` to load the file as a Python module
3. **Extract the function**: `getattr(mod, 'sport_for')` gets the function
4. **Backoff wrapping** (if `backoff=True`):

```python
# Build the original LLM implementation from saved configs
backoff_impl = _build_backoff_impl(interface, workdir)

def learned_with_backoff(*args, **kw):
    result = fn(*args, **kw)          # try the lookup table
    if result is None:                 # input not in the table!
        return backoff_impl(*args, **kw)  # fall back to the LLM
    return result
```

**How backoff reconstruction works:** The learner saved copies of the source config YAMLs. `_build_backoff_impl()` reads them, finds the original `ptools.sport_for` config (e.g., `{method: simulate}`), and rebuilds that implementation automatically. You don't need to specify the fallback — it's derived from the training data's provenance.

### Running the CLI

```bash
uv run -m secretagent.cli.learn rote \
    --interface sport_for \
    --train-dir training \
    results/*
```

Output:

```
collected 80 examples in working directory training/20260323.144500.sport_for__rote
inputs:             47
estimated coverage: 0.85
saved output to training/20260323.144500.sport_for__rote/learned.py
```

"Estimated coverage" = fraction of inputs seen more than once (so the majority vote is meaningful).

### The Complete Learning Loop

```
1. Define Interfaces (ptools.py)
2. Configure with LLM implementations (conf.yaml: method: simulate)
3. Run evaluation with recording (evaluate.record_details=true)
4. Learn from recordings:
   uv run -m secretagent.cli.learn rote --interface sport_for --train-dir training results/*
5. Configure with learned implementations:
   sport_for:
     method: learned
     learner: rote
     backoff: true
6. Run evaluation again — known inputs are instant, novel inputs use LLM
7. Repeat 3-6 to expand the lookup table over time
```

### How to Create a New Learner

Subclass `Learner` and implement three methods:

```python
from secretagent.learn.base import Learner

class MySmartLearner(Learner):
    def __init__(self, interface_name, train_dir):
        super().__init__(interface_name, train_dir, f'{interface_name}__smart')

    def fit(self):
        # self.dataset.cases has all the (input, output) pairs
        # Train your model here (decision tree, neural net, etc.)
        ...
        return self

    def save_code(self):
        # Write learned.py with a function named self.interface_name
        # The function should return None for uncertain inputs (for backoff)
        outpath = Path(self.created_files['learned.py'])
        outpath.write_text(...)
        return outpath

    def report(self):
        return "My smart learner: 95% accuracy, 80% coverage"
```

The `LearnedFunctionFactory` doesn't need any changes — it just imports `learned.py` and calls the function. Data collection, provenance tracking, and backoff all come for free from the base class.

---

## Part 9: The Ptools Pattern

"Ptools" (program tools) is the naming convention for modules that define Interfaces for a benchmark.

### What a ptools module looks like

```python
# ptools.py
from secretagent.core import interface

@interface
def analyze_sentence(sentence: str) -> tuple[str, str, str]:
    """Extract player, action, and event."""

@interface
def sport_for(x: str) -> str:
    """Return the sport for a player/action/event."""

# A workflow that composes the above
def sports_understanding_workflow(sentence):
    player, action, event = analyze_sentence(sentence)
    player_sport = sport_for(player)
    ...
```

### Config-driven binding

The config maps each interface name to its implementation:

```yaml
ptools:
  analyze_sentence:
    method: simulate
  sport_for:
    method: learned
    learner: rote
    backoff: true
  are_sports_in_sentence_consistent:
    method: direct
    fn: ptools.sports_understanding_workflow
```

And `implement_via_config(ptools, config['ptools'])` binds them all at once.

### Why this matters: Ablation Studies

Want to compare GPT-4o vs Claude Haiku? Change one line:
```yaml
llm:
  model: gpt-4o  # or claude-haiku-4-5-20251001
```

Want to try Program of Thought instead of Simulate? Change one line:
```yaml
ptools:
  are_sports_consistent:
    method: program_of_thought  # was: simulate
```

Want to use a learned lookup table with LLM backoff? One line:
```yaml
ptools:
  sport_for:
    method: learned  # was: simulate
    learner: rote
    backoff: true
```

No code changes needed. This makes experiments highly reproducible and easy to compare.

---

## Part 10: End-to-End Walkthrough — Sports Understanding Benchmark

Let's trace a complete experiment from start to finish.

### The Task

Given a sentence like "DeMar DeRozan was called for the goal tend", determine if the sports references are consistent. Goal tend is a basketball term, DeRozan is a basketball player -> yes, consistent.

### File Structure

```
benchmarks/sports_understanding/
  ptools.py              # Interface definitions
  expt.py                # CLI experiment runner
  conf/conf.yaml         # Default config
  data/valid.json        # Test data
  prompt_templates/      # Custom prompt files
  results/               # Where results go (auto-created)
  llm_cache/             # Cached LLM calls (auto-created)
```

### Step 1: The Interfaces (`ptools.py`)

```python
@interface
def analyze_sentence(sentence: str) -> tuple[str, str, str]:
    """Extract player, action, and optional event."""

@interface
def sport_for(x: str) -> str:
    """Return the sport for a player/action/event."""

@interface
def consistent_sports(sport1: str, sport2: str) -> bool:
    """Are these two sport descriptions consistent?"""

@interface
def are_sports_in_sentence_consistent(sentence: str) -> bool:
    """Determine plausibility of a sports-related sentence."""

def sports_understanding_workflow(sentence: str) -> bool:
    """Hand-coded workflow that composes the above."""
    player, action, event = analyze_sentence(sentence)
    player_sport = sport_for(player)
    action_sport = sport_for(action)
    result = consistent_sports(player_sport, action_sport)
    if event:
        event_sport = sport_for(event)
        result = result and consistent_sports(player_sport, event_sport)
    return result
```

### Step 2: The Config (`conf/conf.yaml`)

```yaml
llm:
  model: claude-haiku-4-5-20251001

cachier:
  cache_dir: llm_cache

evaluate:
  result_dir: results
  expt_name: DEFAULT

ptools:
  analyze_sentence:
    method: simulate
  sport_for:
    method: simulate
  consistent_sports:
    method: simulate
  are_sports_in_sentence_consistent:
    method: DEFAULT  # must be overridden on the command line
```

### Step 3: Run the Experiment

```bash
cd benchmarks/sports_understanding

uv run python expt.py run \
  ptools.are_sports_in_sentence_consistent.method=direct \
  ptools.are_sports_in_sentence_consistent.fn=ptools.sports_understanding_workflow \
  evaluate.expt_name=workflow_haiku \
  dataset.n=20 \
  evaluate.record_details=true
```

### Step 4: What Happens Inside `expt.py`

```python
# 1. Load config (YAML + CLI overrides)
config.configure(yaml_file='conf/conf.yaml', dotlist=ctx.args)
config.set_root(Path(__file__).parent)  # resolve relative paths

# 2. Load dataset
dataset = load_dataset('valid')  # loads data/valid.json
dataset = dataset.configure(shuffle_seed=137, n=20)  # shuffle, take 20

# 3. Bind implementations
implement_via_config(ptools, config.require('ptools'))
# This calls:
#   ptools.analyze_sentence.implement_via('simulate')
#   ptools.sport_for.implement_via('simulate')
#   ptools.consistent_sports.implement_via('simulate')
#   ptools.are_sports_in_sentence_consistent.implement_via('direct',
#       fn='ptools.sports_understanding_workflow')

# 4. Run evaluation
evaluator = SportsUnderstandingEvaluator()
csv_path = evaluator.evaluate(dataset, ptools.are_sports_in_sentence_consistent)
```

### Step 5: What Happens for ONE Test Case

Input: `"DeMar DeRozan was called for the goal tend."`

1. `are_sports_in_sentence_consistent(sentence)` -> DirectFactory -> calls `sports_understanding_workflow()`
2. Workflow calls `analyze_sentence(sentence)` -> SimulateFactory -> LLM call -> `('DeMar DeRozan', 'was called for the goal tend', '')`
3. Workflow calls `sport_for('DeMar DeRozan')` -> SimulateFactory -> LLM call -> `'basketball'`
4. Workflow calls `sport_for('was called for the goal tend')` -> SimulateFactory -> LLM call -> `'basketball'`
5. Workflow calls `consistent_sports('basketball', 'basketball')` -> SimulateFactory -> LLM call -> `True`
6. Event is empty string -> no additional check
7. Returns: `True`
8. Expected: `True` -> `correct = 1.0`

All 4 LLM calls are cached. Running the same input again = free.

### Step 6: Results

```
results/20260323.143022.workflow_haiku/
  results.csv       # Metrics per case
  results.jsonl     # Full details with rollouts
  config.yaml       # Snapshot of exact config used
```

### Step 7: Learn from Results

```bash
uv run -m secretagent.cli.learn rote \
  --interface sport_for \
  --train-dir training \
  results/*
```

Output:

```
collected 80 examples in working directory training/20260323.144500.sport_for__rote
inputs:             47
estimated coverage: 0.85
saved output to training/20260323.144500.sport_for__rote/learned.py
```

### Step 8: Re-run with Learned Implementation

```bash
uv run python expt.py run \
  ptools.are_sports_in_sentence_consistent.method=direct \
  ptools.are_sports_in_sentence_consistent.fn=ptools.sports_understanding_workflow \
  ptools.sport_for.method=learned \
  ptools.sport_for.learner=rote \
  ptools.sport_for.backoff=true \
  learn.train_dir=training \
  evaluate.expt_name=workflow_learned \
  dataset.n=20
```

Now `sport_for("LeBron James")` -> instant dictionary lookup. Only novel inputs cost API calls.

### Step 9: Compare Results

```bash
uv run -m secretagent.cli.results pair \
  results/*workflow_haiku* \
  results/*workflow_learned*
```

This runs a paired t-test comparing the two approaches.

---

## Part 11: Architecture Diagram

```
                    ┌─────────────────────────┐
                    │      YAML Config         │
                    │  (llm, ptools, cachier,  │
                    │   orchestrate)           │
                    └─────────┬───────────────┘
                              │ implement_via_config()
    ┌─────────────────────────┼───────────────────────────────┐
    │              │                   │                       │
    │     ┌────────▼────────┐  ┌──────▼──────┐  ┌────────────▼────────────┐
    │     │ Interface       │  │ Interface   │  │ Interface               │
    │     │ analyze_sentence│  │ sport_for   │  │ are_sports_consistent   │
    │     └────────┬────────┘  └──────┬──────┘  └────────────┬────────────┘
    │              │                  │                       │
    │     ┌────────▼────────┐  ┌──────▼──────┐  ┌────────────▼────────────┐
    │     │ simulate        │  │  learned    │  │ orchestrate             │
    │     │ (LLM call)      │  │ (lookup +   │  │ (LLM generates workflow │
    │     │                 │  │  backoff)   │  │  code at bind time)     │
    │     └────────┬────────┘  └──────┬──────┘  └────────────┬────────────┘
    │              │                  │                       │
    │              │           ┌──────▼──────┐               │
    │              │           │ Try table → │               │
    │              │           │ None? → LLM │               │
    │              │           └──────┬──────┘               │
    │              │                  │                       │
    │     ┌────────▼──────────────────▼───────────────────────▼──────┐
    │     │                     llm_util.llm()                       │
    │     │              (litellm + cachier disk cache)               │
    │     └──────────────────────────┬───────────────────────────────┘
    │                                │
    │                        ┌───────▼───────┐
    │                        │  LLM Provider │
    └────────────────────────│  (Anthropic,  │
      orchestrate.model ────>│  Together AI, │
      (powerful LLM for      │   OpenAI,..)  │
       workflow generation)  └───────────────┘
```

**Key insight:** Every box in this diagram looks the same to its caller — just a Python function call. You can swap any implementation via config without touching the code. The orchestrator adds another level: even the *workflow logic itself* can be auto-generated.

---

## Part 12: CLI Tools Reference

### Cost Analysis

Summarize LLM costs from the cachier cache:

```bash
# From a specific cache directory
uv run -m secretagent.cli.costs benchmarks/sports_understanding/llm_cache

# Using the configured cachier.cache_dir from a config file
uv run -m secretagent.cli.costs --config-file conf.yaml
```

### Results Analysis

```bash
# List all experiment results
uv run -m secretagent.cli.results list results/

# Average metrics across the 3 most recent runs of each experiment
uv run -m secretagent.cli.results average results/* --latest 3

# Statistical comparison between two runs (paired t-test)
uv run -m secretagent.cli.results pair results/dir1 results/dir2

# Compare configs across result directories
uv run -m secretagent.cli.results compare_configs results/*
```

### Learning

```bash
uv run -m secretagent.cli.learn rote \
    --interface sport_for \
    --train-dir training \
    --latest 1 \
    results/*
```

Options:
- `--interface`: Which Interface to learn for
- `--train-dir`: Where to save the learned implementation
- `--latest`: Keep only the k most recent result directories per experiment tag
- `--check`: Config constraints like `--check llm.model=gpt-4o`

---

## Summary: Key Design Principles

1. **Everything looks like a function call** — Interfaces have a uniform calling convention regardless of whether an LLM, lookup table, or Python code is behind them.

2. **Late binding** — Implementations are chosen at runtime via config, not hardcoded. This makes experiments easy to configure and compare.

3. **Configuration-driven** — One YAML file controls all behavior. CLI overrides make it easy to tweak parameters without editing files.

4. **Transparent recording** — Every LLM call is tracked with token counts, cost, and latency. Nothing is hidden.

5. **Automatic caching** — Same inputs never cost API calls twice. The lazy caching pattern handles configuration timing elegantly.

6. **Learning loop** — Expensive LLM runs can be distilled into cheap lookup tables. The backoff mechanism ensures you never lose capability for novel inputs.

7. **Composability** — Interfaces can call other Interfaces, creating layered pipelines where each layer has its own implementation strategy.

8. **Orchestration** — Even workflow logic can be auto-generated by an LLM from ptools signatures and a task description, replacing hand-coded orchestration with LLM-generated pipelines. A workflow in Python can orchestrate LLM-powered sub-functions, which can themselves be lookup tables with LLM backoff.
