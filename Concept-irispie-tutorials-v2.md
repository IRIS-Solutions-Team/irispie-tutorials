# Concept irispie tutorials v2

What the tutorial series is, how it is structured, and what it covers.

---

## 1 · Purpose

IrisPie already has reference documentation — generated from docstrings,
organised by class, and correct. What it does not have is anything that
teaches someone who has never used it.

These tutorials fill that gap. They are **not** a second reference:

| Reference docs | Tutorials |
|---|---|
| organised by class | organised by task, easiest first |
| answer "what does this method do?" | answer "how do I do X?" |
| read when you already know what you want | read when you don't |
| complete | selective — the 20% you use daily |

Where the two overlap, the tutorial links to the reference rather than
repeating it.

## 2 · Scope

**The series teaches modelling workflows, end to end.** Writing a model,
solving it, simulating it, confronting it with data, and the machinery
underneath.

General data handling is not part of it — time series objects, date
arithmetic, growth-rate transformations, filtering and seasonal adjustment,
charting. Those have their own series. Here they are *used* wherever a model
needs them and never taught: where one appears for the first time, the
tutorial says in one line what it does, links out, and moves on.

## 3 · Who it is for

**An economist who knows basic Python and has never built a model in IrisPie.**

Assumed: variables, functions, lists, importing a package.
Not assumed: DSGE modelling, state-space methods, anything about IrisPie.

## 4 · Format

**One tutorial = one `.md` file, written in MyST Markdown (jupytext format).**

One choice, every use covered:

| use | how |
|---|---|
| read on GitHub | it is plain Markdown |
| run in Jupyter | `jupytext --to ipynb 02-the-model-file.md` |
| publish as a docs site | MkDocs/Quarto reads MyST directly |
| feed to an LLM | plain text, self-contained sections |
| review a change | diffable text, unlike `.ipynb` |

`.ipynb` files are generated, never committed — notebook JSON does not diff and
stored output goes stale.

Reader setup:

```bash
pip install jupytext ipykernel
jupytext --to ipynb tutorials/02-the-model-file.md
```

Each file opens with the jupytext header, and code goes in fenced cells:

````
```{code-cell} ipython3
your code here
```
````

Files are named `NN-short-slug.md` — two digits, hyphens, lower case. The
number is permanent, because tutorials cite each other by number.

## 5 · Anatomy of a tutorial

**Every tutorial has the same eight parts, in this order.** The reader learns
the rhythm once and then always knows where to look.

| # | Part | Rule |
|---|---|---|
| 1 | **Title + header line** | time needed · what to read first · what comes next |
| 2 | **"By the end you will…"** | one concrete outcome, not a list of topics |
| 3 | **Setup cell** | the same imports and settings in every tutorial |
| 4 | **Numbered sections** | one idea each; prose first, then code — never code first |
| 5 | **⚠️ Break it** | 2–4 deliberate mistakes the reader runs on purpose |
| 6 | **What you did** | the whole notebook as a cheat sheet |
| 7 | **Things to remember** | 3–5 bullets, only the ones that bite |
| 8 | **Next** | links forward, plus one exercise with a checkable answer |

**On "Break it".** The part most documentation skips, and the part that
teaches most. The reader deliberately does the wrong thing and reads the error
while calm — so that when it happens for real at 11pm, they recognise it. Each
item shows the mistake, the message, and one line on what it really means.

**Length.** Target 25–35 minutes, roughly 25–35 code cells. Substantial enough
to teach a whole topic properly, short enough to finish in one sitting.

**One running model.** Every tutorial uses the same small open-economy model —
same variable names, same parameters, same file, growing across the series.
Tutorial 1 runs it end to end; tutorial 2 writes its first equation; tutorial
25 produces a full forecast round with it. By tutorial 8 the reader knows what
the variables mean, so all their attention goes to the new IrisPie idea instead
of to a new toy model.

## 6 · How it is grouped

**Two levels only: level → tutorial.** Five levels, five tutorials each.

The levels follow the modelling workflow — what you are *doing*, not what part
of the library you are touching:

| Level | You are… |
|---|---|
| 1 | writing a model |
| 2 | getting it to a steady state and solving it |
| 3 | simulating it |
| 4 | confronting it with data, and using other model types |
| 5 | going deeper — uncertainty, internals, building models from code |

Rules that keep it honest:

1. **One topic per tutorial**, covered completely — not split across files.
2. **Each tutorial ends where the next begins**, so the reader is handed
   forward rather than stopped.
3. **Traps go where you hit them**, not in a collected "gotchas" page.
4. **No prerequisites beyond the previous tutorial.** Read in order, it always
   works.

## 7 · The curriculum

**25 tutorials, five levels, plus two appendices.**

### Level 1 — Writing a model

| # | Tutorial | Covers |
|---|---|---|
| 1 | Your first IrisPie model | The whole arc end to end: write it, parameterise it, steady, solve, simulate, read the answer |
| 2 | The model file | Transition variables, shocks, parameters, equations, lags and leads, descriptions, log variables |
| 3 | The model language in depth | The `!!` dynamic/steady split, measurement equations, exogenous variables, `{:tag}` attributes, pseudofunctions, `$name$` substitutions, `!steady-autovalues` |
| 4 | Generating models programmatically | The preparser (`!for`, `!if`, `!list`, `!let`), Jinja templating with a Python context, `save_preparsed` to debug it, `!autoswaps`, `!preprocessor` / `!postprocessor` |
| 5 | Creating, parameterising and saving | `from_file` / `from_string`, model flags, `assign` vs `assign_strict`, `get_unassigned_parameters`, shock stds, `rescale_stds`, portable format, pickle and dill |

### Level 2 — Steady state and solution

| # | Tutorial | Covers |
|---|---|---|
| 6 | The steady state | `steady`, flat vs growth models, `get_steady_levels` / `get_steady_changes`, `create_steady_table`, `check_steady` |
| 7 | Steering the steady state | `SteadyPlan` — `fix_level`, `fix_change`, exogenize / endogenize, `swap`, calibrating by hand, `split_into_blocks` to see the recursive structure |
| 8 | Solving the model | `solve_first_order`, what the solution actually is, `get_solution_vectors`, the state-space form, transition vs measurement |
| 9 | When it will not solve | `get_eigenvalues` with `STABLE` / `UNIT` / `UNSTABLE`, Blanchard–Kahn, unit roots, singularity, a steady state that does not exist — each message, its real cause, its fix |
| 10 | Model properties | `get_acov` and `get_acorr`, shock covariances, `DimensionNames` to slice results by variable name, `get_variable_stability` |

### Level 3 — Simulating

| # | Tutorial | Covers |
|---|---|---|
| 11 | Your first simulation | `simulate`, building the input databox, initial conditions, the simulation span, `return_info`, reading the output |
| 12 | Shocks | Anticipated vs unanticipated, how frames split a simulation, the anticipated-shock companions, one-off vs sustained shocks |
| 13 | Nonlinear simulation | `first_order` vs `stacked_time` vs `period_by_period`, `override_tolerance`, solver settings, what to do when it will not converge |
| 14 | Conditioning with plans | `SimulationPlan`, the four explicit forms — `exogenize_anticipated` / `_unanticipated`, `endogenize_anticipated` / `_unanticipated` — `swap_*`, reading a plan back, autoswaps in practice |
| 15 | Comparing and decomposing | Baseline vs scenario, shock decomposition with `minus_control`, attributing a forecast change to its sources |

### Level 4 — Data, filtering and estimation

| # | Tutorial | Covers |
|---|---|---|
| 16 | The Kalman filter | Measurement equations, filtering vs smoothing, diffuse initialization, `likelihood_contributions`, reading the filter output |
| 17 | Estimation | The negative log likelihood, wiring it to an optimiser, likelihood contributions period by period, judging a fit, what to do when it will not identify |
| 18 | Stochastic simulation | Drawing shocks from a covariance matrix, `vary_stds` for time-varying volatility, running many variants, turning them into fan charts |
| 19 | Sequential models | `Sequential` and `Explanatory`, `sequentialize` and `reorder_equations`, the incidence matrix, identities vs behavioural equations, residuals |
| 20 | Vector autoregressions | `RedVAR`, `estimate`, `MinnesotaPriorObs` and `MeanPriorObs`, companion matrices and stability, `resample` with Monte Carlo, bootstrap and wild bootstrap |

### Level 5 — Going deeper

| # | Tutorial | Covers |
|---|---|---|
| 21 | Parameter variants | Many parameterisations in one model object: `expand_num_variants`, `select_variants`, `broadcast_variants`, `iter_variants`, and how every method's `unpack_singleton` behaves |
| 22 | Inside the solution | `systemize` for the unsolved first-order system, the solution matrices, the eigenvalue decomposition, terminal conditions, what `clip_small` throws away |
| 23 | Inside the data contract | `Dataslate` — how a databox becomes the array the solvers use — `slatable_for_simulate`, initial and terminal columns, and how to diagnose a simulation that silently returns nothing |
| 24 | Building models from code | `ModelSource.from_lists` to build a model with no source text, portable-format surgery, generating families of models, `Stacker` for the stacked-time system itself |
| 25 | Capstone: a full forecast round | Filter history → condition with a plan → simulate → compare against baseline → decompose → report |

### Appendices

| # | Tutorial | Covers |
|---|---|---|
| A1 | Coming from MATLAB IRIS | Translation table for the Toolbox modeller |
| A2 | The errors you will hit first | Each message, its real cause, its fix |
