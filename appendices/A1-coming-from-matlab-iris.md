---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# A1 · Coming from MATLAB IRIS

**30 minutes** · reference · usable from *1 · Your first IrisPie model*
onwards

A model file written for the MATLAB IRIS Toolbox is close to a model file
IrisPie will read, and in many cases it is one already. This appendix is
the translation: what carries over untouched, what is spelled differently,
what is called differently, and the handful of places where the two
packages do not line up at all.

Everything in the IrisPie column is executed in this file. The Toolbox
column is written from the Toolbox's documented interface and is not
executed here, so check it against the version you are migrating from.

```{code-cell} ipython3
import numpy as np
import irispie as ip
from irispie import (Simultaneous, RedVAR, Sequential, Databox, Series,
                     SteadyPlan, SimulationPlan, qq)
```

## A Toolbox model file, read unchanged

The block keywords are hyphenated in IrisPie and underscored in the
Toolbox. IrisPie rewrites a single underscore in a keyword to a hyphen
before parsing, so the old spelling is read as written. Line comments
beginning `%` survive as well.

```{code-cell} ipython3
LEGACY = """

%! A model file in the old Toolbox spelling

!transition_variables
    y, pi, i

!transition_shocks
    shk_y, shk_pi, shk_i

!parameters
    a1, a2, b1, b2, c1, c2, c3, pi_tar, r_ss

!transition_equations
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;
    pi - pi_tar = b1*(pi{-1} - pi_tar) + b2*y + shk_pi;
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

"""

old = Simultaneous.from_string(LEGACY, linear=True, flat=True)

print("transition equations:", old.num_transition_equations)
print("max lag, max lead   :", old.max_lag, old.max_lead)
print([q.human for q in old._invariant.quantities])
```

The rewrite takes one underscore between two lower-case words, so
`!measurement_equations`, `!log_variables`, `!steady_autovalues` and
`!exogenous_variables` all arrive correctly. A keyword with two underscores
does not, and neither does camel case.

Three further short forms are accepted, which have no Toolbox equivalent
but are worth knowing because they shorten a quick test model:

```{code-cell} ipython3
SHORT = """
!variables
    y
!shocks
    shk_y
!parameters
    a1
!equations
    y = a1*y{-1} + shk_y;
"""

short = Simultaneous.from_string(SHORT, linear=True, flat=True)
print([(q.human, q.kind.name) for q in short._invariant.quantities])
```

`!variables`, `!shocks` and `!equations` expand to the three transition
blocks.

## The model language, item by item

| Toolbox | IrisPie | Note |
|---|---|---|
| `!transition_variables` | `!transition-variables` | underscore accepted |
| `!transition_shocks` | `!transition-shocks` | underscore accepted |
| `!transition_equations` | `!transition-equations` | underscore accepted |
| `!measurement_variables` | `!measurement-variables` | underscore accepted |
| `!measurement_shocks` | `!measurement-shocks` | underscore accepted |
| `!measurement_equations` | `!measurement-equations` | underscore accepted |
| `!parameters` | `!parameters` | unchanged |
| `!log_variables` | `!log-variables` | underscore accepted |
| `!substitutions` | `!substitutions` | unchanged, tutorial 3 |
| `!links` | `!steady-autovalues` | different keyword, tutorial 3 |
| `!autoexogenise` | `!autoswaps-simulate` | see the last section |
| `x{-1}`, `x{+1}` | `x{-1}`, `x{+1}` | unchanged |
| `!!` steady split | `!!` steady split | unchanged, tutorial 3 |
| `%` comment | `%` comment | `%!` and `#!` are also stripped |
| `!for ?x = … !do … !end` | `!for ?x = … !do … !end` | unchanged, tutorial 4 |
| `!if … !then … !else … !end` | `!if … !then … !else … !end` | unchanged, tutorial 4 |

The pseudofunctions keep their names and expand in the same way:

```{code-cell} ipython3
PSEUDO = """
!variables
    x, dx, gx
!shocks
    shk_x
!parameters
    rho
!equations
    x = rho*x{-1} + shk_x;
    dx = diff(x);
    gx = roc(x);
"""

pseudo = Simultaneous.from_string(PSEUDO, linear=False, flat=True)
for equation in pseudo._invariant.dynamic_equations:
    print("  ", equation.human)
```

`shift`, `diff`, `difflog` (also `diff_log`), `pct`, `roc`, `movsum`,
`movavg` and `movprod` (also `mov_sum`, `mov_avg`, `mov_prod`) are all
available.

## The habit to unlearn

This is the one difference that will cost you an afternoon if it is not
cleared up first. In the Toolbox a model is a value and every operation
returns a new one, so the idiom is `m = sstate(m)`. In IrisPie a model is
an object and the methods change it in place. They return nothing.

```{code-cell} ipython3
m = Simultaneous.from_string(LEGACY, linear=True, flat=True)
m.assign_strict(a1=0.7, a2=0.2, b1=0.1, b2=0.3, c1=0.5, c2=1.5, c3=0.5,
                pi_tar=2, r_ss=1)

print("m.steady() returns      :", m.steady())
m.solve_first_order()
print("and m is solved         :", m.get_solution() is not None)
```

The second consequence is that two names can refer to one model:

```{code-cell} ipython3
alias = m
alias.assign(c2=9.9)
print("changed through the alias:", float(m.get_parameters()["c2"]))

separate = m.copy()
separate.assign(c2=1.5)
print("the copy holds            :", float(separate.get_parameters()["c2"]))
print("and m still holds         :", float(m.get_parameters()["c2"]))

m.assign(c2=1.5)
m.steady()
m.solve_first_order()
```

`m.copy()` is what the Toolbox gave you for free on every assignment. Use
it whenever a model is about to be altered and the original is still
needed.

## The session, call by call

| Toolbox | IrisPie |
|---|---|
| `m = Model.fromFile('f.model')` | `m = Simultaneous.from_file("f.model")` |
| `m = Model.fromString(s)` | `m = Simultaneous.from_string(s)` |
| `'linear=', true` | `linear=True` |
| `m.c2 = 1.5` | `m.assign(c2=1.5)` or `m.assign_strict(c2=1.5)` |
| `m = assign(m, p)` | `m.assign(**p)` |
| `get(m, 'parameters')` | `m.get_parameters()` |
| `m = sstate(m)` | `m.steady()` |
| `chksstate(m)` | `m.check_steady()` |
| `get(m, 'sstateLevel')` | `m.get_steady_levels()` |
| `get(m, 'sstateGrowth')` | `m.get_steady_changes()` |
| `m = solve(m)` | `m.solve_first_order()` |
| `get(m, 'eigenvalues')` | `m.get_eigenvalues()` |
| `isstationary(m)` | `m.get_variable_stability()` |
| `acf(m)` | `m.get_acov()`, `m.get_acorr()` |
| `d = sstatedb(m, range)` | `db = Databox.steady(m, span)` |
| `d = zerodb(m, range)` | `db = Databox.steady(m, span, deviation=True)` |
| `s = simulate(m, d, range)` | `out = m.simulate(db, span)` |
| `'method=', 'stacked'` | `method="stacked_time"` |
| `[~, f] = filter(m, d, range)` | `out = m.kalman_filter(db, span)` |
| `VAR` | `RedVAR` |
| `rpteq` | `Sequential` |

`sstatedb` and `zerodb` differ only in whether the levels or the deviations
are filled in, which is the `deviation` argument:

```{code-cell} ipython3
m.steady()
m.solve_first_order()
SPAN = qq(2025,1) >> qq(2025,4)

levels = Databox.steady(m, SPAN)
deviations = Databox.steady(m, SPAN, deviation=True)

print("Databox.steady            pi:", np.asarray(levels["pi"][SPAN]).ravel())
print("with deviation=True       pi:", np.asarray(deviations["pi"][SPAN]).ravel())
```

## Dates, series and databases

| Toolbox | IrisPie |
|---|---|
| `qq(2025,1)` | `qq(2025,1)` |
| `yy(2025)`, `mm(2025,3)` | `yy(2025)`, `mm(2025,3)` |
| `range = qq(2025,1):qq(2025,4)` | `span = qq(2025,1) >> qq(2025,4)` |
| `tseries(range, values)` | `Series(start=..., values=...)` |
| a `struct` of series | a `Databox` |
| `d.y` | `db["y"]` |
| `dbfun(@(x) …, d)` | an ordinary Python comprehension |

```{code-cell} ipython3
db = Databox()
db["y"] = Series(start=qq(2025,1), values=np.array([0.5, 0.3, 0.2, 0.1]))

print("span      :", [str(p) for p in SPAN])
print("length    :", len(SPAN))
print("db names  :", sorted(db.get_names()))
print("values    :", np.asarray(db["y"][SPAN]).ravel())
```

The colon is a slice operator in Python and cannot be overloaded, which is
why a span is built with `>>` instead.

## Plans

The Toolbox has one `Plan` object used for both jobs. IrisPie has two, and
which one you want depends on whether the thing being pinned is a long-run
value or a path.

```{code-cell} ipython3
steady_plan = SteadyPlan(m)
steady_plan.exogenize("i")
steady_plan.endogenize("r_ss")

sim_plan = SimulationPlan(m, SPAN)
sim_plan.swap_unanticipated(SPAN, ("i", "shk_i"))

print("steady plan  :", steady_plan.get_exogenized_names(),
      steady_plan.get_endogenized_names())
print("simulation   :", type(sim_plan).__name__, "over", len(SPAN), "periods")
```

| Toolbox | IrisPie |
|---|---|
| `plan = Plan(m, range)` | `SteadyPlan(m)` or `SimulationPlan(m, span)` |
| `exogenize(plan, 'i', range)` | `plan.exogenize_unanticipated(span, "i")` |
| `endogenize(plan, 'shk_i', range)` | `plan.endogenize_unanticipated(span, "shk_i")` |
| both at once | `plan.swap_unanticipated(span, ("i", "shk_i"))` |
| `'fix='` on `sstate` | `SteadyPlan.fix_level` |
| `'fixGrowth='` on `sstate` | `fix_change` exists and does not work — tutorial 7 |
| `simulate(m, d, range, plan)` | `m.simulate(db, span, plan=plan)` |

Tutorial 7 covers the steady plan and tutorial 14 the simulation plan. The
accounting is the Toolbox's: every value pinned costs one thing the model
was going to determine.

## ⚠️ Break it

**Writing `m = m.steady()`.**

The reflex from the Toolbox is to catch the return value. The method
returns nothing, so the name is rebound to `None` and the model is lost.

```{code-cell} ipython3
reflex = m.copy()
reflex = reflex.steady()

print("reflex is now:", reflex)

try:
    reflex.solve_first_order()
except Exception as error:
    print(type(error).__name__, "-", str(error))
```

The message names the operation rather than the mistake, and it arrives one
line after the damage. The model object itself was fine; it is the variable
that was thrown away. Call the method as a statement and leave the name
alone.

The same applies to `assign`, `solve_first_order`, `expand_num_variants`,
`select_variants` and every other method that changes the model. Methods
that begin `get_` return something; methods that act on the model do not.

## What you did

```{code-cell} ipython3
# a Toolbox model file, read as written
Simultaneous.from_string(LEGACY, linear=True, flat=True)

# the same accessors under different names
m.get_parameters()
m.get_steady_levels(), m.get_steady_changes()
m.get_eigenvalues()
m.get_acov()

# sstatedb and zerodb
Databox.steady(m, SPAN)
Databox.steady(m, SPAN, deviation=True)

# the two plans
SteadyPlan(m)
SimulationPlan(m, SPAN)

# a model about to be altered
m.copy()
```

## Things to remember

1. **Underscored block keywords are read as written.** A Toolbox model file
   usually needs no edits at all, and `%` comments survive.
2. **Pseudofunctions keep their names.** `diff`, `difflog`, `pct`, `roc`,
   `movavg`, `movsum`, `movprod` and `shift` all expand as before.
3. **Methods change the model and return nothing.** `m = sstate(m)` becomes
   `m.steady()`, and writing `m = m.steady()` destroys the model.
4. **Two names can be one model.** Use `m.copy()` before altering a model
   you still need.
5. **`get(m, 'x')` becomes a named method.** `get_parameters`,
   `get_steady_levels`, `get_steady_changes`, `get_eigenvalues`,
   `get_acov`, `get_variable_stability`.
6. **A span is built with `>>`,** because the colon cannot be overloaded in
   Python.
7. **There are two plan classes**, one for the steady state and one for
   simulations, and they use different method names.
8. **`Simultaneous` has no `estimate`.** `RedVAR` does; a structural model
   is estimated by wiring the likelihood to an optimiser yourself, which is
   tutorial 17.

## What has no translation

Four things do not map, and each is a tutorial rather than a table row.

**Estimation.** The Toolbox estimates a structural model with one call.
IrisPie does not have one: `Simultaneous` has no `estimate` method, and the
likelihood is assembled into an objective function and handed to
`scipy.optimize` by hand. Tutorial 17 does it end to end.

```{code-cell} ipython3
print("Simultaneous.estimate:", hasattr(Simultaneous, "estimate"))
print("RedVAR.estimate      :", hasattr(RedVAR, "estimate"))
```

**Anticipated shocks.** The Toolbox switches between anticipated and
unanticipated with an option on `simulate`. IrisPie makes them two separate
quantities: declaring `shk_i` creates `ant_shk_i` alongside it, and the
plans have `_anticipated` and `_unanticipated` methods rather than a flag.
Tutorial 12 is about the distinction and tutorial 14 about using it.

**Autoexogenise.** The model language has `!autoswaps-simulate`, and
nothing reads it into a `SimulationPlan`. Write the swaps in the plan;
tutorial 14 says so in more detail.

**Reporting equations.** `rpteq` becomes a `Sequential` object, which is a
model in its own right rather than an attachment to one: it is built from
its own source, simulated with its own call, and has residuals and an
equation order. Tutorial 19 covers it.

## Next

**Tutorial 1 · Your first IrisPie model** is the place to start reading,
and most of it will be familiar. **A2 · The errors you will hit first**
collects the messages that are not.
