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

# 5 · Creating, parameterising and saving

**35 minutes** · after *4 · Generating models programmatically* · next: *6 ·
The steady state*

The last four tutorials were about the model file. This one is about the model
**object** — how it gets built, what the flags on the constructor commit you
to, how numbers get into it, and how to put it on disk and get it back.

```{code-cell} ipython3
import irispie as ip
from irispie import Simultaneous, Databox, qq
```

Here is the running model again, written to a file:

```{code-cell} ipython3
SOURCE = """

!transition-variables

    "Output gap, % of potential"          y
    "Inflation, % per year"               pi
    "Policy rate, % per year"             i

!transition-shocks

    "Demand shock"                        shk_y
    "Cost-push shock"                     shk_pi
    "Monetary policy shock"               shk_i

!parameters

    a1, a2, b1, b2, c1, c2, c3, pi_tar, r_ss

!transition-equations

    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;

    "Phillips curve"
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;

    "Policy rule"
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

"""

with open("economy.model", "w", encoding="utf-8") as f:
    f.write(SOURCE)
```

## Where a model comes from

Two constructors, and they take the same arguments:

```{code-cell} ipython3
m = Simultaneous.from_file("economy.model", linear=True, flat=True)
m
```

`from_string` is for tutorials and generated text; `from_file` is for real
work. Both accept the `context` and `save_preparsed` arguments from tutorial 4.

### A model can be split across several files

`from_file` also takes a **list**, and joins the files before parsing. A large
model is much easier to live with when the declarations, the core equations
and the reporting block are separate files:

```{code-cell} ipython3
with open("decl.model", "w", encoding="utf-8") as f:
    f.write(SOURCE.split("!transition-equations")[0])

with open("eqtn.model", "w", encoding="utf-8") as f:
    f.write("!transition-equations" + SOURCE.split("!transition-equations")[1])

split = Simultaneous.from_file(["decl.model", "eqtn.model"], linear=True, flat=True)
split.get_names(kind=ip.TRANSITION_VARIABLE), len(split.get_equations())
```

Same model, two files. The order matters only in that the files are
concatenated, so a block cannot be cut in half across them.

### Asking what the parser did

Both constructors take `return_info=True` and then hand back a second value:

```{code-cell} ipython3
m2, info = Simultaneous.from_string(SOURCE, linear=True, flat=True,
                                    return_info=True)
info["preparser_needed"]
```

`False` here, because this file has no `!for` or `!if` in it. `info` also
carries `preparsed_source`, which is the same text `save_preparsed` writes to
disk — useful when you want it in a variable instead of a file.

## The three flags

Tutorial 1 used two. There are three:

```{code-cell} ipython3
m.get_flags()
```

```{code-cell} ipython3
flags = m.get_flags()
flags.is_linear, flags.is_flat, flags.is_deterministic
```

**`linear`** — the equations are linear in the variables. IrisPie can then
solve the steady state with linear algebra in one step instead of iterating.

**`flat`** — the model has no trend growth; variables settle at constant
levels rather than growing.

**`deterministic`** — the model has no stochastic shocks. This one has a
visible consequence:

```{code-cell} ipython3
det = Simultaneous.from_string(SOURCE, linear=True, flat=True,
                               deterministic=True)
print("stochastic:   ", m.get_names(kind=ip.TRANSITION_STD))
print("deterministic:", det.get_names(kind=ip.TRANSITION_STD))
```

Declare a model deterministic and the standard-deviation quantities are not
created at all. Everything in the next section but one stops existing.

### What `linear` buys you

It is not a formality. Solve the same model both ways:

```{code-cell} ipython3
CALIB = dict(a1=0.7, a2=0.2, b1=0.6, b2=0.3, c1=0.5,
             c2=1.5, c3=0.5, pi_tar=2, r_ss=1)

slow = Simultaneous.from_file("economy.model", linear=False, flat=True)
slow.assign_strict(**CALIB)
slow.steady()
slow.get_steady_levels()
```

The same zero output gap, the same 2% inflation, the same 3% policy rate — and
seven lines of iteration table that the linear version never printed. That is
the nonlinear solver working, and it is why tutorial 1 was silent throughout.

The two directions are not symmetric, which gives you a rule:

| What you declare | When it is false | What you get |
|---|---|---|
| `linear=False` on a model that is linear | understating | the right answer, more slowly |
| `linear=True` on a model that is not | overstating | **a wrong answer, silently** |

Understating a flag costs iteration tables. Overstating one costs the result.
**When you are not sure, say `False`** — you pay in solver output, never in
wrong numbers.

## Filling in the parameters

A freshly built model has names but no numbers:

```{code-cell} ipython3
m.get_unassigned_parameters()
```

All nine, because nothing has been assigned yet. There are two ways to fix
that, and the difference matters.

```{code-cell} ipython3
m.assign(**CALIB)
m.get_unassigned_parameters()
```

Empty. `assign` took them all. Now watch what `assign` does with a name that
does not exist:

```{code-cell} ipython3
m.assign(a11=0.9)
m.get_parameters()
```

**Nothing.** No error, no warning, and `a1` still holds `0.7` — the typo was
silently discarded. `assign_strict` is the same call that refuses:

```{code-cell} ipython3
try:
    m.assign_strict(a11=0.9)
except Exception as e:
    print(type(e).__name__, "-", str(e)[:150])
```

It names the offending key. **Use `assign_strict` for calibrations** and keep
`assign` for the cases where a loose match is what you want — feeding a
databox that holds more than the model needs, for instance.

`get_unassigned_parameters()` is the other half of the safety net. Run it
after calibrating and before solving; an empty result means every parameter
has a number.

## Shock standard deviations

Every shock you declare quietly gets a standard deviation alongside it:

```{code-cell} ipython3
m.get_names(kind=ip.TRANSITION_STD)
```

You never wrote those. They start at 1:

```{code-cell} ipython3
m.get_stds()
```

They are assigned like any other quantity, by name:

```{code-cell} ipython3
m.assign(std_shk_y=0.5, std_shk_pi=0.25, std_shk_i=0.1)
m.get_stds()
```

Two helpers for working with all of them at once. `rescale_stds` multiplies
every standard deviation by a factor — useful for asking "what if the world
were twice as volatile":

```{code-cell} ipython3
m.rescale_stds(2)
m.get_stds()
```

And `reset_stds` puts them all back:

```{code-cell} ipython3
m.reset_stds()
m.get_stds()
```

Note **back to 1, not back to what they were before the rescale.** `reset_stds`
restores the default, it is not an undo. There is also
`get_unassigned_stds()`, the standard-deviation counterpart of
`get_unassigned_parameters()`.

A standard deviation is the model's statement about how big each shock
typically is. That is what lets IrisPie draw shocks at random for a fan chart
in tutorial 18, and what tells the Kalman filter in tutorial 16 how much to
trust each observation against the model's own prediction. Setting them is
part of calibrating a model, not an afterthought — and `rescale_stds` makes
"what if the world were twice as volatile" a one-line question.

## Saving a model and getting it back

Re-parsing and re-solving a large model takes time, and there is no reason to
do it twice. Calibrate and solve one first, so there is something worth
saving:

```{code-cell} ipython3
m.assign_strict(**CALIB)
m.assign(std_shk_y=0.5)
m.steady()
m.solve_first_order()
m.get_steady_levels()
```

**The portable format** turns a model into a plain dictionary of lists and
strings, which means it survives JSON and can be read by something that is not
IrisPie:

```{code-cell} ipython3
import json

portable = m.to_portable()
print("keys:", list(portable.keys()))
print("JSON size:", len(json.dumps(portable)), "characters")
portable["source"]["equations"][0]
```

Each equation comes back as kind, dynamic version, steady version, description
and attributes — which is how tutorial 3 read block tags back. Use it to look
inside a model, to diff two models, or to hand one to a colleague who does not
run Python.

One thing it cannot currently do is reload:

```{code-cell} ipython3
try:
    Simultaneous.from_portable(m.to_portable())
except Exception as e:
    print(type(e).__name__, "-", str(e)[:160])
```

It writes out the anticipated-shock companions IrisPie generated for you, and
`from_portable` generates them again on the way back in, so they collide. That
happens for any model with shocks. The dictionary itself is sound — tutorial
24 strips those entries and reloads it in a few lines — but `from_portable`
on its own will not do it.

**For saving and reloading, use pickle.** It keeps everything:

```{code-cell} ipython3
import pickle

with open("economy.pkl", "wb") as f:
    pickle.dump(m, f)

with open("economy.pkl", "rb") as f:
    reloaded = pickle.load(f)

print("parameter a1: ", float(reloaded.get_parameters()["a1"]))
print("std_shk_y:    ", float(reloaded.get_stds()["std_shk_y"]))
print("steady state: ", {k: round(float(v), 4)
                         for k, v in reloaded.get_steady_levels().items()})
```

**Dill** does the same and copes with objects pickle chokes on — a model
carrying a `context` of your own functions, for instance:

```{code-cell} ipython3
import dill

with open("economy.dill", "wb") as f:
    dill.dump(m, f)

with open("economy.dill", "rb") as f:
    dilled = dill.load(f)

float(dilled.get_parameters()["a1"])
```

### What you get back

The parameters and standard deviations survive, and so does the **first-order
solution**. A reloaded model is not a fresh model that needs setting up again
— it is ready to work. No `steady()`, no `solve_first_order()`, straight to a
simulation:

```{code-cell} ipython3
SPAN = qq(2025,1) >> qq(2034,4)

db = Databox.steady(reloaded, SPAN)
db["shk_y"][qq(2026,1)] = 1.0

out = reloaded.simulate(db, SPAN)
for name in ("y", "pi", "i"):
    print(f"{name:<3} peak {float(out[name].get_data()[:, 0].max()):+.4f}")
```

The same demand shock from tutorial 1, and the same three numbers — `1.0631`,
`3.2283`, `4.7482` — out of a file, with no model-building at all. On a model
that takes a minute to parse and solve, that is the difference between a
forecast round you can iterate on and one you cannot.

## ⚠️ Break it

**A flag that is not true.**

Here is a model with a genuinely nonlinear equation — an inventory ratio with
a quadratic adjustment term:

```{code-cell} ipython3
NONLINEAR = """

!transition-variables
    "Inventory-to-sales ratio"    v

!transition-shocks
    shk_v

!parameters
    rho, kappa, v_bar

!transition-equations
    v = rho*v{-1} + kappa*v{-1}^2 + v_bar + shk_v;

"""

PARS = dict(rho=0.5, kappa=0.001, v_bar=4.0)
```

You can work out the true steady state by hand: `v = 0.5v + 0.001v² + 4`
solves to **8.132268**. Declared honestly, that is what you get:

```{code-cell} ipython3
honest = Simultaneous.from_string(NONLINEAR, linear=False, flat=True)
honest.assign_strict(**PARS)
honest.assign(v=8.0)
honest.steady()
honest.get_steady_levels()
```

Now promise it is linear:

```{code-cell} ipython3
liar = Simultaneous.from_string(NONLINEAR, linear=True, flat=True)
liar.assign_strict(**PARS)
liar.assign(v=8.0)
liar.steady()
liar.get_steady_levels()
```

**8.0.** No error, no warning, and a number close enough to the truth that
nothing about it looks wrong.

Here is what `linear=True` actually does. It replaces every equation with its
**tangent at zero** and solves that linear system exactly. For a model that
really is linear the tangent *is* the equation, so this is both correct and
fast. For this model it is neither: the tangent of `kappa*v^2` at `v = 0` is
flat, because the slope `2*kappa*v` is zero there. The squared term therefore
contributes nothing, leaving `v = rho*v + v_bar`, which gives
`4/(1 - 0.5) = 8`.

Note also that the starting value is **ignored**. Assign `v=0`, `v=8` or
`v=400` before calling `steady()` and the answer is 8.0 every time — there is
no iteration for it to start, only a linear system to solve.

One line catches it:

```{code-cell} ipython3
liar.check_steady(when_fails="silent")
```

`False`. The steady state it found does not satisfy the equations. This is the
whole reason `check_steady()` exists, and the reason to run it every single
time — a wrong flag does not announce itself, and the number it hands you will
usually look perfectly reasonable.

## What you did

```{code-cell} ipython3
# build a model, from one file or several
m = Simultaneous.from_file("economy.model", linear=True, flat=True)
Simultaneous.from_file(["decl.model", "eqtn.model"], linear=True, flat=True)

# the three flags
m.get_flags()

# numbers in, safely
m.get_unassigned_parameters()
m.assign_strict(**CALIB)
m.get_parameters()

# shock standard deviations
m.get_stds()
m.rescale_stds(2)
m.reset_stds()

# save and reload
with open("economy.pkl", "wb") as f:
    pickle.dump(m, f)

# inspect without reloading
m.to_portable()["source"]["equations"][0]
```

## Things to remember

1. **`from_file` takes a list.** Split a big model across files and let
   IrisPie join them.
2. **There are three flags, not two.** `deterministic=True` removes the
   standard-deviation quantities entirely.
3. **Flags are promises, and only one direction is dangerous.** Understating
   costs time; overstating costs the answer. When unsure, say `False`.
4. **Run `check_steady()` after every `steady()`.** It is the only thing that
   catches a wrong flag.
5. **`assign` swallows unknown names silently; `assign_strict` refuses.** Use
   `assign_strict` for calibrations, and `get_unassigned_parameters()` before
   you solve.
6. **`reset_stds()` sets standard deviations to 1**, not to whatever they were
   before.
7. **`to_portable()` is for looking inside a model**, not for reloading it —
   `from_portable` cannot read back what it writes for a model with shocks.
   Tutorial 24 works around it.
8. **Save a solved model with pickle** and it comes back ready to simulate,
   solution and all.

## Exercise

Take the calibrated model, halve the volatility of the world, save it, and
reload it.

- set `std_shk_y = 0.4`, `std_shk_pi = 0.2`, `std_shk_i = 0.1`
- call `rescale_stds(0.5)`
- pickle the model to a file and load it back

Before you run it, predict three things: what the standard deviations are
after the rescale, whether they survive the round trip, and what
`reset_stds()` would give you if you called it on the reloaded model.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

After the rescale the standard deviations are **0.2, 0.1 and 0.05** — each one
halved. `rescale_stds` multiplies, so it compounds with whatever was there
before rather than replacing it.

They **survive the pickle round trip** unchanged, along with the parameters,
the steady state and the first-order solution. That is the point of pickling a
solved model: you can reload it and simulate immediately.

`reset_stds()` on the reloaded model gives **1, 1 and 1** — not 0.4, 0.2 and
0.1. It restores the default, and there is no way back to your own values
except to assign them again. If you want to be able to undo a rescale, keep
the numbers in a Python dictionary and re-assign from that.

The practical habit this points at: keep your calibration in a Python
dictionary, assign from it, and save the solved model with pickle. Then a
rescale is one line, an undo is one line, and rebuilding from scratch is never
necessary.

</details>

## Next

That completes level 1. You can write a model file, read anyone else's, use
everything the language offers, generate one from Python, and get a
parameterised model onto disk and back.

**Tutorial 6 · The steady state** begins level 2 and goes back to something
you have been calling since tutorial 1 without looking inside. What `steady()`
actually solves, the difference between a flat model and one that grows,
`get_steady_levels` against `get_steady_changes`, `create_steady_table` for
reading it all at once, and what `check_steady` is really comparing.
