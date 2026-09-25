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

# 1 · Your first IrisPie model

**30 minutes** · no prerequisites · next: *2 · The model file*

By the end you will have written a small macro model from scratch, found its
steady state, solved it, hit it with a demand shock, and read the answer.

Everything in this tutorial is one continuous session. Run the cells in order.

```{code-cell} ipython3
import irispie as ip
from irispie import Simultaneous, Databox, qq
```

That import line is the same in every tutorial in this series.

## What we are building

Three equations. They are the smallest set that still behaves like a real
macro model.

**Aggregate demand.** The output gap `y` depends on where it was last quarter
and on the real interest rate. When money is expensive, demand falls.

**Phillips curve.** Inflation `pi` depends on past inflation, on expected
future inflation, and on how hot the economy is running.

**Policy rule.** The central bank moves its rate `i` gradually, raising it
when inflation is above target or output is above potential.

Three variables, three shocks, nine parameters. Notice that `y` depends on
*next* quarter's inflation — the model looks forward as well as back, which is
what makes it interesting to solve.

## Write the model

A model is text. You can keep it in a `.model` file, but for a tutorial a
Python string is easier to read alongside the code.

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

    "Output persistence"                  a1
    "Interest rate sensitivity"           a2
    "Inflation persistence"               b1
    "Slope of the Phillips curve"         b2
    "Policy rate smoothing"               c1
    "Policy response to inflation"        c2
    "Policy response to output"           c3
    "Inflation target"                    pi_tar
    "Steady-state real interest rate"     r_ss

!transition-equations

    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;

    "Phillips curve"
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;

    "Policy rule"
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

"""
```

Four things to notice in that text:

- Each block starts with a **keyword beginning with `!`**. Everything after it
  belongs to that block until the next keyword.
- `y{-1}` is **last quarter's** `y`. `pi{+1}` is **next quarter's** `pi`.
  Curly braces, not square ones.
- A **string in quotes** before a name or an equation is its description. It
  is optional, and it is worth writing — it shows up in tables and charts
  later.
- Every equation ends with a **semicolon**.

## Parse it

```{code-cell} ipython3
m = Simultaneous.from_string(SOURCE, linear=True, flat=True)
m
```

`from_string` reads the text and builds a model object. Two flags:

- `linear=True` — the equations are linear, so IrisPie can use faster and more
  reliable methods. Our three equations are.
- `flat=True` — the model has no trend growth; it settles at a constant level
  rather than growing forever.

Both are promises you make about the model. If you get them wrong the model
will still build, and will misbehave later.

The printout tells you what was understood: 3 transition equations, no
measurement equations, a maximum lag of one quarter and a maximum lead of one
quarter. That already confirms the parser saw your `{-1}` and `{+1}`.

```{code-cell} ipython3
m.get_names()
```

Everything the model knows about, in one list. You did not write `std_shk_y` —
IrisPie added a standard deviation for every shock automatically.

## Give it parameters

The model has structure but no numbers yet. Before doing anything else, ask
what is still missing:

```{code-cell} ipython3
m.get_unassigned_parameters()
```

All nine. Now set them:

```{code-cell} ipython3
m.assign_strict(
    a1=0.7,      # output persistence
    a2=0.2,      # how much demand reacts to the real rate
    b1=0.6,      # inflation persistence
    b2=0.3,      # slope of the Phillips curve
    c1=0.5,      # policy smoothing
    c2=1.5,      # response to inflation
    c3=0.5,      # response to output
    pi_tar=2,    # inflation target, % per year
    r_ss=1,      # steady-state real rate, % per year
)
```

Then check again — this should be empty:

```{code-cell} ipython3
m.get_unassigned_parameters()
```

**Use `assign_strict`, not `assign`.** Both set parameters, but `assign`
silently ignores a name the model does not have, so a typo costs you an hour.
`assign_strict` tells you immediately. You will see exactly this in
*Break it* below.

## Find the steady state

The steady state is where the economy settles when nothing disturbs it. Solve
for it:

```{code-cell} ipython3
m.steady()
```

And look at it:

```{code-cell} ipython3
m.create_steady_table()
```

Read the answer against the economics. Inflation sits at the target, 2. The
real rate is 1, so the nominal rate is 3. The output gap is zero — or rather
`-3e-16`, which is how a computer writes zero. If those three numbers were not
what you expected, the model is wrong, and every later step would be wrong
too.

`steady()` solved it. Now verify that the answer really satisfies the
equations:

```{code-cell} ipython3
m.check_steady()
```

`True` means the equations balance at that point. These are two different
questions — "did the solver finish" and "is the answer correct" — and it is
worth asking both.

## Solve it

Because `y` depends on next quarter's inflation, you cannot simply step the
model forward quarter by quarter. It has to be solved first.

```{code-cell} ipython3
m.solve_first_order()
```

No output means it worked. What it produced is a rule that expresses every
variable today in terms of last quarter's variables and this quarter's shocks.

```{code-cell} ipython3
m.get_eigenvalues_stability()
```

Four eigenvalues: three stable, one unstable. That count is not decoration.
For a model with forward-looking variables to have exactly one sensible path,
**the number of unstable eigenvalues must equal the number of forward-looking
variables**. This model looks forward through `pi{+1}` only — one such
variable, one unstable eigenvalue. It matches, so the model has a unique
solution. That is the *Blanchard–Kahn* condition.

Worth knowing now, because it will bite you later: **IrisPie does not check
this for you.** `solve_first_order()` returns quietly whether the condition
holds or not, and you only discover the problem when your simulation refuses
to settle down. Counting eigenvalues here is the check. Tutorial 9 is entirely
about what to do when the count is wrong.

## Build the input data

A simulation needs a span of time and a starting point.

```{code-cell} ipython3
span = qq(2025,1) >> qq(2034,4)
span
```

`qq(2025,1)` is the first quarter of 2025, and `>>` makes the stretch from one
period to another.

Ten years is longer than the shock is interesting for, and that is deliberate.
You need to see the economy come all the way back, not just the first few
quarters. Over a short span a badly behaved model looks fine.

Now the starting data. You could build it by hand, but there is a shortcut
that puts every variable at its steady state across the whole span:

```{code-cell} ipython3
db = Databox.steady(m, span)
db
```

More entries than you wrote. `Databox.steady` has given you every variable,
every shock, every parameter and every standard deviation — everything
`simulate` will look for. Starting from this and changing what you need is
much safer than assembling a databox from nothing.

The `ant_shk_*` entries are the *anticipated* halves of your shocks — shocks
the economy sees coming. Tutorial 12 is about the difference. For now, leave
them alone.

```{code-cell} ipython3
db["pi"]
```

Every quarter sits at 2, the steady state. That is the flat baseline you are
about to disturb.

## Shock it

One line. In the first quarter of 2026, demand comes in one percentage point
stronger than expected:

```{code-cell} ipython3
db["shk_y"][qq(2026,1)] = 1.0
db["shk_y"]
```

Now run it:

```{code-cell} ipython3
out = m.simulate(db, span)
out
```

`simulate` takes the input databox and the span, and returns a new databox.
The input is not modified.

## Read the answer

```{code-cell} ipython3
out["y"]
```

Printed with its dates and its description. The first row, 2024-Q4, is the
quarter before the simulation starts — the initial condition the model needed,
handed back to you so the path is complete.

```{code-cell} ipython3
out["pi"]
```

```{code-cell} ipython3
out["i"]
```

And as a picture:

```{code-cell} ipython3
fig = ip.make_subplots(
    (3, 1),
    figure_title="A one-off demand shock in 2026Q1",
    figure_height=760,
    subplot_titles=[
        "Output gap, % of potential",
        "Inflation, % per year",
        "Policy rate, % per year",
    ],
)

for position, name in enumerate(("y", "pi", "i")):
    out[name].plot(figure=fig, subplot=position, show_figure=False)

fig
```

Now read it as economics, because this is the moment you find out whether the
model works.

Output jumps to **1.06** on impact in 2026Q1, then decays, dips to about
**−0.31**, and comes back. Inflation rises more slowly and peaks at **3.23**
in 2026Q3 — two quarters after output — because prices respond to the pressure
that output creates. The policy rate climbs to **4.75**, and because `c1` makes
the bank move gradually, it stays high while inflation comes back down.

Two things to check, and they are the two things that tell you a model is
sound:

- **The ordering.** Output first, inflation two quarters later. If inflation
  had peaked on impact alongside output, the Phillips curve would not be doing
  its job.
- **It settles.** Every series returns to where it started — `y` to 0, `pi` to
  2, `i` to 3. A shock is temporary, so its effects must die out. If instead
  the swings had grown quarter after quarter, the model would be unusable no
  matter how sensible the first few quarters looked.

That second check is easy to skip, because the first few quarters of a broken
model look exactly like the first few quarters of a good one. Always look at
the whole path.

## ⚠️ Break it

Three mistakes, on purpose, while nothing is at stake. Two of the three
produce messages that do not say what is actually wrong, so it is worth seeing
them once now.

**1. Running `steady()` before setting the parameters.**

```{code-cell} ipython3
broken = Simultaneous.from_string(SOURCE, linear=True, flat=True)
try:
    broken.steady()
except Exception as e:
    print(type(e).__name__, "-", e)
```

Nothing in that message mentions parameters. The solver was handed equations
full of unknowns and gave up on the linear algebra. **Whenever you see a
linear-algebra error out of `steady()`, check `get_unassigned_parameters()`
first.**

**2. Misspelling a parameter name.**

```{code-cell} ipython3
broken = Simultaneous.from_string(SOURCE, linear=True, flat=True)
broken.assign(a11=0.7)                       # meant a1
print("still unassigned:", broken.get_unassigned_parameters())
```

No error, no warning — and `a1` is still unset. The mistake only surfaces
later, as the confusing message you just saw in item 1. With `assign_strict`
you are told at once:

```{code-cell} ipython3
broken = Simultaneous.from_string(SOURCE, linear=True, flat=True)
try:
    broken.assign_strict(a11=0.7)
except Exception as e:
    print(type(e).__name__, "-", e)
```

This is the single most useful habit in this tutorial.

**3. Simulating before solving.**

```{code-cell} ipython3
broken = Simultaneous.from_string(SOURCE, linear=True, flat=True)
broken.assign_strict(a1=0.7, a2=0.2, b1=0.6, b2=0.3, c1=0.5,
                     c2=1.5, c3=0.5, pi_tar=2, r_ss=1)
broken.steady()
try:
    broken.simulate(Databox.steady(broken, span), span)
except Exception as e:
    print(type(e).__name__, "-", e)
```

Also cryptic. It means the solution does not exist yet, because
`solve_first_order()` was never called. The order never changes:

**parse → assign → steady → solve → simulate**

## What you did

The whole session, in one cell:

```{code-cell} ipython3
m = Simultaneous.from_string(SOURCE, linear=True, flat=True)

m.assign_strict(a1=0.7, a2=0.2, b1=0.6, b2=0.3, c1=0.5,
                c2=1.5, c3=0.5, pi_tar=2, r_ss=1)

m.steady()
m.check_steady()
m.solve_first_order()

span = qq(2025,1) >> qq(2034,4)
db = Databox.steady(m, span)
db["shk_y"][qq(2026,1)] = 1.0

out = m.simulate(db, span)
out["y"]
```

Nine lines to build a macro model, shock it, and read the result.

## Things to remember

1. **The order never changes:** parse → assign → steady → solve → simulate.
   Skipping a step gives you an error that does not say which step you skipped.
2. **Use `assign_strict`, not `assign`.** A typo in `assign` is silent.
3. **`{-1}` is a lag, `{+1}` is a lead.** Curly braces.
4. **Start from `Databox.steady(m, span)`.** It builds every entry `simulate`
   expects, including ones you did not write.
5. **Check the steady state against the economics before going further.** If
   inflation is not at target there, nothing after it will be right.

## Exercise

Change `c2` from 1.5 to 3.0 — a central bank that reacts twice as hard to
inflation — and run the whole thing again.

Before you run it, write down a prediction. Does inflation peak higher or
lower? And what happens to output — is the slowdown afterwards deeper, or
shallower?

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

Both peaks come down, and the dip afterwards gets **shallower**, not deeper:

|  | `c2 = 1.5` | `c2 = 3.0` |
|---|---|---|
| inflation peak | 3.23 | **2.65** |
| output peak | 1.06 | **0.92** |
| output trough | −0.31 | **−0.27** |

If you predicted a deeper slowdown — the classic "the bank buys lower
inflation at the cost of lost output" — that is the right instinct applied to
the wrong shock, and it is worth understanding why.

A **demand** shock pushes output and inflation *in the same direction*. Both
are above target at once, so leaning against one leans against the other too.
There is nothing to trade off: the tougher bank simply does better on both,
and the economy returns to steady state sooner.

The trade-off is real, but it needs a shock that pushes them in *opposite*
directions. Try a cost-push shock instead — `db["shk_pi"]` rather than
`db["shk_y"]` — and you get exactly the textbook result:

|  | `c2 = 1.5` | `c2 = 3.0` |
|---|---|---|
| inflation peak | 3.47 | **3.19** |
| output trough | −0.34 | **−0.55** |

Now the harder response does buy lower inflation, and pays for it with a
deeper slowdown.

**The lesson is not about the parameter.** It is that a policy rule cannot be
judged on its own — only against the shocks the economy actually faces. That
is why the rest of this series spends so much time on how you specify a shock.

</details>

## Next

**Tutorial 2 · The model file** takes the source text apart properly: every
declaration block in turn, log variables, what changes when the model is not
linear or not flat, and how to keep a model in its own `.model` file instead
of a Python string.

You now have the shape of every session you will ever run in IrisPie — parse,
assign, steady, solve, simulate. Everything after this is that same loop with
more in it.
