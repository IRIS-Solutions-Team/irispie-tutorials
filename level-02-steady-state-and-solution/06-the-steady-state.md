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

# 6 · The steady state

**35 minutes** · after *5 · Creating, parameterising and saving* · next: *7 ·
Steering the steady state*

You have called `steady()` in every tutorial since the first one and never
looked inside it. This one does. What it solves, why a model that grows needs
a different answer from one that does not, and what `check_steady()` is really
comparing when it hands you `True`.

```{code-cell} ipython3
import irispie as ip
from irispie import Simultaneous, Databox, qq

CALIB = dict(a1=0.7, a2=0.2, b1=0.6, b2=0.3, c1=0.5,
             c2=1.5, c3=0.5, pi_tar=2, r_ss=1)
```

## What a steady state is

The steady state is where the model comes to rest: the values every variable
takes when all the shocks are switched off and nothing is changing any more.
It is the answer to "where is this economy heading", and everything else in
IrisPie is built on top of it — the first-order solution is an expansion
around it, and a simulation is a set of deviations from it.

Here is the running model:

```{code-cell} ipython3
BASE = """

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

m = Simultaneous.from_string(BASE, linear=True, flat=True)
m.assign_strict(**CALIB)
m.steady()
m.get_steady_levels()
```

Three numbers you could have worked out on paper. With no shocks and nothing
changing, the Phillips curve forces `y = 0`, the policy rule then gives
`pi = pi_tar = 2`, and the demand equation gives `i = pi + r_ss = 3`.

That is worth doing once by hand, because it is the check you have when the
model gets big enough that you cannot.

## Levels and changes

A steady state has **two** parts, not one:

```{code-cell} ipython3
m.get_steady_changes()
```

Every change is zero, because this model was built with `flat=True` — a
promise that nothing trends. The **level** is where a variable settles; the
**change** is how fast it moves once it is settled. In a flat model the second
is zero by construction and you can ignore it.

`get_steady()` gives you both halves at once, as `(level, change)` pairs:

```{code-cell} ipython3
m.get_steady()
```

## Reading it all at once

Two calls and a dictionary each is fine for three variables. For a real model
there is a table:

```{code-cell} ipython3
m.create_steady_table()
```

Levels and changes side by side, parameters included. You can narrow it:

```{code-cell} ipython3
m.create_steady_table(kind=ip.TRANSITION_VARIABLE)
```

choose the columns, pick specific `names`, and write the whole thing straight
to disk with `save_to_csv_file=` — which is how a steady state ends up in a
calibration note.

## Models that grow

Not everything settles at a constant. Add a **price level** to the model. It
never stops rising, because inflation never stops:

```{code-cell} ipython3
GROWTH = """

!transition-variables

    "Output gap, % of potential"          y
    "Inflation, % per year"               pi
    "Policy rate, % per year"             i
    "Price level, index"                  P

!transition-shocks

    "Demand shock"                        shk_y
    "Cost-push shock"                     shk_pi
    "Monetary policy shock"               shk_i

!log-variables

    P

!parameters

    a1, a2, b1, b2, c1, c2, c3, pi_tar, r_ss

!transition-equations

    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;

    "Phillips curve"
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;

    "Policy rule"
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

    "The price level accumulates inflation"
    P = P{-1}*exp(pi/400);

"""
```

`P` is declared a **log variable**, because a price index is a positive level
with no natural zero — tutorial 2's rule. And the equation is written with
`exp(pi/400)` rather than the more obvious `(1 + pi/400)`, which keeps the
model linear: in logs it reads `log(P) = log(P{-1}) + pi/400`, a straight
line. The two differ only in the fifth decimal.

Now the flag. This model does **not** settle at a constant, so `flat=True`
would be a lie:

```{code-cell} ipython3
g = Simultaneous.from_string(GROWTH, linear=True, flat=False)
g.assign_strict(**CALIB)
g.steady()
g.get_steady_levels()
```

`y`, `pi` and `i` are exactly where they were. `P` sits at 1 — which is a
choice, not a result: nothing in the model pins the *level* of a price index,
only its growth. Now the half that matters:

```{code-cell} ipython3
g.get_steady_changes()
```

**`P` grows by a factor of 1.00501 every quarter.** Note the units. For a log
variable the steady change is a **gross ratio**; for an ordinary level
variable it is a **difference**, which is why `y`, `pi` and `i` show zero
rather than one.

Annualise it and the model is telling you something you already knew:

```{code-cell} ipython3
change = float(g.get_steady_changes()["P"])
print(f"per quarter: {change:.8f}")
print(f"per year:    {(change**4 - 1)*100:.4f}%")
```

2.02% a year, against an inflation target of 2. The small gap is the
difference between compounding four times and adding up four quarters — not
an error, just arithmetic.

```{code-cell} ipython3
g.create_steady_table(kind=ip.TRANSITION_VARIABLE)
```

A steady state with a column that is no longer all zeros. That is the whole
difference between a flat model and a growing one.

## The steady path

`Databox.steady()` has appeared in every simulation since tutorial 1, quietly
building the input data. Now that the model grows, what it builds is worth
looking at:

```{code-cell} ipython3
SPAN = qq(2025,1) >> qq(2027,4)

steady_db = Databox.steady(g, SPAN)
steady_db["P"]
```

Not a flat line. `Databox.steady()` takes the level *and* the change and rolls
them forward into a path — which for a flat model is the same number repeated,
and for this one is a price index rising every quarter. Check the growth
against the steady change:

```{code-cell} ipython3
p = steady_db["P"].get_data()[:, 0]
[round(float(p[k+1]/p[k]), 8) for k in range(4)]
```

Exactly `1.00501252` each time — the number `get_steady_changes()` reported.
This is what a simulation is a deviation *from*: shock the model and the paths
leave this one, then come back to it.

## What check_steady actually checks

You have been running it after every `steady()` since tutorial 1. Here is what
it is actually testing.

**Every equation you write is stored twice.** One copy is used when simulating
quarter by quarter; the other is used when solving for the steady state.
Normally IrisPie makes the second an exact copy of the first, so the
distinction never comes up. Tutorial 3 showed them side by side, separated by
`!!`, in the output of `get_equations()`.

`steady()` solves the **steady** copies. `check_steady()` then takes that
answer and substitutes it into the **dynamic** copies — the ones that describe
how the economy actually moves. If the answer is a genuine resting point, both
sets hold at once:

```{code-cell} ipython3
g.check_steady(when_fails="silent")
```

### Seeing the difference

That `True` does not tell you which set of equations was tested, because both
agree. To see which one `check_steady` sides with, we need a model where they
**disagree** — so let us build one on purpose.

`!!` lets you write the steady copy yourself. Here the dynamic policy rule is
left alone, and the steady copy is replaced with something absurd: the policy
rate is 99.

```{code-cell} ipython3
liar = BASE.replace(
    "i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;",
    "i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i"
    " !! i = 99;",
)

ml = Simultaneous.from_string(liar, linear=True, flat=True)
ml.assign_strict(**CALIB)
ml.steady()
ml.get_steady_levels()
```

A policy rate of **99** and inflation of **98**. Nothing malfunctioned:
`steady()` used the steady copies, one of which says the rate is 99, and the
demand equation then drags inflation up to match. The number 99 is arbitrary
— what matters is that it contradicts the real policy rule.

Now ask `check_steady` both questions:

```{code-cell} ipython3
print("against the steady equations: ",
      ml.check_steady(equation_switch="steady", when_fails="silent"))
print("against the dynamic equations:",
      ml.check_steady(equation_switch="dynamic", when_fails="silent"))
```

**`True` and `False`.** The answer satisfies the steady copies perfectly — of
course it does, `i = 99` is exactly what they demand. It is the dynamic copies
that reject it, and those are the ones describing the economy you care about.

`dynamic` is the default for precisely that reason. `check_steady()` is not
asking *"did the steady solver do its job?"* — it always did. It is asking
*"would the real, moving model ever come to rest here?"*

### Two more arguments

`tolerance=` loosens the test, which is occasionally useful for a large model
that will not close to machine precision and dangerous everywhere else. And
`when_fails=` controls what happens on a failure — `"error"` by default,
`"silent"` to get a plain `True`/`False`, which is what every tutorial has
been using.

### One thing it does not tell you

It confirms the equations are satisfied. It does **not** confirm you found the
steady state you wanted. A log variable sitting at zero satisfies
`P = P{-1}*exp(pi/400)` perfectly — `0 = 0` — and `check_steady` returns
`True` for it. If a growing model collapses to zeros, the equations are not
what is wrong; the starting point was.

## ⚠️ Break it

**Telling a growing model that it is flat.**

```{code-cell} ipython3
wrong = Simultaneous.from_string(GROWTH, linear=True, flat=True)
wrong.assign_strict(**CALIB)
wrong.steady()
wrong.create_steady_table(kind=ip.TRANSITION_VARIABLE)
```

Look past `P` for a moment. **Inflation is 1.9957.** The target is 2, the
policy rule is untouched, and the steady state of a variable that has nothing
to do with prices has moved as well: the output gap is `0.00068` instead of
zero, and the policy rate is `2.9939` instead of 3.

Nothing about the model changed. One flag did. `flat=True` forces every
steady change to zero, so the solver had to find levels where a price index
does not grow while inflation is positive — which is impossible, so it
distributed the contradiction across the whole model and returned the least
bad compromise.

```{code-cell} ipython3
wrong.check_steady(when_fails="silent")
```

`False`. Without that line you would be reading an inflation target the model
misses by four thousandths, and wondering why.

This is the same lesson as the `linear` flag in tutorial 5, with a sharper
edge: **a wrong flag does not stay in its own corner.** It contaminates
variables that have no connection to the thing you got wrong.

## What you did

```{code-cell} ipython3
# where the model comes to rest
m.get_steady_levels()

# and how fast it moves once it is there
m.get_steady_changes()

# both at once, filtered, and saveable with save_to_csv_file=
m.create_steady_table(kind=ip.TRANSITION_VARIABLE)

# a model that grows needs flat=False
g = Simultaneous.from_string(GROWTH, linear=True, flat=False)

# verify against the dynamic equations, which is the default
g.check_steady(when_fails="silent")
g.check_steady(equation_switch="steady", when_fails="silent")
```

## Things to remember

1. **A steady state has two halves**: a level and a change. In a flat model
   every change is zero.
2. **For a log variable the change is a gross ratio**, for an ordinary
   variable a difference. `1.005` and `0.0` mean the same thing — no growth —
   for different kinds of variable.
3. **`flat=False` for anything that trends**, and the level of a trending
   index is a normalisation you choose, not a result.
4. **`check_steady()` substitutes your answer into the dynamic equations.**
   That is the test worth passing, and it is the default.
5. **It does not tell you the answer is the one you wanted.** A log variable
   at zero passes.
6. **Work the steady state out by hand once**, while the model is still small
   enough. It is the only independent check you will ever have.

## Exercise

Raise the inflation target from 2 to 3 in the growing model, and solve it
again.

Before you run it, predict four things: the steady levels of `y`, `pi` and
`i`, the steady change of `P`, what that is per year, and whether the *level*
of `P` moves.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

`y` stays at **0**, `pi` goes to **3**, and `i` goes to **4** — the Fisher
relation, `pi + r_ss`, with the same real rate as before. A higher target does
not buy any output in the long run; it only raises nominal quantities.

`P` grows by **1.0075281954** per quarter, which is `exp(3/400)`, or
**3.0455%** a year. Faster inflation, faster-rising price level, exactly as it
should be.

The **level** of `P` does not move. It stays at 1, because nothing in the
model pins it — a price index has whatever base year you give it. That is the
useful thing to notice: in a growing model some steady levels are answers and
others are just normalisations, and the model will not tell you which is
which. You have to know.

`check_steady()` returns `True` throughout.

</details>

## Next

**Tutorial 7 · Steering the steady state** is about making the steady state
come out the way you want. `SteadyPlan` lets you fix a level or a change and
let a parameter move instead of the other way round — which is how
calibration is actually done, by naming the long-run values you believe and
solving for the parameters that deliver them. It also covers
`split_into_blocks`, which shows you the recursive structure hiding inside
your equations.
