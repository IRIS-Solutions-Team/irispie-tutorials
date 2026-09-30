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

# 9 · When it will not solve

**40 minutes** · after *8 · Solving the model* · next: *10 · Model
properties*

Tutorial 8 took the solution apart. This one is about what has to be true for
there to be a solution at all, and how to check it in two lines.

That check comes down to the model's **eigenvalues**. They tell you which
forces in the model fade away, which persist forever, and whether the model
has a well-defined answer to begin with — the standard diagnostic for a
linear model, and one of the more useful habits to pick up early.

Running the check is the modeller's call: `solve_first_order()` returns a
solution whether or not the condition holds, because the condition is about
economics rather than arithmetic. By the end of this tutorial you will know
what to look at, what the three labels mean, and what each way of failing
looks like from the outside.

```{code-cell} ipython3
import io

import numpy as np
import irispie as ip
from irispie import Simultaneous, Databox, qq

CALIB = dict(a1=0.7, a2=0.2, b1=0.6, b2=0.3, c1=0.5,
             c2=1.5, c3=0.5, pi_tar=2, r_ss=1)

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

def build(**changes):
    """The running model, with any parameters overridden."""
    model = Simultaneous.from_string(SOURCE, linear=True, flat=True)
    calibration = dict(CALIB)
    calibration.update(changes)
    model.assign_strict(**calibration)
    model.steady()
    model.solve_first_order()
    return model

m = build()
```

## The eigenvalues

Every question in this tutorial is answered by one list of numbers:

```{code-cell} ipython3
m.get_eigenvalues()
```

They are complex, and what matters about each one is its **distance from
zero** — its absolute value. A `transform=` argument applies any function you
like on the way out, and `abs` is the one you will use:

```{code-cell} ipython3
m.get_eigenvalues(transform=abs)
```

Read them against **one**. A root below one is a force that dies away; a root
above one is a force that grows without limit; a root exactly at one is a
force that neither grows nor fades, and leaves a permanent mark.

That is the whole vocabulary. Three of these are below one and one is above.

## Why there are four

Three variables, four eigenvalues. The extra one is not a mistake.

The model looks into the future — `pi{+1}`, in two of the three equations.
Solving it means carrying that expectation along as a quantity in its own
right, so the system IrisPie works with is larger than the list of variables
you wrote, by exactly one element per forward-looking term. Which gives you a
way to count them:

```{code-cell} ipython3
def count_forward_looking(model):
    """How many expectations the solved system carries."""
    return (len(model.get_eigenvalues())
            - len(model.get_solution_vectors().transition_variables))

print("eigenvalues         :", len(m.get_eigenvalues()))
print("transition variables:", len(m.get_solution_vectors().transition_variables))
print("forward-looking     :", count_forward_looking(m))
```

One forward-looking variable, one extra root. Hold on to that number: it is
half of the condition that decides whether the model has a solution at all.

Do not reach for `max_lead` here, tempting though it is. It reports how far
ahead the model looks, not how many things it looks ahead *at* — a model with
both `y{+1}` and `pi{+1}` still has a `max_lead` of 1 while carrying two
expectations, and the count below would come out wrong.

## Stable, unit, unstable

You do not have to compare against one by eye:

```{code-cell} ipython3
m.get_eigenvalues_stability()
```

Three `STABLE` and one `UNSTABLE`, in the same order as the eigenvalues
themselves. The third label, `UNIT_ROOT`, is for roots sitting exactly on
one, and it is worth meeting before the others are put to work.

You can also ask for a single kind:

```{code-cell} ipython3
print("stable:  ", m.get_eigenvalues(kind=ip.STABLE, transform=abs))
print("unit:    ", m.get_eigenvalues(kind=ip.UNIT, transform=abs))
print("unstable:", m.get_eigenvalues(kind=ip.UNSTABLE, transform=abs))
```

## A root at exactly one

A root exactly at one is different from both of the others, and it is often
deliberate. Add a variable that never comes back:

```{code-cell} ipython3
RANDOM_WALK = """

!transition-variables

    "Output gap, % of potential"          y
    "Inflation, % per year"               pi
    "Policy rate, % per year"             i
    "Potential growth, % per year"        g

!transition-shocks

    "Demand shock"                        shk_y
    "Cost-push shock"                     shk_pi
    "Monetary policy shock"               shk_i
    "Potential growth shock"              shk_g

!parameters

    a1, a2, b1, b2, c1, c2, c3, pi_tar, r_ss

!transition-equations

    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;

    "Phillips curve"
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;

    "Policy rule"
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

    "Potential growth is a random walk"
    g = g{-1} + shk_g;

"""

walk = Simultaneous.from_string(RANDOM_WALK, linear=True, flat=True)
walk.assign_strict(**CALIB)
walk.steady()
walk.solve_first_order()

walk.get_eigenvalues_stability()
```

`UNIT_ROOT`, and the root itself is exactly one:

```{code-cell} ipython3
walk.get_eigenvalues(kind=ip.UNIT)
```

This is not a failure. `g = g{-1} + shk_g` says potential growth has no level
to return to — a shock moves it permanently, which is usually exactly what
you mean by potential growth. A unit root is a modelling choice, and it is
the right one for trends, technology, and anything with no natural anchor.

What changes is that the variable is **non-stationary**, and there is a call
that tells you which ones are:

```{code-cell} ipython3
walk.get_variable_stability()
```

`False` for `g`, `True` for everything else. That matters for tutorial 10 —
you cannot report a variance for a variable whose variance is infinite — and
for filtering data in tutorial 16.

That is the vocabulary complete: three labels, and a reading for each. The
rest of the tutorial is about what the *counts* of them mean.

## The Blanchard–Kahn condition

Here is the rule the whole tutorial turns on.

**A model with forward-looking terms has exactly one stable solution when the
number of unstable roots equals the number of forward-looking variables.**

The intuition is a counting argument. Each unstable root is a direction in
which the economy would fly off to infinity. Each forward-looking variable is
a quantity history does not pin down — nothing in the past says what people
expect. Expectations are free to jump, and the only sensible place to jump to
is the one that cancels an explosive direction. So you need exactly one free
jump per explosive direction:

- **too many unstable roots** and there are not enough free jumps to cancel
  them — no stable solution exists, and anything you simulate explodes
- **too few** and there are jumps left over — many stable solutions exist,
  and the one you get is arbitrary

For our model both counts are one:

```{code-cell} ipython3
print("unstable roots :", len(m.get_eigenvalues(kind=ip.UNSTABLE)))
print("forward-looking:", count_forward_looking(m))
```

Equal, so the solution in tutorial 8 was the only one there is.

**This count is yours to run.** `solve_first_order()` does the algebra and
returns the matrices it finds; whether that arrangement of roots is the one
your model was supposed to have is a question about the economics, and
`get_eigenvalues` is the call that answers it. The next section is what it
looks like when the two part company.

## When the count is wrong

Lower `c2` from 1.5 to 0.5. That is a central bank raising the nominal rate
by half a point when inflation rises a point — which *lowers* the real rate,
and so encourages exactly the demand that caused the inflation. Economists
call this a violation of the Taylor principle.

```{code-cell} ipython3
loose = build(c2=0.5)
```

No error. No warning. It solved.

```{code-cell} ipython3
levels = dict(loose.get_steady_levels())

print("steady levels:", {k: round(float(v), 6) for k, v in levels.items()})
print("check_steady :", loose.check_steady(when_fails="silent"))
```

And the steady state is **perfect** — `y = 0`, `pi = 2`, `i = 3`, the same
three numbers as always, because the steady state does not depend on `c2` at
all. `check_steady()` says `True` and is right to.

Now count the roots:

```{code-cell} ipython3
print("stability:", loose.get_eigenvalues_stability())
print("unstable :", len(loose.get_eigenvalues(kind=ip.UNSTABLE)),
      " forward-looking:", count_forward_looking(loose))
```

**Two unstable roots, one forward-looking variable.** Two explosive
directions and one free jump to cancel them with, so no stable solution
exists. The matrices in `loose` are a perfectly good answer to the linear
algebra; they are just not an answer to any economic question, and the root
count is the only step that separates the two.

Simulate with it:

```{code-cell} ipython3
SPAN = qq(2025,1) >> qq(2035,4)

db = Databox.steady(loose, SPAN)
db["shk_y"][qq(2026,1)] = 1.0

path = loose.simulate(db, SPAN)["y"].get_data()[:, 0]

print("first quarters:", np.round(path[1:6], 4))
print("final value:   ", f"{float(path[-1]):.4e}")
```

A one-point demand shock, and eleven years later the output gap is
`-4.2773e+28`. Not a large number — a meaningless one. The simulation ran to
completion, raised nothing, and every figure in it is noise.

You can see it coming in the solution matrix itself, if you look:

```{code-cell} ipython3
np.round(np.asarray(loose.get_solution().T), 4)
```

Entries of 9.6 and 7.7 in a matrix that gets applied once per quarter. The
healthy model's `T` had nothing in it above 0.9.

**Count the roots before you trust a simulation.** It is two lines, and it is
the only thing standing between you and results like these.

## What get_variable_stability does not tell you

`get_variable_stability()` answers one question well — which variables carry
a unit root — and it is tempting to promote it into a general health check.
Ask it about `loose`, the model from the last section:

```{code-cell} ipython3
loose.get_variable_stability()
```

**All `True`** — on the model that simulates to 10^28.

The two questions are genuinely different. `get_variable_stability()` asks
*"does this variable wander off permanently?"*, which is about unit roots.
Blanchard–Kahn asks *"does a solution exist?"*, which is about the count of
unstable roots. A model can be perfectly stationary and have no solution, and
that is precisely what `loose` is.

Only the root count answers the second question.

## Failing before the solution

Everything so far has been about `solve_first_order()` and the roots it
produces. Two failures happen one stage earlier, at `steady()`, and never
reach the eigenvalues at all — so they are worth recognising on sight. They
behave completely differently from each other.

The first is silent. Take a variable that grows by a fixed amount forever, in
a model declared flat:

```{code-cell} ipython3
NO_REST = """

!transition-variables
    x

!transition-shocks
    shk_x

!transition-equations
    x = x{-1} + 1 + shk_x;

"""

drifter = Simultaneous.from_string(NO_REST, linear=True, flat=True)
drifter.steady()

print("steady level:", float(drifter.get_steady_levels()["x"]))
print("check_steady:", drifter.check_steady(when_fails="silent"))
```

There is no value of `x` satisfying `x = x + 1`. The solver returned **zero**
anyway, and `check_steady()` returned `False` — which is the whole reason
tutorial 6 insisted you call it. A steady state of zero together with a
`False` is not a small anomaly to investigate: it means the answer does not
exist.

The second failure is loud. Ask for something with no real solution at all:

```{code-cell} ipython3
IMPOSSIBLE = """

!transition-variables
    x

!transition-shocks
    shk_x

!transition-equations
    exp(x) = -1 + shk_x;

"""

import contextlib

hopeless = Simultaneous.from_string(IMPOSSIBLE, linear=False, flat=True)

try:
    # the solver prints every iteration, and it takes a great many of them
    with contextlib.redirect_stdout(io.StringIO()):
        hopeless.steady()
except Exception as error:
    print(type(error).__module__ + "." + type(error).__name__)
    print(error)
```

`exp(x)` is positive for every real `x`, so there is nothing to find. The
nonlinear solver keeps trying regardless, printing a line per iteration, and
`redirect_stdout` is there only to keep a thousand lines of that out of the
page — drop it and you will see them.

This is the one case in the tutorial that raises, and the message earns its
place: the variant, the block, the offending equation, and which variable
that block was solving for. When the steady solver fails on a model too big
to read, that block listing is where you start. The class is
`datapie.wrongdoings.Error` — from `datapie`, not `irispie` — which is what
to import if you want to catch it specifically.

Read the wording carefully, though. It says **failed to converge**, which is
the solver's account of events: it ran out of iterations. No solver can tell
an answer it failed to find from one that was never there, and here it is the
second. So treat *failed to converge* as a symptom rather than a diagnosis —
a better starting value or a looser tolerance sometimes fixes it, and
sometimes nothing will, because the equation the message names has no
solution to find.

## ⚠️ Break it

**A singular system, which every other diagnostic reports as healthy.**

Add a variable and give it an equation that says nothing:

```{code-cell} ipython3
EMPTY = """

!transition-variables
    x, q

!transition-shocks
    shk_x

!parameters
    rho

!transition-equations
    x = rho*x{-1} + shk_x;
    q = q;

"""

hollow = Simultaneous.from_string(EMPTY, linear=True, flat=True)
hollow.assign_strict(rho=0.8)
hollow.steady()
hollow.solve_first_order()

print("check_steady       :", hollow.check_steady(when_fails="silent"))
print("variable stability :", hollow.get_variable_stability())
```

`True`, and `True` for both variables. Every check you have run so far says
this model is fine. It is not — `q = q` holds for any `q` whatsoever, so
nothing in the model determines it.

The eigenvalues are the only place it shows:

```{code-cell} ipython3
hollow.get_eigenvalues()
```

**`nan`.** Not a large root, not a root at one — no root at all, because the
matrix it would have come from cannot be decomposed. The stability call has
three labels to choose from and none of them fits, so it falls through to
`UNSTABLE`; the `nan` in the eigenvalues themselves is the honest signal, and
it is the one to read.

A `nan` in the eigenvalues means **singular**, and singular almost always
means a redundant equation or a variable nothing pins down. Real cases are
rarely as obvious as `q = q`: two equations that are multiples of one
another, an identity written twice under different names, a variable declared
and then never used.

## What you did

```{code-cell} ipython3
# the list everything comes from
m.get_eigenvalues(transform=abs)

# labelled STABLE / UNIT_ROOT / UNSTABLE
m.get_eigenvalues_stability()

# the Blanchard-Kahn count, which nothing runs for you
(len(m.get_eigenvalues(kind=ip.UNSTABLE))
 == len(m.get_eigenvalues()) - len(m.get_solution_vectors().transition_variables))

# which variables are non-stationary — a different question
walk.get_variable_stability()
```

## Things to remember

1. **Read every eigenvalue against one.** Below, it fades; above, it explodes;
   exactly one, it stays forever.
2. **Unstable roots must equal forward-looking variables.** More and nothing
   works; fewer and the answer is arbitrary.
3. **Nothing checks that for you.** No error, no warning, and a solution
   object either way — the count is yours to run.
4. **`check_steady()` cannot see it.** A steady state can be exactly right
   while the model around it has no solution.
5. **A unit root is usually deliberate**, and `get_variable_stability()` is
   how you find which variables carry one.
6. **`get_variable_stability()` is not a health check.** It returned all
   `True` for the model that simulated to 10^28.
7. **A `nan` eigenvalue means singular** — a redundant equation, or a variable
   nothing determines.

## Exercise

Put the central bank exactly on the knife edge: set `c2` to `1.0`, leaving
everything else alone.

At `c2 = 1.0` the nominal rate moves one-for-one with inflation, so the real
rate never moves at all — the boundary between the working model and the
broken one.

Before you run it, predict two things: whether `check_steady()` passes, and
how many unstable roots you get.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

`check_steady()` passes, as it has for every value of `c2` in this tutorial.
The steady state is `y = 0`, `pi = 2`, `i = 3` regardless, so that call was
never going to catch this.

The root count is the interesting half. Look at
`get_eigenvalues(transform=abs)` and you will find a root sitting a hair away
from `1.0` — and which side of the line it fell on was decided by
floating-point arithmetic, not by economics. The `STABLE` / `UNIT_ROOT` /
`UNSTABLE` label follows whichever way it went.

That is the real lesson, and it is why the Taylor principle is always stated
as `c2 > 1` and never `c2 >= 1`. A model sitting exactly on the boundary is
not a well-posed model — it is one rounding error away from either behaviour,
and no amount of reading the labels will tell you which one you are holding.
Move the parameter off the edge and the question disappears.

</details>

## Next

**Tutorial 10 · Model properties** stops asking whether the model works and
starts asking what it is like. `get_acov` and `get_acorr` give the variances
and correlations the model implies before it has seen any data — how volatile
inflation is at this calibration, how strongly output and the policy rate
move together, how long a shock takes to wash out. It is also where the unit
roots from this tutorial return, because a non-stationary variable has no
variance to report.
