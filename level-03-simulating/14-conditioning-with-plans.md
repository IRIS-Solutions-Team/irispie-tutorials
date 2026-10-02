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

# 14 · Conditioning with plans

**40 minutes** · after *13 · Nonlinear simulation* · next: *15 · Comparing
and decomposing*

Every simulation so far has answered the same kind of question: *given these
shocks, what happens?* Forecasting work is usually the other way round. The
central bank has published a rate path. The exchange rate is assumed flat.
Inflation is taken to hit a number somebody has already agreed. You know part
of the answer and you need the model to supply the rest.

A `SimulationPlan` is how you say so. You name a variable whose path you are
fixing, and a shock that is free to move instead — and the simulation solves
for whatever that shock has to be.

```{code-cell} ipython3
import contextlib
import io

import numpy as np
import irispie as ip
from irispie import Simultaneous, Databox, SimulationPlan, qq

SOURCE = """

!transition-variables

    "Output gap, % of potential"              y
    "Inflation, % per year"                   pi
    "Policy rate, % per year"                 i
    "Real exchange rate gap, %"               q
    "Foreign output gap, %"                   y_w
    "Foreign inflation, % per year"           pi_w
    "Foreign policy rate, % per year"         i_w

!transition-shocks

    shk_y, shk_pi, shk_i, shk_q
    shk_y_w, shk_pi_w, shk_i_w

!parameters

    a1, a2, a3, a4, b1, b2, b3, c1, c2, c3
    pi_tar, r_ss, beta, psi
    d1, d2, d3, pi_w_ss, r_w_ss

!transition-equations

    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + a3*q + a4*y_w + shk_y;

    "Phillips curve with exchange rate pass-through"
    pi - pi_tar = b1*(pi{-1} - pi_tar) + beta*(1-b1)*(pi{+1} - pi_tar)
                + b2*y + b3*q + shk_pi;

    "Policy rule"
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

    "Uncovered interest parity with a risk premium"
    q = q{+1} - ((i - pi{+1}) - (i_w - pi_w{+1}))/4 - psi*q + shk_q;

    "Foreign output gap"
    y_w = d1*y_w{-1} + shk_y_w;

    "Foreign inflation"
    pi_w = d2*pi_w{-1} + (1-d2)*pi_w_ss + shk_pi_w;

    "Foreign policy rate"
    i_w = d3*i_w{-1} + (1-d3)*(pi_w_ss + r_w_ss) + shk_i_w;

"""

CALIB = dict(a1=0.7, a2=0.2, a3=0.1, a4=0.3, b1=0.1, b2=0.3, b3=0.2,
             c1=0.5, c2=1.5, c3=0.5, pi_tar=2, r_ss=1, beta=0.99, psi=0.1,
             d1=0.8, d2=0.7, d3=0.8, pi_w_ss=2, r_w_ss=1)

m = Simultaneous.from_string(SOURCE, linear=True, flat=True)
m.assign_strict(**CALIB)
m.steady()
m.solve_first_order()

SPAN = qq(2025,1) >> qq(2028,4)
HOLD = qq(2025,1) >> qq(2025,4)
```

## The question

A demand shock arrives, and the policy rule does what it was written to do:

```{code-cell} ipython3
free = Databox.steady(m, SPAN)
free["shk_y"][qq(2025,1)] = 1.0
unconditioned = m.simulate(free, SPAN)

print("y:", np.round(unconditioned["y"].get_data()[:5, 0], 4))
print("i:", np.round(unconditioned["i"].get_data()[:5, 0], 4))
```

The rate rises to `3.5254` and output peaks at `0.9057`. That is the model's
own answer.

Now suppose the central bank has announced it will leave the rate at 3 for a
year. The model has no way to express that — the policy rule determines `i`,
and the rule says the rate should rise. Something has to give.

## Fixing a path and freeing a shock

What gives is the policy rule's shock. `shk_i` was invented as a one-off
surprise; here it becomes whatever it needs to be:

```{code-cell} ipython3
plan = SimulationPlan(m, SPAN)

plan.exogenize_unanticipated(HOLD, "i")
plan.endogenize_unanticipated(HOLD, "shk_i")

plan.pretty_print()
```

Two statements, and `pretty_print()` reads back what you asked for. **Exogenize**
means *take this variable out of the model's hands — I will supply its
path.* **Endogenize** means *let the model solve for this instead.*

The pair is the one tutorial 7 used on the steady state. A `SteadyPlan`
exogenizes a long-run value and endogenizes a parameter to pay for it; a
`SimulationPlan` exogenizes a path and endogenizes a shock. Same
accounting, different currency: there the price is a parameter, here it is
a sequence of surprises. The rule that every value you pin costs you
something you were going to let the model decide carries over unchanged.

The variable you fixed needs a path to be fixed *to*, and that goes in the
databox like any other data:

```{code-cell} ipython3
conditioned = Databox.steady(m, SPAN)
conditioned["shk_y"][qq(2025,1)] = 1.0
conditioned["i"][HOLD] = 3.0            # the announced path

held = m.simulate(conditioned, SPAN, plan=plan)

print("i      :", np.round(held["i"].get_data()[:5, 0], 4))
print("y      :", np.round(held["y"].get_data()[:5, 0], 4))
print("shk_i  :", np.round(held["shk_i"].get_data()[:5, 0], 4))
```

**The rate is exactly 3 for four quarters**, and output peaks at `1.0543`
instead of `0.9057` — with the brakes off, the demand shock does more.

The third line is the part people forget to look at. `shk_i` is the cost of
the promise, in the model's own units: `-0.7096` in the first quarter, then
`-0.5728`, `-0.4287`, `-0.3175`. Those are the policy surprises the bank has
to deliver each quarter to keep its word. A plan always tells you this, and
a large number there means your condition is a long way from what the model
wanted to do.

## The same thing in one line

`swap_` does both halves at once, which is how most plans are written:

```{code-cell} ipython3
same = SimulationPlan(m, SPAN)
same.swap_unanticipated(HOLD, ("i", "shk_i"))

check = m.simulate(conditioned, SPAN, plan=same)
print("identical:", np.allclose(check["i"].get_data(), held["i"].get_data()))
```

The pair reads *variable first, shock second* — what you are fixing, and what
pays for it.

## Announced or not

There are two of each verb, and the choice matters as much as it did in
tutorial 12. An unanticipated plan is a central bank that holds the rate and
surprises everyone each quarter by doing so. An anticipated plan is one that
announced the whole year in advance:

```{code-cell} ipython3
for kind in ("unanticipated", "anticipated"):
    p = SimulationPlan(m, SPAN)
    getattr(p, "swap_" + kind)(HOLD, ("i", "shk_i"))
    out = m.simulate(conditioned, SPAN, plan=p)
    print(f"{kind:15s} y {np.round(out['y'].get_data()[:5, 0], 4)}")
    print(f"{'':15s} q {np.round(out['q'].get_data()[:5, 0], 4)}")
```

Output peaks at `1.2809` when the peg is announced, against `1.0543` when it
is not. And look at the exchange rate: it **depreciates** under the announced
peg, `+0.4408`, where under the surprise version it appreciates.

That sign flip is the whole difference. If everyone knows rates will stay low
for a year, the currency has no reason to strengthen and every reason to
weaken — which adds to demand through exports and to inflation through
pass-through, on top of the loose policy itself. Announcing a peg is more
expansionary than simply doing one.

## One instrument per target

The arithmetic is strict: each period where you fix a variable needs exactly
one shock freed in the same period. The plan above fixed four quarters of `i`
and freed four quarters of `shk_i`.

Getting the count wrong does not produce a tidy error:

```{code-cell} ipython3
def try_plan(label, build):
    p = SimulationPlan(m, SPAN)
    build(p)
    try:
        out = m.simulate(conditioned, SPAN, plan=p)
        print(f"{label:34s} i = {np.round(out['i'].get_data()[:5, 0], 3)}")
    except Exception as error:
        print(f"{label:34s} {type(error).__name__}: {error}")

try_plan("fixed i, freed nothing",
         lambda p: p.exogenize_unanticipated(HOLD, "i"))

try_plan("freed shk_i, fixed nothing",
         lambda p: p.endogenize_unanticipated(HOLD, "shk_i"))

try_plan("fixed 4 quarters, freed 2",
         lambda p: (p.exogenize_unanticipated(HOLD, "i"),
                    p.endogenize_unanticipated(qq(2025,1) >> qq(2025,2), "shk_i")))
```

Three different wrong answers. Fixing a variable with nothing freed raises
`Singular matrix` — the system has more conditions than unknowns, and the
message comes from the linear algebra rather than from IrisPie, so it tells
you nothing about plans. Freeing a shock with nothing fixed is **ignored
silently**: the rate follows its own rule and you get the unconditioned
simulation back. Mismatched counts give numbers around `10^16`, which is at
least obvious.

The habit that avoids all three is `swap_`, which cannot produce an unmatched
pair.

## What about autoswaps

The model language has `!autoswaps-simulate`, which looks like a way to
declare these pairs once in the model file rather than in every script.

It is parsed and then discarded. A model file containing it loads without
complaint, and nothing reaches the plan — there is no autoswap anything on
`SimulationPlan`, and a plan built from such a model starts empty. Write the
swaps in the plan.

## ⚠️ Break it

**Fixing a path and forgetting to say what to.**

The plan and the data are separate. The plan says *`i` is mine to supply*;
the databox is where you supply it. Nothing checks that you did:

```{code-cell} ipython3
p = SimulationPlan(m, SPAN)
p.swap_unanticipated(HOLD, ("i", "shk_i"))

forgot = Databox.steady(m, SPAN)
forgot["shk_y"][qq(2025,1)] = 1.0
# meant to write: forgot["i"][HOLD] = 4.0

out = m.simulate(forgot, SPAN, plan=p)
print("i      :", np.round(out["i"].get_data()[:6, 0], 4))
print("y peak :", round(float(out["y"].get_data()[:, 0].max()), 4))
```

No error, and the rate is pinned at a perfectly reasonable 3. That is the
steady-state value `Databox.steady` put there when it built the box, and the
plan dutifully held the rate to it.

Here is the simulation you meant:

```{code-cell} ipython3
intended = Databox.steady(m, SPAN)
intended["shk_y"][qq(2025,1)] = 1.0
intended["i"][HOLD] = 4.0

out = m.simulate(intended, SPAN, plan=p)
print("i      :", np.round(out["i"].get_data()[:6, 0], 4))
print("y peak :", round(float(out["y"].get_data()[:, 0].max()), 4))
```

A rate held at 4 instead of 3, and an output peak of `0.7714` instead of
`1.0543` — a tightening scenario rather than a loosening one. Opposite
conclusions, from a line of data rather than a line of code.

Nothing can catch this for you, because holding the rate at its steady value
is a perfectly sensible thing to want. Check the exogenized path in the
output before reading anything else:

```{code-cell} ipython3
print("the path the plan held:", np.round(out["i"][HOLD].ravel(), 4))
```

It costs one line, and it is the only confirmation you get that the
conditioning did what you intended.

## What you did

```{code-cell} ipython3
plan = SimulationPlan(m, SPAN)

# fix a variable, free a shock, in one statement
plan.swap_unanticipated(HOLD, ("i", "shk_i"))

# or the two halves separately
plan.exogenize_unanticipated(HOLD, "i")
plan.endogenize_unanticipated(HOLD, "shk_i")

# the path you are fixing to lives in the databox
conditioned["i"][HOLD] = 3.0

out = m.simulate(conditioned, SPAN, plan=plan)
```

## Things to remember

1. **A plan says which variable you are supplying and which shock pays for
   it.** Exogenize the one, endogenize the other.
2. **The path you fix to goes in the databox**, not in the plan.
3. **`swap_` does both halves** and cannot leave an unmatched pair.
4. **Read the endogenized shock afterwards.** It is the cost of the
   condition, and a large value means you asked for something the model
   resists.
5. **Anticipated and unanticipated plans give different answers.** An
   announced rate peg is more expansionary than an unannounced one, and here
   it flips the sign of the exchange rate response.
6. **One freed shock per fixed period.** Too few raises `Singular matrix`,
   too many gives nonsense, and freeing with nothing fixed is ignored.
7. **`!autoswaps-simulate` does nothing.** Write the swaps in the plan.

## Exercise

Peg the **exchange rate** instead of the policy rate: hold `q` at zero for
2025 using `shk_q`, with the same demand shock.

Before you run it, predict whether output ends up higher or lower than in the
unconditioned simulation.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**Higher** — output peaks at `0.9265` against `0.9057`, and inflation at
`2.4726` against `2.3986`.

In the unconditioned run the currency appreciates to `-0.3219`, and that
appreciation does two useful things: it damps demand through exports and
pulls inflation down through pass-through. Pegging the rate removes both, so
more of the demand shock reaches output and prices.

The policy rate has to work harder as a result, rising to `3.5861` instead of
`3.5254`.

The risk premium shocks that hold the peg are `0.3819`, `0.3441`, `0.2436`,
`0.1500` — all positive, because keeping a currency from appreciating means
pushing against it every quarter.

This is the standard argument about fixed exchange rates in miniature: the
nominal anchor costs you a stabiliser, and monetary policy has to make up the
difference.

</details>

## Next

**Tutorial 15 · Comparing and decomposing** closes Level 3. You now have two
simulations — a baseline and a conditioned scenario — and the obvious next
question is what accounts for the difference between them. `minus_control`
subtracts one run from another, and a shock decomposition splits a path into
the contribution of each shock, which is how a forecast change gets explained
to somebody who was not in the room.
