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

# 7 · Steering the steady state

**35 minutes** · after *6 · The steady state* · next: *8 · Solving the model*

Tutorial 6 solved the steady state the normal way round: you supply the
parameters, IrisPie tells you where the economy settles. Calibration is the
other way round. You know roughly where the economy settles — twenty years of
data say so — and what you do not know is the parameters that put it there.

A `SteadyPlan` turns the problem around. You name the long-run values you
believe, name the parameters you are willing to let move, and IrisPie solves
for those instead.

```{code-cell} ipython3
import irispie as ip
from irispie import Simultaneous, SteadyPlan

CALIB = dict(a1=0.7, a2=0.2, b1=0.6, b2=0.3, c1=0.5,
             c2=1.5, c3=0.5, pi_tar=2, r_ss=1)

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
```

**One thing before anything else: a steady plan only works on a model built
with `linear=False`.** The linear steady solver has no way to use one, and it
does not complain — it quietly ignores the plan and hands back the ordinary
answer. *Break it* shows exactly what that looks like. Every model in this
tutorial is therefore built nonlinear, which is why you will see iteration
tables throughout.

```{code-cell} ipython3
m = Simultaneous.from_string(BASE, linear=False, flat=True)
m.assign_strict(**CALIB)
```

## What can be steered

Before writing a plan, ask the model what it will allow:

```{code-cell} ipython3
plannable = m.get_steady_plannable()

print("can be exogenized: ", plannable.can_be_exogenized)
print("can be endogenized:", plannable.can_be_endogenized)
print("can be fixed_level:", plannable.can_be_fixed_level)
print("can be fixed_change:", plannable.can_be_fixed_change)
```

The pattern is simple. **Variables** can be exogenized — pinned to a value you
choose. **Parameters** can be endogenized — turned into unknowns for the
solver to find. `can_be_fixed_change` is empty because this model is flat and
nothing has a change to fix.

That is the whole trade. Every value you pin costs you one parameter, because
the number of equations has not changed.

## A calibration you can check by hand

Suppose twenty years of data say the average policy rate is **4%** while
average inflation is **2%**. The model's long run currently disagrees:

```{code-cell} ipython3
m.steady()
m.get_steady_levels()
```

`i` settles at 3, not 4. In this model the steady policy rate is
`pi_tar + r_ss`, so with a 2% target the data are telling you the real
interest rate is **2**, not the 1 you assumed. You could just edit `r_ss` —
but in a model with forty parameters you cannot, so do it properly.

Say what you want, then say what may move:

```{code-cell} ipython3
m.assign(i=4)                 # the value you want i to settle at

plan = SteadyPlan(m)
plan.exogenize("i")           # i is now given
plan.endogenize("r_ss")       # r_ss is now unknown

plan
```

Two entries: one variable pinned, one parameter freed. Now solve with the plan
attached:

```{code-cell} ipython3
m.steady(plan=plan)
m.get_steady_levels()
```

```{code-cell} ipython3
m.get_parameters()["r_ss"]
```

**`r_ss = 2`**, exactly as the arithmetic said. The steady state now has the
policy rate at 4, and `y` and `pi` are untouched.

That is calibration: you asserted something you believe about the long run,
and the model told you what parameter value it implies. The number was easy
to check here on purpose — in a real model it will not be, which is why the
machinery is worth learning on one where it is.

### swap does both in one call

`exogenize` and `endogenize` almost always come in pairs, so there is a
shorthand:

```{code-cell} ipython3
m2 = Simultaneous.from_string(BASE, linear=False, flat=True)
m2.assign_strict(**CALIB)
m2.assign(i=4)

plan2 = SteadyPlan(m2)
plan2.swap(("i", "r_ss"))

m2.steady(plan=plan2)
float(m2.get_parameters()["r_ss"])
```

Note the **pair inside the brackets**. `swap` takes tuples, one per exchange,
so `swap(("i", "r_ss"))` is right and `swap("i", "r_ss")` fails with
`IndexError: string index out of range` — it reads the bare string `"i"` as a
pair and asks for its second element.

## Fixing a level without fixing its growth

On a growing model there is a second, subtler control. Add the price level
from tutorial 6 back in:

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

    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

    "The price level accumulates inflation"
    P = P{-1}*exp(pi/400);

"""

g = Simultaneous.from_string(GROWTH, linear=False, flat=False)
g.assign_strict(**CALIB)
g.get_steady_plannable().can_be_fixed_level
```

Solve it with no plan at all and watch what happens:

```{code-cell} ipython3
g.steady()
print("P level: ", float(g.get_steady_levels()["P"]))
print("P change:", float(g.get_steady_changes()["P"]))
print("check:   ", g.check_steady(when_fails="silent"))
```

**The price level collapsed to zero** — `1.1e-10` is the solver's best
numerical attempt at it, and the growth rate went to 1, meaning no growth at
all. Worse, `check_steady` says `True`, because `0 = 0 × anything` satisfies
the equation perfectly.

This is the degenerate root tutorial 6 warned about. Nothing in the model pins
the *level* of a price index, so the solver is free to pick zero — and zero is
a perfectly good fixed point, just not the one you want.

`fix_level` is the cure. It says "keep this level where I put it, and solve
for the growth rate":

```{code-cell} ipython3
g2 = Simultaneous.from_string(GROWTH, linear=False, flat=False)
g2.assign_strict(**CALIB)
g2.assign(P=1.0)

plan3 = SteadyPlan(g2)
plan3.fix_level("P")

g2.steady(plan=plan3)
print("P level: ", float(g2.get_steady_levels()["P"]))
print("P change:", float(g2.get_steady_changes()["P"]))
print("check:   ", g2.check_steady(when_fails="silent"))
```

Level 1, growth `1.00501252` — the correct answer from tutorial 6, and no
parameter had to be given up, because fixing a level that nothing determines
costs nothing.

### exogenize pins both halves, fix_level pins one

The difference matters exactly here. `exogenize` pins the level **and** the
change to whatever you assigned:

```{code-cell} ipython3
g3 = Simultaneous.from_string(GROWTH, linear=False, flat=False)
g3.assign_strict(**CALIB)
g3.assign(P=(1.0, 1.005))     # (level, change) -- 1.005 is a guess, and slightly wrong

plan4 = SteadyPlan(g3)
plan4.exogenize("P")

g3.steady(plan=plan4)
print("P change:", float(g3.get_steady_changes()["P"]))
print("check:   ", g3.check_steady(when_fails="silent"))
```

`check_steady` is **`False`**. You pinned the growth rate at `1.005`, the
equations want `1.00501252`, and something has to give. `fix_level` avoided
this by pinning only the half you actually knew.

**So: `exogenize` when you know both, `fix_level` when you know where a
variable starts but not how fast it grows.**

### fix_change, the half that does not work

The obvious counterpart is `fix_change`: pin the growth rate and let the
solver find the level. `SteadyPlan` has the method, and it cannot currently
do that. The register it validates against holds the parameters rather than
the variables, which is the same list `can_be_endogenized` returns:

```{code-cell} ipython3
gc = Simultaneous.from_string(GROWTH, linear=False, flat=False)
gc.assign_strict(**CALIB)
gc.assign(P=1.0)

plannable = gc.get_steady_plannable()
print("can_be_fixed_level :", plannable.can_be_fixed_level)
print("can_be_fixed_change:", plannable.can_be_fixed_change)

try:
    SteadyPlan(gc).fix_change("P")
except Exception as error:
    print()
    print("fix_change('P'):",
          " ".join(str(error).replace("⏐", " ").split()))
```

Every variable is refused, so there is nothing whose change can be pinned.
The names it does accept are parameters, and a parameter has no
steady-state change to fix — so accepting one changes nothing:

```{code-cell} ipython3
def solve_growth(fix_change_name=None):
    model = Simultaneous.from_string(GROWTH, linear=False, flat=False)
    model.assign_strict(**CALIB)
    model.assign(P=1.0)
    plan = SteadyPlan(model)
    plan.fix_level("P")
    if fix_change_name:
        plan.fix_change(fix_change_name)
    model.steady(plan=plan)
    return (round(float(model.get_steady_changes()["P"]), 8),
            round(float(model.get_parameters()["pi_tar"]), 8))


plain = solve_growth()
fixed = solve_growth("pi_tar")
print("fix_level only            :", plain)
print("fix_level and fix_change  :", fixed)
print("identical                 :", plain == fixed)
```

The call is accepted, the solve succeeds, and the answer is the same to
eight decimal places. Until the register is corrected, impose a growth rate
the way the rest of this tutorial imposes anything: assign the value you
want and endogenize a parameter to pay for it.

## Seeing the structure inside the model

Steady states are easier to solve when the equations are not all tangled
together. `split_into_blocks=True` asks the solver to look for a recursive
order and work through it in pieces.

The three-equation model is genuinely simultaneous, so there is nothing to
find. Add two reporting variables that read off the core without feeding back
into it:

```{code-cell} ipython3
RECURSIVE = """

!transition-variables

    "Output gap, % of potential"          y
    "Inflation, % per year"               pi
    "Policy rate, % per year"             i
    "Real marginal cost, index"           mc
    "Real interest rate, reported"        rr

!transition-shocks

    "Demand shock"                        shk_y
    "Cost-push shock"                     shk_pi
    "Monetary policy shock"               shk_i

!parameters

    a1, a2, b1, b2, c1, c2, c3, pi_tar, r_ss, gamma

!transition-equations

    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

    "Marginal cost rises with the output gap"
    mc = exp(gamma*y/100);

    "The real rate is the nominal rate less inflation"
    rr = i - pi;

"""

r = Simultaneous.from_string(RECURSIVE, linear=False, flat=True)
r.assign_strict(**CALIB, gamma=1.5)
r.steady(split_into_blocks=True)
r.get_steady_levels()
```

Read the solver output above the answer. **Three blocks**: a 3×3 for `y`, `pi`
and `i`, which really do have to be solved together, then two 1×1 blocks for
`mc` and `rr`, each solved on its own once the core is known.

Compare with the same model solved as one lump:

```{code-cell} ipython3
r2 = Simultaneous.from_string(RECURSIVE, linear=False, flat=True)
r2.assign_strict(**CALIB, gamma=1.5)
r2.steady(split_into_blocks=False)
r2.get_steady_levels()
```

One 5×5 block, same answer. On five equations the difference is nothing; on
five hundred it is the difference between a solver that converges and one that
does not, because a 500×500 nonlinear system is a much harder thing to solve
than a handful of small ones in sequence.

It is also a reading of your own model you cannot get any other way: the block
structure tells you which parts are genuinely simultaneous and which are just
reporting.

## ⚠️ Break it

**A plan on a linear model.**

Everything above used `linear=False`. Here is the identical calibration on a
model declared linear:

```{code-cell} ipython3
oops = Simultaneous.from_string(BASE, linear=True, flat=True)
oops.assign_strict(**CALIB)
oops.assign(i=4)

bad_plan = SteadyPlan(oops)
bad_plan.swap(("i", "r_ss"))

oops.steady(plan=bad_plan)

print("i settled at: ", float(oops.get_steady_levels()["i"]))
print("r_ss is now:  ", float(oops.get_parameters()["r_ss"]))
print("check_steady: ", oops.check_steady(when_fails="silent"))
```

**Nothing happened.** `i` is 3, not 4. `r_ss` is still 1. And
`check_steady()` returns **`True`**, because the steady state IrisPie returned
is a perfectly good steady state — of the model you did not ask about.

No error, no warning, and the usual safety net is no help here: the answer is
internally consistent, it simply ignores your plan. The linear steady solver
takes no `plan` argument at all, and the one you passed was swallowed.

**If a plan appears to do nothing, check the `linear` flag first.** It is the
only explanation that leaves no trace.

## What you did

```{code-cell} ipython3
# ask what the model will let you steer
m.get_steady_plannable().can_be_exogenized
m.get_steady_plannable().can_be_endogenized

# pin a long-run value, free a parameter, solve for it
#     plan = SteadyPlan(m)
#     plan.swap(("i", "r_ss"))        # note the pair
#     m.steady(plan=plan)

# on a growing model: pin the level, let the growth rate solve
#     plan.fix_level("P")

# what each register will accept
m.get_steady_plannable().can_be_fixed_level
m.get_steady_plannable().can_be_fixed_change

# see which equations are genuinely simultaneous
r.steady(split_into_blocks=True)
```

## Things to remember

1. **A steady plan needs `linear=False`.** On a linear model it is silently
   ignored and `check_steady()` still says `True`.
2. **Every value you pin costs one parameter.** Exogenize a variable,
   endogenize a parameter, and the count stays right.
3. **`swap` takes pairs** — `swap(("i", "r_ss"))`, not `swap("i", "r_ss")`.
4. **`exogenize` pins level and change; `fix_level` pins only the level.** On
   a growing variable that is the difference between a right answer and a
   wrong one.
5. **A growing model with no plan can collapse to zero**, and `check_steady()`
   will approve of it. `fix_level` is the fix.
6. **`fix_change` cannot pin a growth rate.** Its register holds the
   parameters rather than the variables, so every variable is refused and
   the parameters it accepts have no change to fix — passing one is a
   silent no-op. Assign the growth rate and endogenize a parameter instead.
7. **`split_into_blocks=True` shows the recursive structure** and makes large
   steady states solvable.

## Exercise

Calibrate two things at once. You believe the long-run policy rate is **4**
and long-run inflation is **3**.

Pin both, free `r_ss` and `pi_tar`, and solve.

Before you run it, predict what each parameter comes out as.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

`pi_tar` goes to **3** and `r_ss` stays at **1**.

Most people expect both to move, because both were freed. But look at what
the model requires: the steady policy rate is `pi_tar + r_ss`, and you asked
for `i = 4` with `pi = 3`. The Phillips curve pins `pi` to `pi_tar`, so
`pi_tar` must be 3 — and then `r_ss` must be `4 - 3 = 1`, which is what it
already was.

Freeing a parameter does not oblige it to move. It only means the solver is
*allowed* to move it, and here it turns out it does not need to.

The plan looks like this:

```python
plan = SteadyPlan(m)
plan.swap(("i", "r_ss"))
plan.swap(("pi", "pi_tar"))
```

and `check_steady()` returns `True` throughout.

</details>

## Next

**Tutorial 8 · Solving the model** leaves the steady state behind and asks
what `solve_first_order()` has been doing in every tutorial since the first
one. What the solution actually *is*, what `get_solution_vectors()` is showing
you, the state-space form the simulations run on, and why transition and
measurement variables are kept apart in it.
