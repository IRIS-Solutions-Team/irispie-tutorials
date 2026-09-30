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

# 8 · Solving the model

**35 minutes** · after *7 · Steering the steady state* · next: *9 · When it
will not solve*

`solve_first_order()` has appeared in every tutorial since the first one,
always in the same place — after `steady()`, before `simulate()` — and has
never been explained. This tutorial opens it up. By the end you will have
stepped the model forward by hand, using nothing but the matrices it produced.

```{code-cell} ipython3
import numpy as np
import irispie as ip
from irispie import Simultaneous, Databox, qq

CALIB = dict(a1=0.7, a2=0.2, b1=0.6, b2=0.3, c1=0.5,
             c2=1.5, c3=0.5, pi_tar=2, r_ss=1)
```

The running model, with a measurement equation added so there is something to
say about the second half of the solution:

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

!measurement-variables

    "Observed inflation"                  obs_pi

!measurement-shocks

    "Inflation measurement error"         mshk_pi

!parameters

    a1, a2, b1, b2, c1, c2, c3, pi_tar, r_ss

!transition-equations

    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;

    "Phillips curve"
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;

    "Policy rule"
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

!measurement-equations

    "Inflation is observed with error"
    obs_pi = pi + mshk_pi;

"""

m = Simultaneous.from_string(SOURCE, linear=True, flat=True)
m.assign_strict(**CALIB)
m.steady()
m.solve_first_order()
```

## What "solving" means

Your model cannot be simulated as written. Look at the demand equation —
`y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;`. Today's output depends on
**next quarter's inflation**. To compute 2026Q1 you
need 2026Q2, which needs 2026Q3, and so on forever. There is no way to march
through time with an equation like that.

Solving the model means rewriting this tangle as something you *can* march
through: a rule that gives you today from yesterday, with no reference to the
future at all. That rule is what `solve_first_order()` produces, and it exists
because the forward-looking terms can be resolved once, in advance, for the
whole model.

## What is in the state vector

Before the matrices, the labels:

```{code-cell} ipython3
m.get_solution_vectors()
```

Five lists, and the ordering of each one is the ordering of the matrix rows
and columns you are about to see:

- **transition variables** — `y`, `pi`, `i`, the economy itself
- **transition shocks** — `shk_y`, `shk_pi`, `shk_i`
- **anticipated shock values** — the silent companions from tutorial 3,
  unused until tutorial 12
- **measurement variables** — `obs_pi`
- **measurement shocks** — `mshk_pi`

## The solution is six matrices

```{code-cell} ipython3
sol = m.get_solution()

for name in ("T", "P", "K", "Z", "H", "D"):
    print(f"{name}: {np.asarray(getattr(sol, name)).shape}")
```

They fit together in two equations. The first moves the economy forward:

```{code-cell} ipython3
print("T — how yesterday's state maps into today's")
print(np.round(np.asarray(sol.T), 4))
print()
print("P — how today's shocks hit")
print(np.round(np.asarray(sol.P), 4))
print()
print("K — the constant that holds the steady state in place")
print(np.round(np.asarray(sol.K), 4))
```

In the notation of the code that is
`xi(t) = T @ xi(t-1) + K + P @ u(t)`, where `xi` holds the transition
variables and `u` the shocks.

The second reads observables off it:

```{code-cell} ipython3
print("Z —", np.asarray(sol.Z))
print("H —", np.asarray(sol.H))
print("D —", np.asarray(sol.D))
```

That one is `y(t) = Z @ xi(t) + D + H @ w(t)`, where `y` holds the
measurement variables and `w` the measurement shocks.

Look at `Z`. It is `[0, 1, 0]` — a row that picks out the **second** transition
variable and ignores the other two. That is the measurement equation
`obs_pi = pi + mshk_pi` written as arithmetic: take `pi`, add the measurement
shock with weight `H = 1`, add nothing else.

## Stepping the model by hand

This is the part worth doing once. Run an ordinary simulation:

```{code-cell} ipython3
SPAN = qq(2025,1) >> qq(2030,4)

db = Databox.steady(m, SPAN)
db["shk_y"][qq(2026,1)] = 1.0

out = m.simulate(db, SPAN)

value = lambda name, date: float(np.asarray(out[name][date]).ravel()[0])
names = ("y", "pi", "i")
```

Take the state at 2025Q4, apply the two matrices and the constant, and see
whether you land where `simulate` did:

```{code-cell} ipython3
previous = np.array([value(n, qq(2025,4)) for n in names])
shocks   = np.array([1.0, 0.0, 0.0])          # shk_y = 1 in 2026Q1

by_hand  = np.asarray(sol.T) @ previous + np.asarray(sol.K) \
         + np.asarray(sol.P) @ shocks

simulated = np.array([value(n, qq(2026,1)) for n in names])

print("by hand: ", np.round(by_hand, 6))
print("simulate:", np.round(simulated, 6))
print("match:   ", np.allclose(by_hand, simulated))
```

**Identical.** `simulate()` is doing exactly this, once per period, and
nothing more. The measurement half works the same way:

```{code-cell} ipython3
by_hand_obs = np.asarray(sol.Z) @ simulated + np.asarray(sol.D) \
            + np.asarray(sol.H) @ np.array([0.0])

print("by hand: ", round(float(by_hand_obs[0]), 6))
print("simulate:", round(value("obs_pi", qq(2026,1)), 6))
```

Once you have seen this, a simulation stops being a black box. It is a matrix
multiplication in a loop.

## The impact response is a column of P

`P` has one column per shock, and each column is what that shock does on the
day it lands:

```{code-cell} ipython3
np.round(np.asarray(sol.P)[:, 0], 6)
```

Compare with the actual simulated response on impact, measured from the steady
state:

```{code-cell} ipython3
steady = m.get_steady_levels()
np.round([value(n, qq(2026,1)) - float(steady[n]) for n in names], 6)
```

The same three numbers — and the first of them, **1.063096**, is the output
gap peak you have been reading since tutorial 1. It was sitting in the top-left
corner of `P` the whole time.

That is a useful habit: to know what a shock does on impact, you do not need
to simulate. Read the column.

## The whole path from two matrices

One column of `P` gives the impact. Apply `T` over and over and you get
everything that happens afterwards — because after the shock lands there are
no more shocks, and the constant drops out once you work in deviations from
the steady state:

```{code-cell} ipython3
horizon = 20

response = np.zeros((horizon, 3))
response[0] = np.asarray(sol.P) @ np.array([1.0, 0.0, 0.0])   # impact

for t in range(1, horizon):
    response[t] = np.asarray(sol.T) @ response[t-1]           # and then decay

np.round(response[:4], 6)
```

No model object, no `simulate`, no parameters — two matrices and a loop. Put
it beside the real simulation to see whether it is telling the truth:

```{code-cell} ipython3
from irispie import Series

SPAN = qq(2026,1) >> qq(2030,4)

db = Databox.steady(m, SPAN)
db["shk_y"][qq(2026,1)] = 1.0
out = m.simulate(db, SPAN)

steady = m.get_steady_levels()

fig = ip.make_subplots(
    (3, 1),
    figure_title="A demand shock, generated two ways",
    figure_height=760,
    subplot_titles=["Output gap", "Inflation", "Policy rate"],
    show_legend=True,
)

for position, name in enumerate(("y", "pi", "i")):
    Series(start=qq(2026,1), values=response[:, position]).plot(
        figure=fig, subplot=position, show_figure=False,
        legend=["from T and P"],
    )
    (out[name] - float(steady[name])).plot(
        figure=fig, subplot=position, show_figure=False,
        legend=["from simulate"],
    )

fig
```

**One line each.** The two curves lie exactly on top of one another — the
largest discrepancy across all sixty numbers is around `5e-14`, which is
floating-point noise.

Read the shapes while they are in front of you. Output jumps on impact and
decays away. Inflation keeps climbing for two more quarters before it turns,
because the Phillips curve is backward-looking in part. The policy rate rises
with it and stays high longest. None of that is in any equation you can point
to any more — it is all in the eigenvalues of `T`, which is tutorial 9.

## Transition and measurement are not symmetric

The two halves of the state space look similar and behave completely
differently. The transition equation **feeds back into itself** — today's
`xi` becomes tomorrow's input, through `T`. The measurement equation does not
appear on the right-hand side of anything:

```{code-cell} ipython3
print("T maps xi(t-1) -> xi(t), shape", np.asarray(sol.T).shape)
print("Z maps xi(t)   -> y(t),  shape", np.asarray(sol.Z).shape)
print()
print("there is no matrix mapping y(t) back into xi — by construction")
```

That is the one-way window tutorial 3 demonstrated, now visible in the
algebra. A measurement error moves `obs_pi` and can never move `pi`, because
there is no route back. It is also why the Kalman filter in tutorial 16 is a
non-trivial problem: it has to run this arrow backwards, and the arrow does
not exist.

## ⚠️ Break it

**Changing a parameter and not re-solving.**

The solution is a snapshot. It is computed once, from the parameters as they
stood at that moment, and stored. It has no link back to them.

```{code-cell} ipython3
def peak_output_gap(model):
    d = Databox.steady(model, SPAN)
    d["shk_y"][qq(2026,1)] = 1.0
    return float(model.simulate(d, SPAN)["y"].get_data()[:, 0].max())

print("with c2 = 1.5:", round(peak_output_gap(m), 6))
```

Now make the central bank much tougher on inflation — `c2` from 1.5 to 3.0 —
and simulate again, forgetting to re-solve. The change goes onto a **copy** of
the model, so that breaking it here leaves `m` alone for the rest of the
tutorial:

```{code-cell} ipython3
tough = m.copy()
tough.assign(c2=3.0)

print("with c2 = 3.0:", round(peak_output_gap(tough), 6))
print("check_steady: ", tough.check_steady(when_fails="silent"))
```

**The same number.** A parameter that governs how aggressively the central
bank responds was doubled, and the simulation did not notice. `check_steady()`
returns `True` and is no help at all — the steady state genuinely did not
change, because `y = 0`, `pi = 2` and `i = 3` hold for any `c2`.

Re-solve and the real answer appears:

```{code-cell} ipython3
tough.steady()
tough.solve_first_order()

print("with c2 = 3.0, re-solved:", round(peak_output_gap(tough), 6))
```

`0.9229` against `1.0631`. A tougher central bank does damp the output
response — you simply could not see it.

**The rule: any time you change a parameter, run `steady()` and
`solve_first_order()` again before simulating.** Nothing enforces it, no
warning is printed, and the stale answer looks perfectly reasonable.

## What you did

```{code-cell} ipython3
# the labels for every row and column
m.get_solution_vectors()

# the six matrices
sol = m.get_solution()
sol.T, sol.P, sol.K     # xi(t) = T @ xi(t-1) + K + P @ u(t)
sol.Z, sol.H, sol.D     # y(t)  = Z @ xi(t)   + D + H @ w(t)

# what a shock does on impact, without simulating
np.asarray(sol.P)[:, 0]
```

## Things to remember

1. **Solving removes the future.** The equations reference `pi{+1}`; the
   solution does not reference anything but yesterday.
2. **The solution is six matrices** and two equations — one to move the state
   forward, one to read observables off it.
3. **`get_solution_vectors()` gives the row and column labels.** Without it
   the matrices are anonymous.
4. **A column of `P` is an impact response.** No simulation needed.
5. **The measurement half is one-way.** Nothing maps observables back into the
   economy, which is exactly the problem the Kalman filter solves.
6. **The solution is a snapshot, not a link.** Change a parameter and you must
   re-run `steady()` and `solve_first_order()`, or you will silently simulate
   the old model.

## Exercise

Step the model forward one more quarter by hand.

You know the state at 2026Q1 from the simulation above. There is **no shock**
in 2026Q2, so the shock term drops out entirely.

Work out `y`, `pi` and `i` for 2026Q2 using only `T`, `K` and the 2026Q1
values, then check against `simulate`.

Before you run it, predict one thing: does the output gap keep rising or start
to fall?

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

```python
q1 = np.array([value(n, qq(2026,1)) for n in names])
by_hand = np.asarray(sol.T) @ q1 + np.asarray(sol.K)
```

gives **`[0.693671, 3.172071, 4.480767]`**, matching `simulate` exactly.

The output gap **falls**, from `1.0631` to `0.6937`. The shock was a one-off,
so nothing is pushing output up any more, and the central bank is now leaning
against it — the policy rate has risen from `3.857` to `4.481`.

Inflation is still climbing, from `2.788` to `3.172`, and will keep climbing
for one more quarter before it turns. That lag is the Phillips curve: today's
inflation depends on yesterday's inflation and on the output gap that has only
just started to close.

Note what you did **not** need: the model file, the parameters, the steady
state, or `simulate`. Three numbers and two matrices are the entire model,
once it is solved.

</details>

## Next

**Tutorial 9 · When it will not solve** is the other half of this one. The
solution you have been reading exists only when the model is well behaved, and
a great deal of real modelling time goes on the cases where it is not.
`get_eigenvalues` and what `STABLE`, `UNIT` and `UNSTABLE` mean, the
Blanchard–Kahn condition and why IrisPie does not check it for you, unit
roots, singular systems, and a steady state that simply does not exist — each
message, what actually caused it, and what to do.
