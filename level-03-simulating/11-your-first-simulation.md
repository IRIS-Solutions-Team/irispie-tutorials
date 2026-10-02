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

# 11 · Your first simulation

**40 minutes** · after *10 · Model properties* · next: *12 · Shocks*

**From this tutorial on, the model is a small open economy.**

Levels 1 and 2 used three equations, which was the right size for learning
what a model file is and how a solution is built. Level 3 is about putting a
model to work, and most of that work — conditioning a forecast, decomposing
one, filtering data against observations — needs an exchange rate and a world
outside. So the model gains both here, and keeps them for the rest of the
series.

The rest of the tutorial is about `simulate`. You have called it since the
first tutorial, always the same way: build a databox with `Databox.steady`,
put a shock in it, run. Here the input gets opened up — what the model needs
before it can step forward, where those values have to be dated, and what
comes back.

## The model from here on

```{code-cell} ipython3
import numpy as np
import irispie as ip
from irispie import Simultaneous, Databox, Series, qq

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

    "Demand shock"                            shk_y
    "Cost-push shock"                         shk_pi
    "Monetary policy shock"                   shk_i
    "Risk premium shock"                      shk_q
    "Foreign demand shock"                    shk_y_w
    "Foreign cost-push shock"                 shk_pi_w
    "Foreign monetary policy shock"           shk_i_w

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

SPAN = qq(2025,1) >> qq(2026,4)

m.get_steady_levels()
```

Seven variables in three groups.

The three you know — output, inflation, the policy rate — are unchanged
except that demand now responds to the exchange rate and to foreign demand.

`q` is the **real** exchange rate gap, in per cent, resting at zero. Interest
parity says a currency expected to appreciate must pay a lower rate, now
compared against the foreign real rate rather than a constant, and `psi*q` is
the risk premium that keeps the gap from wandering off. Pass-through, the
`b3*q` term, says a weaker currency raises import prices and so raises
inflation.

The three foreign variables are simple autoregressive processes. The rest of
the world is not explained by this model; it is something that happens to it.

The Phillips curve is also written in deviations from target now, with a
discount factor `beta` on expected inflation. That is the standard form, and
it matters more here than it did before — tutorial 12 turns on it.

```{code-cell} ipython3
print("unstable roots :", len(m.get_eigenvalues(kind=ip.UNSTABLE)))
print("forward-looking:",
      len(m.get_eigenvalues()) - len(m.get_solution_vectors().transition_variables))
```

Both counts are three, so the Blanchard–Kahn condition holds. The three
variables that appear with a lead are `pi`, `q` and `pi_w`. The closed model
in tutorial 9 had one of each.

## What the model asks for

A solved model steps forward with `xi(t) = T @ xi(t-1) + K + P @ u(t)`, from
tutorial 8. To produce the first period it needs `xi(t-1)` — the quarter
before the simulation starts. The model will name those values:

```{code-cell} ipython3
m.get_initials()
```

**Six names, for seven variables.** `q` is not among them.

That is not an omission. Look back at the interest parity equation: `q`
appears as `q` and as `q{+1}`, never as `q{-1}`. Nothing about yesterday's
exchange rate matters, so there is nothing to supply. The exchange rate is
worked out from where the economy is going, not from where it has been.

`get_initials()` is the only reliable way to tell which variables need one.
Guessing "one lag of everything" would have had you supplying a number the
model never reads.

## A simulation from six numbers

No `Databox.steady`, no shortcuts. Put the model at rest in 2024Q4 and let it
run:

```{code-cell} ipython3
start_here = Databox()

for name, value in (("y", 0.0), ("pi", 2.0), ("i", 3.0),
                    ("y_w", 0.0), ("pi_w", 2.0), ("i_w", 3.0)):
    start_here[name] = Series(start=qq(2024,4), values=[value])

flat = m.simulate(start_here, SPAN)

print("y :", np.round(flat["y"].get_data()[:, 0], 4))
```

A flat line at the steady state, which is the right answer for a model with
no shocks starting from rest. Now give it something to react to:

```{code-cell} ipython3
start_here["shk_y"] = Series(start=qq(2025,1), values=[1.0])

out = m.simulate(start_here, SPAN)

print("y:", np.round(out["y"].get_data()[:, 0], 4))
print("q:", np.round(out["q"].get_data()[:, 0], 4))
```

A one-point demand shock produces an output gap of **0.9057**, from seven
values typed by hand. Shocks go in the same databox as everything else, dated
to the period they arrive in, and every period you leave out is zero.

The exchange rate moves too. `q` is at `-0.3219` in the quarter of the shock:
demand is up, the central bank will raise rates, and a currency about to pay
more is worth more now.

## What the foreign block buys you

The three foreign equations look like padding until you shock one:

```{code-cell} ipython3
abroad = Databox.steady(m, SPAN)
abroad["shk_y_w"][qq(2025,1)] = 1.0
result = m.simulate(abroad, SPAN)

steady = m.get_steady_levels()
for name in ("y_w", "y", "pi", "i", "q"):
    path = result[name].get_data()[:, 0] - float(steady[name])
    print(f"{name:4s} high {path.max():7.4f}   low {path.min():7.4f}")
```

A one-point rise in demand abroad lifts domestic output by **0.3559** and
inflation by **0.2627**, and the central bank tightens by `0.4639`. The
currency appreciates to **-0.3516** — an appreciation is a negative number
here, which is why it shows in the low column rather than the high one.

Exports did the work, through the `a4*y_w` term in the demand equation.

This is the experiment a closed model cannot run at all, and it is why the
foreign block is worth three extra equations.

## Initial conditions are data

It is tempting to treat that first period as just a starting value. But the
model reacts to it exactly as it reacts to a shock.

```{code-cell} ipython3
off_steady = Databox.steady(m, SPAN)
off_steady["pi"][qq(2024,4)] = 5.0

np.round(m.simulate(off_steady, SPAN)["pi"].get_data()[:, 0], 4)
```

There are no shocks here at all. Inflation begins at 5 instead of 2 and is
back near target within two quarters, because this Phillips curve is mostly
forward-looking — `b1` is 0.1, so only a tenth of today's inflation comes
from yesterday's.

When you simulate from real data, this first period is where the data goes.

## The shortcut, and the one thing it hides

`Databox.steady` is what you have been using, and it supplies those initial
conditions by construction — which is why they have never needed a thought:

```{code-cell} ipython3
db = Databox.steady(m, SPAN)

print("names:", len(db.get_names()), "items — variables, shocks, parameters, stds")
print("span :", len(list(SPAN)), "periods")
print("y    :", len(db["y"].get_data()), "periods, starting", db["y"].start)
```

**Ten periods for an eight-period span.** `Databox.steady` adds one quarter
at the start and one at the end: `2024Q4` before the span, `2027Q1` after it.
The one at the start is the initial condition — the `2024Q4` you typed by
hand above. You can turn them off with `prepend_initial=` and
`append_terminal=`, but the one at the start is what the model needs, so
leave it.

The output is narrower than the input:

```{code-cell} ipython3
db["shk_y"][qq(2025,1)] = 1.0
out = m.simulate(db, SPAN)

print("in :", len(db.get_names()), "items")
print("out:", len(out.get_names()), "items")
```

Parameters and standard deviations are dropped, on the grounds that they did
not change. Variables and shocks are kept, because the shocks are part of
what happened.

One more thing to know before you index into it:

```{code-cell} ipython3
print("out['y'][period] ->", out["y"][qq(2025,1)])
print("as a plain float ->", out["y"][qq(2025,1)].item())
```

A `(1, 1)` **array**, not a number — one row per period, one column per
variant, and variants are tutorial 19. `.item()` when you want a float.

Finally, `return_info=True` returns a second object reporting on the run
itself rather than on the economy:

```{code-cell} ipython3
_, info = m.simulate(db, SPAN, return_info=True)

print("method     :", info["method"])
print("exit status:", info["exit_status"])
print("frames     :", info["frames"])
```

`exit_status` is what to check when a script simulates hundreds of times, and
**one frame** means the simulation ran in a single pass. A `first_order`
simulation always does; other methods cut the span into several frames when a
shock arrives as a surprise, which is tutorial 12.

## ⚠️ Break it

**A mistake that gives you a believable answer.**

Here is the shock again, written the way it has been written all tutorial:

```{code-cell} ipython3
right = Databox.steady(m, SPAN)
right["shk_y"][qq(2025,1)] = 1.0

path = m.simulate(right, SPAN)["y"].get_data()[:, 0]
print("peak:", round(float(path.max()), 4), " path:", np.round(path[:5], 3))
```

Now leave off the date. It is one pair of brackets:

```{code-cell} ipython3
wrong = Databox.steady(m, SPAN)
wrong["shk_y"] = 1.0                  # no [qq(2025,1)]

path = m.simulate(wrong, SPAN)["y"].get_data()[:, 0]
print("peak:", round(float(path.max()), 4), " path:", np.round(path[:5], 3))
```

**A peak of 1.88 instead of 0.91**, because assigning to the name rather than
to a period sets the shock in *every* quarter. You asked for one demand shock
and simulated eight of them.

Look at the path before deciding that is obviously wrong. Output rises, then
levels off a little below 2. There is no error, no empty series, no zeros —
it is an ordinary picture of an economy under sustained demand pressure, and
it is what you would expect a simulation to look like. Nothing about it
invites a second look.

That is the failure worth fearing. A result that is missing gets noticed
immediately; a result that is twice too large and entirely plausible does
not.

A misspelled name goes the same way:

```{code-cell} ipython3
typo = Databox.steady(m, SPAN)
typo["shk_Y"] = Series(start=qq(2025,1), values=[1.0])   # capital Y

print("peak:", round(float(m.simulate(typo, SPAN)["y"].get_data()[:, 0].max()), 4))
print("shk_Y is in the databox:", "shk_Y" in typo.get_names())
```

Zero. The databox accepts `shk_Y` as a new item, nothing in the model reads
it, and the simulation runs as though you had asked for nothing.

Both mistakes have the same shape: assigning **to the name** replaces or
creates the whole series, and a databox will hold anything under any name.

Which points at the habit that prevents both. Assign **into a period**
instead, and the same typo cannot get past you:

```{code-cell} ipython3
try:
    caught = Databox.steady(m, SPAN)
    caught["shk_Y"][qq(2025,1)] = 1.0
except KeyError as error:
    print("KeyError:", error)
```

Indexing into a name that is not there raises immediately, and indexing into
a period cannot fill the other seven by accident.
`db["shk_y"][qq(2025,1)] = 1.0` is a few more characters than
`db["shk_y"] = 1.0`, and it is the form that tells you when you are wrong.

## What you did

```{code-cell} ipython3
# what the model needs before it can start — not one lag of everything
m.get_initials()

# the whole model at rest, initial conditions included
db = Databox.steady(m, SPAN)

# a shock is a value in the databox, dated inside the span
db["shk_y"][qq(2025,1)] = 1.0

# a foreign shock reaches the domestic economy through exports
db["shk_y_w"][qq(2025,1)] = 1.0

out = m.simulate(db, SPAN)
```

## Things to remember

1. **The model from here on is a small open economy** — an exchange rate, a
   foreign block, and three forward-looking variables instead of one.
2. **`get_initials()` names what the model needs**, and it is not one lag of
   every variable. `q` has no lag, so it has no initial condition.
3. **Initial conditions go in the period before the span**, shocks go inside
   it.
4. **`Databox.steady` supplies all of it**, which is why none of this has
   needed a thought, and it runs one quarter longer at each end than the
   span.
5. **Initial conditions are data.** Change one and the whole simulation
   changes, with no shock anywhere.
6. **Assign into a period, not to the name.** `db["shk_y"][qq(2025,1)] = 1.0`
   sets one quarter; `db["shk_y"] = 1.0` sets every quarter, and the result
   looks entirely reasonable.
7. **Indexing a `Series` by period gives a `(1, 1)` array.** Use `.item()`.

## Exercise

Run the same demand shock twice: once from the steady state, and once from an
economy already running hot, with the output gap at `2.0` in 2024Q4.

Before you run it, predict whether the peak output gap in the second case is
larger than `0.9057 + 2.0`, smaller, or exactly that.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**Smaller** — the peak is `2.1736`, not `2.9057`.

The two effects do not arrive together. The shock peaks in 2025Q1, and by
then the initial gap of `2.0` has already had a quarter to decay:

| start | 2024Q4 | 2025Q1 | 2025Q2 | 2025Q3 | peak |
|---|---|---|---|---|---|
| steady state | 0.0 | 0.9057 | 0.5130 | 0.2642 | **0.9057** |
| running hot | 2.0 | 2.1736 | 1.2313 | 0.6340 | **2.1736** |

Run the hot start **without** the shock and 2025Q1 comes out at `1.2679`. Add
the shock's own `0.9057` and you get `2.1736` exactly.

That `1.2679` is worth a second look. It is not `2.0 × a1 = 1.4`, even though
`a1` is the coefficient on `y{-1}` in the demand equation. It is
`2.0 × 0.634`, and `0.634` is the top-left entry of `T` — the *solved*
persistence of output, which folds in everything coming back through the
Phillips curve, the policy rule and the exchange rate. The equation
coefficient and the solution coefficient are different numbers, and a
simulation uses the second.

The deeper point is the word *exactly*. The model is linear, so the decay
from the starting point and the response to the shock are computed
independently and added. A different starting point shifts the path; it never
changes the shape of the response. That is what makes the deviations in
tutorial 8 legitimate, and it is precisely what stops being true in tutorial
13.

</details>

## Next

**Tutorial 12 · Shocks** is about the thing this tutorial treated as obvious:
you put a number in a databox and the model reacts. That is half the story,
because it matters enormously *when the economy found out*.

A rate rise announced a year ago and one that arrives with no warning are
different events, even when the number is the same — and IrisPie keeps two
separate copies of every shock for exactly that reason. Those are the
`ant_shk_*` items that have appeared in every databox since tutorial 3. With
an exchange rate in the model the difference is sharper, because a currency
reacts to news the moment it arrives.
