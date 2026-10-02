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

# 15 · Comparing and decomposing

**40 minutes** · after *14 · Conditioning with plans* · next: *16 · The
Kalman filter*

You have two simulations and they disagree. Somebody is going to ask why.

This tutorial is about answering that: subtracting one run from another, and
then splitting the difference into the pieces that caused it. It is the last
step of a forecast round and the one that gets presented, because a number
nobody can explain is a number nobody will use.

```{code-cell} ipython3
import contextlib
import io

import numpy as np
import irispie as ip
from irispie import Simultaneous, Databox, qq

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

SPAN = qq(2025,1) >> qq(2027,4)
```

## A baseline and a scenario

The baseline is whatever you are comparing against. Here it is the model at
rest, which keeps the arithmetic easy to follow; in real work it would be
last round's forecast.

```{code-cell} ipython3
baseline = m.simulate(Databox.steady(m, SPAN), SPAN)

SHOCKS = {
    "shk_y":   (qq(2025,1), 1.0),     # domestic demand
    "shk_pi":  (qq(2025,2), 0.5),     # cost-push
    "shk_y_w": (qq(2025,1), 0.8),     # demand abroad
}

scenario_input = Databox.steady(m, SPAN)
for name, (period, size) in SHOCKS.items():
    scenario_input[name][period] = size

scenario = m.simulate(scenario_input, SPAN)

print("baseline:", np.round(baseline["y"].get_data()[:5, 0], 4))
print("scenario:", np.round(scenario["y"].get_data()[:5, 0], 4))
```

Three things happened at once, and the output gap peaks at `1.1213`. The
question a forecast round actually asks is how much of that came from where.

## Subtracting one run from another

```{code-cell} ipython3
difference = scenario.copy()
difference.minus_control(m, baseline)

print(np.round(difference["y"].get_data()[:5, 0], 4))
```

`minus_control` takes the model as well as the control databox, and the model
is not decoration. It decides how each series is compared:

```{code-cell} ipython3
LOG_MODEL = """

!transition-variables
    y
    P

!transition-shocks
    shk_y

!log-variables
    P

!parameters
    rho

!transition-equations
    y = rho*y{-1} + shk_y;
    P = P{-1}*exp(y/100);

"""

log_model = Simultaneous.from_string(LOG_MODEL, linear=True, flat=False)
log_model.assign_strict(rho=0.8)
log_model.steady()
log_model.solve_first_order()

log_base = log_model.simulate(Databox.steady(log_model, SPAN), SPAN)

shocked = Databox.steady(log_model, SPAN)
shocked["shk_y"][qq(2025,1)] = 1.0
log_scenario = log_model.simulate(shocked, SPAN)

log_difference = log_scenario.copy()
log_difference.minus_control(log_model, log_base)

print("P, minus_control    :", np.round(log_difference["P"].get_data()[:4, 0], 6))
print("P, plain subtraction:",
      np.round((log_scenario["P"] - log_base["P"]).get_data()[:4, 0], 6))
print("y, minus_control    :", np.round(log_difference["y"].get_data()[:4, 0], 6))
```

For the **log variable** `P` the comparison is a **ratio** — `1.01005`, one
per cent above baseline. Plain subtraction gives `0.01005`, which is a
difference in index points and not what anyone means by "how much higher".
For the ordinary variable `y` it is a difference, as you would expect.

That is tutorial 6's rule again: for a log variable the natural comparison is
a gross ratio. `minus_control` knows which variables are which because you
handed it the model.

## Splitting the difference

Nothing in IrisPie decomposes a simulation for you. You do it by running each
shock on its own and comparing each run against the same baseline:

```{code-cell} ipython3
contributions = {}

for name, (period, size) in SHOCKS.items():
    one_shock = Databox.steady(m, SPAN)
    one_shock[name][period] = size

    run = m.simulate(one_shock, SPAN)
    run.minus_control(m, baseline)
    contributions[name] = run["y"].get_data()[:, 0]

    print(f"{name:9s}", np.round(contributions[name][:5], 4))
```

Three paths with three different shapes. Domestic demand is the biggest on
impact and fades quickly. Foreign demand starts smaller and lasts, because
`y_w` is an autoregressive process with `d1 = 0.8` and keeps feeding exports
for years. The cost-push contribution is **negative** — higher inflation
brings a higher policy rate, and that costs output.

Now the part that makes it a decomposition rather than three separate
stories:

```{code-cell} ipython3
total = sum(contributions.values())
actual = difference["y"].get_data()[:, 0]

print("sum of the three:", np.round(total[:5], 4))
print("actual scenario :", np.round(actual[:5], 4))
print("largest gap     :", f"{np.abs(total - actual).max():.3e}")
```

**They add up exactly** — the residual is floating-point dust. That is not a
property of decompositions in general; it is a property of linear models, and
the next-to-last section shows what happens without it.

## The chart that gets presented

```{code-cell} ipython3
LABELS = {"shk_y": "domestic demand",
          "shk_pi": "cost-push",
          "shk_y_w": "foreign demand"}

figure = ip.make_subplots(
    (1, 1),
    figure_title="Output gap: what each shock contributed",
    figure_height=420,
    show_legend=True,
)

for name, (period, size) in SHOCKS.items():
    one_shock = Databox.steady(m, SPAN)
    one_shock[name][period] = size
    run = m.simulate(one_shock, SPAN)
    run.minus_control(m, baseline)
    run["y"].plot(chart_type="bar", figure=figure, subplot=0,
                  show_figure=False, legend=[LABELS[name]])

difference["y"].plot(figure=figure, subplot=0, show_figure=False,
                     legend=["total"])

figure.update_layout(barmode="relative")
figure
```

`barmode="relative"` is what makes it a decomposition rather than three bars
side by side: positive contributions stack upwards, negative ones hang below
the axis, and the top of the stack meets the total line.

Read it left to right and the story tells itself. The first two quarters
belong to domestic demand. From **2025Q3** the foreign contribution is the
larger one — `0.2806` against `0.2642` — and from there on the recovery is
really a foreign story that happens to have started at home. The cost-push
shock subtracts throughout, which is the policy rate doing its job.

That is a slide. It is also an argument, and the argument is only as good as
the baseline you chose.

## When the parts stop adding up

A decomposition assumes the effects are separable: run A alone, run B alone,
add them, get A-and-B. Tutorial 11's exercise showed why that works here —
the model is linear, so responses add.

Give the model a genuine nonlinearity and solve it properly, and they stop:

```{code-cell} ipython3
NONLINEAR = SOURCE.replace(
    "                + b2*y + b3*q + shk_pi;",
    "                + b2*y + b4*y**2 + b3*q + shk_pi;",
).replace(
    "    a1, a2, a3, a4, b1, b2, b3, c1, c2, c3",
    "    a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3",
)

curved = Simultaneous.from_string(NONLINEAR, linear=False, flat=True)
curved.assign_strict(b4=0.05, **CALIB)

with contextlib.redirect_stdout(io.StringIO()):
    curved.steady()
curved.solve_first_order()


def residual(model, method):
    """How far the parts are from the whole, under one simulation method."""
    with contextlib.redirect_stdout(io.StringIO()):
        control = model.simulate(Databox.steady(model, SPAN), SPAN,
                                 method=method, when_fails="silent")
        everything = Databox.steady(model, SPAN)
        for name, (period, size) in SHOCKS.items():
            everything[name][period] = size
        whole = model.simulate(everything, SPAN, method=method,
                               when_fails="silent")
        whole.minus_control(model, control)

        parts = []
        for name, (period, size) in SHOCKS.items():
            alone = Databox.steady(model, SPAN)
            alone[name][period] = size
            run = model.simulate(alone, SPAN, method=method,
                                 when_fails="silent")
            run.minus_control(model, control)
            parts.append(run["y"].get_data()[:, 0])

    return np.abs(sum(parts) - whole["y"].get_data()[:, 0]).max()


print("linear model,    first_order :", f"{residual(m, 'first_order'):.3e}")
print("curved model,    first_order :", f"{residual(curved, 'first_order'):.3e}")
print("curved model,    stacked_time:", f"{residual(curved, 'stacked_time'):.3e}")
```

The middle line is the one to look at twice. The model is nonlinear, the
decomposition still adds up perfectly, and that is **because `first_order` is
a linear approximation** — not because the model is linear. The tidy answer
comes from the method, not from the economy.

Solve the same model properly with `stacked_time` and the parts miss the
whole by `5.6e-03`. That gap is real: with a convex Phillips curve, two
shocks together do more than the two shocks apart, and no split into
single-shock contributions can show it.

So a decomposition is exact for a linear model, and an approximation
everywhere else — a good one for small shocks, and worth reporting a residual
for when the shocks are large.

## ⚠️ Break it

**Comparing against the wrong baseline.**

The decomposition above compared every single-shock run against the *same*
control. It is easy to compare each one against itself instead:

```{code-cell} ipython3
wrong = {}

for name, (period, size) in SHOCKS.items():
    one_shock = Databox.steady(m, SPAN)
    one_shock[name][period] = size

    run = m.simulate(one_shock, SPAN)
    run.minus_control(m, run)          # against itself, not the baseline
    wrong[name] = run["y"].get_data()[:, 0]

print("contributions:", np.round(sum(wrong.values())[:5], 4))
print("actual        :", np.round(actual[:5], 4))
```

Every contribution is zero, so they sum to zero, and the chart would be an
empty axis under a total line that goes somewhere. That one is loud.

The quiet version uses a baseline that is merely *different*:

```{code-cell} ipython3
other_baseline_input = Databox.steady(m, SPAN)
other_baseline_input["shk_i"][qq(2025,1)] = 0.5      # last round had a policy surprise
other_baseline = m.simulate(other_baseline_input, SPAN)

mixed = scenario.copy()
mixed.minus_control(m, other_baseline)

print("against the right baseline:", np.round(actual[:5], 4))
print("against another baseline  :", np.round(mixed["y"].get_data()[:5, 0], 4))
```

Both are legitimate numbers and they answer different questions. The first is
*what did these three shocks do*; the second is *how does this scenario
compare with a round that also had a policy surprise in it* — and the
difference between them is a shock nobody has mentioned.

Nothing in the output records which baseline was used. Name your baselines
after what they are, keep them beside the scenario they belong to, and state
the comparison in the title of every chart you produce from them.

## What you did

```{code-cell} ipython3
# the difference between two runs, with log variables handled properly
difference = scenario.copy()
difference.minus_control(m, baseline)

# one shock at a time, each against the same control
one_shock = Databox.steady(m, SPAN)
one_shock["shk_y"][qq(2025,1)] = 1.0
part = m.simulate(one_shock, SPAN)
part.minus_control(m, baseline)

# stacked positive and negative contributions against the total
part["y"].plot(chart_type="bar", show_figure=False)
```

## Things to remember

1. **`minus_control` needs the model** because it divides log variables and
   subtracts the rest.
2. **There is no built-in decomposition.** Run each shock alone against the
   same baseline and collect the pieces.
3. **The pieces add up exactly for a linear model** — here to `2e-16`.
4. **A tidy decomposition can be an artefact of `first_order`.** The method
   is linear even when the model is not.
5. **With a real nonlinearity the parts miss the whole**, by `5.6e-03` here.
   Report the residual rather than hiding it.
6. **`barmode="relative"`** is what stacks contributions around zero.
7. **The baseline is half the answer.** Two correct decompositions of the
   same scenario against different baselines say different things.

## Exercise

Using the decomposition above, find the first quarter in which **foreign
demand** contributes more to the output gap than domestic demand does.

Before you run it, predict whether it happens inside the first year.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**2025Q3**, the third quarter of the simulation — comfortably inside the
first year.

| quarter | domestic | foreign | cost-push |
|---|---|---|---|
| 2025Q1 | **+0.9057** | +0.2156 | +0.0000 |
| 2025Q2 | **+0.5130** | +0.2847 | −0.0766 |
| 2025Q3 | +0.2642 | **+0.2806** | −0.0872 |

The crossover has nothing to do with the sizes of the two shocks — the
domestic one is larger, at 1.0 against 0.8. It is about persistence. The
domestic shock hits `y` once and then decays at the model's own rate. The
foreign shock hits `y_w`, which is an autoregressive process with
`d1 = 0.8`, and `y_w` keeps feeding domestic demand through `a4*y_w` for
years afterwards.

So a smaller shock to a persistent process overtakes a larger shock to a
transitory one, and it does so quickly. That is the sort of thing a
decomposition is for: the headline number at the peak was almost entirely
domestic, and the story a year later is almost entirely foreign.

</details>

## Next

That is the end of Level 3. You can build the input for a simulation,
distinguish shocks the economy saw coming from shocks it did not, solve the
equations rather than an approximation to them, condition a run on a path you
have been given, and explain the result.

**Tutorial 16 · The Kalman filter** opens Level 4 and turns the machinery
around. Everything so far has started from shocks and produced data. Filtering
starts from data and produces shocks: given what actually happened to
inflation and the exchange rate, what must the economy have been doing
underneath. It is where measurement equations finally earn their place, and
where a forecast round really begins.
