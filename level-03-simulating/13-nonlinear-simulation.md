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

# 13 · Nonlinear simulation

**40 minutes** · after *12 · Shocks* · next: *14 · Conditioning with plans*

Every simulation so far has used `first_order`, which solves a straight-line
approximation of the model built once around the steady state. It is fast,
it always works, and for a model that is genuinely linear it is exact.

This tutorial is about solving the equations themselves. When that matters,
what it costs, and how to tell whether the answer you got is the one you
wanted.

```{code-cell} ipython3
import contextlib
import io

import numpy as np
import plotly.graph_objects as go
import irispie as ip
from irispie import Simultaneous, Databox, qq


def run(model, *args, **kwargs):
    """Simulate without the solver's iteration log, which runs to hundreds of lines."""
    with contextlib.redirect_stdout(io.StringIO()):
        return model.simulate(*args, **kwargs)


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

    a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3
    pi_tar, r_ss, beta, psi
    d1, d2, d3, pi_w_ss, r_w_ss

!transition-equations

    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + a3*q + a4*y_w + shk_y;

    "Phillips curve, convex in the output gap"
    pi - pi_tar = b1*(pi{-1} - pi_tar) + beta*(1-b1)*(pi{+1} - pi_tar)
                + b2*y + b4*y**2 + b3*q + shk_pi;

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

CALIB = dict(a1=0.7, a2=0.2, a3=0.1, a4=0.3, b1=0.1, b2=0.3, b3=0.2, b4=0.05,
             c1=0.5, c2=1.5, c3=0.5, pi_tar=2, r_ss=1, beta=0.99, psi=0.1,
             d1=0.8, d2=0.7, d3=0.8, pi_w_ss=2, r_w_ss=1)

m = Simultaneous.from_string(SOURCE, linear=False, flat=True)
m.assign_strict(**CALIB)

with contextlib.redirect_stdout(io.StringIO()):     # the steady solver logs too
    m.steady()

m.solve_first_order()

SPAN = qq(2025,1) >> qq(2030,4)
```

## One term makes the model nonlinear

The Phillips curve has gained `b4*y**2`. Prices now rise faster when the
economy is already running hot than when it is at rest — a capacity
constraint, in the crudest possible form.

Two things to notice about how it was added.

The model is declared `linear=False`. That is the honest flag now, and
tutorial 5 explained what it costs: the steady state is found by iteration
rather than algebra.

And the steady state itself has not moved:

```{code-cell} ipython3
print(dict((k, round(float(v), 6)) for k, v in m.get_steady_levels().items()))
print("check_steady:", m.check_steady(when_fails="silent"))
```

`y` rests at zero, so `y**2` rests at zero too, and the new term contributes
nothing in the long run. It only does anything while the economy is away from
rest — which is exactly what you want from a term like this, and worth
arranging deliberately, because a nonlinearity that shifts the steady state
changes every number in every other tutorial.

A note on what is not here. The natural nonlinearity for a policy model is a
zero lower bound on the interest rate, written `max(0, ...)`. IrisPie's model
language has `exp`, `log` and `**` but no `max` or `abs`, so a hard floor
cannot be written at all. The convex Phillips curve is the second choice.

## Three ways to simulate

```{code-cell} ipython3
db = Databox.steady(m, SPAN)
db["shk_y"][qq(2026,1)] = 1.0

for method in ("first_order", "stacked_time", "period_by_period"):
    out = run(m, db, SPAN, method=method, when_fails="silent")
    peak = (out["pi"].get_data()[:, 0] - 2).max()
    print(f"{method:17s} inflation peaks at {peak:.5f}")
```

Three answers to the same question.

**`first_order`** uses the solution matrices from tutorial 8. Those were
built by differentiating the model once, at the steady state, so they
describe a straight line through that point. Fast, and always available.

**`stacked_time`** stacks every period of the simulation into one large
system of equations and solves the whole thing at once with Newton's method.
It uses the equations you wrote, not an approximation of them.

**`period_by_period`** solves one period at a time, moving forward. Cheaper
than stacking, and the answer above is wrong — the next section explains why.

## How wrong is the straight line

The gap between the approximation and the real answer depends entirely on how
far from the steady state you go:

```{code-cell} ipython3
print(f"{'shock':>6s} {'first_order':>12s} {'stacked_time':>13s} {'error':>8s}")

for size in (0.5, 1.0, 2.0, 4.0):
    d = Databox.steady(m, SPAN)
    d["shk_y"][qq(2026,1)] = size
    linear = (run(m, d, SPAN)["pi"].get_data()[:, 0] - 2).max()
    exact = (run(m, d, SPAN, method="stacked_time",
                 when_fails="silent")["pi"].get_data()[:, 0] - 2).max()
    print(f"{size:6.1f} {linear:12.5f} {exact:13.5f} {100*(exact-linear)/exact:7.2f}%")
```

**5.6% at half a point, 30.8% at four points.** The error is not a fixed
overhead you can allow for; it grows with the size of the disturbance,
because that is what a convex term does.

Run that sweep finely enough and you can draw the thing the method is named
after. This is not a time series, so it is a plain Plotly chart rather than
`Series.plot` — irispie's plotting draws variables against time, and here the
horizontal axis is the size of the shock:

```{code-cell} ipython3
sizes = np.arange(-4.0, 4.01, 0.5)
straight, actual = [], []

for size in sizes:
    box = Databox.steady(m, SPAN)
    box["shk_y"][qq(2026,1)] = float(size)
    straight.append(float(run(m, box, SPAN)["pi"][qq(2026,1)].item()) - 2)
    actual.append(float(run(m, box, SPAN, method="stacked_time",
                            when_fails="silent")["pi"][qq(2026,1)].item()) - 2)

figure = go.Figure()
figure.add_trace(go.Scatter(x=sizes, y=actual, mode="lines+markers",
                            name="stacked_time", line=dict(width=3)))
figure.add_trace(go.Scatter(x=sizes, y=straight, mode="lines",
                            name="first_order", line=dict(width=3, dash="dot")))
figure.update_layout(
    title="Inflation on impact against the size of the demand shock",
    xaxis_title="shk_y in 2026Q1",
    yaxis_title="inflation, deviation from target",
    height=520, template="plotly_white",
)
figure.add_hline(y=0, line_width=1, line_color="grey")
figure.add_vline(x=0, line_width=1, line_color="grey")

figure
```

The dotted line is straight. Not approximately straight — exactly straight,
because that is what solving a first-order approximation means: double the
shock and you double every number in the answer. `0.1993` at half a point,
`1.5944` at four, and the ratio is the same to the last digit.

The solid line is the model. The two **touch at the origin and separate
either side of it**, which is the clearest statement of what `first_order`
does: it is the tangent to this curve at the steady state. Near the point
of contact the fit is excellent and the speed is free. The further you go,
the less it is worth.

Notice the curve sits **above** the line on both sides. A convex term is
blind to the sign of `y`, so it adds inflation pressure in a slump exactly
as it does in a boom. The approximation is too low in a boom and too high in
a slump — in both directions it misses the same way.

This is the whole case for the slower methods. For routine work around the
steady state the straight line is fine and the matrices make it instant. For
a crisis scenario it is not fine, and the number it gives you is too small in
a way that flatters the forecast.

## Check the term where you did not put it

The left side of that chart is flattening, and it keeps going. The output
terms in the Phillips curve are `b2*y + b4*y**2`, which is `y*(b2 + b4*y)` —
zero when `y = -b2/b4`, and positive below it. With `b2 = 0.3` and
`b4 = 0.05` that threshold is an output gap of `-6`, past which the convex
term outweighs the linear one and a deeper slump produces *more* inflation.

The term was added to make inflation rise faster in a boom. What it does in
a slump was never considered, and the model does not consider it either.

That is a long way from anywhere this model will be simulated, which is why
it is a footnote rather than a section. The habit it argues for is cheap:
when you choose a functional form, work out by hand where it changes sign,
turns around or divides by zero, and check that against the range your
shocks actually reach.

## Why period_by_period is wrong here

`period_by_period` solved each quarter alone and came out at `0.29083` —
further from the truth than the linear approximation it was supposed to
improve on.

The reason is in the equations. Three of them look forward: `pi{+1}` in the
Phillips curve and in demand, `q{+1}` in interest parity, `pi_w{+1}` as well.
Solving one quarter at a time means guessing those and never going back to
check the guess was right.

On a model with no leads there is nothing to guess, and the method is exact:

```{code-cell} ipython3
BACKWARD = """

!transition-variables
    y, pi

!transition-shocks
    shk_y

!parameters
    a1, b1, b2

!transition-equations
    y = a1*y{-1} + shk_y;
    pi = b1*pi{-1} + b2*y + 0.05*y**2;

"""

backward = Simultaneous.from_string(BACKWARD, linear=False, flat=True)
backward.assign_strict(a1=0.7, b1=0.6, b2=0.3)

with contextlib.redirect_stdout(io.StringIO()):
    backward.steady()

backward.solve_first_order()

SHORT = qq(2025,1) >> qq(2028,4)
bd = Databox.steady(backward, SHORT)
bd["shk_y"][qq(2026,1)] = 1.0

for method in ("first_order", "stacked_time", "period_by_period"):
    peak = run(backward, bd, SHORT, method=method,
               when_fails="silent")["pi"].get_data()[:, 0].max()
    print(f"{method:17s} {peak:.6f}")
```

`stacked_time` and `period_by_period` agree to fifteen decimal places, and
`first_order` is the odd one out because it is still a straight line.

So the rule is about the model, not the method: **`period_by_period` is for
models with no forward-looking terms.** Nothing in IrisPie will stop you
using it on a model with leads, and the answer will be quietly wrong.

## Solver settings

`stacked_time` and `period_by_period` both run Newton's method, and
`solver_settings` passes options to it:

```{code-cell} ipython3
d = Databox.steady(m, SPAN)
d["shk_y"][qq(2026,1)] = 1.0

careful = run(
    m, d, SPAN,
    method="stacked_time",
    when_fails="silent",
    solver_settings={"func_tolerance": 1e-10, "max_iterations": 10_000},
)

print("peak:", round(float((careful["pi"].get_data()[:, 0] - 2).max()), 5))
```

The ones worth knowing, with their defaults:

| setting | default | what it does |
|---|---|---|
| `func_tolerance` | `1e-12` | how small the residual has to get |
| `step_tolerance` | `1e-12` | how small a step counts as no progress |
| `max_iterations` | `5000` | when to give up |
| `norm_order` | `inf` | how the residual is measured; `inf` is the largest single one |
| `print_every` | `5` | how often the iteration log prints a line |

Loosening the tolerances is the first thing to reach for on a model that will
not converge, and the last thing you should trust — a looser tolerance does
not make a bad answer good, it makes a bad answer acceptable to the solver.

## Reading the exit status

`stacked_time` reports on itself, and the report deserves a closer look than
it usually gets:

```{code-cell} ipython3
for size in (1.0, 2.0, 5.0, 10.0):
    d = Databox.steady(m, SPAN)
    d["shk_y"][qq(2026,1)] = size
    out, info = run(m, d, SPAN, method="stacked_time",
                    when_fails="silent", return_info=True)
    status = [str(s).split(".")[-1].split(":")[0] for s in info["exit_status"]]
    peak = (out["pi"].get_data()[:, 0] - 2).max()
    print(f"shock {size:5.1f}  peak {peak:8.5f}   {status}")
```

Shocks of 2 and 10 report **Cannot make further progress**, and 1 and 5 do
not. With the default `when_fails="critical"` the first two would have
raised and stopped your script.

Now look at the peaks: `0.44556`, `0.98125`, `3.07836`, `7.96324`. They rise
smoothly with the shock. **All four answers are right**, including the two
that reported failure.

What happens is visible if you let the solver print its iterations: the
residual reaches exactly `0.00000e+00`, and a solver that has landed precisely
on the answer cannot take a smaller step than the one it just took. That is
reported as a failure to progress rather than as success.

So when a stacked-time simulation reports this, check the answer before
believing the status. Compare it against `first_order`, or against the same
simulation with a slightly different shock, and see whether it sits where it
should.

## ⚠️ Break it

**Silencing the complaint.**

Once you have met a few spurious failures, `when_fails="silent"` becomes the
obvious thing to type. It is in every cell in this tutorial, for exactly that
reason.

Here is what it also does:

```{code-cell} ipython3
d = Databox.steady(m, SPAN)
d["shk_y"][qq(2026,1)] = 1.0

truth = (run(m, d, SPAN, method="stacked_time",
             when_fails="silent")["pi"].get_data()[:, 0] - 2).max()

quiet = (run(m, d, SPAN, method="period_by_period",
             when_fails="silent")["pi"].get_data()[:, 0] - 2).max()

print("stacked_time     :", round(float(truth), 5))
print("period_by_period :", round(float(quiet), 5))
print("difference       :", round(float(abs(truth - quiet)), 5))
```

The second number is wrong by `0.155`, a third of the answer, and nothing in
the output says so. `period_by_period` on a forward-looking model does not
fail in a way the solver can detect — it converges perfectly well to the
solution of the wrong problem.

So `when_fails="silent"` hides two different things at once: the spurious
complaints, and the real ones. A simulation that quietly produced nonsense
looks exactly like one that quietly worked.

The habit worth forming is to keep `return_info=True` alongside the silence.
It costs one variable and leaves the status visible:

```{code-cell} ipython3
out, info = run(m, d, SPAN, method="stacked_time",
                when_fails="silent", return_info=True)

print("status:", [str(s).split(".")[-1].split(":")[0] for s in info["exit_status"]])
```

Silence the exception, not the information.

## What you did

```{code-cell} ipython3
# the straight line through the steady state
run(m, db, SPAN, method="first_order")

# every period stacked into one system and solved together
run(m, db, SPAN, method="stacked_time", when_fails="silent")

# one period at a time, for models with no leads
run(backward, bd, SHORT, method="period_by_period", when_fails="silent")

# the Newton settings underneath
out = run(m, db, SPAN, method="stacked_time", when_fails="silent",
          solver_settings={"func_tolerance": 1e-10, "max_iterations": 10_000})
```

## Things to remember

1. **`first_order` is a straight line through the steady state.** Exact for a
   linear model, approximate for everything else.
2. **The approximation error grows with the shock** — here 5.6% at half a
   point and 30.8% at four — and it distorts the shape of the response, not
   just its size.
3. **`stacked_time` solves the equations you wrote**, every period at once.
   It is the method to trust when the answer matters.
4. **`period_by_period` is only for models with no leads.** On a
   forward-looking model it is wrong, and nothing warns you.
5. **Arrange a nonlinearity to vanish at the steady state** if you can, so it
   changes the dynamics without moving the long run — then check what it does
   far away from it. The convex term here reverses the sign of the Phillips
   curve once the output gap passes `-6`.
6. **`solver_settings` holds the Newton options** — `func_tolerance`,
   `step_tolerance`, `max_iterations`, `norm_order`.
7. **A reported failure is not always a failure.** Check the numbers before
   you believe the status, and keep `return_info=True` when you silence the
   exception.

## Exercise

Find roughly where the `first_order` error passes **10%** of the correct
answer, by trying shock sizes between `0.5` and `1.0`.

Before you run it, predict whether the error grows in proportion to the shock
or faster than it.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

The error crosses 10% between a shock of **0.90** and **0.95**:

| shock | first_order | stacked_time | error |
|---|---|---|---|
| 0.60 | 0.23916 | 0.25621 | 6.65% |
| 0.70 | 0.27901 | 0.30217 | 7.66% |
| 0.80 | 0.31887 | 0.34906 | 8.65% |
| 0.90 | — | — | 9.61% |
| 0.95 | — | — | **10.08%** |
| 1.00 | — | — | 10.54% |

It grows **slower** than in proportion, which is the surprising half. The
missing term is quadratic, so you might expect the error to double when the
shock doubles. Between `0.5` and `1.0` the error goes from 5.62% to 10.54%,
which is close to proportional; between `1.0` and `2.0` it goes from 10.54%
to 18.76%, less than double; and from `2.0` to `4.0`, 18.76% to 30.75%.

The reason is the policy rule. A bigger output gap means more inflation,
which means a sharper rate rise, which pulls the gap back down — so the
economy spends less time in the region where `y**2` matters than a bare
quadratic would suggest. The feedback that stabilises the model also limits
how wrong the linear approximation can get.

</details>

## Next

**Tutorial 14 · Conditioning with plans** is about telling a simulation what
the answer has to be. A `SimulationPlan` lets you fix the path of a variable
and let a shock move instead — the exchange rate held at a given level, the
policy rate following a published path, inflation hitting a number you have
been told to assume. It is how a forecast round actually gets made, and it is
the reason the model has four shocks you have barely used.
