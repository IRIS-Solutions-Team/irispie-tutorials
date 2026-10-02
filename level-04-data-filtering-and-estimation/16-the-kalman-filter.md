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

# 16 · The Kalman filter

**50 minutes** · after *15 · Comparing and decomposing* · next: *17 ·
Estimation*

Level 3 ran the model forwards: you supplied shocks and read off the paths.
This level runs it backwards. You have published data for some of the
model's variables, nothing at all for others, and you want the model to tell
you what the others were doing.

The output gap is the standard case. Nobody publishes it, every policy
discussion depends on it, and a model that ties it to inflation and interest
rates is the thing that can estimate it.

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

    shk_y, shk_pi, shk_i, shk_q
    shk_y_w, shk_pi_w, shk_i_w

!measurement-variables

    "Published inflation, % per year"         obs_pi
    "Published policy rate, % per year"       obs_i
    "Real exchange rate index, % gap"         obs_q

!measurement-shocks

    shk_obs_pi, shk_obs_i, shk_obs_q

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

!measurement-equations

    "Inflation as published"
    obs_pi = pi + shk_obs_pi;

    "The policy rate as published"
    obs_i = i + shk_obs_i;

    "The real exchange rate index"
    obs_q = q + shk_obs_q;

"""

CALIB = dict(a1=0.7, a2=0.2, a3=0.1, a4=0.3, b1=0.1, b2=0.3, b3=0.2,
             c1=0.5, c2=1.5, c3=0.5, pi_tar=2, r_ss=1, beta=0.99, psi=0.1,
             d1=0.8, d2=0.7, d3=0.8, pi_w_ss=2, r_w_ss=1)

STDS = dict(std_shk_y=0.4, std_shk_pi=0.3, std_shk_i=0.2, std_shk_q=0.5,
            std_shk_y_w=0.3, std_shk_pi_w=0.2, std_shk_i_w=0.2,
            std_shk_obs_pi=0.1, std_shk_obs_i=0.05, std_shk_obs_q=0.2)

m = Simultaneous.from_string(SOURCE, linear=True, flat=True)
m.assign_strict(**CALIB)
m.assign(**STDS)
m.steady()
m.solve_first_order()

HIST = qq(2015,1) >> qq(2024,4)
```

## The measurement block

Three new sections. `!measurement-variables` are the series you have data
for, `!measurement-equations` say how each one relates to the model, and
`!measurement-shocks` are the errors in the measuring.

```{code-cell} ipython3
print("measurement equations:", m.num_measurement_equations)
print("Z (observables x states):", np.asarray(m.get_solution().Z).shape)
print("H (observables x measurement shocks):", np.asarray(m.get_solution().H).shape)
print()
print("steady:", dict((k, round(float(v), 4)) for k, v in m.get_steady_levels().items()))
```

`Z` and `H` are the second half of the state space from tutorial 8. The
transition equations gave `xi(t) = T @ xi(t-1) + K + P @ u(t)`, and the
measurement block adds `y(t) = Z @ xi(t) + D + H @ w(t)`.

Three observables, seven states. Each measurement equation here is a model
variable plus an error, and the errors are small — `0.1` on inflation,
`0.05` on the policy rate, `0.2` on the exchange rate index. These are
published series, measured about as well as published series are.

**The output gap has no measurement equation.** Nothing observes it, now or
ever, and that is the problem the rest of this tutorial is about. Whatever
the filter ends up saying about `y` it will have worked out from the three
series above and the equations connecting them.

## Where the data comes from

This tutorial needs forty quarters of observations and has no suitable real
dataset to hand, so the data is made: run the model with shocks drawn from
the standard deviations assigned above, then keep only the three published
series.

```{code-cell} ipython3
rng = np.random.default_rng(0)

shocks_in = Databox.steady(m, HIST)
for name in ("shk_y", "shk_pi", "shk_i", "shk_q",
             "shk_y_w", "shk_pi_w", "shk_i_w"):
    shocks_in[name] = Series(start=HIST[0],
                             values=rng.normal(0, STDS["std_" + name], len(HIST)))

economy = m.simulate(shocks_in, HIST)

observed = Databox()
for state, obs_name in (("pi", "obs_pi"), ("i", "obs_i"), ("q", "obs_q")):
    observed[obs_name] = economy[state] + Series(
        start=HIST[0],
        values=rng.normal(0, STDS["std_shk_" + obs_name], len(HIST)))

print("the filter will see :", sorted(observed.get_names()))
print("it will not see     :", ["y", "y_w", "pi_w", "i_w", "and every shock"])
```

Because the data was made, there is a series sitting in `economy["y"]` that
the filter never sees. Nothing in this tutorial scores the filter against
it, and it is worth saying why.

**There is no true output gap.** Potential output is not a quantity anyone
observes, publishes or ever will; it is a construct, and the gap is whatever
the model you chose says it is. Run two reasonable models on the same data
and you get two different gaps, neither of them wrong. The hidden series in
`economy["y"]` exists because we invented the economy that produced it,
which is a fact about this simulation and not about economics.

So the filter is judged here the way it has to be judged at work: on what it
says about its own uncertainty, on how much its answer moves when more data
arrives, and on whether its one-step forecast errors look the way the model
says they should. All three are computable on real data, and none of them
needs an answer key.

## Running it

```{code-cell} ipython3
out, info = m.kalman_filter(observed, HIST, return_info=True)

print("output :", sorted(out.get_names()))
print("info   :", sorted(info.keys()))
```

Eight items come back, in three families and a residual.

`predict_med` and `predict_std` are the estimate of each state at period `t`
using data up to `t-1`. `update_med` and `update_std` add the observations
at `t` itself. `smooth_med` and `smooth_std` use the whole sample, including
everything that happened afterwards. `predict_err` is the difference between
what was observed and what had been predicted, and `predict_mse_obs` carries
the matrices behind it.

The `_med` series are the estimates and the `_std` series are their standard
errors. Both are ordinary `Databox` objects full of ordinary `Series`, so
everything from Level 1 works on them.

## What the filter is doing

Each quarter the filter takes two steps.

**Predict.** Move the state forward with the model and nothing else, which
gives `xi` one step ahead, and form what that implies for the observables
through `Z`.

**Update.** The observations arrive. The difference between them and what
was predicted is the *prediction error*, written `v(t)` below, and the state
is corrected in proportion to it: `xi(t|t) = xi(t|t-1) + G @ v(t)`, where
`xi(t|t-1)` is the prediction and `xi(t|t)` the estimate after the data.

`G` is the Kalman gain, and the whole method is in how it is built:
`G = Q @ Z' @ inv(F)`, where `Q` is the filter's uncertainty about the state
and `F = Z @ Q @ Z' + H @ cov_w @ H'` is its uncertainty about the
observables. Loosely: the more confident the model and the noisier the data,
the less a surprise moves the answer.

This is why a variable nobody observes still gets an estimate. `Z` has no
row for the output gap, but `G` has one, because the gap is correlated with
inflation and the policy rate through the model. **A surprise in a series
you do observe moves your estimate of one you do not:**

```{code-cell} ipython3
surprise = np.asarray(out["predict_err"]["obs_pi"][HIST]).ravel()
before = np.asarray(out["predict_med"]["y"][HIST]).ravel()
after = np.asarray(out["update_med"]["y"][HIST]).ravel()
usable = ~np.isnan(surprise)

print("correlation between an inflation surprise and the revision to the gap:",
      round(float(np.corrcoef(surprise[usable], (after - before)[usable])[0, 1]), 3))

worst = int(np.nanargmax(np.abs(surprise)))
print()
print(f"largest surprise, {str(HIST[worst])}:")
print(f"  inflation came in   {surprise[worst]:+.4f} against the prediction")
print(f"  gap estimate moved  {before[worst]:+.4f}  ->  {after[worst]:+.4f}")
```

That is the Phillips curve read backwards. Inflation came in weaker than the
model expected, and the explanation the model offers is that there was more
slack than had been assumed, so the estimate of the gap falls.

## Watching the two steps

```{code-cell} ipython3
ZOOM = qq(2018,1) >> qq(2021,4)

steps = ip.make_subplots(
    (2, 1),
    figure_title="A surprise in what you observe moves what you do not",
    subplot_titles=["Inflation: published against predicted",
                    "Output gap: the estimate before and after"],
    figure_height=640,
    show_legend=True,
)

out["predict_med"]["pi"].plot(
    span=ZOOM, figure=steps, subplot=0, show_figure=False,
    legend=["predicted by the model"],
    update_traces={"mode": "lines",
                   "line": {"width": 2, "dash": "dash", "color": "rgb(230,120,40)"}})
observed["obs_pi"].plot(
    span=ZOOM, figure=steps, subplot=0, show_figure=False,
    legend=["published"],
    update_traces={"mode": "markers",
                   "marker": {"size": 9, "color": "rgb(40,160,90)"}})

out["predict_med"]["y"].plot(
    span=ZOOM, figure=steps, subplot=1, show_figure=False,
    legend=["before the data"],
    update_traces={"mode": "lines",
                   "line": {"width": 2, "dash": "dash", "color": "rgb(230,120,40)"}})
out["update_med"]["y"].plot(
    span=ZOOM, figure=steps, subplot=1, show_figure=False,
    legend=["after the data"],
    update_traces={"mode": "lines",
                   "line": {"width": 3, "color": "rgb(99,110,250)"}})

steps
```

Read the panels together. Wherever a green dot sits above the dashed line in
the top panel, inflation beat the model's expectation, and the bottom panel
shows the gap estimate revised up in the same quarter. The filter has no
other way to learn about the gap, and this is the whole of it.

What is exactly true at every period is the ordering of the uncertainty:

```{code-cell} ipython3
ladder = [(step, np.asarray(out[step]["y"][HIST]).ravel())
          for step in ("predict_std", "update_std", "smooth_std")]

for step, values in ladder:
    print(f"  {step:12} mean {np.nanmean(values):.4f}   first period {values[0]:.4f}")

print()
print("  predict >= update at every period:",
      bool(np.all(ladder[0][1] >= ladder[1][1] - 1e-12)))
print("  update  >= smooth  at every period:",
      bool(np.all(ladder[1][1] >= ladder[2][1] - 1e-12)))
```

Knowing the model and the past puts the standard error on the gap at
`0.4462`; adding the quarter's three observations takes it to `0.3387`;
adding everything that happened afterwards takes it to `0.3160`.

The first period is worth a second look. Before any data at all the
prediction's standard error is `0.5050`, and that is not a coincidence — it
is the model's unconditional standard deviation for the output gap, the
number tutorial 18 recovers from `get_acov`. With nothing observed, the best
the filter can say is what the model says in general.

Use `smooth` for history, which is almost always what you want. Use `update`
when you need the estimate somebody would have had at the time: judging a
past forecast, or testing a policy rule that could only react to what was
known then.

## The whole sample at once

```{code-cell} ipython3
lower = out["smooth_med"]["y"] - out["smooth_std"]["y"]
upper = out["smooth_med"]["y"] + out["smooth_std"]["y"]

figure = ip.make_subplots(
    (1, 1),
    figure_title="The output gap, estimated in real time and finally",
    figure_height=520,
    show_legend=True,
)

band = {"mode": "lines", "line": {"width": 0}}
lower.plot(span=HIST, figure=figure, subplot=0, show_figure=False,
           legend=["_lower"],
           update_traces=dict(band, showlegend=False, hoverinfo="skip"))
upper.plot(span=HIST, figure=figure, subplot=0, show_figure=False,
           legend=["± 1 standard error"],
           update_traces=dict(band, fill="tonexty",
                              fillcolor="rgba(99,110,250,0.20)"))

out["update_med"]["y"].plot(span=HIST, figure=figure, subplot=0, show_figure=False,
                            legend=["real time"],
                            update_traces={"mode": "lines",
                                           "line": {"width": 2, "dash": "dash",
                                                    "color": "rgb(230,120,40)"}})
out["smooth_med"]["y"].plot(span=HIST, figure=figure, subplot=0, show_figure=False,
                            legend=["final"],
                            update_traces={"mode": "lines", "line": {"width": 3}})

figure
```

This is the chart a filtering exercise produces. Nothing in it was observed.

## How much the data told you

A standard error of `0.3160` on the output gap is hard to read on its own.
It becomes readable against what you knew before the data arrived, which is
the model's unconditional standard deviation from `get_acov`:

```{code-cell} ipython3
acov = np.asarray(m.get_acov())[0]
rows = [str(name) for name in m.get_acov_dimension_names().rows]
prior = np.sqrt(np.diag(acov))

print(f"{'variable':9} {'model alone':>12} {'after filtering':>16} {'narrowed by':>12}")
for name in ("y", "pi", "i", "q", "y_w", "pi_w", "i_w"):
    before_data = prior[rows.index(name)]
    after_data = float(np.nanmean(np.asarray(out["smooth_std"][name][HIST]).ravel()))
    print(f"{name:9} {before_data:12.4f} {after_data:16.4f} "
          f"{100*(1-after_data/before_data):11.0f}%")
```

That right-hand column is the honest summary of what forty quarters of three
series bought you, and it needs nothing you do not have at work.

The policy rate narrows by 92 per cent and inflation by 78, which they
should: both are published with small measurement errors. The output gap
narrows by **37 per cent** — nothing observes it, so every bit of that comes
from the model's equations tying it to things that are observed. The data
removes about a third of your uncertainty about the gap and leaves the rest.

Then look at `pi_w`. **Four per cent.** Nothing in the three observed series
speaks to foreign inflation: it is absent from the Phillips curve and
reaches the exchange rate only as `pi_w{+1}` inside a term divided by four.
The filter returns an estimate anyway, because it returns an estimate for
everything.

That is the warning to carry out of this tutorial. **A filtered series is
not evidence that your data identified it.** Every state comes back with a
number attached, including the ones nothing constrains, and this table is
how you tell which is which.

## Real time against final

The two lines on the chart are the same quantity computed twice: what you
would have said at the time, and what you say now. The distance between them
is the revision, and for an output gap it causes more trouble than anything
else about it:

```{code-cell} ipython3
real_time = np.asarray(out["update_med"]["y"][HIST]).ravel()
final = np.asarray(out["smooth_med"]["y"][HIST]).ravel()
revision = final - real_time

print(f"  standard deviation of the revision : {revision.std():.4f}")
print(f"  standard deviation of the estimate : {final.std():.4f}")
print(f"  revision as a share of the signal  : {revision.std()/final.std():.2f}")
print(f"  largest single revision            : {np.abs(revision).max():.4f}")
print(f"  quarters where the sign changed    : "
      f"{100*np.mean(np.sign(real_time) != np.sign(final)):.0f}%")
```

**Revisions are about a third as large as the thing being estimated**, and
in eight per cent of quarters the revision changes the sign: at the time you
would have said the economy was above potential, and now you would say it
was below.

No new data about those quarters ever arrived. The inflation and interest
rate figures are the same ones. What changed is that the filter has since
seen what came next, and persistence means later quarters carry information
about earlier ones.

This is why a policy rule written on a filtered gap is harder to run than it
looks, and it is measurable without knowing any true gap, because both
numbers come out of your own filter.

## Checking the filter without an answer key

Two more things are worth looking at before you believe a filtered history,
and both come out of the filter itself.

The first is the one-step prediction errors. If the model is an adequate
description of the data, what the filter fails to predict should be
unpredictable — in particular it should not be correlated with its own past,
because any pattern there is something the model could have used and did
not:

```{code-cell} ipython3
print(f"{'series':10} {'mean':>8} {'std':>8} {'lag-1 autocorr':>16}")
for name in ("obs_pi", "obs_i", "obs_q"):
    errors = np.asarray(out["predict_err"][name][HIST]).ravel()
    errors = errors[~np.isnan(errors)]
    autocorr = np.corrcoef(errors[:-1], errors[1:])[0, 1]
    print(f"{name:10} {errors.mean():+8.4f} {errors.std():8.4f} {autocorr:+16.3f}")
```

All three autocorrelations are small, which is what a correctly specified
filter looks like. A value of `0.5` in that last column would say the model
is leaving a usable pattern on the table, and the fix would be in the
equations rather than in the filter.

The second is the shocks the filter had to invent to fit the data:

```{code-cell} ipython3
print(f"{'shock':10} {'you assumed':>12} {'filter needed':>14}")
for name in ("shk_y", "shk_pi", "shk_i", "shk_q"):
    needed = float(np.nanstd(np.asarray(out["smooth_med"][name][HIST]).ravel()))
    print(f"{name:10} {STDS['std_' + name]:12.2f} {needed:14.4f}")
```

Every one is smaller than assumed, and that is the expected direction: a
smoothed shock is a conditional mean rather than a draw, so it is pulled
toward zero. The check is one-sided. **A smoothed shock series larger than
the standard deviation you assigned means the model is being forced**, and
the equation it belongs to is the one to look at first.

## The likelihood

The filter computes, as a by-product, how surprising the data were:

```{code-cell} ipython3
print("neg log likelihood :", round(float(info["neg_log_likelihood"]), 4))

contributions = info["neg_log_likelihood_contributions"]
values = np.asarray(contributions.get_data()).ravel()

print("contributions      :", len(values), "periods, summing to",
      round(float(np.nansum(values)), 4))

worst_periods = np.argsort(-np.nan_to_num(values))[:3]
print("hardest to explain :", [str(HIST[int(k)]) for k in sorted(worst_periods)])
```

One number for the whole sample, and `likelihood_contributions` splits it
across periods, summing back to the total. A large contribution is a quarter
the model did not see coming.

Treat it as a diagnostic now and as an objective later. Tutorial 17 hands
this number to an optimiser and calls the result estimation.

## A ragged edge

Published data does not arrive in a rectangle. The current quarter is not
out yet, or one series is late. The filter takes this as it comes:

```{code-cell} ipython3
ragged = observed.copy()
for period in (qq(2024,3), qq(2024,4)):
    for name in ("obs_pi", "obs_i", "obs_q"):
        ragged[name][period] = np.nan

late = m.kalman_filter(ragged, HIST)

print("last four quarters, standard error on the gap")
print("  all data   :", np.round(np.asarray(out["smooth_std"]["y"][HIST]).ravel()[-4:], 4))
print("  nothing out:", np.round(np.asarray(late["smooth_std"]["y"][HIST]).ravel()[-4:], 4))
```

No argument, no flag, no reshaping. Write `nan` where there is no
observation and the filter skips that measurement equation for that period.

The standard errors do the right thing. For the two unpublished quarters
they rise from `0.3170` and `0.3382` to `0.4445` and `0.4795` — the estimate
still exists, carried forward by the model, but on much thinner evidence.
The two quarters *before* the gap also widen slightly, which is the
smoothing step losing the later observations it would otherwise have
borrowed.

## How the filter starts

The recursion needs a state and an uncertainty to begin from. For a
stationary model there is an obvious answer — the model's own unconditional
distribution — and IrisPie uses it without being asked.

A model with a unit root has no unconditional distribution to start from,
and there are three ways out. `diffuse_method` chooses between them:

```{code-cell} ipython3
RANDOM_WALK = """
!transition-variables
    trend, gap
!transition-shocks
    shk_trend, shk_gap
!measurement-variables
    obs
!measurement-shocks
    shk_obs
!parameters
    rho
!transition-equations
    trend = trend{-1} + shk_trend;
    gap = rho*gap{-1} + shk_gap;
!measurement-equations
    obs = trend + gap + shk_obs;
"""

rw = Simultaneous.from_string(RANDOM_WALK, linear=True, flat=True)
rw.assign_strict(rho=0.7)
rw.assign(std_shk_trend=0.3, std_shk_gap=0.5, std_shk_obs=0.1)
rw.steady()
rw.solve_first_order()

print("unit roots in the solution:", rw.get_solution().num_unit_roots)

noise = np.random.default_rng(1)
walk = Databox()
walk["obs"] = Series(start=HIST[0],
                     values=np.cumsum(noise.normal(0, 0.3, len(HIST)))
                            + noise.normal(0, 0.5, len(HIST)))

print()
print(f"{'diffuse_method':16} {'scale':>8} {'neg log lik':>12} {'std, trend 2015Q1':>20}")
for method in ("fixed_unknown", "approx_diffuse", "fixed_zero"):
    for scale in (None, 1e2):
        result, rw_info = rw.kalman_filter(walk, HIST, diffuse_method=method,
                                           diffuse_scale=scale, return_info=True)
        print(f"{method:16} {str(scale):>8} "
              f"{float(rw_info['neg_log_likelihood']):12.4f} "
              f"{float(result['smooth_std']['trend'][qq(2015,1)].item()):20.4f}")
```

**`fixed_unknown` is the default and the one to leave alone.** It treats the
initial value of each unit-root direction as an unknown constant and lets
the data determine it, which is the exact treatment. Under this method
`diffuse_scale` is set to zero internally and has no effect — which is why
passing it can look as though it does nothing.

**`approx_diffuse`** is the older device: put a very large variance on those
directions instead, `1e8` by default, and let the first observations shrink
it. `diffuse_scale` is the knob for that variance, and this is the only
method that reads it. It fits worse here, `46.1440` against `36.8306`,
because an arbitrary prior is never fully undone.

**`fixed_zero`** pins the unit-root directions at zero. Cheapest, and
defensible only if you really believe the starting level.

The practical rule is short: leave `diffuse_method` alone, and if you pass
`diffuse_scale` and nothing changes, this is why.

## ⚠️ Break it

**Filtering with the standard deviations you never set.**

`assign_strict` would have refused a missing parameter. Standard deviations
are not parameters, they are assigned separately, and they have defaults:

```{code-cell} ipython3
lazy = Simultaneous.from_string(SOURCE, linear=True, flat=True)
lazy.assign_strict(**CALIB)          # the calibration, and nothing else
lazy.steady()
lazy.solve_first_order()

print("standard deviations it will use:",
      np.round(np.asarray(lazy.get_stdvec_measurement_shocks()), 3))

lazy_out, lazy_info = lazy.kalman_filter(observed, HIST, return_info=True)
careless = np.asarray(lazy_out["smooth_med"]["y"][HIST]).ravel()

print()
print(f"  correlation between the two gap estimates : "
      f"{np.corrcoef(final, careless)[0, 1]:.4f}")
print(f"  reported standard error  "
      f"{np.nanmean(np.asarray(out['smooth_std']['y'][HIST]).ravel()):.4f}"
      f"  against {np.nanmean(np.asarray(lazy_out['smooth_std']['y'][HIST]).ravel()):.4f}")
print(f"  neg log likelihood       {float(info['neg_log_likelihood']):.4f}"
      f"  against {float(lazy_info['neg_log_likelihood']):.4f}")
```

Every standard deviation is `1.0`, because that is the default and nothing
objected. The model solved, the filter ran, no warning was printed.

The gap estimate still correlates `0.969` with the properly specified one,
so a chart of it looks broadly right. Everything else has moved. The
reported standard error is `1.0119` instead of `0.3160`, and the likelihood
is `174.16` instead of `42.83`.

What you are filtering is a different economy. Ask that model what the
output gap normally does and it answers `1.41` per cent, against `0.51` for
the one you meant to write:

```{code-cell} ipython3
for label, model in (("assigned", m), ("default", lazy)):
    model_acov = np.asarray(model.get_acov())[0]
    model_rows = [str(name) for name in model.get_acov_dimension_names().rows]
    print(f"  {label:9} unconditional sd of the output gap "
          f"{np.sqrt(np.diag(model_acov))[model_rows.index('y')]:.4f}")
```

Standard deviations are not a detail of the filter, they are a statement
about how volatile the economy is, and leaving them at `1.0` is such a
statement rather than the absence of one. Print them before you filter, and
put the model's own unconditional standard deviations next to what you
believe about the data.

## What you did

```{code-cell} ipython3
# the measurement block says what you observe and how well
#   !measurement-variables / !measurement-equations / !measurement-shocks

# standard deviations are assigned separately from parameters
m.assign(std_shk_y=0.4, std_shk_obs_pi=0.1)

# run it
out, info = m.kalman_filter(observed, HIST, return_info=True)

# estimates, and their standard errors
out["smooth_med"]["y"]      # using the whole sample
out["update_med"]["y"]      # using data up to that quarter
out["smooth_std"]["y"]      # how much to trust it

# what the filter failed to predict, one step ahead
out["predict_err"]["obs_pi"]

# the objective tutorial 17 will optimise
info["neg_log_likelihood"]
```

## Things to remember

1. **The measurement block connects the model to the data.** Each equation
   maps model variables to one observed series, with a shock for the error
   in measuring.
2. **A variable with no measurement equation still gets an estimate**, built
   entirely from the equations tying it to things that are observed.
3. **`smooth` for history, `update` for what was knowable at the time.**
   `predict` is the one-step-ahead forecast, and mostly a diagnostic.
4. **There is no true output gap to recover.** The gap is whatever your
   model says it is, so judge a filter on its uncertainty, its revisions and
   its forecast errors, never on an answer key.
5. **Read `_std` against the unconditional standard deviation from
   `get_acov`** — that ratio is what the data bought you. Here the gap
   narrowed by 37 per cent and foreign inflation by four.
6. **Revisions are large.** Real time against final differed by a third of
   the signal, and changed the sign of the gap in eight per cent of
   quarters.
7. **Prediction errors should not be autocorrelated**, and smoothed shocks
   should not be larger than the standard deviations you assigned.
8. **Missing observations need no special handling** — write `nan` and the
   standard errors widen where the data is absent.
9. **`diffuse_method` defaults to `fixed_unknown`**, the exact treatment,
   which ignores `diffuse_scale` by design.
10. **Standard deviations default to `1.0` and nothing warns.** They are a
    claim about how volatile the economy is.

## Exercise

The output gap is identified only through the equations linking it to the
three observed series. Work out which of the three is doing that work: run
the filter on each pair and on each single series, and compare the standard
error on the gap against the `0.5050` the model gives on its own.

Before you run it, predict which series matters most.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**The policy rate, by a distance — and inflation turns out to be nearly
redundant.**

| observables kept | std on the gap | narrowed by |
|---|---|---|
| `obs_pi`, `obs_i`, `obs_q` | `0.3160` | 37% |
| `obs_i`, `obs_q` | `0.3169` | 37% |
| `obs_pi`, `obs_i` | `0.3410` | 32% |
| `obs_pi`, `obs_q` | `0.3506` | 31% |
| `obs_i` alone | `0.3480` | 31% |
| `obs_pi` alone | `0.4099` | 19% |
| `obs_q` alone | `0.4337` | 14% |

**Dropping inflation entirely costs almost nothing** — `0.3160` becomes
`0.3169`. That is the result worth sitting with, because inflation is where
you would expect the gap to show up.

The policy rule is the reason. The rate responds to inflation *and* to the
gap, through `c2*(pi - pi_tar) + c3*y`, and it is published with a
measurement error of `0.05` against inflation's `0.1`. Observing the rate
accurately tells you about a combination of the two, and once the model has
that combination, observing inflation separately adds a second and noisier
look at something already largely pinned down.

The general point is that **observables are not independent sources of
information.** What a series is worth depends on what the model already
implies about it from the others, and a series can be close to redundant
without being irrelevant. The only way to find out is the table above, which
is a few lines of code and worth running before anyone spends a quarter
improving a data feed.

</details>

## Next

**Tutorial 17 · Estimation** takes `neg_log_likelihood` and stops treating
it as a diagnostic. Hand it to an optimiser, let it choose the parameters
rather than you, and the filter becomes a way of fitting the model to data
instead of a way of reading it. The failures there are mostly about
identification — parameters the likelihood cannot distinguish, which is the
formal version of what foreign inflation ran into here.
