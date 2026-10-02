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

# 17 · Estimation

**50 minutes** · after *16 · The Kalman filter* · next: *18 · Stochastic
simulation*

The filter in tutorial 16 returned a quantity that was not used: the
negative log likelihood, which measures how improbable the observed data are
under the current parameters. It changes when the parameters change.
Estimation is the selection of the parameters that minimise it.

There is no `estimate` method on `Simultaneous`. The objective function is
written by the user and passed to an optimiser. This requires more code than
a single call, and it makes the quantity being minimised explicit.

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from irispie import Simultaneous, Databox, Series, qq

SOURCE = """

!transition-variables
    y, pi, i, q, y_w, pi_w, i_w

!transition-shocks
    shk_y, shk_pi, shk_i, shk_q, shk_y_w, shk_pi_w, shk_i_w

!measurement-variables
    obs_pi, obs_i, obs_q

!measurement-shocks
    shk_obs_pi, shk_obs_i, shk_obs_q

!parameters
    a1, a2, a3, a4, b1, b2, b3, c1, c2, c3
    pi_tar, r_ss, beta, psi
    d1, d2, d3, pi_w_ss, r_w_ss

!transition-equations
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + a3*q + a4*y_w + shk_y;

    pi - pi_tar = b1*(pi{-1} - pi_tar) + beta*(1-b1)*(pi{+1} - pi_tar)
                + b2*y + b3*q + shk_pi;

    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

    q = q{+1} - ((i - pi{+1}) - (i_w - pi_w{+1}))/4 - psi*q + shk_q;

    y_w = d1*y_w{-1} + shk_y_w;
    pi_w = d2*pi_w{-1} + (1-d2)*pi_w_ss + shk_pi_w;
    i_w = d3*i_w{-1} + (1-d3)*(pi_w_ss + r_w_ss) + shk_i_w;

!measurement-equations
    obs_pi = pi + shk_obs_pi;
    obs_i = i + shk_obs_i;
    obs_q = q + shk_obs_q;

"""

CALIB = dict(a1=0.7, a2=0.2, a3=0.1, a4=0.3, b1=0.1, b2=0.3, b3=0.2,
             c1=0.5, c2=1.5, c3=0.5, pi_tar=2, r_ss=1, beta=0.99, psi=0.1,
             d1=0.8, d2=0.7, d3=0.8, pi_w_ss=2, r_w_ss=1)

STDS = dict(std_shk_y=0.4, std_shk_pi=0.3, std_shk_i=0.2, std_shk_q=0.5,
            std_shk_y_w=0.3, std_shk_pi_w=0.2, std_shk_i_w=0.2,
            std_shk_obs_pi=0.1, std_shk_obs_i=0.05, std_shk_obs_q=0.2)


def build(calib, stds):
    """A solved model at the parameters given."""
    model = Simultaneous.from_string(SOURCE, linear=True, flat=True)
    model.assign_strict(**calib)
    model.assign(**stds)
    model.steady()
    model.solve_first_order()
    return model


HIST = qq(2015,1) >> qq(2024,4)
generating_model = build(CALIB, STDS)

rng = np.random.default_rng(0)
shocks_in = Databox.steady(generating_model, HIST)
for name in ("shk_y", "shk_pi", "shk_i", "shk_q",
             "shk_y_w", "shk_pi_w", "shk_i_w"):
    shocks_in[name] = Series(start=HIST[0],
                             values=rng.normal(0, STDS["std_" + name], len(HIST)))

economy = generating_model.simulate(shocks_in, HIST)

observed = Databox()
for state, obs_name in (("pi", "obs_pi"), ("i", "obs_i"), ("q", "obs_q")):
    observed[obs_name] = economy[state] + Series(
        start=HIST[0],
        values=rng.normal(0, STDS["std_shk_" + obs_name], len(HIST)))
```

The data is the same forty quarters as tutorial 16, generated from the same
seed and the same three published series. Because the data was constructed,
the parameters that produced it are known. This permits a comparison that is
not available in applied work: whether the optimiser converged on the values
that generated the sample, and not merely whether it converged. That
comparison cannot be made with real data, so the profiles presented later in
this tutorial are of greater practical importance than the estimates.

## The objective

An objective function receives a vector of parameter values from the
optimiser and returns a single number. The intermediate steps — assigning
the values, re-solving the model, re-running the filter — must be written
explicitly:

```{code-cell} ipython3
NAMES      = ["a1", "b2", "c2", "std_shk_y"]
GENERATING = [0.7, 0.3, 1.5, 0.4]
BOUNDS     = [(0.05, 0.95), (0.02, 1.5), (1.01, 4.0), (0.05, 2.0)]

evaluations = {"count": 0}


def neg_log_likelihood(x):
    """How badly the parameters in x explain the observed data."""
    evaluations["count"] += 1
    proposed = dict(zip(NAMES, x))
    calib = dict(CALIB)
    stds = dict(STDS)
    for name, value in proposed.items():
        (stds if name.startswith("std_") else calib)[name] = value
    try:
        model = build(calib, stds)
        _, info = model.kalman_filter(observed, HIST, return_info=True)
        value = float(info["neg_log_likelihood"])
        return value if np.isfinite(value) else 1e10
    except Exception:
        return 1e10


print("at the generating parameters:", round(neg_log_likelihood(GENERATING), 4))
```

Three things in that function are not optional.

**Rebuild and re-solve every time.** `assign` changes a parameter; it does
not change the solution matrices that were computed from the old one. The
filter uses the solution, so without `steady()` and `solve_first_order()`
the returned value is unrelated to the parameters supplied. The Break it
section at the end of this tutorial demonstrates this error.

**Handle the failures.** An optimiser will evaluate parameter values that
were not anticipated, and some of them will not solve. An uncaught exception
terminates the entire estimation.

**Return a large finite number, not `inf` or `nan`.** Most optimisers cannot
recover from an infinite objective value: they either terminate or propagate
`nan` through the subsequent arithmetic. A large finite penalty such as
`1e10` is treated as an ordinary bad value and the search moves away from
it.

## Profile one parameter before optimising all of them

An optimiser returns a result regardless of whether the problem is well
posed. The least expensive precaution is to vary one parameter at a time
before optimising over all of them:

```{code-cell} ipython3
def profile(name, values):
    """The likelihood along one estimated parameter, others at generating values."""
    position = NAMES.index(name)
    out = []
    for value in values:
        point = list(GENERATING)
        point[position] = value
        out.append(neg_log_likelihood(point))
    return np.array(out)


for value in (0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
    point = list(GENERATING)
    point[NAMES.index("c2")] = value
    print(f"  c2 = {value:4.1f}   {neg_log_likelihood(point):9.4f}")
```

This is the profile of a well identified parameter. The likelihood is worst
at both ends of the range and best near `1.5`, and the penalty for departing
from that value is substantial: nearly four units of log likelihood for an
error of half a point, and twenty-eight at `0.8`. An optimiser will locate
the minimum reliably.

## Running it

```{code-cell} ipython3
start = [0.5, 0.15, 2.0, 0.8]
evaluations["count"] = 0

result = minimize(neg_log_likelihood, start, method="Nelder-Mead",
                  bounds=BOUNDS,
                  options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})

print(f"converged: {result.success}   after {evaluations['count']} evaluations")
print()
print(f"  {'parameter':12} {'start':>8} {'estimate':>10} {'generating':>11}")
for name, s, estimate, true_value in zip(NAMES, start, result.x, GENERATING):
    print(f"  {name:12} {s:8.3f} {estimate:10.4f} {true_value:11.3f}")
print()
print(f"  neg log likelihood at the start       : {neg_log_likelihood(start):9.4f}")
print(f"  neg log likelihood at the estimate    : {result.fun:9.4f}")
print(f"  neg log likelihood at the generating  : {neg_log_likelihood(GENERATING):9.4f}")
```

`Nelder-Mead` is an appropriate default here. It requires no derivatives,
which matters because the objective is a model solution followed by a filter
rather than a closed-form expression, and it tolerates the `1e10` penalty.
Each evaluation takes approximately twenty milliseconds, so the estimation
completes in a few seconds.

Two of the four estimates are close to the generating values: `a1` at
`0.7274` against `0.700`, and `c2` at `1.5279` against `1.500`.

Two are not. `b2` is estimated at `0.2446` against `0.300`, and the standard
deviation of the demand shock at `0.2929` against `0.400`, lower by
approximately one quarter.

## The likelihood prefers the wrong answer

The first explanation to test is a local minimum. It can be ruled out:

```{code-cell} ipython3
print(f"  {'a1 start':>9}  {'a1 estimate':>12} {'neg log lik':>12}")
for a1_start in (0.1, 0.3, 0.5, 0.7, 0.9):
    again = minimize(neg_log_likelihood, [a1_start, 0.3, 1.5, 0.4],
                     method="Nelder-Mead", bounds=BOUNDS,
                     options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
    print(f"  {a1_start:9.2f}  {again.x[0]:12.4f} {again.fun:12.4f}")
```

Five starting values spanning the permitted range all converge to the same
estimate and the same likelihood to four decimal places. This is the global
optimum.

Compare the three likelihood values reported above. At the estimate the
negative log likelihood is **`42.1961`**; at the generating parameters it is
**`42.8323`**. The estimated parameters therefore explain this sample
*better than the parameters that produced it*.

This is the expected behaviour of maximum likelihood rather than a defect.
The method selects the parameters that best explain the observed sample,
which with forty quarters is not the same as the parameters that generated
it. The profiles identify which parameters are susceptible:

```{code-cell} ipython3
for name, grid in (("b2", (0.20, 0.25, 0.30, 0.40, 0.60)),
                   ("std_shk_y", (0.25, 0.30, 0.40, 0.50, 0.70))):
    print(f"  {name}:")
    for value, nll in zip(grid, profile(name, grid)):
        print(f"    {value:5.2f}  {nll:9.4f}")
```

Both profiles are shallow over the relevant range. Moving `b2` from `0.20`
to `0.30` changes the objective by four tenths of a log likelihood unit, and
moving `std_shk_y` from `0.30` to `0.40` by half a unit. The corresponding
figure for `c2` was nearly four units for half a point. A difference of half
a unit is smaller than the variation a single sample of this length can
produce, so the estimate is determined largely by the particular realisation
of the shocks.

**No diagnostic computed from the likelihood will detect this.** The
optimiser reported convergence, the objective improved, and all restarts
agreed.

## A parameter the data does not identify

`b2` is weakly identified. The contrasting case is a parameter that is not
identified at all. Tutorial 16 established that the data reduced uncertainty
about foreign inflation by four per cent, because the three observed series
contain almost no information about it. The parameter governing that process
should therefore have almost no effect on the likelihood:

```{code-cell} ipython3
def profile_free(name, values):
    """The likelihood along any model parameter, not only an estimated one."""
    out = []
    for value in values:
        calib = dict(CALIB)
        calib[name] = value
        try:
            _, info = build(calib, STDS).kalman_filter(observed, HIST,
                                                       return_info=True)
            out.append(float(info["neg_log_likelihood"]))
        except Exception:
            out.append(np.nan)
    return np.array(out)


for value in (0.1, 0.3, 0.5, 0.7, 0.9):
    print(f"  d2 = {value:4.1f}   {profile_free('d2', [value])[0]:9.4f}")
```

Between `0.1` and `0.5` the likelihood changes by three thousandths.
Varying the persistence of foreign inflation across most of its range has no
material effect on the fit, because the observed series contain no
information about that variable.

```{code-cell} ipython3
c2_grid = np.arange(1.05, 3.01, 0.05)
d2_grid = np.arange(0.05, 0.96, 0.05)

figure = make_subplots(
    rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
    subplot_titles=("c2 — policy response to inflation",
                    "d2 — persistence of foreign inflation"),
)
figure.add_trace(go.Scatter(x=c2_grid, y=profile("c2", c2_grid),
                            mode="lines", line=dict(width=3),
                            showlegend=False), row=1, col=1)
figure.add_trace(go.Scatter(x=d2_grid, y=profile_free("d2", d2_grid),
                            mode="lines", line=dict(width=3),
                            showlegend=False), row=1, col=2)
figure.add_vline(x=1.5, line_width=1, line_dash="dot",
                 line_color="grey", row=1, col=1)
figure.add_vline(x=0.7, line_width=1, line_dash="dot",
                 line_color="grey", row=1, col=2)
figure.update_layout(
    title="The likelihood along two parameters, on one vertical scale",
    height=520, template="plotly_white",
)
figure.update_xaxes(title_text="parameter value", row=1, col=1)
figure.update_xaxes(title_text="parameter value", row=1, col=2)
figure.update_yaxes(title_text="negative log likelihood", row=1, col=1)

figure
```

The two panels share a vertical scale, which is what makes the comparison
meaningful. The dotted line in each marks the generating value. The profile
for the policy parameter spans more than twenty units, with its minimum at
the generating value. The profile for foreign inflation is flat across most
of its range and rises only as the parameter approaches one, where a process
close to a unit root behaves differently irrespective of the data.

An optimiser applied to the second parameter will return a value and report
convergence. **A flat profile is the signature of an unidentified parameter,
and profiling is how it is detected before the estimate is reported.**

## Locating the misfit

The likelihood decomposes across periods, which converts a single measure of
fit into a diagnostic:

```{code-cell} ipython3
best = dict(CALIB)
best_stds = dict(STDS)
for name, value in zip(NAMES, result.x):
    (best_stds if name.startswith("std_") else best)[name] = value

_, fitted = build(best, best_stds).kalman_filter(observed, HIST, return_info=True)
_, at_generating = generating_model.kalman_filter(observed, HIST, return_info=True)

fitted_c = np.asarray(fitted["neg_log_likelihood_contributions"].get_data()).ravel()
generating_c = np.asarray(
    at_generating["neg_log_likelihood_contributions"].get_data()).ravel()

print(f"total   at the estimate {np.nansum(fitted_c):9.4f}"
      f"   at the generating values {np.nansum(generating_c):9.4f}")
print()
print("the five quarters the estimated model explains worst")
for k in sorted(np.argsort(-np.nan_to_num(fitted_c))[:5]):
    print(f"  {str(HIST[int(k)]):9} {fitted_c[int(k)]:8.4f}")
```

A large contribution identifies a quarter whose observations the model did
not predict. A small number of such quarters is normal. Several consecutive
ones usually indicate an event the model has no equation for, and the
appropriate responses are a dummy variable, a shock with a time-varying
standard deviation, or an explicit statement of the limitation.

## ⚠️ Break it

**Assigning a parameter without re-solving.**

Tutorial 8 ran this mistake on a simulation: `c2` was doubled, the
simulated path did not move, and `check_steady` reported `True`. Inside an
optimiser the same mistake is worse, because there is a second thing to
mislead you.

```{code-cell} ipython3
stale = build(CALIB, STDS)              # solved once, at the calibration

for value in (1.5, 2.5, 3.5):
    stale.assign(c2=value)              # assigned, never re-solved
    _, info = stale.kalman_filter(observed, HIST, return_info=True)
    print(f"  c2 assigned {value:4.1f}   neg log likelihood "
          f"{float(info['neg_log_likelihood']):10.4f}")

_, fresh_info = build({**CALIB, "c2": 2.5}, STDS).kalman_filter(
    observed, HIST, return_info=True)
print(f"  c2 re-solved  2.5   neg log likelihood "
      f"{float(fresh_info['neg_log_likelihood']):10.4f}")
```

The same number three times. `assign` put the new value on the model, the
model kept the solution matrices it already had, and the filter used those.
Re-solved properly, `c2 = 2.5` gives `53.7974` rather than `42.8323` — a
difference of eleven units that the stale version could not see.

Consider this function placed inside `minimize`. The objective returns the
same value for every input, so the optimiser takes one step, detects no
improvement in any direction, and terminates. It reports **success** and
returns the starting values. Nothing in the output indicates that the
parameters were never evaluated.

The following check requires one line and should precede any lengthy
estimation:

```{code-cell} ipython3
spread = [neg_log_likelihood([a1, 0.3, 1.5, 0.4]) for a1 in (0.2, 0.5, 0.8)]
print("objective at three different a1:", np.round(spread, 4))
print("all identical?", len(set(np.round(spread, 10))) == 1)
```

If the objective does not respond to a change in a parameter, no result
derived from it is meaningful.

## What you did

```{code-cell} ipython3
# an objective: numbers in, one number out, rebuild and re-solve in between
def objective(x):
    calib = dict(CALIB)
    calib.update(zip(["a1", "b2", "c2"], x))
    try:
        model = build(calib, STDS)               # steady() and solve_first_order()
        _, info = model.kalman_filter(observed, HIST, return_info=True)
        return float(info["neg_log_likelihood"])
    except Exception:
        return 1e10                              # a wall, not infinity


# profile before you optimise
profile("c2", np.arange(1.05, 3.01, 0.5))

# then optimise
minimize(objective, [0.5, 0.15, 2.0], method="Nelder-Mead",
         bounds=[(0.05, 0.95), (0.02, 1.5), (1.01, 4.0)])

# the contributions, period by period
np.asarray(fitted["neg_log_likelihood_contributions"].get_data()).ravel()[:4]
```

## Things to remember

1. **The objective must be written by the user.** There is no `estimate`
   method on `Simultaneous`: assign, re-solve, filter, return
   `neg_log_likelihood`.
2. **Re-solve inside the objective.** `assign` alone leaves the previous
   solution in place, and the likelihood then does not respond to the
   parameters.
3. **Return a large finite number on failure**, rather than `inf` or an
   uncaught exception.
4. **Profile one parameter at a time before optimising.** A steep profile
   indicates an identified parameter; a flat one does not.
5. **Convergence does not imply correctness.** The estimate reported here
   fitted better than the parameters that generated the data, from five
   different starting values.
6. **A shallow profile is a warning.** `b2` and `std_shk_y` changed the
   likelihood by less than one unit across the relevant range, and both were
   estimated below the generating values — `b2` by one fifth, `std_shk_y`
   by one quarter.
7. **An unidentified parameter is still assigned an estimate.** Varying `d2`
   across most of its range changed the likelihood by thousandths.
8. **Likelihood contributions localise the misfit** to individual quarters,
   which is more informative than the total.

## Exercise

Add `d2` to the estimated set, with bounds `(0.05, 0.95)` and a generating
value of `0.7`, and run the estimation from three different starting values
for that parameter.

Before running it, predict whether the three runs will agree.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**They do not agree, and the likelihood does not distinguish between the
results.**

| `d2` start | `d2` estimate | neg log likelihood |
|---|---|---|
| `0.20` | `0.2219` | `42.1057` |
| `0.50` | `0.3696` | `42.1053` |
| `0.80` | `0.3696` | `42.1053` |

Two distinct results, `0.2219` and `0.3696`, separated by four
ten-thousandths of a log likelihood unit. Neither is close to the `0.7` that
generated the data, and the optimiser reported convergence in every case.

This is the characteristic behaviour of an unidentified parameter under
estimation. The objective surface is flat enough that the stopping point is
determined by the optimiser's step sizes and convergence tolerance rather
than by the data. Tightening `fatol` moves the results, and so does changing
the starting values.

The appropriate procedure is the one described earlier in this tutorial:
profile the parameter first, observe the flat surface, and exclude it from
the estimated set. Calibrate it instead, or take a value from the
literature, and state that the data were uninformative about it. An estimate
the data did not determine is worse than no estimate, because it is reported
alongside estimates that were determined.

</details>

## Next

**Tutorial 18 · Stochastic simulation** uses the standard deviations rather
than estimating them. Shocks are drawn from the covariance matrix instead of
being set individually, the model is run many times, and the result is a
distribution rather than a single path. This is the basis of fan charts, and
`vary_stds` allows volatility to change over the forecast horizon.
