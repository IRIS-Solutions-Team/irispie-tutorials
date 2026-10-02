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

# 25 · Capstone: a full forecast round

**60 minutes** · after *24 · Building models from code* · the last tutorial
in the series

A forecast round is six steps: read the history through the model, project
it forward, impose the judgements the round has agreed on, measure what the
judgements did, attribute the result to its sources, and publish the
numbers. Every one of those steps has appeared somewhere in this series.
This tutorial runs them once, end to end, on the model that tutorials 16 to
24 have been using, and spends most of its length on the two places where a
round goes wrong without raising anything: the attribution, and the size of
the surprises the judgements require.

```{code-cell} ipython3
import numpy as np
import irispie as ip
from irispie import Simultaneous, Databox, Series, SimulationPlan, qq

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

m = Simultaneous.from_string(SOURCE, linear=True, flat=True)
m.assign_strict(**CALIB)
m.assign(**STDS)
m.steady()
m.solve_first_order()

HIST = qq(2015,1) >> qq(2024,4)
FCAST = qq(2025,1) >> qq(2027,4)
PLEDGE = qq(2025,1) >> qq(2025,4)
JUMP_OFF = qq(2024,4)

STATES = ("y", "pi", "i", "q", "y_w", "pi_w", "i_w")
SHOCKS = tuple("shk_" + name for name in STATES)

rng = np.random.default_rng(0)

shocks_in = Databox.steady(m, HIST)
for name in SHOCKS:
    shocks_in[name] = Series(start=HIST[0],
                             values=rng.normal(0, STDS["std_" + name], len(HIST)))

economy = m.simulate(shocks_in, HIST)

observed = Databox()
for state, obs_name in (("pi", "obs_pi"), ("i", "obs_i"), ("q", "obs_q")):
    observed[obs_name] = economy[state] + Series(
        start=HIST[0],
        values=rng.normal(0, STDS["std_shk_" + obs_name], len(HIST)))

print("history  :", HIST[0], "to", HIST[-1], f"({len(HIST)} quarters)")
print("forecast :", FCAST[0], "to", FCAST[-1], f"({len(FCAST)} quarters)")
print("observed :", sorted(observed.get_names()))
```

## Reading the history

The round starts where tutorial 16 finished. The filter turns three
published series into a complete picture of the economy, including the
variables nobody publishes, and the last column of that picture is the
state the forecast starts from.

```{code-cell} ipython3
filtered = m.kalman_filter(observed, HIST)

print("jump-off state at", JUMP_OFF)
for name in STATES:
    level = float(filtered["smooth_med"][name][JUMP_OFF].item())
    error = float(filtered["smooth_std"][name][JUMP_OFF].item())
    print(f"   {name:5} {level:8.4f}   standard error {error:.4f}")
```

The economy is above trend, inflation is above the target of two, and the
policy rate is above its neutral level of three. None of that was read off
a published series. The output gap in particular has no measurement
equation at all, and the `0.7337` comes from the three series that do.

The standard errors matter here more than anywhere else in the round.
Every forecast number below inherits them, and the gap carries a standard
error of a third of a percentage point into a forecast that will be
published to one decimal place.

## The baseline

The baseline is the model left alone: the filtered state at the jump-off,
no shocks after it, and whatever the equations do. Building the input
databox is the contract from tutorial 23 — the forecast needs one period
before `2025-Q1`, and the smoothed series supply it.

```{code-cell} ipython3
jump = Databox.steady(m, FCAST)
for name in STATES:
    jump[name] = filtered["smooth_med"][name]

baseline = m.simulate(jump, FCAST)


def quarters(run, name, count=8):
    return np.round(np.asarray(run[name][FCAST]).ravel()[:count], 4)


print("baseline, first eight quarters")
for name in ("pi", "i", "y"):
    print(f"   {name:4}", quarters(baseline, name))
print()
print("steady:", {k: round(float(v), 2)
                  for k, v in m.get_steady_levels().items() if k in ("pi", "i", "y")})
```

Inflation falls from `2.19` back towards two, the rate from `3.62` back
towards three, and the gap closes. A baseline from a stationary model is
always this shape, and the only interesting thing about it is how fast it
converges, which is a property of the calibration and not of the data.

## The judgements

Two judgements go into this round. The central bank has announced it will
hold the policy rate at `3.75` for four quarters, which is above where the
rule would take it. And the round has agreed on a one-off cost increase of
one percentage point in the first quarter.

The pledge is a plan, and the cost increase is a shock. Both are built
exactly as tutorial 14 built them: `swap_unanticipated` names the variable
being fixed and the shock that pays for it, and the path itself goes into
the databox like any other data.

```{code-cell} ipython3
plan = SimulationPlan(m, FCAST)
plan.swap_unanticipated(PLEDGE, ("i", "shk_i"))

judged = jump.copy()
judged["i"][PLEDGE] = 3.75
judged["shk_pi"][qq(2025,1)] = 1.0

forecast = m.simulate(judged, FCAST, plan=plan)

print("conditioned forecast, first eight quarters")
for name in ("i", "pi", "y"):
    print(f"   {name:4}", quarters(forecast, name))
print()
print("the shock that pays for the pledge")
print("   shk_i", quarters(forecast, "shk_i"))
```

The rate is exactly `3.75` for four quarters and then returns to the rule.
Inflation jumps to `3.23` on the cost increase and then undershoots, down
to `1.73` by the end of 2025, because the pledge keeps policy tight after
the cost increase has passed. The gap turns negative in the third quarter.

The last line is the one to read before anything else. `shk_i` is the
policy surprise the pledge requires in each quarter, and the section
*Break it* comes back to how large those numbers are.

## What the judgements did

This is tutorial 15's subtraction, run on a real pair. `minus_control`
turns two runs into the difference between them, with the model supplied so
that it knows which variables are differences and which are ratios.

```{code-cell} ipython3
difference = forecast.copy()
difference.minus_control(m, baseline)

Y2025 = qq(2025,1) >> qq(2025,4)

print("2025, quarter by quarter")
print("            ", "  ".join(f"{str(p):>8}" for p in Y2025))
for name in ("pi", "i", "y"):
    row = np.asarray(difference[name][Y2025]).ravel()
    print(f"   {name:4}     ", "  ".join(f"{v:8.2f}" for v in row))
```

Inflation is a full point higher in the first quarter and below the
baseline from the third onwards. The rate is higher throughout, by more
each quarter as the baseline falls away from the pledged level. The gap is
lower throughout and worsening.

## The forecast chart

```{code-cell} ipython3
def from_jump_off(run, name):
    """The forecast path with the last filtered observation attached."""
    return Series(
        start=JUMP_OFF,
        values=np.hstack((filtered["smooth_med"][name][JUMP_OFF].item(),
                          np.asarray(run[name][FCAST]).ravel())))


figure = ip.make_subplots(
    (2, 1), subplot_titles=("Inflation", "Policy rate"),
    figure_title="Forecast round, conditioned on a four-quarter rate pledge",
    figure_height=620, show_legend=True,
)

for row, name in enumerate(("pi", "i")):
    filtered["smooth_med"][name].plot(
        figure=figure, subplot=row, span=qq(2023,1) >> JUMP_OFF,
        show_figure=False, legend=["filtered history"])
    from_jump_off(baseline, name).plot(
        figure=figure, subplot=row, show_figure=False, legend=["baseline"])
    from_jump_off(forecast, name).plot(
        figure=figure, subplot=row, show_figure=False, legend=["conditioned"])

for trace in figure.data[3:]:
    trace.showlegend = False

figure
```

Attaching the jump-off observation to the front of each forecast is what
makes the two lines leave the history at the same point instead of
appearing to start from nowhere. The history is the smoothed path, not the
published series, which is the correct thing to show next to a forecast
produced by the same model.

## Why the obvious attribution is wrong

Tutorial 15 decomposed a forecast by running each shock on its own against
a common baseline and adding the results, and showed that on a linear model
the parts sum to the whole exactly. The round here agreed on two
judgements rather than two shocks, so the natural move is the same one: run
each judgement separately and report the two answers.

```{code-cell} ipython3
pledge_only_plan = SimulationPlan(m, FCAST)
pledge_only_plan.swap_unanticipated(PLEDGE, ("i", "shk_i"))
pledge_in = jump.copy()
pledge_in["i"][PLEDGE] = 3.75
pledge_only = m.simulate(pledge_in, FCAST, plan=pledge_only_plan)

cost_in = jump.copy()
cost_in["shk_pi"][qq(2025,1)] = 1.0
cost_only = m.simulate(cost_in, FCAST)


def deviation(run, control):
    out = run.copy()
    out.minus_control(m, control)
    return out


separate = [deviation(pledge_only, baseline), deviation(cost_only, baseline)]

print("largest gap between the two judgements added up and the two together")
for name in ("y", "pi", "i"):
    pieces = sum(np.asarray(part[name][FCAST]).ravel() for part in separate)
    whole = np.asarray(difference[name][FCAST]).ravel()
    print(f"   {name:4} {np.max(np.abs(pieces - whole)):.4f}")
```

The two judgements do not add up, and they are out by more than half a
point on the policy rate. The model is linear and the arithmetic is
correct, so the fault is in the question.

A shock is additive; a condition is not. Under the pledge the rate is
`3.75` whatever else happens, so the cost increase no longer raises it, and
the part of the cost increase that worked through the policy response in
the run on its own has nowhere to go in the run with the pledge. Adding the
two runs double counts a policy reaction that the plan has removed.

## Attributing the change to shocks

The way out is in the output of the conditioned run. A plan is a device for
finding shocks, and once it has found them the plan can be discarded: the
same paths come back out of a plain simulation driven by those shocks.

```{code-cell} ipython3
replay_in = jump.copy()
for name in SHOCKS:
    replay_in[name] = forecast[name]
replay = m.simulate(replay_in, FCAST)

print("conditioned run reproduced with no plan, only its shocks")
for name in ("y", "pi", "i"):
    planned = np.asarray(forecast[name][FCAST]).ravel()
    replayed = np.asarray(replay[name][FCAST]).ravel()
    print(f"   {name:4} largest difference {np.max(np.abs(planned - replayed)):.2e}")
```

With the plan gone the forecast is an initial condition plus a set of
shocks, which is exactly the situation tutorial 15's decomposition
assumes, and both parts are additive. The decomposition has three parts: what the
economy was already doing at the jump-off, the cost increase, and the
policy surprises the pledge required.

```{code-cell} ipython3
no_history = m.simulate(Databox.steady(m, FCAST), FCAST)

parts = {"jump-off state": deviation(baseline, no_history)}
for name, label in (("shk_pi", "cost increase"), ("shk_i", "rate pledge")):
    one = jump.copy()
    one[name] = forecast[name]
    parts[label] = deviation(m.simulate(one, FCAST), baseline)

total = deviation(forecast, no_history)

print("do the three parts add to the whole")
for name in ("y", "pi", "i"):
    stacked = sum(np.asarray(part[name][FCAST]).ravel() for part in parts.values())
    whole = np.asarray(total[name][FCAST]).ravel()
    print(f"   {name:4} largest difference {np.max(np.abs(stacked - whole)):.2e}")

print()
print("inflation, deviation from steady state")
print("              ", "  ".join(f"{str(p):>7}" for p in Y2025))
for label, part in parts.items():
    row = np.asarray(part["pi"][Y2025]).ravel()
    print(f"   {label:15}", "  ".join(f"{v:7.3f}" for v in row))
row = np.asarray(total["pi"][Y2025]).ravel()
print(f"   {'total':15}", "  ".join(f"{v:7.3f}" for v in row))
```

The parts add to the whole to machine precision. Each line is a sentence
the round can defend: inflation is `1.23` points above steady in the first
quarter, of which `0.87` is the cost increase, `0.19` is where the economy
already was, and `0.18` is the pledge. By the end of the year the cost
increase has gone and the pledge is holding inflation `0.26` below steady.

## The decomposition chart

```{code-cell} ipython3
figure = ip.make_subplots(
    (1, 1),
    figure_title="Inflation: deviation from steady state, by source",
    figure_height=460, show_legend=True,
)

for label, part in parts.items():
    part["pi"].plot(chart_type="bar", figure=figure, subplot=0,
                    span=FCAST, show_figure=False, legend=[label])

total["pi"].plot(figure=figure, subplot=0, span=FCAST,
                 show_figure=False, legend=["total"])

figure.update_layout(barmode="relative")
figure
```

`barmode="relative"` stacks the positive contributions above the axis and
hangs the negative ones below, so the bars meet the total line. The line
is the check: if it does not sit on top of the stack, a source has been
left out.

## The report

What gets published is not the quarterly path. It is a small table of
annual averages with the baseline next to it, and that table is what the
rest of the round is for.

```{code-cell} ipython3
YEARS = {2025: qq(2025,1) >> qq(2025,4),
         2026: qq(2026,1) >> qq(2026,4),
         2027: qq(2027,1) >> qq(2027,4)}

LABELS = {"pi": "Inflation", "i": "Policy rate", "y": "Output gap"}

print(f"{'':14}{'':6}", "  ".join(f"{year:>10}" for year in YEARS))
for name, label in LABELS.items():
    for title, run in (("baseline", baseline), ("conditioned", forecast)):
        row = [float(np.mean(np.asarray(run[name][span]).ravel()))
               for span in YEARS.values()]
        print(f"{label if title == 'baseline' else '':14}{title:12}",
              "  ".join(f"{v:10.2f}" for v in row))
    row = [float(np.mean(np.asarray(difference[name][span]).ravel()))
           for span in YEARS.values()]
    print(f"{'':14}{'difference':12}", "  ".join(f"{v:10.2f}" for v in row))
    print()
```

Rounded to the published decimal place, the round says this: the pledge and
the cost increase together add `0.13` to average inflation in 2025 and take
`0.08` off it in 2026, hold the policy rate `0.36` higher in 2025, and cost
`0.15` of output gap in 2025 and `0.13` in 2026. One year of higher
inflation, bought with two years of a weaker economy.

## ⚠️ Break it

**A scenario that needs a three-sigma surprise every quarter for a year.**

Nothing above has checked whether the judgements are plausible. The pledge
is imposed, the model finds the shocks that deliver it, and the shocks are
reported without comment. They are the only place where an implausible
condition shows up.

```{code-cell} ipython3
print("implied shocks against their calibrated standard deviations")
for name in ("shk_i", "shk_pi"):
    path = np.asarray(forecast[name][PLEDGE]).ravel()
    sd = STDS["std_" + name]
    print(f"   {name:7} sd {sd:.2f}")
    print(f"       level   ", "  ".join(f"{v:7.3f}" for v in path))
    print(f"       in sd   ", "  ".join(f"{v:7.2f}" for v in path / sd))
```

The pledge requires a policy surprise of `-3.24` standard deviations in the
first quarter, `1.33` and `2.48` in the next two, and `3.12` in the fourth.
The first and last of those are three-sigma events, which the calibration
says happen once in several hundred quarters, and the sequence also
reverses sign: the bank must first surprise far below its own rule and then
far above it. Four consecutive surprises of that size and shape are not a
forecast scenario, they are a different policy rule.

Nothing in IrisPie raises here, and nothing will. The plan did exactly what
it was asked. The check is arithmetic you add yourself, and it belongs in
every round: divide each endogenized shock by its standard deviation and
report the largest. A condition that needs a two-sigma surprise is worth a
footnote; one that needs three is worth re-opening the discussion about
whether the rule, rather than the shock, is what should change.

## What you did

```{code-cell} ipython3
# read the history through the model and take the jump-off state
m.kalman_filter(observed, HIST)
filtered["smooth_med"]["y"][JUMP_OFF], filtered["smooth_std"]["y"][JUMP_OFF]

# project it forward with nothing imposed
m.simulate(jump, FCAST)

# impose the round's judgements
SimulationPlan(m, FCAST).swap_unanticipated(PLEDGE, ("i", "shk_i"))
m.simulate(judged, FCAST, plan=plan)

# measure them against the baseline, and read the implied shocks
deviation(forecast, baseline)["pi"][Y2025]
np.asarray(forecast["shk_i"][PLEDGE]).ravel() / STDS["std_shk_i"]

# replay the plan as shocks, then attribute source by source
m.simulate(replay_in, FCAST)
deviation(forecast, no_history)["pi"][Y2025]
```

## Things to remember

1. **The round is six steps and each one is a separate object.** Filter
   output, a baseline run, a plan, a conditioned run, a difference databox,
   a table.
2. **The jump-off state comes from the smoother, not from the data.** The
   unobserved variables have no other source, and the forecast needs the
   period before the span starts.
3. **Carry the filter's standard errors into the discussion.** The output
   gap here is `0.7337` with a standard error of `0.3382`, and the forecast
   is published to one decimal place.
4. **A baseline from a stationary model goes back to steady.** That is not
   a result; the result is everything measured against it.
5. **Conditions are not additive.** Running two judgements separately and
   adding them disagrees with running them together, here by more than half
   a point on the policy rate, because the plan removes a policy reaction
   that the separate run still contains.
6. **Decompose the shocks, not the judgements.** Replay the conditioned run
   with no plan and its own shocks, confirm the paths are identical, then
   attribute shock by shock. The parts then add to the whole exactly.
7. **Include the jump-off state as a contribution.** Measured against a run
   from steady state with no history, it is often the largest single source
   in the first year.
8. **Divide every endogenized shock by its standard deviation.** It is the
   only diagnostic that tells you whether the scenario is a scenario.

## Exercise

The bank is considering extending the pledge from four quarters to eight,
at the same rate of `3.75`, with the same cost increase in the first
quarter. Run it, and report the average inflation and output gap in 2025
and 2026, and the largest implied policy surprise in standard deviations.

Predict the 2025 figures before running it.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**2025 does not change at all.**

| | 2025 inflation | 2026 inflation | 2025 gap | 2026 gap | largest `shk_i` |
|---|---|---|---|---|---|
| baseline | 2.089 | 2.018 | 0.267 | 0.065 | — |
| four-quarter pledge | 2.220 | 1.934 | 0.115 | −0.069 | 3.24 sd |
| eight-quarter pledge | 2.220 | 1.552 | 0.115 | −0.435 | 4.53 sd |

Build the longer plan with `swap_unanticipated` over
`qq(2025,1) >> qq(2026,4)` and set `judged["i"]` to `3.75` over the same
span.

The 2025 columns are identical to eight decimal places because the swap is
*unanticipated*, which is tutorial 12's distinction arriving at the point
where it decides a published number. Each quarter's surprise arrives in
that quarter and in no earlier one, so a decision about 2026 cannot reach
into 2025. Everything
the extension does happens in 2026 and after, where inflation falls a
further `0.38` and the gap a further `0.37`.

The cost of the extension is in the last column. The surprises needed grow
to `4.53` standard deviations, which is further past the point at which the
scenario stops being one.

Swapping `swap_unanticipated` for `swap_anticipated` makes 2025 move, and
makes the eight-quarter version unusable: a fully credible two-year peg of
the policy rate in a forward-looking model loses determinacy, and the run
returns an average inflation rate of `27.1` in 2025. That is the model
reporting that the question has no sensible answer, not a bug.

</details>

## Next

That is the series. Twenty-five tutorials, from a model written in a string
to a forecast round defended line by line. The appendices follow:
**A1 · Coming from MATLAB IRIS** is a translation table for the Toolbox
modeller, and **A2 · The errors you will hit first** lists each message,
its real cause and its fix.
