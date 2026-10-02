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

# 12 · Shocks

**35 minutes** · after *11 · Your first simulation* · next: *13 · Nonlinear
simulation*

Every shock in IrisPie comes in two versions: one the economy sees coming,
and one it does not. This tutorial is about the difference between them, and
when it matters.

A rate rise the central bank announced a year ago and one that arrives with
no warning are different events, even when the number is the same. People who
know something is coming act before it happens. The two copies are how you
say which kind you mean, and they are the `ant_shk_*` items that have
appeared in every databox since tutorial 3.

```{code-cell} ipython3
import io
import contextlib

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

SPAN = qq(2025,1) >> qq(2028,4)
steady = m.get_steady_levels()
```

## Two copies of every shock

You declared seven shocks. The solved model carries fourteen:

```{code-cell} ipython3
vectors = m.get_solution_vectors()

print("what you declared  :", vectors.transition_shocks)
print("what the model got :", vectors.anticipated_shock_values)
```

Each shock has a companion with `ant_` in front of it. `shk_y` is the
surprise version; `ant_shk_y` is the version everyone saw coming. They are
separate items in the databox and you fill in whichever one you mean.

## The same number, two ways

Put a one-point demand shock in 2026Q1 — a year into the simulation — first
as a surprise:

```{code-cell} ipython3
surprise = Databox.steady(m, SPAN)
surprise["shk_y"][qq(2026,1)] = 1.0
out_surprise = m.simulate(surprise, SPAN)

print("y:", np.round(out_surprise["y"].get_data()[:, 0][:8], 4))
print("q:", np.round(out_surprise["q"].get_data()[:, 0][:8], 4))
```

Both lines are flat until the shock arrives, then output jumps to **0.9057**
and the currency appreciates to **-0.3219**. The economy sits still until the
quarter of the shock, because nothing has told it to do otherwise.

Now the same number, in the other copy:

```{code-cell} ipython3
expected = Databox.steady(m, SPAN)
expected["ant_shk_y"][qq(2026,1)] = 1.0
out_expected = m.simulate(expected, SPAN)

print("y:", np.round(out_expected["y"].get_data()[:, 0][:8], 4))
print("q:", np.round(out_expected["q"].get_data()[:, 0][:8], 4))
```

**The exchange rate moves a year early.** `q` is at `-0.1894` four quarters
before anything happens, and keeps appreciating through `-0.2275`, `-0.2674`,
`-0.3106`.

Nothing has been shocked in those quarters. The currency moves because of
interest parity: rates are going to rise, a currency that will pay more is
worth more now, so it appreciates the moment the news arrives. The exchange
rate is the variable that trades on expectations, and it is the one that
reacts first.

Output barely moves beforehand, and what little it does is *downwards* —
`-0.0036`, `-0.0116`, `-0.0217`, `-0.0335`. The stronger currency is already
costing exports before the demand shock arrives to make up for it.

## What anticipation actually changes

```{code-cell} ipython3
def deviations(output):
    return {name: output[name] - float(steady[name])
            for name in ("y", "pi", "i", "q")}

unexpected, foreseen = deviations(out_surprise), deviations(out_expected)

figure = ip.make_subplots(
    (2, 2),
    figure_title="A demand shock in 2026Q1: surprise against expected",
    figure_height=620,
    subplot_titles=["Output gap", "Inflation", "Policy rate", "Exchange rate"],
    show_legend=True,
)

for position, name in enumerate(("y", "pi", "i", "q")):
    unexpected[name].plot(figure=figure, subplot=position,
                          show_figure=False, legend=["surprise"])
    foreseen[name].plot(figure=figure, subplot=position,
                        show_figure=False, legend=["expected"])

for index, trace in enumerate(figure.data):      # one legend pair, not four
    trace.showlegend = index < 2

figure
```

Read the first two panels together. Output peaks at **0.8480** instead of
`0.9057`, and inflation at **0.3710** instead of `0.3986`. **Both are
smaller**, which is the result to expect: an economy that sees a disturbance
coming has time to adjust, so less of it shows up in output and prices.

The other two panels are where the adjustment went. The policy rate peaks
*higher*, at `0.6438` against `0.5519`, because the central bank starts
tightening a year early and keeps at it. The currency appreciates further, to
`-0.3576` against `-0.3219`.

So anticipation does not make a shock disappear. It moves the response out of
output and inflation and into the two variables that can jump — the exchange
rate immediately, and the policy rate in advance.

## Anticipated means known from the start

There is a subtlety that is easy to get wrong. "Anticipated" does not mean
*announced on some date you choose*. It means **known from the first period
of the simulation span**.

So the warning the economy gets is the distance from the start of your span
to the date of the shock:

```{code-cell} ipython3
for start in (qq(2026,1), qq(2025,3), qq(2025,1), qq(2024,1)):
    span = start >> qq(2028,4)
    d = Databox.steady(m, span)
    d["ant_shk_y"][qq(2026,1)] = 1.0
    value = m.simulate(d, span)["y"][qq(2026,1)].item()
    print(f"{qq(2026,1) - start} quarters of warning:  y at 2026Q1 = {value:.4f}")
```

More warning, smaller response, settling down once the economy has had long
enough. With no warning at all the anticipated shock gives `0.9057` — exactly
the surprise answer, because a shock you learn about on the day it arrives is
a surprise.

Notice how quickly it converges. Four quarters of warning gets you `0.8480`
and eight gets `0.8470`; after a year there is almost nothing left to gain
from knowing earlier.

To announce a shock *part way through* a simulation you need a simulation
plan, which is tutorial 14.

## One quarter or several

A shock is one number in one period. Four quarters of demand pressure is four
numbers:

```{code-cell} ipython3
sustained = Databox.steady(m, SPAN)
for quarter in range(4):
    sustained["shk_y"][qq(2026,1).shift(quarter)] = 1.0

path = m.simulate(sustained, SPAN)["y"].get_data()[:, 0]
print("four surprise quarters: peak", round(float(path.max()), 4))
print(np.round(path[3:10], 3))
```

The effects accumulate: `0.906`, then `1.419`, `1.683`, `1.806`, as each new
quarter adds to what remains of the previous ones.

The same four quarters, seen coming:

```{code-cell} ipython3
sustained_expected = Databox.steady(m, SPAN)
for quarter in range(4):
    sustained_expected["ant_shk_y"][qq(2026,1).shift(quarter)] = 1.0

path = m.simulate(sustained_expected, SPAN)["y"].get_data()[:, 0]
print("four expected quarters: peak", round(float(path.max()), 4))
print(np.round(path[3:10], 3))
```

**1.65 against 1.81.** The same pattern as the single shock, and for the same
reason.

## They add together

Both copies can hold a value at once, and because the model is linear the two
responses simply add:

```{code-cell} ipython3
def path_of(**items):
    d = Databox.steady(m, SPAN)
    for name, period in items.items():
        d[name][period] = 1.0
    return m.simulate(d, SPAN)["y"].get_data()[:, 0]

only_surprise = path_of(shk_y=qq(2026,1))
only_expected = path_of(ant_shk_y=qq(2026,1))
together = path_of(shk_y=qq(2026,1), ant_shk_y=qq(2026,1))

print("surprise alone :", round(float(only_surprise.max()), 4))
print("expected alone :", round(float(only_expected.max()), 4))
print("both together  :", round(float(together.max()), 4))
print("both == sum    :", np.allclose(together, only_surprise + only_expected))
```

Two separate events in the same quarter, one of which the economy expected
and one of which it did not.

## Frames

`return_info` reports frames, and this is where they come from:

```{code-cell} ipython3
d = Databox.steady(m, SPAN)
d["shk_y"][qq(2026,1)] = 1.0

_, info = m.simulate(d, SPAN, return_info=True)
print("first_order        :", info["frames"])

_, info = m.simulate(d, SPAN, return_info=True, force_split_frames=True)
print("force_split_frames :", info["frames"])

with contextlib.redirect_stdout(io.StringIO()):     # the solver logs every pass
    _, info = m.simulate(d, SPAN, return_info=True, method="stacked_time")

print("stacked_time       :", info["frames"])
```

A **frame** is one stretch of the span that can be solved in a single pass. A
surprise ends a frame: everything the economy believed up to that quarter has
to be thrown away and the rest recomputed from the new situation. So the span
is cut at 2026Q1, where the surprise arrives.

`first_order` does not need to split — a linear solution gives the same
answer either way — so it runs one frame unless you ask for more.
`stacked_time` splits on its own. The numbers are identical:

```{code-cell} ipython3
one = m.simulate(d, SPAN)["y"].get_data()[:, 0]
two = m.simulate(d, SPAN, force_split_frames=True)["y"].get_data()[:, 0]

print("largest difference:", np.abs(one - two).max())
```

A purely anticipated simulation has nothing to surprise anyone, so it stays
in one frame whatever method you use.

## ⚠️ Break it

**Changing your mind and leaving the old copy behind.**

This is the most ordinary way to get the two copies wrong. You set up a
surprise shock, decide it should have been announced instead, and add the
anticipated version:

```{code-cell} ipython3
changed_mind = Databox.steady(m, SPAN)
changed_mind["shk_y"][qq(2026,1)] = 1.0        # written first

changed_mind["ant_shk_y"][qq(2026,1)] = 1.0    # second thoughts

path = m.simulate(changed_mind, SPAN)["y"].get_data()[:, 0]
print("peak:", round(float(path.max()), 4))
print("path:", np.round(path[:8], 3))
```

**0.8480 was the intention; 1.7537 is the answer.** The previous section
showed why: the two copies add, so the databox now holds two shocks rather
than one.

The path gives nothing away. Output is flat, then jumps, then decays — the
shape of every demand shock in this tutorial. The only tell is a number
roughly twice what it should be, in a model where you may have no strong
prior about the right size.

Clearing the old copy is the fix, and it has to be explicit:

```{code-cell} ipython3
changed_mind["shk_y"][qq(2026,1)] = 0.0

print("peak after clearing:", round(float(
    m.simulate(changed_mind, SPAN)["y"].get_data()[:, 0].max()), 4))
```

The habit worth forming is to build the databox fresh from `Databox.steady`
whenever you change what the experiment is, rather than editing the one you
already have. It costs a line and removes the whole class of mistake.

## What you did

```{code-cell} ipython3
# the surprise copy and the expected copy
m.get_solution_vectors().transition_shocks
m.get_solution_vectors().anticipated_shock_values

d = Databox.steady(m, SPAN)

# nothing happens until the quarter it arrives in
d["shk_y"][qq(2026,1)] = 1.0

# the economy starts adjusting from the first period of the span
d["ant_shk_y"][qq(2026,1)] = 1.0

# where the frames come from
_, info = m.simulate(d, SPAN, return_info=True, force_split_frames=True)
```

## Things to remember

1. **Every shock has two copies**: `shk_x` for a surprise, `ant_shk_x` for
   something seen coming.
2. **A surprise does nothing until it arrives.** The path is flat up to the
   quarter of the shock.
3. **The exchange rate reacts first**, because interest parity prices in a
   future rate change the moment it is known.
4. **Anticipation damps output and inflation** and raises the policy rate and
   the exchange rate. The response moves to the variables that can jump.
5. **Anticipated means known from the first period of the span**, not
   announced on a date of your choosing. For that, use a plan — tutorial 14.
6. **Both copies can hold a value at once** and the responses add, which is
   also the easiest way to double a shock by accident.
7. **A frame is one stretch solved in a single pass**, and a surprise ends
   one. `first_order` uses a single frame and gets the same answer either
   way.

## Exercise

Put a one-point **monetary policy** shock in 2026Q1, first as a surprise and
then as something expected, and look at inflation and the exchange rate
rather than output.

Before you run it, predict the sign of inflation in 2025Q1 — a year before an
expected tightening.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

Inflation is **below target a year early**, at `-0.2976` from steady state,
and it stays roughly there until the rise arrives: `-0.2976`, `-0.3055`,
`-0.2870`, `-0.2662`.

The surprise version does nothing at all in those quarters — exactly `0.0`
until the quarter the shock arrives, when inflation drops to `-0.2763`.

A tightening everyone knows is coming is already working before it happens,
and in this model you can see the route it takes. The currency appreciates
immediately, to `-0.0905` in the first quarter, which makes imports cheaper
and pulls inflation down through the pass-through term. Firms setting prices
today are also looking at weaker demand tomorrow. The central bank gets a
good part of the effect it wants before changing the rate at all.

This is why the two copies of each shock exist. A model that could only
handle surprises would have nothing to say about forward guidance, which is
most of what a central bank does between meetings.

</details>

## Next

**Tutorial 13 · Nonlinear simulation** changes the machinery underneath.
Everything so far has used `first_order`, which is why the responses in this
tutorial added together exactly. `stacked_time` and `period_by_period` solve
the equations themselves instead of a linear approximation to them. They take
longer, they can fail to converge, and they are what you need once a model
has a real nonlinearity in it.
