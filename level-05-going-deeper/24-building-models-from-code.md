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

# 24 · Building models from code

**50 minutes** · after *23 · Inside the data contract* · next: *25 ·
Capstone: a full forecast round*

Every model so far arrived as a block of source text handed to
`from_string`. There is a second entrance. A model is a list of quantities
and a list of equations, and both lists can be assembled in Python and
passed straight to the constructor. That makes a model something a program
can write, and it makes an existing model something a program can read,
edit and compare.

```{code-cell} ipython3
import copy
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from irispie import Simultaneous, Databox, Series, Stacker, qq
from irispie.sources import ModelSource
from irispie import quantities as _quantities
from irispie import equations as _equations

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


def build(source=SOURCE, calib=CALIB):
    model = Simultaneous.from_string(source, linear=True, flat=True)
    model.assign_strict(**calib)
    model.assign(**STDS)
    model.steady()
    model.solve_first_order()
    return model


def unconditional_sds(model):
    acov = np.asarray(model.get_acov())[0]
    rows = [str(name) for name in model.get_acov_dimension_names().rows]
    return dict(zip(rows, np.sqrt(np.diag(acov))))


m = build()
HIST = qq(2015,1) >> qq(2024,4)
```

## The parts a model is made of

`ModelSource.from_lists` takes one keyword argument per declaration section.
Each quantity is a three-element tuple of description, name and attributes;
each equation is a three-element tuple of description, a pair of equation
strings, and attributes. `Simultaneous.from_source` then turns the result
into a model object, exactly as `from_string` does once it has finished
parsing.

```{code-cell} ipython3
source = ModelSource.from_lists(
    transition_variables=[("Output gap", "y", None),
                          ("Inflation", "pi", None)],
    transition_shocks=[("Demand shock", "shk_y", None),
                       ("Cost shock", "shk_pi", None)],
    parameters=[("Gap persistence", "a1", None),
                ("Inflation persistence", "b1", None),
                ("Slope", "b2", None)],
    transition_equations=[
        ("Output gap", ("y = a1*y[-1] + shk_y", None), None),
        ("Phillips curve", ("pi = b1*pi[-1] + b2*y + shk_pi", None), None),
    ],
)

print("quantities:", source.all_names)
print("equations :", [eqn.human for eqn in source.dynamic_equations])

tiny = Simultaneous.from_source(source, linear=True, flat=True)
tiny.assign(a1=0.7, b1=0.4, b2=0.3)
tiny.steady()
tiny.solve_first_order()

print()
print("T:")
print(np.round(np.asarray(tiny.get_solution().T), 4))
```

The pair in each equation tuple is the dynamic form and the steady form. A
`None` in second position means the steady form is the dynamic one, which
is what every equation in this series has used so far. The two strings are
stored separately from here on, and the section on surgery below returns to
what that separation costs.

## What the lists expect

The lists are handed to the constructor after parsing rather than before
it, so the equation strings must already be in the form the parser
produces. Two things differ from the text written in a
`!transition-equations` section. A shift is written in square brackets,
`y[-1]` rather than `y{-1}`, and there is no closing semicolon.

```{code-cell} ipython3
for eqn in m._invariant.dynamic_equations[:2]:
    print(eqn.human)
```

That is the stored form of the first two equations of the model built from
text at the top. The braces have become brackets, the whitespace has gone,
and each shock has been paired with its anticipated counterpart. Writing
the brace form into `from_lists` is reported, although the report names the
syntax rather than the braces:

```{code-cell} ipython3
try:
    Simultaneous.from_source(ModelSource.from_lists(
        transition_variables=[("", "y", None)],
        transition_shocks=[("", "shk_y", None)],
        parameters=[("", "a1", None)],
        transition_equations=[("", ("y = a1*y{-1} + shk_y", None), None)],
    ), linear=True, flat=True, )
except Exception as error:
    print(type(error).__name__)
    print(str(error).replace("⏐", " ").strip())
```

## Generating a family of models

The payoff for building models in code is the model you would not type out.
The function below returns a model of `n` sector inflations, each following
its own first-order process with its own shock, and an average inflation
rate defined over all of them. Nothing about it changes with `n` except the
length of the lists.

This is deliberately the same model tutorial 4 built, and it is worth
putting the two side by side. Tutorial 4 wrote a source string and let the
preparser repeat the block, with `!for ?S = core, food, energy !do` and a
Python list substituted into the text. Here no source text exists at any
point: the lists are the model.

The choice between them is about where the structure lives. Source text
with `!for` stays readable, stays diffable, and is what a colleague can
open and check, so it is the right default for a model a team maintains.
`from_lists` wins when the structure is a result rather than a decision —
when the number of sectors comes from a dataset, when a study sweeps over
specifications, or when the model is built inside a loop and never read by
a person. The sweep below is the second case: five models of up to
twenty-five sectors, none of them written down.

```{code-cell} ipython3
def sector_model(n, rho=0.7, sigma=0.3):
    """An n-sector inflation model and its average."""
    sectors = range(1, n+1)
    total = "+".join(f"pi_{k}" for k in sectors)
    source = ModelSource.from_lists(
        transition_variables=(
            [("Average inflation", "pi", None)]
            + [(f"Inflation in sector {k}", f"pi_{k}", None) for k in sectors]
        ),
        transition_shocks=[
            (f"Shock to sector {k}", f"shk_{k}", None) for k in sectors
        ],
        parameters=[("Sector persistence", "rho", None)],
        transition_equations=(
            [("Average inflation", (f"pi = ({total})/{n}", None), None)]
            + [(f"Sector {k}", (f"pi_{k} = rho*pi_{k}[-1] + shk_{k}", None), None)
               for k in sectors]
        ),
    )
    model = Simultaneous.from_source(source, linear=True, flat=True)
    model.assign(rho=rho)
    model.assign(**{f"std_shk_{k}": sigma for k in sectors})
    model.steady()
    model.solve_first_order()
    return model


COUNTS = (1, 2, 4, 10, 25)
measured = []

print("sectors   variables   sd(pi)   sigma/sqrt(n(1-rho^2))")
for n in COUNTS:
    model = sector_model(n)
    sd = unconditional_sds(model)["pi"]
    analytic = 0.3 / np.sqrt(n) / np.sqrt(1 - 0.7**2)
    measured.append(sd)
    print(f"{n:7d}   {1+n:9d}   {sd:6.4f}   {analytic:21.4f}")
```

The twenty-five-sector model has twenty-six equations and twenty-five
shocks, and the call that produced it is one line. Its average inflation is
a fifth as volatile as the single-sector version, which is the square root
of twenty-five, because the sector shocks are independent and the average
diversifies them away.

```{code-cell} ipython3
grid = np.arange(1, 26)
figure = go.Figure()
figure.add_trace(go.Scatter(
    x=grid, y=0.3/np.sqrt(grid)/np.sqrt(1 - 0.7**2),
    mode="lines", line=dict(width=3, dash="dot", color="rgb(150,150,160)"),
    name="sigma / sqrt(n (1 - rho^2))",
))
figure.add_trace(go.Scatter(
    x=list(COUNTS), y=measured, mode="markers",
    marker=dict(size=13, color="rgb(99,110,250)"),
    name="get_acov on the generated model",
))
figure.update_layout(
    title="Unconditional volatility of average inflation, by number of sectors",
    xaxis_title="sectors", yaxis_title="standard deviation",
    height=460, template="plotly_white",
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                yanchor="bottom"),
)

figure
```

The markers sit on the line because the model is linear and the shocks are
independent, so the covariance that `get_acov` computes from the solution
matrices is the one the algebra gives. The point of the chart is that the
agreement holds at every `n`, which is the check worth running whenever a
generator starts producing models too large to read.

## The portable format

Tutorial 5 introduced `to_portable` and left it with a warning: the
dictionary it produces cannot be handed back to `from_portable`, so the
format is good for reading a model and not for reloading one. That is true
of the two calls as they stand. It is not true of the dictionary, and this
section is the rest of the story.

`to_portable` converts a model into a dictionary of strings, numbers and
lists. There are no objects in it and no pickled state, so it can be
written as JSON, stored in a repository, compared, and read by something
that is not Python.

```{code-cell} ipython3
portable = m.to_portable()

print("top level  :", list(portable.keys()))
print("format     :", portable["portable_format"])
print("source     :", list(portable["source"].keys()))
print("flags      :", portable["source"]["flags"])
print("quantities :", len(portable["source"]["quantities"]))
print("equations  :", len(portable["source"]["equations"]))
print("variants   :", len(portable["variants"]))
print()
print("first quantity:", portable["source"]["quantities"][0])
print("first equation:", portable["source"]["equations"][0])
print()
print("variant entries:", len(portable["variants"][0]))
for name in ("y", "a1", "std_shk_y"):
    print("  ", name, portable["variants"][0][name])
```

A quantity is a five-element row of kind code, name, log flag, description
and attributes. An equation is a five-element row of kind code, dynamic
string, steady string, description and attributes. The kind codes are `#x`
for a transition variable, `#y` for a measurement variable, `#u` for a
transition shock, `#w` for a measurement shock, `#p` for a parameter and
`#z` for an exogenous variable; equations are `#T`, `#M` and `#A` for
transition, measurement and steady autovalues.

The variant is a flat mapping from name to a level-and-change pair, and it
carries the standard deviations alongside the parameters. The whole model
is a few kilobytes of text:

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "model.json"
    m.to_portable_file(str(path), json_settings={"indent": 2})
    text = path.read_text()

print("characters on disk:", len(text))
print()
print("\n".join(text.splitlines()[:8]))
```

## Reading a portable model back

Two details stand between the dictionary and a working model. The first is
the one tutorial 5 ran into; the second it never reached, because the first
one raises.

The quantity list contains one extra kind, `#v`, holding the anticipated
counterpart of every shock. Those names are generated from the shocks when
a model is constructed, so passing them back in declares each of them
twice. The flags are the second detail: they are written into the
dictionary and read out of it, and then not applied, so a model that was
linear and flat comes back as neither.

```{code-cell} ipython3
kinds = sorted({row[0] for row in portable["source"]["quantities"]})
print("kind codes present:", kinds)
print("the #v rows:", [r[1] for r in portable["source"]["quantities"]
                       if r[0] == "#v"][:3], "...")

try:
    Simultaneous.from_portable(portable)
except Exception as error:
    print()
    print(type(error).__name__ + ":",
          str(error).replace("⏐", " ").split("\n")[0].strip())

stripped = copy.deepcopy(portable)
stripped["source"]["quantities"] = [
    row for row in stripped["source"]["quantities"] if row[0] != "#v"
]
for variant in stripped["variants"]:
    for name in [n for n in variant if n.startswith("ant_")]:
        del variant[name]

print()
print("flags on the original     :", m.get_flags())
print("flags after from_portable :",
      Simultaneous.from_portable(stripped).get_flags())
```

Both are handled by a loader of a few lines, which drops the `#v` rows and
passes the flags where they belong, to `from_source`. With it, the portable
format does reload, and tutorial 5's advice to use pickle instead applies
only to `from_portable` called directly:

```{code-cell} ipython3
def load_portable(portable):
    """Rebuild a Simultaneous from a portable dictionary, flags included."""
    body = portable["source"]
    rows = [row for row in body["quantities"] if row[0] != "#v"]
    dynamic, steady = _equations.from_portable(body["equations"], )
    source = ModelSource(
        quantities=_quantities.from_portable(rows, ),
        dynamic_equations=dynamic,
        steady_equations=steady,
    )
    model = Simultaneous.from_source(source, check_syntax=False,
                                     **body["flags"], )
    for variant, values in zip(model.iter_own_variants(),
                               portable["variants"], ):
        variant.assign_strict({k: v for k, v in values.items()
                               if not k.startswith("ant_")}, )
    return model


restored = load_portable(portable)
restored.steady()
restored.solve_first_order()

print("flags     :", restored.get_flags())
print("same T    :", bool(np.allclose(np.asarray(m.get_solution().T),
                                      np.asarray(restored.get_solution().T))))
print("a1        :", restored.get_parameters()["a1"])
print("std_shk_y :", restored.get_stds()["std_shk_y"])
```

A solution is not part of the portable dictionary, so `steady` and
`solve_first_order` have to be called again on the way back in. That is the
correct arrangement: the dictionary holds what was written down, not what
was computed from it.

## Surgery on a portable model

Because the equations are strings in a list, a change to the model is a
change to a string. The function below rewrites every equation and returns
a new dictionary, leaving the original untouched.

```{code-cell} ipython3
def rewrite(portable, old, new):
    out = copy.deepcopy(portable)
    out["source"]["equations"] = [
        [kind, dynamic.replace(old, new),
         (steady.replace(old, new) if steady else steady), text, attributes]
        for kind, dynamic, steady, text, attributes
        in out["source"]["equations"]
    ]
    return out


edited = rewrite(portable, "+b3*q", "")
closed = load_portable(edited)
closed.steady()
closed.solve_first_order()

print("edited equation:")
print("  ", edited["source"]["equations"][1][1])
print()
print("b3 is still a declared parameter:", "b3" in closed.get_parameters())
print()
left, right = unconditional_sds(m), unconditional_sds(closed)
print("           as written   without b3*q")
for name in ("y", "pi", "i", "q"):
    print(f"   sd({name:2})    {left[name]:8.4f}   {right[name]:12.4f}")
```

Removing the exchange rate from the Phillips curve raises the volatility of
inflation by nearly a quarter and lowers that of the output gap by six per cent,
and the policy rate and the exchange rate both become more volatile as
well. The parameter `b3` stays in
the quantity list and in the variant, declared and unused, which is legal
and is what a hand edit of the source text would also leave behind.

Because the dictionary is plain data, two of them can also be compared
directly: the equation strings are a set, and the variant is a mapping, so
the difference between two versions of a production model is a few lines of
Python rather than a reading exercise.

## The stacked-time system

`Stacker` assembles the model's first-order solution over a whole span at
once, as a single vector holding every transition variable in every period
and a single covariance matrix over that vector. The data is made the same
way as in tutorial 16: run the model with drawn shocks, then keep the three
published series.

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

stacker = Stacker.from_simultaneous(m, HIST)
RECENT = qq(2024,1) >> qq(2024,4)

print("transition variables:", stacker.transition_variable_names)
print("periods             :", len(stacker.base_periods))
print("stacked vector      :", len(stacker.stacked_vector), "entries")
print("first five          :", stacker.stacked_vector[:5])
print("max_lag, max_lead   :", stacker.max_lag, stacker.max_lead)
```

Seven variables over forty quarters is a vector of two hundred and eighty.
`calculate_marginal` conditions that vector on whatever the databox
observes and returns its mean and its covariance.

```{code-cell} ipython3
mean, cov = stacker.calculate_marginal(observed)
mean, cov = np.asarray(mean), np.asarray(cov)
names = stacker.stacked_vector
gap = [k for k, name in enumerate(names) if name.startswith("y[")]

smoothed = m.kalman_filter(observed, HIST)
smooth_med = np.asarray(smoothed["smooth_med"]["y"][HIST]).ravel()
smooth_std = np.asarray(smoothed["smooth_std"]["y"][HIST]).ravel()

print("mean:", mean.shape, "  covariance:", cov.shape)
print()
print("last four quarters")
print("                ", "  ".join(f"{str(p):>9}" for p in RECENT))
print("stacker mean    ", "  ".join(f"{v:9.4f}" for v in mean[gap][-4:]))
print("smoother median ", "  ".join(f"{v:9.4f}" for v in smooth_med[-4:]))
print("stacker sd      ", "  ".join(
    f"{v:9.4f}" for v in np.sqrt(np.diag(cov))[gap][-4:]))
print("smoother sd     ", "  ".join(f"{v:9.4f}" for v in smooth_std[-4:]))
print()
print("largest disagreement over all forty quarters")
print("   mean:", f"{np.max(np.abs(mean[gap] - smooth_med)):.2e}")
print("   sd  :", f"{np.max(np.abs(np.sqrt(np.diag(cov))[gap] - smooth_std)):.2e}")
```

The largest disagreement anywhere in the forty quarters is of the order of
`1e-15`, which is the expected result: the stacked system and the Kalman
smoother are two arrangements of the same calculation. What the stacked
form adds is everything off the diagonal.

## What the off-diagonal entries are for

The smoother reports a standard error for each period separately. The
stacked covariance also reports how the errors in different periods move
together, and that is what any statement about a stretch of the path
depends on.

```{code-cell} ipython3
sd = np.sqrt(np.diag(cov))
last_four = gap[-4:]
block = cov[np.ix_(last_four, last_four)]
correlation = block / np.outer(sd[last_four], sd[last_four])

print("correlation of the output gap estimate across the last four quarters")
print("        ", "  ".join(f"{str(p):>9}" for p in RECENT))
for row, period in enumerate(RECENT):
    print(f"{str(period):8}", "  ".join(f"{v:9.4f}" for v in correlation[row]))

weights = np.zeros(len(names))
weights[last_four] = 0.25
joint = float(np.sqrt(weights @ cov @ weights))
independent = float(np.sqrt(np.sum((0.25**2) * np.diag(cov)[last_four])))

print()
print(f"average gap over the four quarters : {float(weights @ mean):.4f}")
print(f"  standard error, full covariance  : {joint:.4f}")
print(f"  standard error, diagonal only    : {independent:.4f}")
print(f"  ratio                            : {joint/independent:.4f}")
```

Adjacent quarters correlate at about `0.37` and the correlation decays with
distance, which is the band structure a model with one lag produces. Ignore
it and the standard error on the four-quarter average comes out thirty-one
per cent too small. Any figure that averages, differences or cumulates the
filtered path needs the off-diagonal entries, and the stacked system is
where they are.

## ⚠️ Break it

**A model whose steady state does not satisfy its own dynamic equations.**

The portable format stores two strings per equation, the dynamic one and
the steady one. Editing a string is editing one of the two, and nothing
checks that the other still agrees.

```{code-cell} ipython3
half_edited = copy.deepcopy(portable)
half_edited["source"]["equations"] = [
    ([kind, dynamic.replace("pi_tar+r_ss", "pi_tar+2*r_ss"),
      steady, text, attributes]
     if dynamic.startswith("i=c1*i")
     else [kind, dynamic, steady, text, attributes])
    for kind, dynamic, steady, text, attributes
    in half_edited["source"]["equations"]
]

for row in half_edited["source"]["equations"]:
    if row[1].startswith("i=c1*i"):
        print("dynamic:", row[1])
        print("steady :", row[2])

broken = load_portable(half_edited)
broken.steady()
broken.solve_first_order()

span = qq(2025,1) >> qq(2030,4)
path = broken.simulate(Databox.steady(broken, span), span)

print()
print("steady i, pi    :", round(float(broken.get_steady_levels()["i"]), 4),
      round(float(broken.get_steady_levels()["pi"]), 4))
for name in ("i", "pi"):
    values = np.asarray(path[name][span]).ravel()
    print(f"simulated {name:3}   :", np.round(values[:4], 4),
          "...", round(float(values[-1]), 4))
print("check_steady    :", broken.check_steady(when_fails="silent"))
print("check_steady, m :", m.check_steady(when_fails="silent"))
```

The policy rule now has twice the real rate in its dynamic form and the
original real rate in its steady form. The model builds, solves and
simulates, and nothing raises.

The simulation above starts every variable at the model's own reported
steady state and applies no shocks, so it should stay there. Instead the
policy rate leaves three immediately and settles near one, and inflation
leaves two and settles near zero. The steady state was computed from the
steady equations and the dynamics were solved from the dynamic equations,
and the two are never asked whether they describe the same model.

`check_steady` is the one call that asks. It evaluates the dynamic
equations at the steady state and returns `False` when the residuals are
not zero. Run it after any surgery, and after any hand edit of a source
file that touches an equation written with a separate steady form.

## What you did

```{code-cell} ipython3
# a model assembled from lists instead of source text
ModelSource.from_lists(
    transition_variables=[("Output gap", "y", None)],
    transition_shocks=[("Demand shock", "shk_y", None)],
    parameters=[("Gap persistence", "a1", None)],
    transition_equations=[("Output gap", ("y = a1*y[-1] + shk_y", None), None)],
)

# the stored form of an equation, which is what the lists must supply
m._invariant.dynamic_equations[0].human

# the model as plain data, and the same data on disk
m.to_portable()
portable["source"]["quantities"], portable["source"]["equations"]
portable["variants"][0]

# reading it back with the two corrections applied
load_portable(portable)

# the first-order solution stacked over a whole span
Stacker.from_simultaneous(m, HIST)
stacker.stacked_vector
stacker.calculate_marginal(observed)

# whether a steady state satisfies the dynamic equations
m.check_steady(when_fails="silent")
```

## Things to remember

1. **`ModelSource.from_lists` with `Simultaneous.from_source` is the second
   entrance.** One keyword argument per declaration section, three-element
   tuples throughout.
2. **The lists take the parsed form.** Shifts in square brackets, `y[-1]`,
   and no closing semicolon. A `None` steady string means the steady form
   equals the dynamic form.
3. **Generated models are worth checking against algebra at several
   sizes.** The sector family matches `sigma/sqrt(n(1-rho^2))` at every `n`,
   which is the evidence that the generator builds what it claims.
4. **`to_portable` is plain data.** Five-element rows for quantities and
   equations, a flat name-to-value mapping per variant, and JSON on disk.
5. **Reading it back needs two corrections.** Drop the `#v` rows and pass
   `flags` to `from_source`. `from_portable` on its own raises on the
   duplicate anticipated-shock names, which is the limit tutorial 5
   reported, and once those rows are removed it returns a model that is no
   longer linear or flat.
6. **A comparison of two portable dictionaries is a comparison of two
   models.** Which equations moved, and which numbers were recalibrated.
7. **`Stacker` reproduces the smoother and adds the off-diagonal
   entries.** Use it for any statement about an average, a difference or a
   cumulation over several periods.
8. **`check_steady` is the test for an inconsistent edit.** The dynamic and
   steady forms are stored separately and are never compared for you.

## Exercise

A parameter is to be renamed across the whole model: `c2`, the policy
rule's response to inflation, becomes `phi_pi`. The name appears in three
separate places in the portable dictionary. Write
`rename(portable, old, new)` that handles all three, load the result, and
confirm that the model solves to the same `T` as the original.

Name the three places before writing anything.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**The quantity rows, both equation strings, and the variant keys.**

In the quantity rows the name is element one. In the equations it is
element one and element two, the dynamic and steady strings, and it must be
replaced on a word boundary so that `c2` does not match inside another
name. In each variant it is a dictionary key. Build a new dictionary in all
three cases rather than editing in place, because the quantity and equation
rows arrive as tuples.

With all three done, `load_portable` returns a model carrying `phi_pi` at
`1.5`, with no `c2` anywhere, and `solve_first_order` produces a `T`
identical to the original — a rename changes the label and nothing else.

Leave any one of the three out and the load fails. Each omission has its
own message, and the three are worth telling apart:

| renamed | message |
|---|---|
| quantities only | `These names are used in equations but not declared: * c2` |
| equations only | `These names are used in equations but not declared: * phi_pi` |
| quantities and equations, variants left | `Cannot assign these names (nonexistent in the model object): * c2` |

The first two fail while the model is being constructed and name the place
to look. The third gets a working model built and fails only when the
values are poured in, which is why it is the one that takes longest to
recognise.

</details>

## Next

**Tutorial 25 · Capstone: a full forecast round** runs the whole series
once, end to end: filter the history, condition the forecast with a plan,
simulate, compare against a baseline, decompose the difference into its
sources, and publish the table.
