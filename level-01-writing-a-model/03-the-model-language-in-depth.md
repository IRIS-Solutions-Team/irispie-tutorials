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

# 3 · The model language in depth

**35 minutes** · after *2 · The model file* · next: *4 · Generating models
programmatically*

Tutorial 2 covered the blocks every model file needs. This one covers the
rest: expressions you name once and reuse, transformations the parser writes
for you, the block that ties a model to observed data, and variables the model
takes as given rather than explaining.

```{code-cell} ipython3
import irispie as ip
from irispie import Simultaneous, Databox, qq
```

Here is the model from tutorials 1 and 2 again, and the calibration that goes
with it:

```{code-cell} ipython3
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

CALIB = dict(a1=0.7, a2=0.2, b1=0.6, b2=0.3, c1=0.5,
             c2=1.5, c3=0.5, pi_tar=2, r_ss=1)

SPAN = qq(2025,1) >> qq(2034,4)
```

## Why the equations read back differently

Print the equations of the model you just built:

```{code-cell} ipython3
m = Simultaneous.from_string(BASE, linear=True, flat=True)
print(m.get_equations()[0])
```

You wrote one equation and two things came back that you never typed.

**`!!` splits every equation in two.** The left half is used when simulating
through time, the right half when solving for the steady state. IrisPie always
stores both, and unless you say otherwise the steady half is a copy of the
dynamic one — which is why you will almost never see `!!` in a model file, and
always see it when you print one back.

**`shk_y` became `(shk_y + ant_shk_y)`**, on the dynamic side only. Every shock
gets a silent companion holding its *anticipated* value, so one equation can
carry both a surprise and a pre-announced change. Tutorial 12 uses them; until
then you can ignore them.

You can write the steady half yourself when the dynamic form makes no sense in
the long run — an equation in growth rates, say, or one with a moving average
in it. Here the policy rule gets the Fisher equation instead, with the
smoothing and feedback terms that vanish in steady state simply dropped:

```{code-cell} ipython3
SPLIT = BASE.replace(
    "i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;",
    "i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i"
    " !! i = pi_tar + r_ss;",
)

ms = Simultaneous.from_string(SPLIT, linear=True, flat=True)
ms.assign_strict(**CALIB)
ms.steady()
ms.get_steady_levels()
```

Same steady state as before — `y` at zero, `pi` at target, `i` at 3% — because
the Fisher equation says exactly what the dynamic rule implies once everything
settles.

The catch is that nothing checks it. IrisPie does not verify that your steady
half agrees with your dynamic half, or with anything at all. *Break it* shows
what a wrong one costs.

## Naming an expression once

The real interest rate `i - pi{+1} - r_ss` appears in the demand equation. In
a real model such an expression appears in five places, and changing it means
finding all five.

`!substitutions` gives it a name, referenced as `$name$`:

```{code-cell} ipython3
SUBS = """

!transition-variables
    y, pi, i

!transition-shocks
    shk_y, shk_pi, shk_i

!parameters
    a1, a2, b1, b2, c1, c2, c3, pi_tar, r_ss

!substitutions
    real_rate := i - pi{+1} - r_ss;
    taylor := pi_tar + r_ss + c2*(pi - pi_tar) + c3*y;

!transition-equations
    y = a1*y{-1} - a2*($real_rate$) + shk_y;
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;
    i = c1*i{-1} + (1-c1)*($taylor$) + shk_i;

"""

msub = Simultaneous.from_string(SUBS, linear=True, flat=True)
for equation in msub.get_equations():
    print(equation)
```

Look closely: **the substitutions are gone.** They were expanded while the
file was being read, and what came out is character-for-character the model
you get from `BASE`. A substitution is a convenience for whoever edits the
file, not a feature of the model.

Note the two symbols: you *define* with `:=` and *use* with `$name$`.

## Transformations the parser writes for you

Writing `x - x{-1}` gets old, and writing the four-quarter version of it gets
worse. IrisPie has pseudofunctions that expand into ordinary algebra:

```{code-cell} ipython3
PSEUDO = """

!transition-variables
    x, dx, dlx, px, rx, mx, sx

!transition-shocks
    shk_x

!parameters
    rho

!transition-equations
    x = rho*x{-1} + shk_x;
    dx  = diff(x);
    dlx = diff_log(x);
    px  = pct(x);
    rx  = roc(x);
    mx  = mov_avg(x);
    sx  = mov_sum(x, -2);

"""

mp = Simultaneous.from_string(PSEUDO, linear=False, flat=True)
for equation in mp.get_equations():
    print(equation)
```

| You write | You get | Reads as |
|---|---|---|
| `diff(x)` | `(x) - (x[-1])` | change |
| `diff_log(x)` | `log(x) - log(x[-1])` | log change |
| `pct(x)` | `100*(x)/(x[-1]) - 100` | percent change |
| `roc(x)` | `(x)/(x[-1])` | gross rate of change |
| `mov_avg(x)` | `((x) + (x[-1]) + (x[-2]) + (x[-3])) / 4` | moving average |
| `mov_sum(x)` | `(x) + (x[-1]) + (x[-2]) + (x[-3])` | moving sum |
| `mov_prod(x)` | product of the same four terms | moving product |
| `shift(x, -2)` | `x[-2]` | an arbitrary shift |

Every one of them takes an optional second argument giving how far back to
reach, which is how `mov_sum(x, -2)` came out as a two-term sum.

### The default reach is not what you expect

**`diff`, `diff_log`, `pct` and `roc` default to `-1`. The three moving
functions default to `-4`.**

And `-4` is fixed. It does not follow the model's frequency — a monthly model
gets four months, not twelve. So `mov_avg(x)` in a monthly model is a
four-month average, which is almost certainly not what you meant.

The damage is not just a wrong number. Reaching back four periods changes how
much history your input data has to supply, silently — *Break it* below runs
it.

## Telling the model what you observe

So far every variable has been a model concept. The output gap is not
something a statistical office publishes — it is an unobservable your model
invents. To confront the model with data you need to say which observed
series correspond to what.

That is the measurement block, and it has its own variables, equations and
shocks:

```{code-cell} ipython3
MEAS = BASE + """
!measurement-variables

    "Observed output gap"                 obs_y
    "Observed inflation"                  obs_pi

!measurement-shocks

    "Output gap measurement error"        mshk_y

!measurement-equations

    "The output gap is observed with error"
    obs_y = y + mshk_y;

    "Inflation is observed cleanly"
    obs_pi = pi;

"""

mm = Simultaneous.from_string(MEAS, linear=True, flat=True)
mm.assign_strict(**CALIB)
mm.steady()
mm.get_steady_levels()
```

The measurement block shows up in the solution as a separate stage:

```{code-cell} ipython3
mm.solve_first_order()
mm.get_solution_vectors()
```

Transition variables and shocks drive the economy. Measurement variables and
shocks sit on top, reading it off. The important property is that the reading
is **one-way**. Shock the measurement error and nothing in the economy moves:

```{code-cell} ipython3
db = Databox.steady(mm, SPAN)
db["mshk_y"][qq(2026,1)] = 0.5

out = mm.simulate(db, SPAN)
print("y      peak:", float(out["y"].get_data()[:, 0].max()))
print("obs_y  peak:", float(out["obs_y"].get_data()[:, 0].max()))
```

The observed gap jumps by 0.5; the actual gap does not move at all. That is
exactly what a measurement error should be — a fault in the thermometer, not
in the weather. It is also what makes the Kalman filter in tutorial 16
possible: the filter's whole job is to work backwards from `obs_y` to `y`,
and it can only do that if the two are genuinely different things.

## Variables the model does not explain

Sometimes you want a variable that enters the equations but that the model
makes no attempt to explain — a path you supply from outside. A fiscal
impulse, say:

```{code-cell} ipython3
EXO = """

!transition-variables
    y, pi, i

!transition-shocks
    shk_y, shk_pi, shk_i

!exogenous-variables
    "Fiscal impulse, % of potential"      g

!parameters
    a1, a2, a4, b1, b2, c1, c2, c3, pi_tar, r_ss

!transition-equations
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + a4*g + shk_y;
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

"""

me = Simultaneous.from_string(EXO, linear=True, flat=True)
me.assign_strict(**CALIB, a4=0.5)
me.assign(g=0.0)
me.steady()
me.solve_first_order()
me.get_steady_levels()
```

An exogenous variable needs no equation — that is the whole point — but it
does need a steady-state value, which is why `assign(g=0.0)` is there. Note
that `get_unassigned_parameters()` will **not** remind you: `g` is not a
parameter, so it is the one kind of quantity that needs a value and is not
covered by that check.

Now feed in a fiscal expansion and simulate:

```{code-cell} ipython3
db = Databox.steady(me, SPAN)
db["g"][qq(2026,1)] = 1.0

out = me.simulate(db, SPAN)
print("g  peak:", float(out["g"].get_data()[:, 0].max()))
print("y  peak:", float(out["y"].get_data()[:, 0].max()))
```

**Nothing happened.** The fiscal impulse is sitting in the output at 1.0 and
the demand equation says `+ a4*g` with `a4 = 0.5`, yet the output gap is still
sitting on its steady state of zero — the `e-16` is floating-point dust, not a
response. No error was raised and no warning printed.

Here is why:

```{code-cell} ipython3
me.get_solution_vectors()
```

`g` is not in the solution. The first-order solution is built around the
transition variables and shocks only; exogenous variables are held at their
steady values and the path you supplied is carried through to the output
untouched, as if it were a comment.

`simulate` defaults to `method="first_order"`. Ask for a method that works
with the equations themselves and the impulse lands.

**This needs IrisPie 0.80.1 or newer.** On earlier versions the same call fails
with `Simulation failed to complete — Cannot make further progress`, which
says nothing about exogenous variables. If you see that, check your version
with `ip.__version__` before looking for anything else.

```{code-cell} ipython3
db = Databox.steady(me, SPAN)
db["g"][qq(2026,1)] = 1.0

out = me.simulate(db, SPAN, method="stacked_time")
print("y  peak:", float(out["y"].get_data()[:, 0].max()))
```

`0.507` — the direct effect of `a4*g = 0.5` plus a little general equilibrium
on top.

The table above it is the stacked-time solver reporting its iterations; it
solved the whole 120-period system twice and stopped when the residual hit
`4e-16`. There is no way to switch it off — `use_iter_printer=False` is
accepted and ignored — so get used to seeing it whenever you leave the default
method. Tutorial 13 is about choosing between the methods.

**So treat `!exogenous-variables` as an advanced feature with a sharp edge.**
If what you want is a driving process and you intend to use the default
first-order simulation, declare an ordinary transition variable with its own
equation instead:

```{code-cell} ipython3
AR = """

!transition-variables
    y, pi, i
    "Fiscal impulse, % of potential"      g

!transition-shocks
    shk_y, shk_pi, shk_i
    "Fiscal shock"                        shk_g

!parameters
    a1, a2, a4, b1, b2, c1, c2, c3, pi_tar, r_ss, rho_g

!transition-equations
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + a4*g + shk_y;
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;
    g = rho_g*g{-1} + shk_g;

"""

mar = Simultaneous.from_string(AR, linear=True, flat=True)
mar.assign_strict(**CALIB, a4=0.5, rho_g=0.8)
mar.steady()
mar.solve_first_order()

db = Databox.steady(mar, SPAN)
db["shk_g"][qq(2026,1)] = 1.0

out = mar.simulate(db, SPAN)
print("g  peak:", float(out["g"].get_data()[:, 0].max()))
print("y  peak:", float(out["y"].get_data()[:, 0].max()))
```

`0.98` — and that is the **default** method, no `stacked_time` needed. The
variable is part of the solution now, it responds to its own shock, and it
works with everything downstream.

That version is part of the solution, responds to `shk_g`, and works with
every simulation method.

## Worth knowing

Two blocks you will meet in other people's files more often than you will
write yourself. Neither is needed for anything later in the series.

**Block tags** label a block, and everything in it inherits the label:

```{code-cell} ipython3
TAGGED = """

!transition-variables
    y, pi, i

!transition-shocks
    shk_y, shk_pi, shk_i

!parameters
    a1, a2, b1, b2, c1, c2, c3, pi_tar, r_ss

!transition-equations{:demand}
    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;

!transition-equations{:supply :nominal}
    "Phillips curve"
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;

!transition-equations{:policy :nominal}
    "Policy rule"
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

"""

mt = Simultaneous.from_string(TAGGED, linear=True, flat=True)
for *_, description, attributes in mt.to_portable()["source"]["equations"]:
    print(f"{attributes:<20} {description}")
```

The tag goes on the **keyword**, not the equation, so you repeat the keyword to
start a differently tagged block. There must be **no space** before the brace
or the file will not parse. Several tags share one pair of braces, each with
its own colon.

What you can do with them afterwards is, for now, very little: they survive
into `to_portable()["source"]["equations"]` as the fifth field of each row, and
no method filters equations by tag. Treat a tag as a **structured comment** —
it makes a long file navigable for a human reader.

**Steady autovalues** are quantities worked out *after* the steady state has
been solved, by evaluating an expression against the answer:

```{code-cell} ipython3
AUTO = BASE + """
!exogenous-variables

    "Steady real interest rate, derived"  r_bar

!steady-autovalues

    r_bar := i - pi;

"""

ma = Simultaneous.from_string(AUTO, linear=True, flat=True)
ma.assign_strict(**CALIB)
ma.steady()
ma.get_steady_levels()
```

`r_bar` comes out at exactly **1.0** — the `r_ss` the calibration assumed. Two
routes to one number, and when they disagree your calibration is inconsistent.
That is what autovalues are for: derived quantities and calibration checks.

The left-hand side must be a **parameter or an exogenous variable**, never a
transition variable, because an autovalue is not an equation and cannot stand
in for the one a transition variable is required to have. And it is not part
of the system: an autovalue can depend on the steady state, never the reverse.

## ⚠️ Break it

**1. A steady half that lies.**

```{code-cell} ipython3
liar = BASE.replace(
    "i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;",
    "i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i"
    " !! i = 99;",
)

ml = Simultaneous.from_string(liar, linear=True, flat=True)
ml.assign_strict(**CALIB)
ml.steady()
ml.get_steady_levels()
```

No error. The steady state dutifully reports a policy rate of 99% and
inflation of 98%, because that is what you told it the long run looks like.

```{code-cell} ipython3
ml.check_steady(when_fails="silent")
```

`False`. The dynamic equations are not satisfied at the steady state the
steady equations produced. Which is why tutorial 1 ran `check_steady()` after
`steady()` without dwelling on why: **do it every time, no exceptions.** It is
the only thing standing between a wrong steady half and a set of results that
look perfectly plausible.

**2. A moving average that quietly reaches back a year.**

```{code-cell} ipython3
mp.max_lag, mp.get_initials()
```

That model has one equation with `mov_avg(x)` in it, and the model now reaches
**three quarters further back** than anything you wrote. `get_initials()` is
the exact list of values your input data must supply before a simulation can
start, and it just grew from one entry to three.

Nothing warned you, and nothing will. The moment a file contains `mov_avg`,
`mov_sum` or `mov_prod` without an explicit second argument, check
`get_initials()` and make sure your data goes back far enough.

## What you did

```{code-cell} ipython3
# every equation is stored twice: dynamic !! steady
m.get_equations()

# write the steady half yourself
#     pi = b1*pi{-1} + ... + shk_pi !! pi = pi_tar;

# name an expression once, use it as $name$
#     !substitutions
#         real_rate := i - pi{+1} - r_ss;

# transformations the parser expands for you
#     diff, diff_log, pct, roc      default shift -1
#     mov_avg, mov_sum, mov_prod    default shift -4
#     shift(x, -2)

# connect the model to observed data
mm.get_solution_vectors()

# variables supplied from outside
me.get_names(kind=ip.EXOGENOUS_VARIABLE)

# quantities derived from the solved steady state
ma.get_steady_levels()
```

## Things to remember

1. **Every equation is two equations.** `!!` separates the dynamic half from
   the steady half, and the steady half is a copy unless you write one — and
   nothing checks it except `check_steady()`.
2. **Substitutions disappear at parse time.** `$name$` never reaches the model.
3. **Moving pseudofunctions reach back four periods by default**, whatever the
   model's frequency, which silently changes what your data must supply.
4. **Measurement equations are a one-way window.** Measurement shocks move
   what you observe and never move the economy.
5. **Exogenous variables are ignored by first-order simulation.** Use
   `method="stacked_time"`, or give the variable its own equation instead.

## Exercise

Give the Phillips curve an explicit steady half saying that in the long run
inflation simply equals the target — append `!! pi = pi_tar;` to it inside
`BASE`, the same way the policy rule was changed earlier in this tutorial.

Build the model, parameterise it with `CALIB`, and solve the steady state.

Before you run it, predict two things: what the steady state will be, and
whether `check_steady()` will pass.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

The steady state is **unchanged** — `y` at 0, `pi` at 2, `i` at 3 — and
`check_steady()` returns **True**.

That is the result worth understanding. You replaced the Phillips curve's
steady half with something that looks nothing like it, and the answer did not
move. It did not move because `pi = pi_tar` is precisely what the dynamic
Phillips curve implies once `y = 0` and inflation stops changing. A correct
explicit steady equation is a restatement, not a new assumption.

Compare that with *Break it*, where `!! i = 99;` was **not** a restatement and
`check_steady()` caught it at once. The difference between a useful `!!` and a
broken one is invisible in the source. Only `check_steady()` can tell them
apart.

The simulated paths are identical to tutorial 1 as well — `y` peaking at
1.0631, `pi` at 3.2283, `i` at 4.7482. Nothing in this tutorial changed the
economics of the model. It changed how the file says it.

</details>

## Next

**Tutorial 4 · Generating models programmatically** stops writing equations by
hand. The preparser gives the file its own small language — `!for` loops,
`!if` conditions, `!list` groups — and Jinja templating lets a Python
dictionary drive what the file contains. You will build a three-sector model
from one sector list, and use `save_preparsed` to see what the parser actually
received when it goes wrong.

You can now read and write everything the model language offers. Tutorial 4 is
about not having to type it.
