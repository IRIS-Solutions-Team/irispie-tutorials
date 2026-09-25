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

# 4 · Generating models programmatically

**45 minutes** · after *3 · The model language in depth* · next: *5 · Creating,
parameterising and saving*

Every model file so far was typed out in full. That stops working the moment
the model has three sectors, or five regions, or twelve age cohorts — you end
up copying a block of equations and editing names by hand, which is how models
acquire the kind of bug nobody finds for a year.

The model file has its own small language for this: loops, conditions and
expressions that are resolved *before* the model parser ever sees the file.
This tutorial uses it to write a model that would be tedious to type, and
spends a good deal of time on how to see what it actually produced.

```{code-cell} ipython3
import irispie as ip
from irispie import Simultaneous, Databox, qq
```

## The model we are going to build

The running model has had one inflation rate. Real forecasting models split it
up, because food and energy prices behave nothing like the rest. So:

- **headline inflation** `pi` is a weighted average of sector inflation rates
- each sector has **its own Phillips curve**, with its own persistence and its
  own slope
- demand and the policy rule are unchanged, and the central bank still
  responds to headline inflation

With three sectors that is three near-identical declarations in the variables
block, three in the shocks block, three groups of parameters, and three
near-identical Phillips curves. Every one of them differs only in a name.

## Looping over names

`!for ... !do ... !end` repeats a chunk of file, substituting a token each
time. The control name starts with `?`:

```{code-cell} ipython3
LOOP = """

!transition-variables
    !for ?S = core, food, energy !do
        "Inflation in ?S, % per year"     pi_?S
    !end

!transition-shocks
    !for ?S = core, food, energy !do
        "Cost-push shock in ?S"           shk_pi_?S
    !end

!parameters
    !for ?S = core, food, energy !do
        rho_?S
    !end

!transition-equations
    !for ?S = core, food, energy !do
        pi_?S = rho_?S*pi_?S{-1} + shk_pi_?S;
    !end

"""

mloop = Simultaneous.from_string(LOOP, linear=True, flat=True)
mloop.get_names(kind=ip.TRANSITION_VARIABLE)
```

```{code-cell} ipython3
for equation in mloop.get_equations():
    print(equation)
```

Nine declarations and three equations out of four short blocks.

Three details about the control name:

**The name is arbitrary** — `?S` here, but `?X` or `?sector` work the same.
Write `?` on its own and you get the default control name, which is fine for
a single loop and dangerous inside a nested one, for reasons *Break it* makes
vivid.

**Tokens are separated by anything non-word.** `core, food, energy` and
`core food energy` are the same list.

**`?(name)` gives you case conversion.** Write the control as `?(s)` and `?{s}`
becomes the uppercased token, `?[s]` the lowercased one — handy when the same
name has to appear in a description as well as in a variable, as
`"?{s} inflation"` giving `"CORE inflation"`.

It comes with one trap. A case control cannot be followed by a curly-brace
lag: `pi_?(s){-1}` leaves the `{-1}` unconverted and the model parser later
rejects an equation you never wrote. Curly shifts are rewritten before the
loops run, and the rewriter does not recognise one that follows a closing
bracket. Write `shift(pi_?(s), -1)` instead — pseudofunctions are resolved
last, long after `?(s)` has become `core`.

One error message is worth memorising now: if a loop is missing its `!end`,
IrisPie reports `ValueError: 0 is not in list` — no line number, no mention of
`!for`. It comes from the preparser losing its place, and it always means an
unclosed directive.

## Seeing what the parser actually received

Generated files go wrong in ways that are impossible to diagnose from the
error message, because the text that failed is text you never wrote. So learn
this before you need it.

`save_preparsed` writes the expanded file to disk:

```{code-cell} ipython3
def preparsed(source, **kwargs):
    """Build the model, and return the source the model parser really saw."""
    kwargs.setdefault("linear", True)
    kwargs.setdefault("flat", True)
    Simultaneous.from_string(source, save_preparsed="_preparsed.model",
                             **kwargs)
    with open("_preparsed.model", encoding="utf-8") as f:
        return f.read().strip()


print(preparsed(LOOP))
```

That is the file after every loop has been unrolled — and note the time shifts
have already become `[-1]` rather than `{-1}`, because the preparser
standardises those too.

**The file is written even when the build fails.** That is the whole point:
the moment a generated model refuses to build, the preparsed file is sitting
there showing you why. *Break it* leans on this.

## Driving the file from Python

Hard-coding `core, food, energy` in four places is better than hard-coding
nine names, but it is still four places. Anything in angle brackets `<...>` is
evaluated as a **Python expression** against a `context` dictionary you pass
in:

```{code-cell} ipython3
SECTORS = ["core", "food", "energy"]

SRC = """

!transition-variables

    "Output gap, % of potential"          y
    "Headline inflation, % per year"      pi
    "Policy rate, % per year"             i

    !for ?S = <sectors> !do
        "Inflation in ?S, % per year"     pi_?S
    !end

!transition-shocks

    "Demand shock"                        shk_y
    "Monetary policy shock"               shk_i

    !for ?S = <sectors> !do
        "Cost-push shock in ?S"           shk_pi_?S
    !end

!parameters

    a1, a2, c1, c2, c3, pi_tar, r_ss

    !for ?S = <sectors> !do
        b1_?S, b2_?S, w_?S
    !end

!transition-equations

    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;

    "Headline inflation is a weighted average of the sectors"
    pi = <" + ".join(f"w_{s}*pi_{s}" for s in sectors)>;

    !for ?S = <sectors> !do
        "Phillips curve for ?S"
        pi_?S = b1_?S*pi_?S{-1} + (1-b1_?S)*pi{+1} + b2_?S*y + shk_pi_?S;
    !end

    "Policy rule"
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

"""

print(preparsed(SRC, context={"sectors": SECTORS}))
```

The sector list now appears **once**, in Python. Two different mechanisms used
it:

**`!for ?S = <sectors>`** — the loop's token list is a Python expression, so
the list itself comes from the context.

**`pi = <" + ".join(f"w_{s}*pi_{s}" for s in sectors)>;`** — an arbitrary
Python expression whose *result* is pasted into the file. The weighted-average
equation is not written anywhere; it is computed. Add a sector to `SECTORS`
and the equation grows a term on its own.

That second trick is the one worth remembering. A loop repeats a whole line; a
`<...>` expression builds a piece of one, which is what you need for sums,
products and lists of arguments.

## Switching parts of the file on and off

`!if ... !then ... !else ... !end` keeps or drops a chunk:

```{code-cell} ipython3
FLAG = """

!transition-variables
    y, pi

!transition-shocks
    shk_y, shk_pi

!parameters
    a1, b1, b2

!transition-equations
    y = a1*y{-1} + shk_y;
    !if sticky_prices !then
        pi = b1*pi{-1} + b2*y + shk_pi;
    !else
        pi = b1*pi{-1} + shk_pi;
    !end

"""

print(preparsed(FLAG, context={"sticky_prices": False}))
```

```{code-cell} ipython3
print(preparsed(FLAG, context={"sticky_prices": True}))
```

One file, two models. This is how you keep a simple variant and a full variant
in step instead of maintaining two files that slowly diverge.

Notice that `b2` is declared unconditionally, even though the simple variant
never uses it. That is deliberate, and the reason is a defect worth knowing
about: **if any `!if` in a file has an `!else`, every `!if` in that file needs
one.** Mix the two forms and the file builds for some context values and not
others:

```{code-cell} ipython3
mixed = """
!transition-variables
    y, pi
!transition-shocks
    shk_y, shk_pi
!parameters
    a1, b1
    !if extra !then
        b2
    !end
!transition-equations
    y = a1*y{-1} + shk_y;
    !if extra !then
        pi = b1*pi{-1} + b2*y + shk_pi;
    !else
        pi = b1*pi{-1} + shk_pi;
    !end
"""

for extra in (False, True):
    try:
        Simultaneous.from_string(mixed, linear=True, flat=True,
                                 context={"extra": extra})
        print(f"extra={extra!s:<5} built fine")
    except Exception as e:
        print(f"extra={extra!s:<5} {type(e).__name__} - {str(e)[:60]}")
```

The identical file builds with `extra=False` and dies with `extra=True`. The
search for a matching `!else` does not respect block boundaries: with the first
condition true it runs past that block's own `!end` and latches onto the
`!else` belonging to the second `!if`. So a generated model can pass every
configuration you tested and fail on the one you did not.

**The condition is plain Python, with no angle brackets.** Write
`!if sticky_prices !then`, not `!if <sticky_prices> !then` — the second fails,
because `<...>` expressions are resolved *after* the conditions have already
been decided. Which brings us to the ordering.

## Jinja, the other templating layer

There is a second, completely separate templating system in the same file:
Jinja2, with the usual `{{ ... }}` and `{% ... %}`. It does the same job:

```{code-cell} ipython3
JINJA = """

!transition-variables
{% for s in sectors %}
    pi_{{ s }}
{% endfor %}

!transition-shocks
{% for s in sectors %}
    shk_{{ s }}
{% endfor %}

!parameters
    rho

!transition-equations
{% for s in sectors %}
    pi_{{ s }} = rho*pi_{{ s }}{-1} + shk_{{ s }};
{% endfor %}
"""

print(preparsed(JINJA, context={"sectors": ["core", "food"]}))
```

Same result, different syntax. What matters is that **Jinja runs first** —
before anything else in the file is touched — so Jinja can write `!for`
directives, but `!for` can never write Jinja. The full order explains most of
the surprises:

| Order | Layer | Sees |
|---|---|---|
| 1 | Block comments `#{ ... #}` removed | — |
| 2 | **Jinja2** `{{ }}` `{% %}` | the whole file, `context` |
| 3 | Line comments removed | — |
| 4 | `{-1}` rewritten to `[-1]` | — |
| 5 | **`!for` / `!if`** | `context`; `<...>` inside token lists only |
| 6 | **`<...>` expressions** | `context` |
| 7 | `!list(\`type)` expanded | — |
| 8 | Pseudofunctions expanded | — |

Two consequences you will hit: `<...>` works in a `!for` token list but not in
an `!if` condition, and a `<...>` expression cannot produce a `!for` loop
because loops were resolved a step earlier.

If you only want one of the two, pick Jinja — it is a real templating language,
documented elsewhere, and people already know it. The `!` directives matter
mainly because existing model files are full of them.

## Does the generated model actually work?

A generated model is an ordinary model. Nothing downstream can tell the
difference:

```{code-cell} ipython3
m = Simultaneous.from_string(SRC, linear=True, flat=True,
                             context={"sectors": SECTORS})

CALIB = dict(
    a1=0.7, a2=0.2, c1=0.5, c2=1.5, c3=0.5, pi_tar=2, r_ss=1,
    b1_core=0.7,   b2_core=0.2,   w_core=0.70,
    b1_food=0.5,   b2_food=0.3,   w_food=0.15,
    b1_energy=0.3, b2_energy=0.5, w_energy=0.15,
)

m.assign_strict(**CALIB)
m.steady()
m.get_steady_levels()
```

```{code-cell} ipython3
m.check_steady(when_fails="silent")
```

Every sector sits at the 2% target, headline inflation with it, output at zero
and the policy rate at 3% — the same steady state as the one-sector model,
which is the right answer, because the weights sum to one.

```{code-cell} ipython3
m.solve_first_order()

from collections import Counter
Counter(str(s).split(".")[-1] for s in m.get_eigenvalues_stability())
```

One unstable eigenvalue against one forward-looking variable — only `pi`
appears with a lead — so the model is determinate. Now hit energy with a
one-off cost-push shock:

```{code-cell} ipython3
SPAN = qq(2025,1) >> qq(2034,4)

db = Databox.steady(m, SPAN)
db["shk_pi_energy"][qq(2026,1)] = 1.0

out = m.simulate(db, SPAN)

for name in ("pi_energy", "pi_food", "pi_core", "pi", "y", "i"):
    series = out[name].get_data()[:, 0]
    print(f"{name:<10} peak {float(series.max()):+7.4f}   "
          f"trough {float(series.min()):+7.4f}")
```

Energy inflation jumps a full point on impact. Headline inflation rises about
`0.16` — roughly the 0.15 weight times the shock, plus a little spillover
through the shared output gap. Core barely moves. The central bank lifts rates
by about a tenth of a point, output dips by three hundredths, and everything
comes back.

That is the behaviour you want from a sector split: a relative-price shock
should not look like a general inflation problem, and here it does not.

```{code-cell} ipython3
fig = ip.make_subplots(
    (2, 1),
    figure_title="A one-off energy cost-push shock in 2026Q1",
    figure_height=560,
    subplot_titles=["Sector and headline inflation, % per year",
                    "Policy rate, % per year"],
)
for name in ("pi_energy", "pi_food", "pi_core", "pi"):
    out[name].plot(figure=fig, subplot=0, show_figure=False)
out["i"].plot(figure=fig, subplot=1, show_figure=False)
fig
```

## Worth knowing

Two corners of the language you will meet in other people's files and are
unlikely to write yourself. Nothing later in the series depends on either.

**`!list` groups names by a backtick tag**, and recalls the group anywhere a
list of names is expected:

```{code-cell} ipython3
LIST = """

!transition-variables
    "Real GDP, index"        gdp`level
    "Real wage, index"       w`level
    "Inflation, % per year"  pi`rate

!transition-shocks
    shk_gdp, shk_w, shk_pi

!parameters
    rho

!log-variables
    !list(`level)

!transition-equations
    gdp = gdp{-1}^rho*exp(shk_gdp);
    w = w{-1}^rho*exp(shk_w);
    pi = rho*pi{-1} + shk_pi;

"""

print(preparsed(LIST, linear=False))
```

Tagging a new variable as a level adds it to the log-variables block with
nothing else to edit. **But the order is not stable** — the tags go into a
Python set, and five runs of one file gave five different orderings. Harmless
where order carries no meaning, fatal where it does. `` !list `` also cannot
feed a `!for` loop, since lists expand after the loops have run. A Python list
in the context does the same job and keeps its order.

**Four keywords parse and then do nothing** in IrisPie 0.80.1:

| Keyword | Status |
|---|---|
| `!let` | not implemented — not even in the preparser's keyword list |
| `!autoswaps-simulate` / `!autoswaps-steady` | parsed, then discarded |
| `!preprocessor` / `!postprocessor` | parsed, then discarded |

They go missing without a warning — this model declares a preprocessor and a
postprocessor, and neither leaves a trace:

```{code-cell} ipython3
GHOST = """

!transition-variables
    y

!transition-shocks
    shk_y

!parameters
    a1

!transition-equations
    y = a1*y{-1} + shk_y;

!preprocessor
    pre_y = 2*y;

!postprocessor
    post_y = 3*y;

"""

mg = Simultaneous.from_string(GHOST, linear=True, flat=True)
mg.get_names()
```

No `pre_y`, no `post_y`, and nothing said two blocks were thrown away.
Autoswaps behave the same way. To exogenize and endogenize variables today,
build a `SimulationPlan` in Python — tutorial 14.

## ⚠️ Break it

**A control name that eats more than it should.**

The preparser does blind text replacement. It has no idea what a variable name
is, or where one ends. Here are nested loops with the outer one using the bare
`?`:

```{code-cell} ipython3
collision = """
!transition-variables
    !for ?= us, jp !do
    !for ?S = core, food !do
        pi_?S_?
    !end
    !end
!transition-shocks
    shk
!parameters
    a
!transition-equations
    !for ?= us, jp !do
    !for ?S = core, food !do
        pi_?S_? = a*pi_?S_?{-1} + shk;
    !end
    !end
"""
try:
    Simultaneous.from_string(collision, linear=True, flat=True,
                             save_preparsed="_preparsed.model")
except Exception as e:
    print(type(e).__name__, "-", str(e)[:120])
```

`These names are declared multiple times` — true, and completely unhelpful,
because the names it means are names you never wrote. This is what
`save_preparsed` is for:

```{code-cell} ipython3
with open("_preparsed.model", encoding="utf-8") as f:
    print(f.read().strip())
```

Four variables, two distinct. `?` is a prefix of `?S`, so when the outer loop
replaced every `?` with `us` it turned `?S` into `usS` as well. The inner loop
then had nothing left to substitute and emitted the same equation twice.

The same blindness bites your prose. Use the bare `?` and a description
reading `"Is ? really needed?"` comes out as `"Is y really neededy"` — the
question mark ending the sentence was a control name too. No error, ever; you
find out when someone reads a chart legend.

**Never use the bare `?`.** Give every control a distinct name — `?C`, `?S` —
and neither is a prefix of the other, nor of your punctuation.

## What you did

```{code-cell} ipython3
# repeat a block over a list of names
#     !for ?S = core, food, energy !do
#         pi_?S = rho_?S*pi_?S{-1} + shk_pi_?S;
#     !end

# get the list from Python instead of hard-coding it
m = Simultaneous.from_string(SRC, linear=True, flat=True,
                             context={"sectors": SECTORS})

# build a fragment of an equation with a Python expression
#     pi = <" + ".join(f"w_{s}*pi_{s}" for s in sectors)>;

# keep or drop a block
#     !if sticky_prices !then ... !else ... !end

# the same job in Jinja, which runs before everything else
#     {% for s in sectors %} ... {% endfor %}

# group names by a tag
#     gdp`level ... !list(`level)

# see what the model parser actually received -- written even on failure
Simultaneous.from_string(SRC, linear=True, flat=True,
                         context={"sectors": SECTORS},
                         save_preparsed="_preparsed.model")
```

## Things to remember

1. **`save_preparsed` is the debugger.** A generated model that will not build
   writes its preparsed file anyway, and the answer is almost always in it.
2. **Never use the bare `?`.** It is a prefix of `?S`, so in a nested loop it
   silently mangles the inner control — and it eats question marks in your
   descriptions too.
3. **The layers run in a fixed order**: Jinja, then `!for` and `!if`, then
   `<...>` expressions, then `!list`. A later layer cannot feed an earlier one,
   which is why `<...>` works in a `!for` token list but not in an `!if`
   condition.
4. **`<...>` builds fragments, `!for` repeats lines.** Sums and argument lists
   want the first; blocks of declarations want the second.
5. **If any `!if` in a file has an `!else`, give them all one.** Mixing the two
   forms breaks for some context values and not others, so it survives testing
   and fails later.
6. **A generated model is an ordinary model.** Nothing downstream knows or
   cares how the file was produced.

## Exercise

Add a fourth sector, **services**, to the generated model — and do it by
touching as little as possible.

Services inflation should be the stickiest of the four and the least sensitive
to the output gap: `b1_services = 0.8`, `b2_services = 0.15`. Give it a weight
of `0.10`, and take it out of core, so `w_core` drops to `0.60`.

Before you run it, predict: how many lines of `SRC` do you have to change, how
many transition equations will the model have, and what will the steady state
be?

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**You change no lines of `SRC` at all.** The only edit is to the Python list:

```python
SECTORS = ["core", "food", "energy", "services"]
```

That is the whole point of the exercise. The declarations, the shocks, the
parameters and the Phillips curve all come from `<sectors>`, and the headline
equation is assembled by a `<...>` expression, so all five places update
themselves. Confirm it with `preparsed(SRC, context={"sectors": SECTORS})`,
which prints the headline equation with a fourth term added:
`w_core*pi_core + w_food*pi_food + w_energy*pi_energy + w_services*pi_services`.

The model has **seven** transition equations: demand, headline, four Phillips
curves, and the policy rule.

The steady state is **unchanged** — `y` at 0, `i` at 3, and every inflation
rate at 2 — because the weights still sum to one. It is worth checking that
they do: get them wrong and the steady state quietly shifts, `check_steady()`
still returns `True`, and nothing tells you, because a weighted average with
the wrong weights is still a perfectly consistent equation.

The model stays determinate: eight eigenvalues, seven stable and one unstable,
against the single forward-looking variable `pi`.

Re-running the energy shock, headline inflation peaks at `2.1624` against
`2.1621` with three sectors — almost identical, since services took its weight
from core and both barely respond. Services inflation peaks at `2.0063`, the
smallest response of the four, exactly as a high `b1` and a low `b2` should
give you.

</details>

## Next

**Tutorial 5 · Creating, parameterising and saving** closes out level 1. It
covers `from_file` and `from_string` properly, what each model flag promises,
`assign` against `assign_strict`, finding what you forgot with
`get_unassigned_parameters`, setting shock standard deviations and rescaling
them, and getting a model onto disk and back — the portable format, pickle and
dill.

You can now generate a model of any size. Tutorial 5 is about handling the
model object itself.
