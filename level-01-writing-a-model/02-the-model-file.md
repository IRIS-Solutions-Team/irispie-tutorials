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

# 2 · The model file

**30 minutes** · after *1 · Your first IrisPie model* · next: *3 · The model
language in depth*

By the end you will be able to read any IrisPie model file line by line, and
keep your own model in a proper file instead of a Python string.

Tutorial 1 used a model without really explaining it. This one takes the same
model apart.

```{code-cell} ipython3
import irispie as ip
from irispie import Simultaneous, Databox, qq
```

## Move the model out of the notebook

Here is the model from tutorial 1 again, unchanged:

```{code-cell} ipython3
SOURCE = """

!transition-variables

    "Output gap, % of potential"          y
    "Inflation, % per year"               pi
    "Policy rate, % per year"             i

!transition-shocks

    "Demand shock"                        shk_y
    "Cost-push shock"                     shk_pi
    "Monetary policy shock"               shk_i

!parameters

    "Output persistence"                  a1
    "Interest rate sensitivity"           a2
    "Inflation persistence"               b1
    "Slope of the Phillips curve"         b2
    "Policy rate smoothing"               c1
    "Policy response to inflation"        c2
    "Policy response to output"           c3
    "Inflation target"                    pi_tar
    "Steady-state real interest rate"     r_ss

!transition-equations

    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;

    "Phillips curve"
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;

    "Policy rule"
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

"""
```

A string was fine for one tutorial. For real work the model belongs in its own
file, conventionally with a `.model` extension:

```{code-cell} ipython3
with open("economy.model", "w", encoding="utf-8") as f:
    f.write(SOURCE)
```

And then it loads with `from_file` instead of `from_string`:

```{code-cell} ipython3
m = Simultaneous.from_file("economy.model", linear=True, flat=True)
m
```

Identical model, same two flags. Everything from here works the same way.

Why bother: the file can be version-controlled on its own, so a change to an
equation shows up as a one-line diff rather than a change buried in a
notebook. Several notebooks can load the same model. And your editor will
treat it as a text file you can search.

## The declaration blocks

Every name a model uses has to be declared first, in a block that says what
kind of thing it is. There are three in this file.

**`!transition-variables`** — the things the model explains. Also called
endogenous variables. Here: the output gap, inflation, the policy rate.

**`!transition-shocks`** — the things that disturb it. One per equation is
typical.

**`!parameters`** — numbers that do not change over time, which you set with
`assign_strict`.

You can read each list back:

```{code-cell} ipython3
m.get_names(kind=ip.TRANSITION_VARIABLE)
```

```{code-cell} ipython3
m.get_names(kind=ip.TRANSITION_SHOCK)
```

```{code-cell} ipython3
m.get_names(kind=ip.PARAMETER)
```

Names follow Python's rules: letters, digits and underscores, not starting
with a digit. Within a block you can separate names with commas, newlines or
both — the layout above is for humans, not the parser.

### What IrisPie adds by itself

```{code-cell} ipython3
m.get_names(kind=ip.TRANSITION_STD)
```

You never wrote `std_shk_y`. Declaring a shock creates a standard deviation
for it automatically, named `std_` plus the shock name. Tutorial 1 saw the
`ant_shk_*` names appear the same way.

This is worth knowing because those names are real: they show up in databoxes,
they can be assigned, and they will confuse you the first time you see one you
did not write.

## Descriptions

The quoted string before a name is its description:

```{code-cell} ipython3
m.create_name_to_description()
```

Equations can have them too — the string on the line above the equation:

```{code-cell} ipython3
m.get_equation_descriptions()
```

Descriptions are optional and the model runs fine without them. Write them
anyway. They cost one line, they survive into charts and tables, and six
months later they are the difference between a readable model and a puzzle.

## Equations, lags and leads

Three rules for the equation block:

- **Every equation ends with a semicolon.** This is how the parser knows where
  one stops. Forgetting it does not produce a "missing semicolon" error — see
  *Break it*.
- **`{-1}` is a lag, `{+1}` is a lead.** Curly braces. `y{-2}` is two quarters
  back.
- **Anything already declared can appear on either side.** There is no
  requirement that the variable on the left be "caused" by the right.

Read the equations back and something has changed:

```{code-cell} ipython3
m.get_equations()
```

Two things to notice. The braces have become square brackets — `{-1}` is
source syntax, `[-1]` is what IrisPie stores. And each equation now appears
twice, separated by `!!`. That is the dynamic version and the steady-state
version; tutorial 3 is about why they can differ.

How far back and forward does the model reach?

```{code-cell} ipython3
m.max_lag, m.max_lead
```

One quarter each way. That is a direct consequence of the `{-1}` and `{+1}`
you wrote, and it is worth checking after editing a model — if you meant to
add a two-quarter lag and `max_lag` is still `-1`, your edit did not take.

The lags also determine what data a simulation needs before it can start:

```{code-cell} ipython3
m.get_initials()
```

These are the **initial conditions**. In tutorial 1 the output databox began
one quarter before the simulation span, and this is why: the model cannot
compute 2025Q1 without knowing 2024Q4.

## Log variables

Some quantities are naturally read in percentages rather than units. A price
index, a level of GDP, a productivity index — for these you usually want a 1%
move to mean the same thing whether the level is 10 or 1000, and you never
want the solver to try a negative value.

That is what `!log-variables` does. Here is a small model of its own:

```{code-cell} ipython3
PRODUCTIVITY = """

!transition-variables
    "Productivity, index"     a

!transition-shocks
    "Productivity shock"      shk_a

!parameters
    rho, a_ss

!log-variables
    a

!transition-equations
    a = a_ss * (a{-1}/a_ss)^rho * exp(shk_a);

"""
```

```{code-cell} ipython3
ma = Simultaneous.from_string(PRODUCTIVITY, linear=False, flat=True)
ma.assign_strict(rho=0.8, a_ss=100)
```

Note `linear=False`. That equation has a power and an exponential in it, so it
is genuinely nonlinear, and saying otherwise would be a lie. Tutorial 5 covers
the flags properly, including what IrisPie does when you tell it one that is
not true.

A nonlinear solver needs somewhere to start. Give it the answer you expect:

```{code-cell} ipython3
ma.assign(a=100)
ma.steady()
```

Those iteration tables are the nonlinear solver working. Tutorial 1 never
showed them because a linear model is solved in one step.

```{code-cell} ipython3
ma.get_steady_levels()
```

```{code-cell} ipython3
ma.get_log_status()
```

`a` is flagged as a log variable; everything else would show `False`.

Now the part that matters. Shock it by `0.01` and watch what that means:

```{code-cell} ipython3
ma.solve_first_order()

span = qq(2025,1) >> qq(2029,4)
db = Databox.steady(ma, span)
db["shk_a"][qq(2026,1)] = 0.01

out = ma.simulate(db, span)
out["a"]
```

Productivity goes from 100 to about **101.005** — a **one percent** rise, not
a rise of 0.01 units. For a log variable the shock is proportional. That is
the whole point: the same shock means the same percentage whatever the level.

## ⚠️ Break it

Three mistakes in a model file, all of them ones you will actually make.

**1. Forgetting a semicolon.**

```{code-cell} ipython3
no_semicolon = SOURCE.replace("+ shk_y;", "+ shk_y")
try:
    Simultaneous.from_string(no_semicolon, linear=True, flat=True)
except Exception as e:
    print(type(e).__name__, "-", e)
```

An `IncompleteParseError` — the parser got partway through the file and then
could not continue. It never says "missing semicolon", but it tells you two
things that find it immediately: a **line and column**, and the text where it
gave up, which is the **description of the equation that went wrong**.

Here that text is `"Aggregate demand"`, so the semicolon is missing from the
aggregate demand equation. Break the Phillips curve instead and it quotes
`"Phillips curve"`.

Which is a practical argument for descriptions: they are not only
documentation. When the parser fails it reads one back to you, and a named
equation is far easier to find than a line number in a long file.

**2. Misspelling a block keyword.**

```{code-cell} ipython3
typo = SOURCE.replace("!transition-variables", "!transition-variable")
try:
    Simultaneous.from_string(typo, linear=True, flat=True)
except Exception as e:
    print(type(e).__name__, "-", e)
```

A `ParseError` rather than a helpful message — but it gives you the line and
column, which is enough. The block keywords are fixed strings and
`!transition-variables` is plural.

**3. The wrong kind of brackets.**

```{code-cell} ipython3
wrong_brackets = SOURCE.replace("a1*y{-1}", "a1*y(-1)")
oops = Simultaneous.from_string(wrong_brackets, linear=True, flat=True)
oops
```

**No error.** The model builds. But look:

```{code-cell} ipython3
oops.get_initials()
```

`y[-1]` is gone. IrisPie read `y(-1)` as calling a function named `y`, not as
a lag, so the lag silently vanished from the model. It falls over later, with
a message that says nothing about brackets:

```{code-cell} ipython3
try:
    oops.assign_strict(a1=0.7, a2=0.2, b1=0.6, b2=0.3, c1=0.5,
                       c2=1.5, c3=0.5, pi_tar=2, r_ss=1)
    oops.steady()
except Exception as e:
    print(type(e).__name__, "-", e)
```

`get_initials()` is the quick way to catch this: if a lag you wrote is not in
that list, it was not understood as a lag.

## What you did

```{code-cell} ipython3
# a model lives in its own file
m = Simultaneous.from_file("economy.model", linear=True, flat=True)

# read its structure back
m.get_names(kind=ip.TRANSITION_VARIABLE)
m.get_names(kind=ip.PARAMETER)
m.create_name_to_description()
m.get_equation_descriptions()
m.get_equations()

# how far it reaches in time, and what it needs to start
m.max_lag, m.max_lead
m.get_initials()

# the flags it was built with
m.is_linear, m.is_flat
```

## Things to remember

1. **Keep the model in a `.model` file** and load it with `from_file`. Strings
   are for tutorials.
2. **Every name must be declared before it is used**, in a block that says
   what kind of thing it is.
3. **Every equation ends with a semicolon.** A missing one shows up as a
   complaint about a name that looks like two names joined together.
4. **`{-1}` and `{+1}`, curly braces.** Round brackets parse without error and
   silently destroy the lag — check `get_initials()`.

## Exercise

Add a second lag to the policy rule, so the central bank looks two quarters
back as well as one:

```
i = c1*i{-1} + c1b*i{-2} + (1-c1-c1b)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;
```

Declare the new parameter `c1b`, set it to `0.2` and reduce `c1` to `0.4` so
the weights still behave. Then load the model and check what changed.

Before you run it, predict: what will `max_lag` be, and what will
`get_initials()` return?

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

`max_lag` becomes **−2**, and `get_initials()` grows from three entries to
four:

```
['y[-1]', 'pi[-1]', 'i[-1]', 'i[-2]']
```

Only `i` gains a second entry, because only `i` is now used two quarters back.

The consequence is the one that catches people out: **the model now needs two
quarters of history before it can start.** In tutorial 1 the output databox
began one period before the simulation span. With this change it begins two,
and a databox with only one quarter of history is no longer enough.

That is the general rule. `get_initials()` is not trivia — it is the exact
list of values your input data has to supply before a simulation can compute
anything.

If you forgot to declare `c1b`, you will have seen
`These names are used in equations but not declared: * c1b` — one of the few
IrisPie messages that tells you plainly what is wrong and how to fix it.

</details>

## Next

**Tutorial 3 · The model language in depth** covers everything the file can
contain beyond the basics: the `!!` split between dynamic and steady-state
equations that you saw in `get_equations()`, measurement equations, exogenous
variables, `{:tag}` attributes, the built-in pseudofunctions like `diff` and
`pct`, and `$name$` substitutions for repeated expressions.

You can now read a model file. Tutorial 3 is about writing a more expressive
one.
