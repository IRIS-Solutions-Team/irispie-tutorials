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

# A2 · The errors you will hit first

**35 minutes** · reference · usable from *1 · Your first IrisPie model*
onwards

Every message in this appendix is produced by the code on this page, so
the wording is the wording you will see. They are grouped by the stage that
raises them, because the stage is usually the fastest way to narrow down
the cause: a message from the parser means the file is wrong, a message
from `steady` means the file is fine and the economics is not.

The last section is the longest and the most useful. It collects the cases
where nothing is raised at all.

```{code-cell} ipython3
import contextlib
import io

import numpy as np
from irispie import Simultaneous, Databox, Series, SteadyPlan, qq


def report(label, call):
    """Run something expected to fail, and print what came back."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            call()
    except Exception as error:
        kind = type(error).__module__ + "." + type(error).__name__
        text = " ".join(str(error).replace("⏐", " ").split())
        print(f"{label}\n    {kind}\n    {text[:94]}\n")
        return
    print(f"{label}\n    nothing raised\n")


GOOD = """
!variables
    y, pi
!shocks
    shk_y, shk_pi
!parameters
    a1, b1, b2
!equations
    y = a1*y{-1} + shk_y;
    pi = b1*pi{-1} + b2*y + shk_pi;
"""


def good_model():
    model = Simultaneous.from_string(GOOD, linear=True, flat=True)
    model.assign(a1=0.7, b1=0.4, b2=0.3)
    model.steady()
    model.solve_first_order()
    return model


SPAN = qq(2025,1) >> qq(2025,4)
print("the working model:", good_model())
```

## How to read one

Almost everything IrisPie raises is `datapie.wrongdoings.Error` or its
subclass `datapie.wrongdoings.Critical` — from `datapie`, the data package
underneath, not from `irispie`. Catching `Error` catches both.

```{code-cell} ipython3
from datapie.wrongdoings import Error, Critical

print("Critical is an Error:", issubclass(Critical, Error))
print("both are Exceptions :", issubclass(Error, Exception))
```

The message is a headline followed by a bulleted list, each bullet marked
with `⏐`. The headline says what rule was broken and the bullets say which
names or equations broke it. The bullets are the part to read first: they
name the thing to go and fix.

Anything that is a plain `ValueError`, `AttributeError`, `IndexError` or
`TypeError` came from NumPy, SciPy or Python itself, which usually means
the model object was fine and the call was wrong.

## Writing the file

```{code-cell} ipython3
report("an unknown block keyword",
       lambda: Simultaneous.from_string(
           GOOD.replace("!shocks", "!shoks"), linear=True, flat=True))

report("a missing semicolon",
       lambda: Simultaneous.from_string(
           GOOD.replace("+ shk_y;", "+ shk_y"), linear=True, flat=True))
```

The first is a parsing error and names the text it stopped at, which is the
line to inspect. `parsimonious` is the parser library; the class name is of
no further interest.

The second is the one to memorise, and the *Break it* section below comes
back to it.

## Building the model

```{code-cell} ipython3
report("a name used but never declared",
       lambda: Simultaneous.from_string(
           GOOD.replace("b2*y", "b2*yy"), linear=True, flat=True))

report("a name declared twice",
       lambda: Simultaneous.from_string(
           GOOD.replace("    y, pi", "    y, pi, y"), linear=True, flat=True))

report("fewer equations than variables",
       lambda: Simultaneous.from_string(
           GOOD.replace("    pi = b1*pi{-1} + b2*y + shk_pi;", ""),
           linear=True, flat=True))

report("a log-variable that does not exist",
       lambda: Simultaneous.from_string(
           GOOD + "\n!log-variables\n    nosuch\n", linear=True, flat=True))
```

| message | real cause | fix |
|---|---|---|
| `These names are used in equations but not declared` | a typo in an equation, or a variable you meant to declare | declare it, or correct the spelling |
| `These names are declared multiple times` | the same name in two blocks, or twice in one | remove the duplicate |
| `Inconsistent numbers of variables and equations` | a transition equation deleted or commented out | the counts must match exactly, block by block |
| `Illegal name(s) on the log-variables list` | a name on `!log-variables` that is not a loggable variable | parameters and shocks cannot be logged |

The equation count is checked separately for the transition and the
measurement block, so a model can be short one transition equation while
the measurement block balances.

## Parameters and data

```{code-cell} ipython3
report("assign_strict with a misspelled name",
       lambda: good_model().assign_strict(a_one=0.5))

report("a number where a series is expected",
       lambda: good_model().simulate(
           Databox(y=3.0, pi=2.0, shk_y=0.0, shk_pi=0.0), SPAN))
```

`assign_strict` refuses a name the model does not have. `assign` accepts
it, which is the subject of the last section.

The databox validator reports the name and what was wrong with it. It
checks the **type** only, so a series of the right type containing nothing
but `nan` passes — tutorial 23 is about what happens next.

## The steady state

```{code-cell} ipython3
report("steady before the parameters are assigned",
       lambda: Simultaneous.from_string(GOOD, linear=False,
                                        flat=True).steady())

NO_SOLUTION = """
!variables
    x
!shocks
    shk_x
!equations
    exp(x) = -1 + shk_x;
"""
report("a steady state that does not exist",
       lambda: Simultaneous.from_string(NO_SOLUTION, linear=False,
                                        flat=True).steady())
```

`Non-finite values in these equations when evaluating steady state` means
a parameter is still unset: the equation was evaluated with `nan` in it.
`get_unassigned_parameters()` lists them.

`Steady state calculations failed to converge` is the solver's account of
events and not a diagnosis. It ran out of iterations, which happens both
when the answer is hard to find and when there is none — here `exp(x)` is
positive for every real `x`, so there is none. The message names the
variant, the block, the equations in that block and the variables it was
solving for, and on a large model that block listing is where to start.

## Solving

```{code-cell} ipython3
report("solving before the steady state",
       lambda: Simultaneous.from_string(GOOD, linear=True,
                                        flat=True).solve_first_order())
```

`array must not contain infs or NaNs` is NumPy complaining, one stage
downstream of the real problem: the steady state was never computed, so the
matrices were built around `nan`. Call `steady()` first.

One more belongs to this stage and does not reproduce on a model this
small. `Inconsistency in classification of unit roots; modify (increase)
the tolerance level` means two tests of the same roots disagreed at the
tolerance supplied. It appears when `tolerance` has been lowered by hand on
a model with roots close to the unit circle; tutorial 22 produces it and
explains what the setting does. The default suits nearly every model and
the message is the signal to put it back.

## Simulating

```{code-cell} ipython3
report("simulating before solving",
       lambda: Simultaneous.from_string(GOOD, linear=True, flat=True)
                           .simulate(Databox.steady(good_model(), SPAN), SPAN))

report("swap given two arguments instead of a pair",
       lambda: SteadyPlan(good_model()).swap("y", "a1"))
```

`'NoneType' object has no attribute 'cov_u'` is the unsolved model again:
`get_solution()` returned `None` and the simulator went looking inside it.
Any `AttributeError` mentioning `NoneType` during a simulation means
`solve_first_order()` was not called, or its result was thrown away.

`IndexError: string index out of range` from a plan means `swap` was given
two arguments where it wanted one pair. It reads the bare string `"y"` as
the pair and asks for its second element. The correct form is
`swap(("y", "a1"))`.

## The errors you do not get

These are worse than any message above, because the run completes and the
output has the right shape.

```{code-cell} ipython3
m = good_model()

print("1. assign accepts a name the model does not have")
m.assign(a_one=0.5, std_shk_DOES_NOT_EXIST=1.0)
print("   nothing raised")
print("   a_one in the model      :", "a_one" in m.get_parameters())
print("   the bogus std in the model:",
      "std_shk_DOES_NOT_EXIST" in m.get_stds())

print()
print("2. a parameter changed without re-solving")
first = np.asarray(m.get_solution().T).copy()
m.assign(a1=0.1)
print("   T unchanged after assign:",
      bool(np.allclose(first, np.asarray(m.get_solution().T))))
m.steady()
m.solve_first_order()
print("   T changed after re-solving:",
      not bool(np.allclose(first, np.asarray(m.get_solution().T))))

print()
print("3. a Series built from a list keeps the first value only")
from_list = Series(start=qq(2025,1), values=[1.0, 2.0, 3.0])
from_array = Series(start=qq(2025,1), values=np.array([1.0, 2.0, 3.0]))
print("   list :", np.asarray(from_list.get_data()).ravel())
print("   array:", np.asarray(from_array.get_data()).ravel())
```

```{code-cell} ipython3
clean = good_model()

print("4. a databox missing a variable returns nan, not an error")
shocks_only = Databox()
full = Databox.steady(clean, SPAN)
for name in full.get_names():
    if str(name).startswith("shk"):
        shocks_only[name] = full[name]

out = clean.simulate(shocks_only, SPAN)
print("   y =", np.asarray(out["y"][SPAN]).ravel())

print()
print("5. a model with an explosive root solves and simulates")
unstable = Simultaneous.from_string(GOOD, linear=True, flat=True)
unstable.assign(a1=1.4, b1=0.4, b2=0.3)
unstable.steady()
unstable.solve_first_order()
LONG = qq(2025,1) >> qq(2035,4)
db = Databox.steady(unstable, LONG)
db["shk_y"][qq(2026,1)] = 1.0
path = np.asarray(unstable.simulate(db, LONG)["y"][LONG]).ravel()
print("   largest eigenvalue modulus:",
      round(float(max(abs(np.asarray(unstable.get_eigenvalues())))), 4))
print("   y after eleven years      :", f"{float(path[-1]):.4e}")
```

| what happens | how to catch it |
|---|---|
| `assign` accepts any name, including `std_` names for shocks that do not exist, and discards it | use `assign_strict` for calibrations, and `get_unassigned_parameters()` before solving |
| the solution is a snapshot and does not follow the parameters | `steady()` and `solve_first_order()` after every change; tutorial 17 shows the objective function that returns the same number for every input |
| `Series(values=[…])` with a list keeps the first value | pass `np.array([…])` or a tuple |
| a missing variable becomes a row of `nan` and propagates | build the slate and look for all-`nan` rows; tutorial 23 has the helper |
| an explosive or Blanchard–Kahn-violating model solves and simulates | count the roots: `get_eigenvalues(kind=ip.UNSTABLE)` against the forward-looking variables; tutorial 9 |

The last one is the most dangerous in the list. A root of `1.4` is enough:
nothing raises, the simulation completes, and a single one-point shock has
grown to `5.0e+05` eleven years later. The model here has no leads, so it
is explosiveness rather than a Blanchard–Kahn failure, and tutorial 9 does
the root counting properly on a model that has both.

## ⚠️ Break it

**A missing semicolon, and the message that sends you to the wrong file.**

```{code-cell} ipython3
MISSING = GOOD.replace("+ shk_y;", "+ shk_y")

report("the file with one semicolon removed", lambda:
       Simultaneous.from_string(MISSING, linear=True, flat=True))

print("the two lines the parser read as one:")
for line in MISSING.splitlines():
    if "=" in line:
        print("   ", line.strip())
```

The complaint is about `shk_ypi`, a name that appears nowhere in the file.
Without the semicolon the two equations are one, the end of the first ran
into the start of the second, and `shk_y` and `pi` were read as a single
identifier.

The lesson generalises. **When a reported name is not in your file, you
have a separator problem, not a spelling problem** — a missing semicolon,
a missing comma in a declaration block, or a line break where the parser
expected none. Search for the two halves of the invented name instead of
the name itself; they are on consecutive lines.

## What you did

```{code-cell} ipython3
# the exception classes worth catching
from datapie.wrongdoings import Error, Critical

# the checks that turn a silent failure into a loud one
m = good_model()
m.get_unassigned_parameters()
m.check_steady(when_fails="silent")
m.get_eigenvalues()

# the strict versions of the forgiving calls
m.assign_strict(a1=0.7)
```

## Things to remember

1. **The bullets name the thing to fix.** The headline says which rule
   broke; the `⏐` list says where.
2. **`Critical` is a subclass of `Error`, both from `datapie`.** Catching
   `Error` catches both.
3. **A plain `ValueError` or `AttributeError` is one stage downstream.**
   `array must not contain infs or NaNs` and `'NoneType' has no attribute`
   both usually mean the model was never solved.
4. **A reported name that is not in your file is a separator problem.**
   Two names fused into one.
5. **`failed to converge` is a symptom, not a diagnosis.** No solver can
   tell an answer it could not find from one that was never there.
6. **`assign` accepts anything; `assign_strict` does not.** Calibrate with
   the strict one.
7. **The solution does not follow the parameters.** Re-solve after every
   change.
8. **The worst failures raise nothing.** A missing variable gives `nan`, a
   model with no stable solution gives numbers, and a list passed as
   `values` gives one period.

## Next

The messages here are collected out of the tutorials that explain them.
**Tutorial 9 · When it will not solve** covers the eigenvalue counting,
**tutorial 23 · Inside the data contract** the `nan` output, and
**tutorial 17 · Estimation** what a stale solution does to an optimiser.
