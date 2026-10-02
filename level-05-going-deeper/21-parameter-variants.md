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

# 21 · Parameter variants

**40 minutes** · after *20 · Vector autoregressions* · next: *22 · Inside
the solution*

Sensitivity analysis is usually written as a loop: copy the model, change a
parameter, solve, simulate, store the result, repeat. IrisPie offers an
alternative. A single model object can hold several parameterisations at
once, each with its own steady state and its own solution, and the ordinary
methods operate on all of them in one call.

These are called *variants*. Tutorials 18 and 20 used them to carry Monte
Carlo draws; this tutorial uses them for their original purpose, and
explains the behaviour that every other method in the package inherits from
them.

```{code-cell} ipython3
import itertools

import numpy as np
from irispie import Simultaneous, Databox, Series, qq

SOURCE = """

!transition-variables
    y, pi, i, q, y_w, pi_w, i_w

!transition-shocks
    shk_y, shk_pi, shk_i, shk_q, shk_y_w, shk_pi_w, shk_i_w

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

"""

CALIB = dict(a1=0.7, a2=0.2, a3=0.1, a4=0.3, b1=0.1, b2=0.3, b3=0.2,
             c1=0.5, c2=1.5, c3=0.5, pi_tar=2, r_ss=1, beta=0.99, psi=0.1,
             d1=0.8, d2=0.7, d3=0.8, pi_w_ss=2, r_w_ss=1)

base = Simultaneous.from_string(SOURCE, linear=True, flat=True)
base.assign_strict(**CALIB)
base.steady()
base.solve_first_order()

SPAN = qq(2025,1) >> qq(2027,4)

print("variants:", base.num_variants, "  singleton:", base.is_singleton)
```

## Creating variants

`expand_num_variants` sets the number of parameterisations the object
carries. Each is a complete copy of the parameter values, and a parameter
assigned a list is distributed across them:

```{code-cell} ipython3
policy = base.copy()
policy.expand_num_variants(3)
policy.assign(c2=[1.2, 1.5, 2.5])

policy.steady()
policy.solve_first_order()

print("variants  :", policy.num_variants)
print("singleton :", policy.is_singleton)
print("c2        :", policy.get_parameters()["c2"])
print("a1        :", policy.get_parameters()["a1"])
```

Three policy rules: one that responds weakly to inflation, the original, and
one that responds aggressively. `a1` was not given a list, so all three
variants share its value.

`steady` and `solve_first_order` were each called once and applied to all
three. There is no loop anywhere, and there is no separate model object for
each case.

## Each variant is a complete model

A variant carries its own steady state, not only its own parameters. `c2`
does not enter the steady state, so the three agree; a parameter that does
enter it produces three different long runs:

```{code-cell} ipython3
targets = base.copy()
targets.expand_num_variants(3)
targets.assign(pi_tar=[1.0, 2.0, 3.0])
targets.steady()
targets.solve_first_order()

for name in ("pi", "i"):
    print(f"  steady {name}:",
          [round(float(x), 4) for x in targets.get_steady_levels()[name]])
```

Three inflation targets give three steady inflation rates and three steady
nominal rates, each one point apart, which is the Fisher relation holding in
each variant separately.

## Running every variant in one call

```{code-cell} ipython3
db = Databox.steady(policy, SPAN)
db["shk_y"][qq(2025,1)] = 1.0

out = policy.simulate(db, SPAN)

print("output series variants:", out["y"].num_variants)
print("data shape            :", out["y"].get_data().shape)
print()
header = "".join(f"{f'c2 = {value}':>12}"
                 for value in policy.get_parameters()["c2"])
print(f"{'on impact':>12}" + header)
for name in ("y", "pi", "i"):
    row = out[name].get_data()[1, :]
    print(f"{name:>12}" + "".join(f"{value:12.4f}" for value in row))
```

One `simulate` call, three columns of output. The same demand shock raises
output by `0.9209` under the weak rule and `0.8688` under the aggressive
one, because a central bank that responds harder to inflation damps more of
the disturbance. Inflation on impact falls from `2.4332` to `2.3243` across
the three.

The input databox had one variant and the model had three. `Databox.steady`
read the variant count from the model, which is why the shock applied to all
of them.

## The return shape depends on the variant count

This is the behaviour to understand before using variants anywhere else,
because every method in the package has it.

```{code-cell} ipython3
single = base.copy()
single.steady()
single.solve_first_order()

print("get_eigenvalues on one variant   :",
      type(single.get_eigenvalues()).__name__,
      "of length", len(single.get_eigenvalues()))
print("get_eigenvalues on three variants:",
      type(policy.get_eigenvalues()).__name__,
      "of length", len(policy.get_eigenvalues()))
```

The same call returns ten eigenvalues from the singleton and three entries
from the three-variant model, one entry per variant. The length means
something different in each case.

This is `unpack_singleton`, which defaults to `True`: a model with one
variant returns the result itself, and a model with several returns a list.
Every method that can produce per-variant output takes the argument, and
setting it to `False` makes the shape constant — a singleton then returns a
one-element list like any other model.

Code that will meet both cases should pass `unpack_singleton=False` and
index unconditionally. The failure arises when a function written for a
singleton is later applied to a multi-variant model, which is the subject of
the Break it below.

## Iterating over variants

There are two iterators and they do different things. Only one of them
stops.

```{code-cell} ipython3
print("iter_own_variants:")
for position, variant in enumerate(policy.iter_own_variants()):
    print(f"  item {position}: c2 = {variant.get_parameters()['c2']}"
          f"   variants in it: {variant.num_variants}")

print()
print("iter_variants, first seven items:")
for position, variant in enumerate(itertools.islice(policy.iter_variants(), 7)):
    print(f"  item {position}: c2 = {variant.get_parameters()['c2']}")
```

`iter_own_variants` yields each variant once, as a singleton model, and
stops. It is the one to use.

`iter_variants` yields each variant and then **repeats the last one for
ever**. It is a broadcasting helper, written so that a one-variant model can
be zipped against many data variants without the caller checking lengths,
and it is used that way inside the simulation code. Called directly in a
`for` loop it never terminates, and `list(model.iter_variants())` does not
return.

## Selecting and matching variants

```{code-cell} ipython3
pair = policy.copy()
pair.select_variants(indices=[0, 2])
print("select_variants([0, 2]) :", pair.num_variants,
      pair.get_parameters()["c2"])

one_of_them = policy.get_variant(1)
print("get_variant(1)          :", one_of_them.num_variants,
      one_of_them.get_parameters()["c2"],
      " singleton:", one_of_them.is_singleton)

other = base.copy()
other.broadcast_variants(policy)
print("broadcast_variants      :", other.num_variants,
      other.get_parameters()["c2"])
```

`select_variants` keeps a subset in place. `get_variant` extracts one as a
new singleton model, which is the usual way to pull out the case worth
reporting. `broadcast_variants` matches the variant *count* of another
object by copying the existing values, which is how two objects are brought
to the same shape before being used together; it does not copy the other
object's parameter values.

## ⚠️ Break it

**A helper written for one variant, used on several.**

Tutorial 9 counted forward-looking variables by subtracting the number of
transition variables from the number of eigenvalues:

```{code-cell} ipython3
def count_forward_looking(model):
    """How many expectations the solved system carries."""
    return (len(model.get_eigenvalues())
            - len(model.get_solution_vectors().transition_variables))


print("on one variant   :", count_forward_looking(single))
print("on three variants:", count_forward_looking(policy))
```

The function is correct, and on the three-variant model it returns `-4`.

`len(model.get_eigenvalues())` is ten on the singleton, because the list
holds eigenvalues. On the three-variant model it is three, because the list
now holds one entry per variant and the eigenvalues are one level further
down. The subtraction then compares a variant count with a variable count
and produces a number that means nothing.

Nothing raises. A negative count of forward-looking variables would be
caught by a reader, but the same mistake with three variants and ten
transition variables could as easily have produced a plausible positive
number.

The repair is to decide which object the function is for:

```{code-cell} ipython3
def count_forward_looking_safely(model):
    """Works on any variant count; returns one number per variant."""
    eigenvalues = model.get_eigenvalues(unpack_singleton=False)
    num_transition = len(model.get_solution_vectors().transition_variables)
    return [len(group) - num_transition for group in eigenvalues]


print("on one variant   :", count_forward_looking_safely(single))
print("on three variants:", count_forward_looking_safely(policy))
```

`unpack_singleton=False` makes the outer list always mean variants, so the
inner `len` always means eigenvalues, and the function returns one answer
per variant in both cases.

## What you did

```{code-cell} ipython3
# several parameterisations in one object
model = base.copy()
model.expand_num_variants(3)
model.assign(c2=[1.2, 1.5, 2.5])

# solved and simulated once, for all of them
model.steady()
model.solve_first_order()
model.simulate(db, SPAN)

# per-variant results
model.get_parameters()["c2"]
model.get_steady_levels()["i"]
model.get_eigenvalues(unpack_singleton=False)

# iterate, select, extract
list(model.iter_own_variants())
model.select_variants(indices=[0, 2])
model.get_variant(1)
```

## Things to remember

1. **One object can hold many parameterisations.** `expand_num_variants`
   sets the count and a list passed to `assign` is distributed across them.
2. **Every variant has its own steady state and solution.** `steady` and
   `solve_first_order` are called once and apply to all.
3. **`simulate` returns one column per variant**, from a single call.
4. **`unpack_singleton` changes the return shape.** A singleton returns the
   result; several variants return a list of results.
5. **Pass `unpack_singleton=False` in any function that may meet both**, so
   that the outer level always means variants.
6. **Use `iter_own_variants`.** `iter_variants` repeats the last variant for
   ever and will not terminate in a `for` loop.
7. **`get_variant` extracts a singleton; `select_variants` keeps a subset in
   place; `broadcast_variants` matches a count**, not a set of values.

## Exercise

A model has been expanded to three variants. Assign `c2` a list of two
values, then a list of four, and record what the model ends up holding in
each case.

Before running it, predict whether either raises.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**Neither raises. The short list is padded and the long list is truncated.**

| assigned | `c2` afterwards |
|---|---|
| `[1.2, 1.5]` | `[1.2, 1.5, 1.5]` |
| `[1.2, 1.5, 2.5, 3.0]` | `[1.2, 1.5, 2.5]` |

Two values for three variants gives the third variant a copy of the second.
Four values for three variants discards the fourth. Both are silent, and
both produce a model that is internally consistent and not the one intended.

The padding is the same convention as `iter_variants`: a shorter sequence is
extended by repeating its last element. It is deliberate, and it is what
makes `assign(c2=1.5)` work on a model of any size, since a scalar is the
limiting case of a short list.

The consequence is that a list whose length is derived from something else
in the program — a grid of calibrations, a set of estimation results, a
column read from a file — must have its length checked against
`num_variants` before it is assigned. The check is one line and the failure
is otherwise invisible:

| check | result |
|---|---|
| `len(values) == model.num_variants` | `False` for both cases above |

</details>

## Next

**Tutorial 22 · Inside the solution** opens the first-order solution itself.
`systemize` returns the unsolved system, the solution matrices are
decomposed into their stable and unstable parts, and terminal conditions and
`clip_small` determine what is kept and what is discarded along the way.
