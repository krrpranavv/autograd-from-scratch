# Adding `sin` to all three engines

Try the exercise yourself first (Exercise 3 in the guide ([GUIDE.md](../GUIDE.md))). This file is the
worked answer to check against, not a substitute for doing it.

The three facts you need: the value is $\sin x$, the first derivative is
$\cos x$, and the second derivative is $-\sin x$. Each engine encodes those
same facts in its own form.

## 1. `autograd/engine.py`: reverse mode

Add this method to `Tensor`, next to the other elementwise ops (after `tanh`,
before `gelu`):

```python
def sin(self):
    out = Tensor(np.sin(self.data), (self,), "sin")

    def _backward():
        self.grad += out.grad * np.cos(self.data)

    out._backward = _backward
    return out
```

The closure multiplies the incoming gradient by the local derivative
$\cos x$ and accumulates with `+=`, like every other op. No `_unbroadcast`
is needed because the shape never changes.

## 2. `autograd/dual.py`: forward mode

Add this method to `Dual`, after `tanh`:

```python
def sin(self):
    p = self.primal
    val = p.sin() if isinstance(p, Tensor) else np.sin(p)
    c = np.cos(p.data if isinstance(p, Tensor) else p)
    return Dual(val, c * self.tangent)
```

The tangent rule is the same chain rule pushed forward: tangent out equals
$\cos(x)$ times tangent in. The `isinstance` branches follow the pattern of
`Dual.relu`: they let the primal be a reverse-mode `Tensor`. One note: the
cosine here is a plain array, so first-order results (jvp, vjp, the adjoint
identity) are exact, but a forward-over-reverse Hessian-vector product
through `sin` would treat $\cos x$ as a constant and miss its derivative.
Making `hvp` exact through `sin` needs a `Tensor.cos` and a `_sin`/`_cos`
routing helper like `_tanh`.

## 3. `autograd/secondorder.py`: order-2 duals

Add this method to `Dual2`, next to `tanh` and `relu`:

```python
def sin(self):
    return self._unary(np.sin(self.primal), np.cos(self.primal), -np.sin(self.primal))
```

`_unary(val, d1, d2)` takes $g(x)$, $g'(x)$, $g''(x)$ and applies the
twice-differentiated chain rule for you.

## Checking it

The repo gives you three oracles. All of the snippets below were run against
patched copies of the three files; the printed numbers are from that run.

Finite differences on the reverse-mode gradient (for $f = \sum \sin x_i$ the
analytic gradient is $\cos x$):

```python
import numpy as np
from autograd.engine import Tensor

rng = np.random.default_rng(0)
a = rng.standard_normal((3, 4))
x = Tensor(a.copy())
x.sin().backward()                      # seed of ones: gradient of the sum
assert np.allclose(x.grad, np.cos(a))
eps, num = 1e-6, np.zeros_like(a)
it = np.nditer(a, flags=["multi_index"])
for _ in it:
    i = it.multi_index
    bp, bm = a.copy(), a.copy()
    bp[i] += eps
    bm[i] -= eps
    num[i] = (np.sin(bp).sum() - np.sin(bm).sum()) / (2 * eps)
assert np.allclose(x.grad, num, atol=1e-5)   # max diff observed: 1.5e-10
```

The adjoint identity $\langle u, Jv \rangle = \langle J^\top u, v \rangle$,
which exercises the `Tensor` and `Dual` versions against each other:

```python
from autograd.dual import adjoint_gap

W = rng.standard_normal((4, 3))

def f(z):
    return (z @ W).sin()                 # runs on Dual and on Tensor

for _ in range(20):
    x = rng.standard_normal((2, 4))
    u = rng.standard_normal((2, 3))
    v = rng.standard_normal((2, 4))
    assert adjoint_gap(f, x, u, v) < 1e-10   # worst gap observed: 1.8e-15
```

Curvature from `Dual2` (for $f = \sum \sin x_i$ the Hessian is
$\mathrm{diag}(-\sin x)$, so $v^\top H v = -\sum_i \sin(x_i)\, v_i^2$):

```python
from autograd.secondorder import directional_curvature

aa, vv = rng.standard_normal(5), rng.standard_normal(5)
t2, t1 = directional_curvature(lambda z: z.sin().sum(), aa, vv)
assert np.isclose(t1, (np.cos(aa) * vv).sum())            # error observed: 0.0
assert np.isclose(t2, -(np.sin(aa) * vv * vv).sum())      # error observed: 0.0
```

For a fourth oracle, `tests/test_engine.py` shows the pattern for checking an
op against `torch` (`_check_against_torch` with `lambda x: x.sin().sum()` on
both sides), and `tests/test_dual.py::test_dual_and_tensor_expose_the_same_ops`
has the op set you can add `"sin"` to so the engines cannot drift apart.
