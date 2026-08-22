# Solutions

Worked answers for the exercises in the guide ([GUIDE.md](../GUIDE.md)). Try each exercise before
opening its file; the point of the exercises is the attempt, and every one of
them comes with an oracle (finite differences, the adjoint identity, the
existing tests) that tells you on its own whether your answer is right.

| File | Exercise |
|------|----------|
| `01_add_sin.md` | Add `sin` to `Tensor`, `Dual`, and `Dual2`, with the checks |
| `02_break_newton.md` | Run Newton into a saddle, explain it, damp it |
| `03_zero_grad.md` | The double-backward demo and why `+=` forces `zero_grad` |

## Warm-up: gradients of f = tanh(a*b + c) by hand

For $a = 2$, $b = -3$, $c = 10$: the inner value is $ab + c = 4$ and
$f = \tanh(4) = 0.999329$. With $t = \tanh(4)$, the local derivative is
$1 - t^2 = 0.001341$, and the chain rule gives

$$\frac{\partial f}{\partial c} = 1 - t^2 = 0.00134, \qquad
\frac{\partial f}{\partial a} = b\,(1 - t^2) = -0.00402, \qquad
\frac{\partial f}{\partial b} = a\,(1 - t^2) = 0.00268.$$

Verified against the engine (this also reproduces the `autograd/viz.py` example graph):

```python
from autograd.engine import Tensor
a, b, c = Tensor(2.0), Tensor(-3.0), Tensor(10.0)
f = (a * b + c).tanh()
f.backward()
print(a.grad, b.grad, c.grad)   # -0.004023  0.002682  0.001341
```

## Exercise 1: gradients of g = relu(a*b + a) by hand

For $a = 2$, $b = -3$: the inner value is $ab + a = -6 + 2 = -4$, so the relu
is inactive ($g = 0$) and its local derivative is 0. Every upstream gradient
is multiplied by that 0, so all three gradients are 0, even though $a$ feeds
the expression twice. That is the instructive part: the two-path accumulation
for $a$ (which would give $b + 1 = -2$ inside the relu) is real, but a dead
relu kills the whole path.

```python
from autograd.engine import Tensor
a, b = Tensor(2.0), Tensor(-3.0)
g = (a * b + a).relu()
g.backward()
print(a.grad, b.grad)   # 0.0  0.0
```

Change $a$ to a value that keeps the relu active (try $a = 2$, $b = 1$: inner
value 4) and the same two-path reasoning gives
$\partial g/\partial a = b + 1 = 2$ and $\partial g/\partial b = a = 2$,
which `backward()` confirms.

## Sketches for the open exercises

These stay open; the notes below are starting points, not answers.

Condition-number sweep (guide Exercise 6). Use the quadratic
$f(x) = \tfrac12 (x_0^2 + \kappa x_1^2)$ and sweep
$\kappa \in \{1, 10, 10^2, 10^3, 10^4\}$. Newton solves it in one step at any
$\kappa$; gradient descent with the best fixed step converges at rate
$(\kappa - 1)/(\kappa + 1)$, so steps to a fixed accuracy grow roughly
linearly in $\kappa$. Count both and plot. For the size question, note that
`hessian()` in `autograd/secondorder.py` costs $n(n+1)/2$ curvature evaluations plus
an $O(n^3)$ solve; time it against the iteration count gradient descent
needs as $n$ grows.

Hv on the GPT (guide Exercise 5). `examples/landscape.py` is the template: it
flattens the MLP's parameters into one vector, defines `loss(vector)` by
unflattening into the model, and feeds that to `hvp` / `top_eigenvalue`. Do
the same with `examples/train_gpt.py`'s loss. Each `Hv` is one forward and one
backward over the whole model, so shrink the model and the number of power
iterations first, then scale up. The 2D version walks the loss over the span
of the top two eigenvectors (get the second by power-iterating on
$H v - \lambda_1 (v_1^\top v) v_1$, the deflated operator).

Op-count benchmark (guide Exercise 7). Wall clock depends on the machine;
op counts do not. Add a module-level counter to `autograd/engine.py` and `autograd/dual.py`
(increment once per op call), reset it, run `jacobian_forward` and
`jacobian_reverse` from `autograd/dual.py`, and read it back. Forward mode should
scale with the input count $n$ (one pass per column), reverse with the
output count $m$ (one pass per row), crossing at $n = m$, which is what
`examples/benchmark.py` measures in seconds.
