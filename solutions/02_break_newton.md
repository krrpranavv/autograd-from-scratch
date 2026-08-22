# Breaking Newton on a saddle

Try the exercise yourself first (Exercise 4 in the guide ([GUIDE.md](../GUIDE.md))). This file is the
worked answer to check against.

## What happens

Run `newton_minimize` on $f(x) = x_0^2 - x_1^2$:

```python
import numpy as np
from autograd.secondorder import newton_minimize, hessian

def f(x):
    return x[0] ** 2 - x[1] ** 2

for start in ([1.0, 1.0], [1.2, 0.9], [-3.0, 5.0]):
    x, hist = newton_minimize(f, start, steps=4)
    print(start, "->", hist[0][0])

print(hessian(f, [1.0, 1.0]))
```

Output from running this against the repo's `autograd/secondorder.py`:

```
[1.0, 1.0]  -> [0. 0.]
[1.2, 0.9]  -> [0. 0.]
[-3.0, 5.0] -> [0. 0.]
[[ 2.  0.]
 [ 0. -2.]]
```

From every start, one step lands exactly on $(0, 0)$ and stays there. But
$(0,0)$ is not a minimum of $f$; it is a saddle ($f$ decreases without bound
along $x_1$).

## Why, from the Hessian

The gradient is $\nabla f = (2x_0,\, -2x_1)$ and the Hessian is constant:

$$H = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}$$

The Newton step is $x \leftarrow x - H^{-1} \nabla f$. Here
$H^{-1}\nabla f = (x_0, x_1) = x$, so the update is $x \leftarrow x - x = 0$
from any starting point. The quadratic is its own second-order model, so one
step solves $\nabla f = 0$ exactly.

That is the failure: Newton solves for a critical point, not a minimum. Along
$x_0$ (eigenvalue $+2$) it steps downhill; along $x_1$ (eigenvalue $-2$) the
negative curvature flips the sign of the step, so it walks uphill toward the
saddle. Wherever $H$ is indefinite, $-H^{-1}\nabla f$ is not a descent
direction, and `newton_minimize`'s docstring says as much. This is why
production optimizers do not use raw Newton.

## The fix: damp the Hessian

Replace $H$ with $H + \mu I$, with $\mu$ large enough that the matrix is
positive definite. Then the step is a descent direction again: as
$\mu \to \infty$ it turns into a small gradient-descent step, and as
$\mu \to 0$ (where $H$ is already positive definite) it returns to Newton.

```python
from autograd.engine import Tensor
from autograd.secondorder import gradient, hessian

def newton_damped(f, x0, steps=8, mu=1.0):
    x = np.asarray(x0, np.float64).copy()
    hist = []
    for _ in range(steps):
        g = gradient(f, x)
        H = hessian(f, x)
        lam_min = np.linalg.eigvalsh(H).min()
        shift = max(0.0, -lam_min) + mu     # make H + shift*I positive definite
        x = x - np.linalg.solve(H + shift * np.eye(x.size), g)
        hist.append((x.copy(), float(f(Tensor(x)).data)))
    return x, hist
```

Running it on the same $f$ from $[1, 1]$ (output from the same session):

```
step 1: x = [+0.6000, +3.0000], f = -8.6400
step 2: x = [+0.3600, +9.0000], f = -80.8704
step 3: x = [+0.2160, +27.0000], f = -728.9533
...
```

Now every step decreases $f$: $x_0$ contracts toward 0 and $x_1$ grows,
following the downhill direction the saddle offers. On this particular $f$
there is no minimum to converge to ($f$ is unbounded below), so descending
forever is the correct behavior; on a function that does have a minimum, $H$
becomes positive definite near it, the shift stops being needed, and the fast
Newton convergence comes back.

Two related mitigations in the wild: trust-region methods, which bound the
step instead of shifting the Hessian (Levenberg-Marquardt sits between the
two), and the truncated-CG approach in this repo's `autograd/hvp.py` (`newton_cg`
stops the inner solve when it detects negative curvature, see `_cg`).
