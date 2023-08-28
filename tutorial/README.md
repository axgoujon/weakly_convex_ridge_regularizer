Solving an Inverse Problem
--------------
Given a forward operator $\mathbf{H}$ and some measurements $\mathbf{y}$, the WCRR-NN is plugged into the variational problem

<img src="https://latex.codecogs.com/svg.image?\mathrm{argmin}_{\mathbf{x}}&space;\frac{1}{2}\|\mathbf{H}\mathbf{x}&space;-&space;\mathbf{y}\|_2^2&space;&plus;\lambda&space;R(\mathbf{x}, \sigma)." />

The regularizer $R$ is smooth, and the (non necessarily convex) optimization problem can be solved with gradient-based methods, e.g. the safe-guarded accelerated gradient-descent discussed in the preprint, which guarantees convergence to a critical point. 

For these algorithms, the following functions are useful:
- `model.conv_layer.spectral_norm(mode="power_method", n_steps=500)` to update the spectral-norm of the convolution layer cached. **Once this function has been called, the convolution layer is automatically normalized to have unit norm**.
Nb: given $\lambda$, the Lipschitz bound should on the regularizer should be `lmbd * model.get_mu()`, and it does not depend on $\sigma$.
- `model.grad(x, sigma)` to compute $\nabla R(\mathbf{x}, \sigma)$.
Nb: given $\lambda$, the gradient actually used is `lmbd * model.grad(x)`, i.e. $\lambda$ is not included into the model.

In addition,
- `model.cost(x, sigma)` gives the regularization cost. Again, $\lambda$ needs to be manually added as in `lmbd * model.cost(x, sigma)`.

Tuning $\lambda$ and $\sigma$
--------------
The tuning can be done with the coarse to fine method given for CRR-NNs in [paper](https://ieeexplore.ieee.org/abstract/document/10223264).