# Learning Weakly Convex Regularizers for Convergent Image-Reconstruction Algorithms

Implementation of the WCRR-NN as presented in this [preprint](https://arxiv.org/abs/2308.10542). For any bug report/question/help [contact us](mailto:alexis.goujon@epfl.ch) or create an issue.
## WCRR-NN in short
The WCRR framework provides a method to solve imaging inverse problems with an explicit regularizer $R$ that
1. is **data-driven** (trained on denoising, by default with BSD images and deep-equilibrium ~ bilevel optimization),
2. is **interpretable** (sparsity-promoting regularizer and ~13k parameters => not a black box),
3. is **$1$-weakly convex**, i.e. $R + 1/2\|\cdot\|^2$ is convex,
4. has **Lipschitz continuous gradients**.

The inverse problem is converted into the optimization problem

<img src="https://latex.codecogs.com/svg.image?\mathrm{argmin}_{\mathbf{x}}&space;\frac{1}{2}\|\mathbf{H}\mathbf{x}&space;-&space;\mathbf{y}\|_2^2&space;&plus;\lambda&space;R(\mathbf{x}, \sigma)," />

where $\mathbf{H}$ is the forward operator. Because of 4., the problem is solved with gradient-based algorithms with two different approaches
1. $\sqrt{\lambda_{\min}(\mathbf{H}^T \mathbf{H})} \geq \lambda$: the problem is convex, **global minimization** is possible with accelerated gradient descent for instance. E.g. in denoising with $\lambda=1$.
2. Otherwise: **convergence** is guaranteed, but only to **critical points**, with SAGD (see paper) for instance. E.g. ill-posed problems.


WCRR extends their convex counter-parts CRR ([repo](https://github.com/axgoujon/convex_ridge_regularizers) by learning a less constrained regularizer (not convex). It also differs in the training (deep-equilibrium + share activations + multi-noise + no regularization on the splines). For completeness this repository also includes a convex model trained similar to the WCRR-NN. 
#

The repository is organized as follows:
- **trained_models:** contains a pretrained WCRR-NN and a CRR-NN.
- **tutorial:** to understand and deploy the trained WCRR-NN for solving inverse problems.
- **denoising:** to test the WCRR-NN on denoising.
- **training:** to reproduce training and extend it to other image domains.
- **under_the_hood:** to visualize the WCRR-NN (filters and activations).
- **models:** contains the spline modules, the WCRR-NN class, some optimization schemes...

<span style="color:green;">**The use of WCRR-NN to solve inverse problems with SAGD is illustrated in the CRR-NN github repository [here](https://github.com/axgoujon/convex_ridge_regularizers).**</span>


Some requirements (depending on what you use)
--------------
* python >= 3.8
* pytorch >= 1.12
* (optional) CUDA
* numpy
* pandas
* tensorboard
* matplotlib
* Pillow