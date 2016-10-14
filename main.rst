Automatic Differentiation and Cosmic Initial Condition
======================================================

One project at Berkeley Center for Cosmological Physics is to study the 
recovery the cosmic initial condition from observations of the later time universe.

Cosmic initial condition is the density fluctuation of the universe about 13.7 billion years ago,
when the main form of energy in the universe was still dominated by the cosmic microwave background (CMB).

Due to the finite speed of light, any direct measurements of the CMB, 
including space based programs such as Planck, WMAP and COBE, and ground based programs such as ACT-Pole, PolarBear, 
can only observe a thin slice of cosmic initial condition.

For the rest of Universe, we are only able to observe an evolved state: the close to us, the older the Universe we observe
has grown to.
The recovery of the full cosmic initial condition is thus an inversion problem:

.. math::

    x = S^{-1}(y) .

where :math:`x` is the unknown initial condition, and :math:`y` is the observation. :math:`S` is the dynamical model
of the universe. At the resolution we can currently probe, :math:`S` is determined mostly by gravity. 


We must realize two difficulties of the problem:

1. The observation comes with noise; there is also uncertainty in the forward model :math:`S`. We therefore restate the problem
as an optimization problem,

.. math::

    \mathrm{ minimize}_{x = \hat{x}} \chi^2(x) = \left|\frac{S(x) - y}{\sigma}\right|^2

The solution :math:`x=\hat{x}` is our best estimate of the cosmic initial condition.

2. It has a very large dimensionality. :math:`x` and :math:`y` are fields defined on a 3-dimenional space.
For even moderate resolution, for example a mesh of :math:`128^3` points, the total number of elements become millions.

3. The model of structure formation, :math:`S` is nonlinear, and non-perturbative. We use
ordienary differential equation (ODE) solvers to follow the generative modelling function
:math:`S(x)`. A particular simple family of solvers are Particle-Mesh solvers, we refer the readers to the classical book
Computer Simulation Using Particles by Hockney and Eastwood for further references.

For an non-linear optimization problem of high dimension, we need to
use the gradient information of the objective function :math:`\chi^2(x)`. 

There are generic software tools to automatically evaluate the gradient of any function. We were hoping to use these
generic software tools in our problem,
and surveyed three packages, Tensorflow, Theono, and autograd. Unfortunately we find all three of them
lacking the elements to describe our ODE solver (particle mesh solver).

Therefore, in this blog entry, we will discuss and derive the missing pieces.
We will first make an attempt to define the useful gradient related operators in `Automatic Differentiation`,
before listing the two most relevant missing operators, discrete fourier transform and resampling that are used
in the Particle-Mesh solver we use.

Interested parties can implement them to
existing automatic differential packages to make them readily useful to a wider audience.

Automatic Differentiation
-------------------------

Automatic Differentiation (AD) has been the buzz word in recent years. 
It is a relatively new technology in astronomy: a few month ago at the 2016 AstroHackWeek
in Berkeley, the attendees organized a special session to explore the landscape of automatic differentiation software.
This blog was partially inspired by the discussion.

The recent popularity of AD is partially due to the movement of deep learning.
Training large neural networks demand effective and efficient optimization algorithms, because
the dimenionality of the problem (number of neural nodes) are huge.
Popular optimization algorithms (e.g. Gradient Descent -- the only embeded optimizer in TensorFlow, or L-BFGS which we use
in the cosmic initial condition problem) operate on the evaluation of the gradient.

A large landscape of AD beyond deep learning is in the context of inversion of dynamical systems.
Many physical problems can be written as solutions to time evolution of differential equations,
including, for example,  the evolution of the Universe, the atomsphere and ocean (weather / climate forecasting),
and space travelling (orbits).
The solution to these equations can be written as the product of a sequence of evolution operators (nested function evaluations).
Then AD can be applied to evaluate the gradients.
In these problems, AD is usually tailored to a specialized form that suits to the particular system.
A generic AD software is not used. (c.f. Sengupta et al. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120812/)

De-mysterifying AD
------------------

.. figure:: autodiff-func.svg
    :width: 50%
    :align: right

    Figure: Illustration of the evaluation sequences of automatic differentiation.

The basics of AD start from the chain-rule of differentiation, which claims that

If we have two functions :math:`y=f(x)` and :math:`z=g(y)=g(f(x))`, then

.. math::

    \frac{\partial z_j }{\partial x_i} = \sum_k \frac{\partial z_j}{\partial y_k} \frac{\partial y_k}{\partial x_i}
                        = \frac{\partial z_j}{\partial y_k} \cdot \frac{\partial y_k}{\partial x_i}

We see that the chain-rule converts gradient of nested functions to a sequence of tensor products.

Let's now consider a scalar that comes from nested evaluation of :math:`n` functions,

.. math::

    F(x) := \left(f^1 \odot \cdots \odot f^n \right)(x) = f^n(f^{n-1}(\cdots (f^1(x)) \cdots ))) .

:math:`f^i` maps to concepts in real world problems:

- as a time step in a dynamical system; then the nested functions are simply evolving the dyanmical system forward in time.

- as a layer in the neural network; then the nested functions are simply stacking layers of neural network.

We shall name the intemediate variables :math:`r^{(i)}`,

.. math::

    r^n = F(x)

    r^i = f^i(r^{i-1})

    r^0 = x

This function is illustrated in the `function evaluation` section of the figure.

Applying chain rule to :math:`\nabla F`, we find that

.. math::

    \nabla_j F = \frac{\partial F}{\partial r^0_j} = 
        \left[\Pi_{i=1, n} \frac{\partial f^i}{\partial r^{i-1}}\right]_j

where :math:`\Pi` represents tensor contractions on the corresponding dimension.
(known as the Einstein summation rule, c.f. `numpy.einsum`)

Automatic differentation software evaluates this expression for us.
The optimal evaluation is still a open question.

We will look at two popularly used schemes, the `reverse accumulation/backpropagation` scheme and
the `forward accumulation` scheme. Both are described in the Wikipedia entry of Automatic Differentiation.

Here will will motivate these schemes slightly differently, by defining two different types of functional operators.

Backward
++++++++

For a function `f` defined on the domain :math:`f : X \to Y`, we define gradient-adjoint-dot operator as

.. math::

    \Psi[f](v) = \sum_i v_i \frac{\partial f_i}{\partial x_j}

It is implied that :math:`v \in Y` and the domain of :math:`\Psi[f]` is :math:`\Psi[f] : Y \to X`.

Notice how the summation eliminate the indexing of the function; while the indexing for the gradient remains.

Using :math:`\Psi^i = \Psi[f^i]`, the chain-rule above can be reorganized as a sequence of function evaluations
of :math:`\Psi^i`

.. math::

    \nabla F_j = (\Psi^1 \cdots (\Psi^{n-1}(\nabla_j f^n))\cdots)_j

The process is illustrated in Section `backpropagation` of the figure. 
We see that at each evaluation of :math:`\Psi^i`, we
obtain the gradient of :math:`F` relative to the intermiedate variable :math:`r^i`, :math:`\nabla_{r^i} F`. Because we apply
:math:`\Psi^i` in the decreasing order of :math:`i`, 
this method is called the `backward propagation` or `reverse accumulation`.

This method is also called `adjoint method` in the analysis of dynamical systems, because the summation is along the `adjoint`
index of the jacobian :math:`\frac{\partial f_i}{\partial x_j}`.
The main drawback of backpropagation is
that it requires one to store the intemediate results of along the line in order to compute the gradient-adjoint-dot operator.
However, the method gives the full gradient against the free variables `x_j` after one full accumulation, making it at advantage
in certain problems than the `forward accumulation` we will describe next.

Most popular automatic differentiation software packages (TensorFlow, Theono, or autograd) implements the
gradient-adjoint-dot operator as the gradient element of supported functions.


Forward
+++++++

In contrast, we can define an gradient-dot operator,

.. math::

    \Gamma[f](u) = \sum_j \frac{\partial f_i}{\partial x_j} u_{j}.

It is implied that :math:`u \in X` and the domain of :math:`\Gamma[f]` is :math:`\Gamma[f] : X \to Y`.

Notice the summation is over the indexing of the free variable, :math:`x_j`. Hence the name does not have `adjoint` like the previous
operator. One way to think of :math:`\Gamma[f]` is that it rotates :math:`u` by the jacobian.

With the gradient-dot operator of :math:`\Gamma^i = \Gamma[f^i]`, we can write down the `forward accumulation` rule of AD:

.. math::

    \sum_j \nabla_j F u_j = \Gamma^n (\cdots (\Gamma^1(u)) \cdots)

This process is illustrated in the `Forward accumulation` section of the figure.
We see that at each evaluation of :math:`\Gamma^i`, we obtain the directional
derivative of :math:`r^i` along :math:`u`, :math:`\sum \frac{\partial r^i}{\partial x_j} u_j`. The accumulation goes along the increasing
order of :math:`i`, making the name `forward accumulation` a suitable one.

The advantage of forward accumulation is that one can evaluate the gradient as the function :math:`F` is evaluated, and no intemediate
results need to be saved. This is clearly a useful feature when the number of nesting (layers of neural network or number of time steps)
is high.
However, the cost is we can only obtained a directional derivative. In some applications it is useful (e.g. computing Hession for Newton-CG or trust-region
Newton-CG methods). When the full gradient is desired, one need to run
the `forward accumulation` many times - as many as the number of the free parameters, which could be prohibatively high.

We shall note that this method is also called `forward senstivity` in the analysis of dynamical systems.

Two Useful Operators in Particle-Mesh solvers
---------------------------------------------

In this section we write down two families of gradient-adjoint-dot operators that are useful in AD of cosmological simulations.
The first family is the Discrete Fourier transforms. The second family is the resampling windows. At the time of this blog,
no popular AD software implement all of these gradient-adjoint-dot operators. We will list them in this section for further 
references.

Discrete Fourier Transform
++++++++++++++++++++++++++

Discrete Fourier transform is the discretized version of Fourier Transform.
It is a commonly used density matrix operator in the modelling of physical process.
This is mostly because finite differentiation can be written as multiplication
in the spectrum space.

The gradients involve complex numbers which are tuples of two real numbes. We therefore do not include a proof
in this blog. The gradient that is conveniently used is

.. math::

    \nabla_z = \frac{\partial}{\partial x} + \imath \frac{\partial}{\partial y}

for :math:`z = x + \imath y`. It is related to the Wirtinger derivatives (Fourier transform is a harmonic function).

The gradient-adjoint-dot operator of a discrete fourier transform
is its dual transform. Specifically,

.. math::

    \Psi[\mathrm{fft}](V) = \mathrm{ifft}(V)

    \Psi[\mathrm{rfft}](V) = \mathrm{irfft}(V)

    \Psi[\mathrm{ifft}](V) = \mathrm{fft}(V)

    \Psi[\mathrm{irfft}](V)_j = \left\{
                \begin{matrix}
                        \mathrm{rfft}(V)_j & \mathrm{ if } j = N - j, \\
                            2 \mathrm{rfft}(V) & \mathrm{ if } j \neq N - j.
                \end{matrix} \right.


where :math:`\Psi` is the gradient-adjoint-dot operator. Notably, the free variable `X` do not show up in the 
final expressions. This is because Fourier transforms are linear operators. We also notice that the gradient of
complex to real transform has an additional factor of 2 for most modes.
This is because the hermitian conjugate frequency mode also contributes to the gradient.

The complex version of Discrete Fourier Transform is implemented in TensorFlow (GPU only), Theono, and autograd. Though
it appears the version in autograd is incorrect. The real-complex transforms 
are not implemented in any of the packages.

Resampling Windows
++++++++++++++++++

The resampling window converts a field representation between particles and meshes.
It is written as

.. math::

    B_j(p, q, A) = \sum_i W(p^i, q^j) A_i

where :math:`p^i` is the position of `i`-th particle/grid point and :math:`q^j` is the position
of `j`-th particle/grid point; both are usually vectors themselves (the universe has 3 spatial dimensions).
:math:`W` is the resampling window function. A popular form is the
cloud in cell window, which represents a linear interpolation:

.. math::

    W(x, y) = \Pi_{a} (1 - h^{-1}\left|x_a - y_a\right|)

for a given size of the window :math:`h`.

Most windows are seperatable, which means they can be written as a product of
a scalar function :math:`W_1`,

.. math::

    W(x, y) = \Pi_{a} W_1(\left|x_a - y_a\right|),

For these windows,

.. math::

    \frac{\partial W}{\partial x_a} = \frac{\partial W}{\partial y_a} = 
    W_1^\prime(\left|x_a - y_a\right|) \Pi_{b \neq a} W1(\left|x_b - y_b\right|) 

We can then write down the gradient-adjoint-dot operator of the window

.. math::

    \Psi[B, p](v)_{(i,a)} = \sum_j \frac{\partial W(p^i, q^j)}{\partial p^i_a} A_i v_j

    \Psi[B, q](v)_{(j,a)} = \sum_i \frac{\partial W(p^i, q^j)}{\partial q^j_a} A_i v_j

    \Psi[B, A](v)_i =  \sum_j W(p^i - q^j) v_j

The first gradient corresponds to the displacement of the source. The second gradient corresponds to
the displacment of the destination. The third gradient corresponds to the evolution of the field.
Usually in a particle mesh simulation, either one of the source and the destination is a fixed grid, and
the corresponding gradient vanishes.

They are a bit complicated because we need to loop of the spatial dimension index :math:`a`.

Unlike the partial support of Fourier Transforms, none of the three packages we surveyed
(TensorFlow, Theono and autograd) recognizes these resampling window operators.
We are still a bit away from being able to implement our problem on top of existing generic AD software packages.

