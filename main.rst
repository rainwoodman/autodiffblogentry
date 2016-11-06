..  Comment To be published at BIDS blog.
..  Build with
..    rst2html --math-output=mathjax main.rst > main.html

Automatic Differentiation and Cosmology Simulation
==================================================

One project at `Berkeley Center for Cosmological Physics <http://bccp.berkley.edu>`_ studies the 
reconstruction of the cosmic initial condition based on observations of the later-time universe.

The cosmic initial condition is the density fluctuation of the universe about 13.7 billion years ago,
when the main form of energy in the universe mainly consisted of cosmic microwave background (CMB).
Due to the finite speed of light, any direct measurements of the `CMB <https://en.wikipedia.org/wiki/Cosmic_microwave_background>`_, 
including space-based programs, such as Planck, WMAP, and COBE, and ground-based programs, such as ACT-Pole and PolarBear, 
can only observe a thin slice of the cosmic initial condition.
For the rest of universe, we are only able to observe an evolved state. The closer to us the observed universe is to us physically, the the older it is.


Data about the latest universe usually comes as a catalogue of galaxies. (e.g., `Malavasi et al. <https://arxiv.org/abs/1509.08964>`_)
The slightly older universe was captured by the measurements of Lyman-alpha Forest. (e.g., `Lee et al. <https://arxiv.org/abs/1409.5632>`_)
Gravitational lensing measures a projection along the line of sight (e.g., `Amara et al. <https://arxiv.org/abs/1205.1064>`_).

The recovery of the full cosmic initial condition is an inversion problem, which reverts the time evolution from the observed
field :math:`y`,

.. math::

    x = S^{-1}(y) ,

where :math:`x` is the unknown initial condition, :math:`y` is the observation, and :math:`S` is the dynamical model
of the universe. At the resolution we can currently probe, :math:`S` is mostly determined by gravity. 


There are three difficulties in this problem:

1. The observation comes with noise; there is also uncertainty in the forward model :math:`S`. A noisy inversion problem
can be written as an optimization problem,

.. math::

    \mathrm{ minimize}_{x = \hat{x}} \chi^2(x) = \left|\frac{S(x) - y}{\sigma}\right|^2 ,

where :math:`\sigma` quantifies the level of noise.

The solution :math:`x=\hat{x}` is our best estimate of the cosmic initial condition. There are ways of deriving the uncertainty
of :math:`\hat{x}`.

2. It has very large dimensionality. :math:`x` and :math:`y` are fields defined in a three-dimenional space. 
For example, for a mesh of :math:`128^3` points, the number of elements in the vectors :math:`x` and :math:`y` is in the millions.

3. The model of structure formation (:math:`S`) is nonlinear and becomes non-perturbative quickly as the resolution increases.
We use ordinary differential equation (ODE) solvers to follow the evolution of the structure.
A particularly simple family of solvers that is frequently used in cosmology are particle-mesh solvers.
We refer readers to the classical book
`Computer Simulation Using Particles <http://dl.acm.org/citation.cfm?id=62815>`_ by Hockney and Eastwood for further reference.
We are therefore facing a non-linear optimization problem in a high-dimensional space.
The gradient of the objective function :math:`\chi^2(x)` is a crucial ingredient for solving such a problem.

There are generic software tools (automatic differentiation software) to automatically evaluate the gradient of any function.
We were hoping to use these generic software tools in our problem.
We tried three packages, `Tensorflow <https://www.tensorflow.org/>`_, `Theono <http://deeplearning.net/software/theano/>`_,
and `autograd <https://github.com/HIPS/autograd>`_.
We encountered quite a few problems when we tried to build a particle-mesh solver with these packages:
we found that all three of them lack the elements needed to describe our particle-mesh solver.

This motivated the writing of this blog.
We will review how automatic differentiation (AD) works.
Next, we will build gradient operators that are useful in a particle-mesh solver.
In the long term, we would like to patch the generic AD software packages to include these operators.

Automatic Differentiation
-------------------------

AD is a relatively new technology in astronomy and cosmology despite
its growing popularity in machine learning. At the `2016 AstroHackWeek <http://astrohackweek.org/2016/>`_,
the attendees organized a session to explore the AD software landscape. One idea was that 
we should try to use AD more in astronomy if we are to define the boundary of the technology.
This blog was partially inspired by the discussion among the astronomers during that session.

The recent popularity of AD is partially due to the movement of deep learning.
Training large neural networks demands effective and efficient optimization algorithms because
the dimensionality of the problem (number of neural nodes) is large.
Popular optimization algorithms (e.g., gradient descent -- the only embedded optimizer in TensorFlow -- or L-BFGS, which we use
in the cosmic initial condition problem) demands evaluation of the gradient.

A large portion of AD lies beyond deep learning in the context of inversion of dynamical systems.
Many physical problems can be written as solutions to a set of time-dependent differential equations.
The solution to these equations can be written as the product of a sequence of evolution operators
(nested function evaluations).
AD can be applied to evaluate the gradient of the final condition in respect to the intial condition, which is also called
`sensitivity analysis` in this context. 
Due to the complicity of the problem, AD in inversion problems is
usually tailored to a specialized form that suits the particular dynamical system
(cf. `Sengupta et al. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120812/>`_).
In these problems, however, generic AD software is not widely used
because such software usually does not implement the necessary operators.
Another issue is there is no universal fast algorithm to recognize shortcuts, or optimizations, in
the evaluation sequence of AD.

We ran into these problems when we attempted to apply generic AD software to particle-mesh simulations. 
For particle-mesh simulations, we need the AD software to
support Discrete Fourier Transforms and window resampling operations.

In this article, we will try to bridge the gap by writing down these operators; we shall leave
the problem of finding an optimal AD evaluation sequence for the future.

De-Mysterifying AD
------------------

We will first revisit what AD actually does. 

.. figure:: autodiff-func.svg
    :width: 50%
    :align: right

    Figure: Illustration of the evaluation sequences of automatic differentiation.

The basics of AD start from the chain rule of differentiation, which claims the following:

    If we have two functions :math:`y=f(x)` and :math:`z=g(y)=g(f(x))`, then

    .. math::

        \frac{\partial z_j }{\partial x_i} = \sum_k \frac{\partial z_j}{\partial y_k} \frac{\partial y_k}{\partial x_i}
                            = \frac{\partial z_j}{\partial y_k} \cdot \frac{\partial y_k}{\partial x_i} .

We see that the chain rule converts a gradient of nested functions to a sequence of tensor products.

Let's now consider a scalar that comes from the nested evaluation of :math:`n` functions,

.. math::

    F(x) := \left(f^1 \odot \cdots \odot f^n \right)(x) = f^n(f^{n-1}(\cdots (f^1(x)) \cdots ))) .

:math:`f^i` maps to concepts in real-world problems:

- as a time step in a dynamical system, the nested functions are simply moving the dyanmical system forward in time.

- as a layer in the neural network, the nested functions are simply stacking layers of the neural network.

We will name the intemediate variables :math:`r^{(i)}`,

.. math::

    r^n = F(x) ,

    r^i = f^i(r^{i-1}) ,

    r^0 = x .

This function is illustrated in the `function evaluation` section of the figure.

Applying the chain rule to :math:`\nabla F`, we find that

.. math::

    \nabla_j F = \frac{\partial F}{\partial r^0_j} = 
        \left[\prod_{i=1, n} \frac{\partial f^i}{\partial r^{i-1}}\right]_j ,

where :math:`\prod` represents a tensor product on the corresponding dimension
(known as the Einstein summation rule, cf. `numpy.einsum`).
AD software constructs and evaluates this long tensor product expression for us.

There are many ways to evaluate this expression.
We will look at two popular schemes: the `reverse-accumulation/back-propagation` scheme and
the `forward-accumulation` scheme. Both are described in the Wikipedia entry for `AD <https://en.wikipedia.org/wiki/Automatic_differentiation>`_.

Here, we will motivate these schemes by defining two different types of functional operators.

Backward
++++++++

For a function `f` defined on the domain :math:`f : X \to Y`, we define the gradient-adjoint-dot operator as

.. math::

    \Psi[f, x](v) = \sum_i v_i \frac{\partial f_i}{\partial x_j} .

It is implied that :math:`v \in Y` and the domain of :math:`\Psi[f, x]` is :math:`\Psi[f, x] : Y \to X`.

Notice how the summation eliminates the indexing of the function, while the indexing for the gradient remains.

Using :math:`\Psi^i = \Psi[f^i, r^i]`, the chain rule above can be re-organized as a sequence of function evaluations
of :math:`\Psi^i`

.. math::

    \nabla F_j = (\Psi^1 \cdots (\Psi^{n-1}(\nabla_j f^n))\cdots)_j .

The process is illustrated in the `back-propagation`section of the figure. 
We see that for each evaluation of :math:`\Psi^i`, we
obtain the gradient of :math:`F` relative to the intermiedate variable :math:`r^i`, :math:`\nabla_{r^i} F`. Because we apply
:math:`\Psi^i` in the decreasing order of :math:`i`, 
this method is called `backward propagation` or `reverse accumulation`.

This method is also called the `adjoint method` in the analysis of dynamical systems because the summation is along the `adjoint`
index of the jacobian :math:`\frac{\partial f_i}{\partial x_j}`.
The main drawback of back propagation is
that it requires one to store the intermediate results along with the function evaluation in order to compute the
gradient-adjoint-dot operators :math:`\Psi^i` depends on :math:`r^i`, which needs to be evaluated before the back propagation.
However, the method gives the full gradient against the free variables `x_j` after one full accumulation, making it at advantageous
for certain problems compared to `forward accumulation`, which we describe next.

In all three AD software packages we checked (TensorFlow, Theono, or autograd), a method to
look up the the gradient-adjoint-dot operator is provided, either as a member of the operator entity or as an external
dictionary.


Forward
+++++++

In contrast, we can define a gradient-dot-operator as

.. math::

    \Gamma[f, x](u) = \sum_j \frac{\partial f_i}{\partial x_j} u_{j} .

It is implied that :math:`u \in X` and the domain of :math:`\Gamma[f, x]` is :math:`\Gamma[f, x] : X \to Y`.

Notice the summation is over the indexing of the free variable, :math:`x_j`. Hence, the name does not have `adjoint` like the previous
operator. One way to think of :math:`\Gamma[f]` is that it rotates :math:`u` by the jacobian.

With the gradient-dot operator of :math:`\Gamma^i = \Gamma[f^i, r^i]`, we can write down the `forward accumulation` rule of AD:

.. math::

    \sum_j \nabla_j F u_j = \Gamma^n (\cdots (\Gamma^1(u)) \cdots) .

This process is illustrated in the section on `forward accumulation` in the figure.
We see that for each evaluation of :math:`\Gamma^i`, we obtain the directional
derivative of :math:`r^i` along :math:`u`, :math:`\sum \frac{\partial r^i}{\partial x_j} u_j`. The accumulation goes along the increasing
order of :math:`i`, making the name `forward accumulation` a suitable one.

The advantage of forward accumulation is that one can evaluate the gradient as the function :math:`F` is evaluated, and no intemediate
results need to be saved: we see that when :math:`\Gamma^i` is requested, :math:`r_i` is already evaluated.
This is clearly a useful feature when nesting (layers of neural networks or number of time steps)
is high.
However, the cost is we can only obtain a directional derivative. For some applications, this is useful (e.g., computing Hession for Newton-CG or trust-region
Newton-CG methods). When the full gradient is desired, one needd to run
the `forward accumulation` many times - as many times as the number of free parameters, which could be prohibatively high.

We shall note that this method is also called `forward senstivity` in the analysis of dynamical systems.

Two Useful Operators in Particle-Mesh solvers
---------------------------------------------

In this section, we present two families of gradient-adjoint-dot operators that are useful for the AD of cosmological simulations.
The first family is the Discrete Fourier Transforms, and the second family is resampling windows. At the time of this blog,
no popular AD software implements all of these gradient-adjoint-dot operators. We will list them in this section for further 
reference.

Discrete Fourier Transform
++++++++++++++++++++++++++

Discrete Fourier Transform is the discretized version of Fourier Transform.
It is a commonly used density matrix operator in the modelling of physical process.
This is mostly because spatial differentiation can be written as multiplication
in the spectrum space.

.. math::

    \nabla \phi (x) = \mathrm{ifft}(k \cdot \mathrm{fft}(\phi)(k))(x)

The gradients involve complex numbers, which are tuples of two real numbes.
The gradient that is conveniently used is

.. math::

    \nabla_z = \frac{\partial}{\partial x} + \imath \frac{\partial}{\partial y} ,

for :math:`z = x + \imath y`. It is related to the Wirtinger derivatives (Fourier transform is a harmonic function).

The gradient-adjoint-dot operator of a discrete fourier transform
is its dual transform. Specifically,

.. math::

    \Psi[\mathrm{fft}, X](V) = \mathrm{ifft}(V) ,

    \Psi[\mathrm{ifft}, Y](V) = \mathrm{fft}(V) ,


where :math:`\Psi` is the gradient-adjoint-dot operator. Notably, the free variables :math:`X` and :math:`Y`
do not show up in the final expressions.
This is because Fourier transforms are linear operators. 

We do not include a formal proof in this blog. (The proof is relatively simple)

Gradients of the real fourier transforms (`ifft` and `irfft`) are slightly more complicated.
Of the complex vector in a real fourier transform, only about half of the complex numbers are indepedent.
The other half is the hermitian conjugates.
(See `What FFTW really computes <http://www.fftw.org/doc/The-1d-Real_002ddata-DFT.html#The-1d-Real_002ddata-DFT>`_)

Due to this hermitian property of the complex vector, `irfft` has two types of gradients:

- gradient over the full complex vector, which follows the same rules as the complex fourier transform

  .. math::

    \Psi[\mathrm{rfft}, X](V) = \mathrm{irfft}(V) ,

    \Psi^{\mathrm{full}}[\mathrm{irfft}, Y](V)_j = \mathrm{rfft}(V)_j,

- gradient over the compressed complex vector

The full complex vector and the compressed complex vector are related by a 
`decompress` operation, which introduces a factor of 2 to the fourier modes
with a hermitian conjugate mode.

  .. math::

    \mathrm{decompress}(Y) = Y

    \Psi[\mathrm{decompress}, Y](V)_j = V_j \left\{
                \begin{matrix}
                            1 & \mathrm{ if } j = N - j, \\
                            2 & \mathrm{ if } j \neq N - j
                \end{matrix} \right. = V_j \mathsf{D}(j, N) .

    \Psi^{\mathrm{compressed}}[\mathrm{irfft}, Y](V)_j
           = \Psi[\mathrm{decompress}, Y](V)_j
             \Psi^{\mathrm{full}}[\mathrm{irfft}, Y](V)_j
           = \mathsf{D}(j, N) \mathrm{rfft}(V)_j,

The complex version of Discrete Fourier Transform is implemented in TensorFlow (GPU only), Theono, and autograd, though
it appears the version in autograd is incorrect. The real-complex transforms (`rfft` and `irfft`)
are not implemented in any of the packages, neither is the `decompress` operation
defined in any of these packages.

Resampling Windows
++++++++++++++++++

The resampling window converts a field representation between particles and meshes.
It is written as

.. math::

    B_j(p, q, A) = \sum_i W(p^i, q^j) A_i ,

where :math:`p^i` is the position of `i`-th particle/mesh point and :math:`q^j` is the position
of `j`-th mesh/particle point; both are usually vectors themselves (the universe has three spatial dimensions).
:math:`p^i` and :math:`q^i` are themselves vectors with a spatial dimension (we will use
the integer symbol :math:`a` to index this dimension).

- `paint`: When :math:`p^i` is the position of particles
  and :math:`q^j` is the position of the mesh points,
  the operation is called a `paint`.

- `readout`: When :math:`p^i` is the position of the mesh points and
  :math:`q^j` is the position of mesh points, the operation is called a `readout`.

:math:`W` is the resampling window function. A popular form is the
cloud-in-cell window, which represents a linear interpolation:

.. math::

    W(x, y) = \prod_{a} (1 - h^{-1}\left|x_a - y_a\right|) ,

for a given size of the window :math:`h`. We have used :math:`a` as the index of the spatial
dimensions.

Most windows are seperatable, which means they can be written as a product of
a scalar function :math:`W_1`,

.. math::

    W(x, y) = \prod_{a} W_1(\left|x_a - y_a\right|),

For these windows,

.. math::

    \frac{\partial W}{\partial x_a} = \frac{\partial W}{\partial y_a} = 
    W_1^\prime(\left|x_a - y_a\right|) \prod_{b \neq a} W1(\left|x_b - y_b\right|) .

We can then write down the gradient-adjoint-dot operator of the window

.. math::

    \Psi[B, \{p, q, A\}]_p(v)_{(i,a)} = \sum_j \frac{\partial W(p^i, q^j)}{\partial p^i_a} A_i v_j ,

    \Psi[B, \{p, q, A\}]_q(v)_{(j,a)} = \sum_i \frac{\partial W(p^i, q^j)}{\partial q^j_a} A_i v_j ,

    \Psi[B, \{p, q, A\}]_A(v)_i =  \sum_j W(p^i - q^j) v_j .

- The first gradient corresponds to the displacement of the source.

- The second gradient corresponds to the displacment of the destination.

- The third gradient corresponds to the evolution of the field.

This looks complicated, and it is.
However, in a particle mesh simulation, more often than not
either one of the sources or the destination is a fixed grid which does not move.
In this case :math:`v = 0`, and we do not need to compute the corresponding gradient term.

We also note that it is possible to extend these expressions
to Smoothed Particle Hydrodynamics (SPH). In SPH, :math:`h` is another free variable. We leave this
to future work, but note that the symmetric
property of the hydro-dynamical force kernel may introduce additional complication.

None of the three packages we surveyed
(TensorFlow, Theono and autograd) recognizes these resampling window operators. 
The resampling window operators cannot be easily expressed as diagonal matrix operators.

Work in progress
----------------

We will implement these operators in our cosmological forward modelling software FastPM
(https://github.com/rainwoodman/fastpm/), where we plan to manually implement the gradients
with the AD rules.

In a longer term, we would like to implement these operators in a generic AD software to take advantage
of the automated differentiation in full.

*Acknowledgement*

The author received help on the algebra from Chirag Modi, Grigor Aslanyan, and Yin Li from Berkeley Center for Cosmological Physics.
The author received help on writing from Ali Ferguson at Berkeley Institute for Data Science.
