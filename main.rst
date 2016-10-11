Differentation of Fast Fourier Transforms
=========================================

Discrete Fourier transform is a commonly used density matrix operator in the modelling of
physical process. (Examples)

Computation of the gradient of the Fast Fourier transform operator
is thus of practical utility. In this blog entry we derive the gradient operator
of discrete Fourier transform for three commonly used cases: 
complex-to-complex, real-to-complex and complex-to-real.

In some sense, this short article is an extension to the FFTW documentation page at [fftw]_.

Summary
-------

Regardless of transformation type, the gradient-dot operator of discrete fourier transform
is the dual transform. Specifically,

.. math::

    V \odot \nabla_X \mathrm{fft}(X) = \mathrm{ifft}(V)

    V \odot \nabla_X \mathrm{rfft}(X) = \mathrm{irfft}(V)

    V \odot \nabla_Y \mathrm{ifft}(Y) = \mathrm{fft}(V)

    V \odot \nabla_Y \mathrm{irfft}(Y) = \mathrm{rfft}(V)

where :math:`V` and `X` are properly defined real or complex vectors. Interestingly, the variables
(:math:`X` in forward and :math:`Y` in backward) do not appear in the final expressions. This is because
fourier transform is a linear operator, even though a quite dense one it is: in a discrete fourier transform,
each term cross talks to every term; people sometimes call this a global shuffling ( map-reduce)
or a all to all transpose (massively-parallel computing).

We will elaborate how these gradient-dot operators are obtained, and what it means 'properly defined' in the following
sections. But first let's take a look at an example.

Examples
--------

Convolution of two fields.

.. math::

    W = \| irfft(rfft(X) \odot rfft(Y)) \|^2

For simplicity, we shall define a real vector

.. math::

    w_x = irfft(rfft(X) \odot rfft(Y))

    w_k = rfft(X) \odot rfft(Y)

and we have

.. math::

The gradient of W regarding to X is

.. math::

    \nabla_X W = \nabla_wx W \odot \nabla_X wx
               = \nabla_wx W \odot \nabla_X [irfft(wk)]
               = \nabla_wx W \odot \nabla_wk [irfft(wk)] \odot \nabla_X(wk)
               = rfft(\nabla_wx) \odot \nabla_X(wk)
               = rfft(\nabla_wx) \odot \nabla_X(rfft(X)) \odot rfft(Y)
               =  irfft(rfft(\nabla_wx)) \odot rfft(Y)
             
Discrete Fourier transforms
---------------------------

Discrete Fourier transform calculates the matrix product between a
Fourier transformation matrix and the data vector. We will follow the FFTW convention to write down
the unnormalized transforms:

.. math::

    \mathrm{fft}(X) : Y_j = \sum_i F_{ij} X_i

    \mathrm{ifft}(Y) : X_i = \sum_j F_{ij}^\dagger Y_j

where :math:`X_i`, :math:`Y_j` are complex numbers, and the matrix elements
are :math:`F_{ij} = \exp^{- 2 \imath \pi ij / N}` is the transformation matrix.
The forward FFT takes a negative sign.

For multi-dimension FFTs, the product :math:`ij / N` can be replaced by summations over dimensions. The basic
linear form of the transformation remains.

Periodicity says that :math:`i` and :math:`i+k N` (:math:`k` is an integer) refers to the same component. We notice that

.. math::

    F_{ij} = F_{-i, -j} = F^\dagger_{-i, j} = F^\dagger_{i, -j}

To avoid confusions over taking gradients of complex numbers, it is useful to rewrite the transforms in the real and imaginary
components, explicitly:

.. math::

    \mathrm{fft}(X) : Y^0_j = \sum_i F_{ij}^0 X_i^0 - \sum_i F_{ij}^1 X_i^1
                      Y^1_j = \sum_i F_{ij}^1 X_i^0 + \sum_i F_{ij}^0 X_i^1

    \mathrm{ifft}(Y) : X^0_i = \sum_j   F_{ij}^0 Y_j^0 + \sum_i F_{ij}^1 X_i^1
                       X^1_i = \sum_j - F_{ij}^1 Y_j^0 + \sum_i F_{ij}^0 X_i^1

We have expanded with :math:`u = u^0 + \imath u^1`.


The real to complex transformation is

.. math::

    \mathrm{rfft}(X) : Y_j^{0} =  \sum_i F_{ij}^{0} X_i^0
                       Y_j^{1} =  \sum_i F_{ij}^{1} X_i^0

In this case, we see that :math:`Y_j = Y_j^0 + \imath Y_j^1` is hermitian:

.. math::

    Y_{-j} = Y_{j}^\dagger

The complex to real transformation is

.. math::

    \mathrm{irfft)(Y) :
    X_i^{0} = \sum_j F_{ij}^{0} Y_j^0 + F_{ij}^{1} Y_j^1

The complex-to-relal transform is meaningful only if :math:`Y` is hermitian. Otherwise the
imaginary part of :math:`X`, :math:`X^1` is nonzero.

Usually the hermitian complex array :math:`Y` is stored in a compress format, where roughly half of
the complex components are stored in memory. Even in this case, we shall be aware that
not all of them are independent. The hermitian conjugation property shall be properly maintained.

Gradient opeartors
------------------

Consider an operator :math:`Y = F(X)`. The gradient is

.. math::

    F^\prime(X)_{ji} = \frac{\partial Y_j}{\partial X_i} \|_{X}


The gradient operator contracts :math:`F^\prime` along the dimensions of :math:`Y` with a
chaining vector `V`,

.. math::

    \nabla_i F(X, V) = V \cdot F^\prime(X) = \sum_j V_j^\dagger \frac{\partial Y_j}{\partial X_i}

Thus, a gradient operator is sometimes called the jacobian-product operator.
We notice that the gradient is a matrix of size :math:`d(X) d(Y)`, while the
gradient operator is a vector, of size :math:`d(X)`. Since the chain rule only
requres a dot product, the gradient operator compresses the storage without loosing
generality. 

.. note::

    For complex numbers :math:`a \odot b = \sum_i a_i^\dagger b_i`.

As in the previous section, we will expand the real and imaginary components explicitly

.. math::
    :expanded:

    \nabla_i^{0, 1} F(X, V) =\sum_j V_j^0 \frac{\partial Y_j^0}{\partial X_i^{0, 1}}
                       +       V_j^1 \frac{\partial Y_j^1}{\partial X_i^{0, 1}}

Complex to Complex
------------------

We notice that all complex numbers in a complex to complex transform are independent free variables.


We will work out the forward transform first. Notice that

.. math::

    \frac{\partial{Y_j^0}}{\partial{X_i^0}} = F_{ij}^0

    \frac{\partial{Y_j^0}}{\partial{X_i^1}} = - F_{ij}^1

    \frac{\partial{Y_j^1}}{\partial{X_i^0}} = F_{ij}^1

    \frac{\partial{Y_j^1}}{\partial{X_i^1}} = F_{ij}^0

This simply shows that the fourier transform is a harmonic function.

Plugging these into eq:`expanded`, and compare these to the expanded version of ifft, we find that

.. math::

    \nabla_i^0 \mathrm{fft}(X, V) = \sum_j V_j^0   F_{ij}^0 + V_j^1 F_{ij}^1 = \mathrm{ifft}^0(V)

    \nabla_i^1 \mathrm{fft}(X, V) = \sum_j V_j^0 - F_{ij}^1 + V_j^1 F_{ij}^0 = \mathrm{ifft}^1(V)

We then construct a complex vector from the above two components:

.. math::

    \nabla_i \mathrm{fft}(X, V) = \nabla_i^0 \mathrm{fft}(X, V) + \imath \nabla_i^1 \mathrm{fft}(X, V)
      = \mathrm{ifft}(V)

Therefore, the gradient operators of forward and backward discrete fourier transforms are simply themselves applied to
the chaining vector :mathrm:`V`.

We shall not confuse these operators with the Wirtinger derivatives https://en.wikipedia.org/wiki/Wirtinger_derivatives .
Here we are essentially treating each complex number as a tuple of real numbers. Care must be take when propagating the
chain rules.

Common auto differentiation software packages implements the gradient of complex-to-complex
Fast Fourier transform operators.

A particular clear example is in the fft module of the autograd Python package:

https://github.com/amarshah/complex_RNN/blob/master/fftconv.py#L107

Note that autograd claims differently.

https://github.com/HIPS/autograd/blob/c6e62b6bbf4faa14b2a55fe556a57cbf242278f9/autograd/numpy/fft.py#L12

Real to Complex
---------------

In a Real to Complex transform, the real numbers :math:`X_i` are independent. Therefore

.. math::

    \nabla_i \mathrm{rfft}(X, V) = \sum_j V_j^0 F_{ij}^0 + V_j^1 F_{ij}^1 = \mathrm{irfft}_i (V)

We note that :math:`V` is complex, but the gradient operator gives real (the same type as :math:`X`).

Complex to Real
---------------

.. math::

    \nabla_j^0 \mathrm{irfft}(Y, V) = \sum_i V_i F_{ij}^0 = \mathrm{rfft}_j^0 (V)

    \nabla_j^1 \mathrm{irfft}(Y, V) = \sum_i V_i F_{ij}^1 = \mathrm{rfft}_j^1 (V)

Following the practice in complex-to-complex, we can define the complex vector

.. math::

    \nabla_j \mathrm{irfft}(Y, V) = \mathrm{rfft}_j(V)

The gradient operator of irfft is rfft. We can check that V is real, and Y is complex.


.. [fftw] http://www.fftw.org/doc/What-FFTW-Really-Computes.html
