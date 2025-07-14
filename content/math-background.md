# Maths background for MSc AI

This MSc program assumes that you have some university-level mathematics experience.
For example, as covered by first-year undergraduate mathematics courses taken by
informatics, physics, or engineering students in Edinburgh.

It is expected that you are used to manipulating algebraic expressions, and
solving for unknowns. For example, it should be straightforward for you to
rearrange an expression like
$$
    y = 3 + \log x^3 z,
$$
to give an explicit expression for $x$ in terms of the other variables.

The three main areas of mathematics we need are probability, linear algebra, and
calculus. Most of the results you should know are summarised on the following
cribsheet:\
<https://homepages.inf.ed.ac.uk/imurray2/pub/cribsheet.pdf>

These areas are covered in the book
[*Mathematics for Machine Learning*](https://mml-book.github.io/) (Deisenroth et\ al.),
but there's more there than you need. The rest of this document gives more
specific details of what you need for most of our courses, with some other reading
options.


# Probability

As stated in the cribsheet, MacKay's free textbook provides a terse introduction
to probability, as does Murphy Book\ 1 Chapter\ 2, or Barber Chapter\ 1.
Alternatively, Sharon Goldwater has a longer, more tutorial introduction:\
<https://homepages.inf.ed.ac.uk/sgwater/math_tutorials.html>

You *must* know the sum and product rules of probability: their equations, what
they mean, and how to apply them for discrete and real-valued variables.

Expectations, or averages of random quantities, are also important.
[Detailed notes on expectations are provided](https://mlpr.inf.ed.ac.uk/2024/notes/w0g_expectations.html)
in the background section of the MLPR course notes. Please make sure you can do
the exercises.


# Linear Algebra

An undergraduate linear algebra course will usually discuss abstract linear
spaces and operators. This course largely focusses on concrete operations on
matrices and vectors expressed as arrays of numbers, as we can explicitly
compute in NumPy.

You need to be able to do basic algebraic manipulation of matrices and vectors,
and know how matrix multiplication works. You should also have a geometric
understanding of these operations, which can be relevant to understanding their
application to machine learning. If you're unsure, please work through David
Barber's tutorial:

<https://www.inf.ed.ac.uk/teaching/courses/mlpr/notes/mlpr-supplementary-maths.pdf>

A shortened version of this tutorial also appears as an appendix of his textbook.

You will *not* need to be able to numerically compute matrix inverses,
determinants, or eigenvalues of matrices by hand for this course. You can safely
skip those exercises!

There are many possible introductions to linear algebra. Another terse one is 
[Chapter\ 2](https://www.deeplearningbook.org/contents/linear_algebra.html) of
Goodfellow et al.'s [Deep Learning textbook](https://www.deeplearningbook.org/).
A nice series of videos is [3blue1brown](https://www.3blue1brown.com/)'s [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).


# Differentiation

You should know how to differentiate algebraic expressions. Computer algebra
systems can do this stuff for us, and you may also learn about automatic numerical
differentiation later in some courses. However, in simple cases it's still common
for researchers to differentiate expressions with pen and paper as part of their
working.

The cribsheet summarizes the basic results you are expected to know. If the rules
don't make sense, you will need to consult an undergraduate level or advanced
high-school level maths textbook, or a tutorial series such as those from Khan
Academy.

Some students may not have seen or remember *partial derivatives*. For example:
$$
    \pdd{xy^2}{x} = y^2, \quad
    \pdd{xy^2}{y} = 2xy.
$$
The curly $\partial$ simply means that you treat all other variables as
constants when you are doing the differentiation.

Partial derivatives can be combined to create total derivatives. For example,
imagine moving around the circumference of a unit circle by changing an angle
$\theta$. Your $(x,y)$ position is given by $x \te \cos\theta$ and $y \te
\sin\theta$. To compute the change in a function $f(\bx)$ due to an
infinitesimal change $\mathrm{d}\theta$ in the angle, you can use the chain rule
of differentiation:
$$
    \mathrm{d}{f} = \pdd{f}{x} \mathrm{d}{x}  +  \pdd{f}{y} \mathrm{d}{y}
\qquad\text{or}\qquad
    \tdd{f}{\theta} = \pdd{f}{x} \tdd{x}{\theta}  +  \pdd{f}{y} \tdd{y}{\theta}.
$$
In this case you could also substitute expressions for $x$ and $y$, to find
$f(\theta)$ and differentiate with respect to\ $\theta$. You could try both
methods to differentiate $f(x,y) \te xy^2$ with respect to\ $\theta$. You should
get the same answer! The chain rule approach is needed in many machine learning
settings.

You should be comfortable enough with both vectors and derivatives that you
wouldn't find it intimidating to work with a vector containing derivatives. For
example, if we want to find partial derivatives of a function $f(\bx)$ with
respect to each element of a vector $\bx = [x_1~x_2]^\top$, then the vector
$\nabla_\bx f$ is defined as:
$$
    \nabla_\bx f = 
        \left[ \begin{array}{c} \pdd{f}{x_1}  \pdd{f}{x_2} \end{array} \right].
$$
We will do mathematics containing such expressions. For example, given the chain
rule of differentiation above, you should be happy that
$$
    \tdd{f}{\theta} = \left(\nabla_\bx f\right)^\top 
        \left[ \begin{array}{c} \tdd{x_1}{\theta}  \tdd{x_2}{\theta} \end{array} \right].
$$

This material is covered by undergraduate-level mathematics textbooks such as
*"Mathematical Methods for Physics and Engineering"*, Riley, Hobson, Bence.
Although there are many other possible textbooks and online tutorials.

[3blue1brown](https://www.3blue1brown.com/) also have an [Essence of
Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
playlist.


# Integration

You should also know enough about integration to understand the sum rule and
expectations for real-valued variables.

The two most common situations in machine learning are: 1)\ an integral is
impossible to solve with pen and paper, there is no closed-form solution;
or 2)\ an integral is easy, there is a trick to write down the answer.

For example, the Gaussian distribution (discussed in much more detail later in
the notes) with mean $\mu$ and variance $\sigma^2$ has probability density function:
$$
    \N(x;  \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}}  e^{-\frac{1}{2\sigma^2} (x - \mu)^2}.
$$
We may have to compute various integrals involving this function. For example:
$$
    I = \int_{-\infty}^\infty  (x + x^2)  \N(x;  \mu, \sigma^2) \intd{x}.
$$
We can express this integral in terms of expectations for which we already
know the answers:
$$
    I = \E[x] + \E[x^2] = \mu  +  (\mathrm{var}[x] \tp \E[x]^2)
    = \mu + \sigma^2 + \mu^2.
$$

Summary: there's no need to revise everything about integration covered in a
calculus course. You're not going to need most of the advanced tricks for solving integrals,
such as clever trigonometric substitutions or "contour integration".
However, you do need to be comfortable enough with what integration is,
and with probability theory, so that you can follow and produce mathematical
arguments like the one above.

