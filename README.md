# Mapping field

Fields in mathematics come in many shapes or forms, from the standard Rational, Real and Complex numbers 
to the less standard, but just as interesting finite field. We are used to do many operations with these
fields, to the point that we introduced variables for elements in the field, which we can "fill in" later.
In other words, instead of saying that (1+5)+2 = 1+(5+2) we write (x+y)+z = x+(y+z) and when we need, we
plug in the numbers in x,y,z.

Given any field $\mathbb{F}$ we can consider the ring (almost a field) of all functions $\mathbb{F} \to \mathbb{F}$. 
The field operations from $\mathbb{F}$ are extended to these functions. Here too we would want to have
equations with "function variables" which we can fill in later. For example, we can write 

$$f(f(x,y),z) = f(x,f(y,z)),$$

and then later choose $f$ to be addition or multiplication.

This library emerged from this type of problem which comes up naturally in mathematics. Another application is
as follows. If I want to solve functional equations like 

$$y*f(x) = y*x - f(1/x)$$

we might note that we can write $f(x) = x - f(1/x)/y$, then substitute $x$ with $1/x$ to obtain

$$y*f(x) = y*x - 1/x + f(x)/y$$

from which we can find out $f(x)$. 