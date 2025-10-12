# Mapping field

Fields in mathematics come in many shapes or forms, from the standard Rational, Real and Complex numbers 
to the less standard, but just as interesting finite field. We are used to do many operations with these
fields, to the point that we introduced variables for elements in the field, which we can "fill in" later.
In other words, instead of saying that (1+5)+2 = 1+(5+2) we write (x+y)+z = x+(y+z) and when we need, we
plug in the numbers in x,y,z.

Given any field $\mathbb{F}$ we can consider the ring (almost a field) of all functions $\mathbb{F} \to \mathbb{F}$. 
The field operations from $\mathbb{F}$ are extended to these functions. Here too we would want to have
equations with "function variables" which we can fill in later. For example, we can write 

```math
f(f(x,y),z) = f(x,f(y,z)),
```

and then later choose $f$ to be addition or multiplication.

This library emerged from this type of problem which comes up naturally in mathematics. 

Another application is to help solve functional equations. For example, for 

```math
y\cdot f(x) = y\cdot x - f(1/x)
```

we might note that we can write $f(x) = x - f(1/x)/y$, then substitute $x$ with $1/x$ to obtain

```math
y\cdot f(x) = y\cdot x - 1/x + f(x)/y
```

from which we can find out $f(x)$.  While it is already possible to do it with the current code, it is hard 
follow it visually, since it will look something like "(x-( (( 1/x )-( f(x)/y ))/y ))" and it is not always easy to decide how to "simplify" the formula. 
More over, it will require a simple interface that us mere humans can use, for example, moving element from both sides of the
equation to the other side (and thus simplifying the formula manually), isolating functions, automatically adding in exercises like the one above
that $f(x)$ is actually a function of both $x,y$ (though we think of $y$ as "fixed" and then find $f$), etc.

---

**My homepage**: [https://prove-me-wrong.com/](https://prove-me-wrong.com/)

**Contact**:	 [totallyRealField@gmail.com](mailto:totallyRealField@gmail.com)

**Ofir David**
