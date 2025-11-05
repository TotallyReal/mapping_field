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


# The mechanism

## The MapElement class

The `MapElement` class is the main class for functions. It has two main features:
### Calling the function:
Given a `f: MapElement` we can actually call `f(entry_1,...,entry_n)` to generate a new `MapElement' of this function with the entries. There are 3 ways of calling this function:
- Positional: `f(a_1, ..., a_n)`       
- Keywords:   `f(x_1 = a_1, ..., x_n = a_n, f_1 = F_1, ..., f_k = F_k)`
- Dictionary: `f({x_1 : a_1, ..., x_n : a_n, f_1 : F_1, ..., f_k : F_k})`
   
Note that the positional way only work for variable assignments. See `MapElement.__call__` for more details,
and a bit more keywords you can use when calling a function.

### Simplifying the function
Calling `f.simplify2()` (where the `2` will be removed eventually...) will return a simplified version of the function
(and in general should not change `f` itself).
When creating new types of simplifications, the main mechanism uses `Processor` simplifier methods:

```python
(f: MapElment, var_dict: VarDict) -> Optional[MapElement]
```

It get the function `f` to be simplified, an assignment of variables `var_dict`, and returns the simplified version, 
or `None` if the simplifier couldn't find any. In particular, each `MapElement` class has such a method under `f._simplify2()`.

You can register new simplifiers directly into functions (in which case, we don't need the `f` variables), 
or to classes. For example, we can add a $\cos^2(t) + \sin^2(t) = 1$ simplification to the `Add` function. Suppose
that we created a function 
```python
inner_trig(elem: MapElement, trig: str) -> Optional[MapElement]
```
which checks if the given `elem` is a trigonometry function of type `trig` and if so returns the inner function.
So for example `inner_trig(cos^2(t+5), 'cos^2') = t+5`. Then we can write a simplifier using the following:

```python
def trig_simplifier(var_dict: VarDict) -> Optional[MapElement]:
    entries = [var_dict[v] for v in Add.vars]
    
    if inner_trig(entries[0],'cos^2') == inner_trig(entries[1],'sin^2') != None:
        return MapElementConstant.one
    if inner_trig(entries[1],'cos^2') == inner_trig(entries[0],'sin^2') != None:
        return MapElementConstant.one
    
    return None

Add.register_simplifier(trig_simplifier)
```

All of the standard arithmetic operations already have built in calls for simplifiers using 'positional' calls
like with the standard dunder functions. For example, writing `x+y` will call automatically to `x.add(y)` and if 
it cannot simplify the expression, then to `y.radd(x)`.

---

**My homepage**: [https://prove-me-wrong.com/](https://prove-me-wrong.com/)

**Contact**:	 [totallyRealField@gmail.com](mailto:totallyRealField@gmail.com)

**Ofir David**
