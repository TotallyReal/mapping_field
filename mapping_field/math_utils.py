
def _inner_gcd(n: int, m: int):  # n>m>0
    """
    Assuming that n>m>0, finds the gcd(n,m)=d and returns a triple (d,a,b) such that
            d = a*n + b*m   .
    """
    q = n // m
    r = n - q * m
    if r == 0:
        return m, 0, 1
    else:
        res = _inner_gcd(m, r)
        return res[0], res[2], res[1] - q * res[2]


def gcd(n: int, m: int):
    """
    returns a triple (d,a,b) such that
        d = gcd(n,m) = a*n + b*m.
    If both n=m=0, then returns (0,1,0), so that 0=gcd(0,0).
    """
    if m == 0:
        return n, 1, 0
    if n == 0:
        return m, 0, 1

    signN = 1 if n > 0 else -1
    signM = 1 if m > 0 else -1
    n *= signN
    m *= signM

    if n < m:
        res = _inner_gcd(m, n)
        return res[0], res[2] * signN, res[1] * signM
    else:
        res = _inner_gcd(n, m)
        return res[0], res[1] * signN, res[2] * signM