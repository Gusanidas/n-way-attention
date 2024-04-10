import random
from collections import defaultdict
import string


def generate_bool_parenthesis(depth=1):
    if depth <1:
        return ("T", True) if random.random() <0.5 else ("F", False)
    left, lv = generate_bool_parenthesis(depth=random.randint(0,1)*depth-1)
    right, rv = generate_bool_parenthesis(depth=random.randint(0,1)*depth-1)
    op = random.randint(1,10)
    if op == 1:
        s = f"( {left} and {right} )"
        v = lv and rv
    elif op == 2:
        s = f"( {left} or {right} )"
        v = lv or rv
    elif op == 3:
        s = f"( {left} ^ {right} )"
        v = lv^rv
    elif op == 4:
        s = f"( {left} nor {right} )"
        v = not( lv or rv)
    elif op == 5:
        s = f"( {left} v {right} )"
        v = not (lv ^ rv)
    elif op == 6:
        s = f"( {left} e {right} )"
        v = lv and (not rv)
    elif op == 7:
        s = f"( {left} k {right} )"
        v = (not lv) and rv
    elif op == 8:
        s = f"( {left} w {right} )"
        v = lv or (not rv)
    elif op == 9:
        s = f"( {left} i {right} )"
        v = (not lv) or rv
    else:
        s = f"( {left} n {right} )"
        v = not (lv and rv)
    return s, v

def generate_bool_expr(depth=1):
    expr, v = generate_bool_parenthesis(depth=depth)
    sv = "T" if v else "F"
    return f"{expr} = {sv}"


def generate_arith_parenthesis(depth=1, nmax=10):
    if depth <1:
        n = random.randint(0, nmax)
        return str(n), n
    left, lv = generate_arith_parenthesis(depth=random.randint(0, 1)*(depth-1),nmax=nmax)
    right, rv = generate_arith_parenthesis(depth=random.randint(0, 1)*(depth-1),nmax=nmax)
    op = random.randint(1,10)
    if op ==1 or op ==6:
        s = f"( {left} + {right} )"
        v = lv + rv
    elif op == 2:
        n = random.randint(0, nmax)
        s = f"( {left} + {right} + {n} )"
        v = lv + rv + n
    elif op == 3:
        n = random.randint(0, nmax)
        s = f"( {left} - {right} - {n} )"
        v = lv - rv - n
    elif op == 4:
        n = random.randint(0, nmax)
        s = f"( {left} - {right} + {n} )"
        v = lv - rv + n
    elif op == 5:
        n = random.randint(0, nmax)
        s = f"( {left} + {right} - {n} )"
        v = lv + rv - n
    elif op == 7:
        n = random.randint(0,3)
        s = f"( {left} + {n} * {right} )"
        v = lv + n*rv
    elif op == 8:
        n = random.randint(0,3)
        s = f"( {left} - {n} * {right} )"
        v = lv - n*rv
    else:
        s = f"( {left} - {right} )"
        v = lv-rv
    return s, v

def generate_arithmetic_expr(depth=1, vmax =50,nmax=19):
    while True:
        expr, v = generate_arith_parenthesis(depth=depth, nmax=nmax)
        if abs(v)<vmax:
            break
    
    return f"{expr} = {v}"

def length_of_LIS(nums):
    """
    Calculates the length of the longest increasing subsequence in a list of numbers.
    """
    if not nums:
        return 0
    LIS = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                LIS[i] = max(LIS[i], LIS[j] + 1)
    return max(LIS)


def generate_lis_arr(n=15, nmax=45):
    base = random.randint(0,nmax//2)
    size = random.randint(nmax//3, nmax-base)
    arr = random.sample(range(base,size+base), k=n)
    l = random.randint(0,n)
    if l == n:
        arr = sorted(arr)
    elif l>3:
        idx = sorted(random.sample(range(n), k=l))
        subs = sorted([arr[j] for j in idx])
        for i, j in enumerate(idx):
            arr[j] = subs[i]
    if random.random()<0.3:
        arr = arr[::-1]
    return arr


def generate_lis(depth=15, nmax = 60):
    arr = generate_lis_arr(n=depth, nmax=nmax)
    r = length_of_LIS(arr)
    s = " "
    for e in arr:
        s += f"{e} , "
    s = s[:-2] + f"= {r}"
    return s


def longestPalindromeSubseq(s: str) -> int:
    """
    Finds the length of the longest palindromic subsequence in a given string.
    Utilizes memoization to improve performance for repeated subproblem checks.
    """
    d = defaultdict(int)
    def f(s):
        if s not in d:
            for c in set(s):
                i, j = s.find(c), s.rfind(c)
                d[s] = max(d[s], 1 if i == j else 2 + f(s[i+1:j]))
        return d[s]
    return f(s)

def make_palindrome(n):
    """
    Generates a palindrome of a specified length. If n is odd, includes a single
    character in the middle; otherwise, generates a mirrored string.
    """
    if n < 1:
        return []
    first_half = random.choices(string.ascii_lowercase, k=n//2)
    if n % 2:
        middle_char = random.choices(string.ascii_lowercase, k=1)
        return first_half + middle_char + first_half[::-1]
    else:
        return first_half + first_half[::-1]

def get_subpal(n=10):
    """
    Generates a random string of length n and injects a palindrome of random length
    into it at a random position. Adjusts the probability of palindrome injection based
    on the string length.
    """
    x = random.random()
    s = random.choices(string.ascii_lowercase, k=n)
    if x < 0.6 and n > 5:
        l = random.randint(4, n)
        pal = make_palindrome(l)
        idx = sorted(random.sample(range(n), l))
        for i, e in enumerate(idx):
            s[e] = pal[i]
    elif x > 0.9 and 2 < n < 25:
        s = random.sample(string.ascii_lowercase, n)
        if x > 0.95:
            idx = random.randint(0, n-2)
            s[idx] = s[idx+1]
    return ''.join(s)

def generate_subpal(depth=10, *args, **kwargs):
    """
    Generates a string with a potential sub-palindrome and calculates the length
    of the longest palindromic subsequence. Returns a formatted string showing the
    generated string and the length of its longest palindromic subsequence.
    """
    s = get_subpal(n=depth)
    d = longestPalindromeSubseq(s)
    r = " "
    for e in s:
        r += f"{e} , "
    r = r[:-2] + f"= {d}"
    return r


def knapsack_01(values, weights, W):
    n = len(values)
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][W]

def generate_knapsack(depth=4):
    n=depth
    weights = [random.randint(1,20) for i in range(n)]
    values = [random.randint(1,25) for i in range(n)]
    total_weight = random.randint(3,80)
    k = knapsack_01(values, weights, total_weight)
    r = " "
    for e in weights:
        r += f"{e} , "
    r = r[:-2] + " -  "
    for e in values:
        r += f"{e} , "
    r = r[:-2] + f" . {total_weight} = {k}"
    return r


if __name__ == "__main__":
    print("bool_expr")
    print(generate_bool_expr(depth=2))
    print(generate_bool_expr(depth=3))
    print("arith expr")
    print(generate_arithmetic_expr(depth=2))
    print(generate_arithmetic_expr(depth=3))
    print("lis")
    print(generate_lis(depth=10))
    print(generate_lis(depth=15))
    print("subpal")
    print(generate_subpal(depth=10))
    print(generate_subpal(depth=15))
    print("knapsack")
    print(generate_knapsack(depth=4))
    print(generate_knapsack(depth=5))