#!/usr/bin/python3
from numpy import *
from numpy.random import *
from math import *
import matplotlib.pyplot as plt

def sq(x):
    return x * x

def f(x, t=0):
    k = 4 * (t - 1)
    c1 = pow(sq(x[0] - 0.25) + sq(x[1] - 0.25), k)
    c2 = pow(sq(x[0] - 0.75) + sq(x[1] - 0.25), k)
    c3 = pow(sq(x[0] - 0.75) + sq(x[1] - 0.75), k)
    c4 = pow(sq(x[0] - 0.25) + sq(x[1] - 0.75), k)
    c5 = pow(sq(x[0] - 0.5 ) + sq(x[1] - 0.5 ), k)
    return c1 + c2 + c3 + c4 + c5

def mutate(p):
    if random() < 0.5:
        return random(), random()
    else:
        dx = normal(0, 0.0001)
        dy = normal(0, 0.0001)
        x = p[0] + dx
        y = p[1] + dy
        x = x - 1 if x > 1 else x + 1 if x < 0 else x
        y = y - 1 if y > 1 else y + 1 if y < 0 else y
        return x, y

def mh_step(x, t=0):
    y = mutate(x)
    f1 = f(y, t)
    f2 = f(x, t)
    a = min(1, 0 if f1 == 0 else 1 if f2 == 0 else f1 / f2)
    if random() < a:
        return True, y
    return False, x

def sim_mh():
    # MH
    x0 = random(), random()
    x = x0
    z = []
    M = 100000
    c = 0

    for i in range(0, M):
        accept, x = mh_step(x)
        if accept:
            c += 1
        z.append(x)

    print("MH Acceptance rate:", 100 * c / M, "%")
    plt.hexbin([x for x, y in z], [y for x, y in z])

def gen_temps(K):
    return flipud(append(cumprod(repeat(0.9, K - 1)), 0))

def sim_pt_mh():
    # PTMH
    K = 40                # Number of temperatures (of chains)
    N = 2                 # Number of steps before exchange
    M = 100000             # Number of MH steps over all chains

    T = gen_temps(K)
    if K < 200:
        print("PTMH Temperatures:")
        print(T)

    z = [[] for _ in range(0, K)]
    x = list(zip(random(K), random(K)))
    c = repeat(0, K)
    C = int(M / (K * N))

    for i in range(0, C): 
        for j in range(0, K):
            # Simulate N steps (could be done in parallel)
            for n in range(0, N):
                accept, x[j] = mh_step(x[j], T[j])
                if accept:
                    c[j] += 1
                z[j].append(x[j])
    
        # Exchange
        for j in range(K - 1, 0, -1):
            k1 = f(x[j - 1], T[j]) * f(x[j], T[j - 1])
            k2 = f(x[j - 1], T[j - 1]) * f(x[j], T[j])
            a = min(1, 0 if k1 == 0 else 1 if k2 == 0 else k1 / k2)
            if random() < a:
                x[j], x[j - 1] = x[j - 1], x[j]

    print("PTMH Target acceptance rate:", 100 * c[0] / (C * N), "%")
    print("PTMH Highest Temp. acceptance rate:", 100 * c[-1] / (C * N), "%")

    plt.hexbin([x for x, y in z[0]], [y for x, y in z[0]])

def sim_fopt_mh():
    # FOPTMH
    K = 40                # Number of temperatures (of chains)
    N = 2                 # Number of steps before exchange
    C = 2                 # Number of PTMH steps before temperature update
    M = 100000             # Number of MH steps over all chains

    T = gen_temps(K)
    if K < 200:
        print("Initial FOPTMH Temperatures:")
        print(T)

    z = [[] for _ in range(0, K)]
    x = list(zip(random(K), random(K)))
    c = repeat(0, K)
    Nu = repeat(0, K)
    Nd = repeat(0, K)
    S = int(M / (K * N * C))

    for s in range(0, S):
        for i in range(0, C):
            # For all K chains, simulate N steps (could be done in parallel)
            for j in range(0, K):
                for n in range(0, N):
                    accept, x[j] = mh_step(x[j], T[j])
                    if accept:
                        c[j] += 1
                    z[j].append(x[j])

            # Exchange
            for j in range(K - 1, 0, -1):
                k1 = f(x[j - 1], T[j]) * f(x[j], T[j - 1])
                k2 = f(x[j - 1], T[j - 1]) * f(x[j], T[j])
                a = min(1, 0 if k1 == 0 else 1 if k2 == 0 else k1 / k2)
                if random() < a:
                    x[j], x[j - 1] = x[j - 1], x[j]
                    Nu[j] += 1
                    Nd[j - 1] += 1
            for j in range(0, K):
                j1 = randint(0, K)
                j2 = randint(0, K)
                if j1 > j2:
                    j1, j2 = j2, j1
                k1 = f(x[j1], T[j2]) * f(x[j2], T[j1])
                k2 = f(x[j1], T[j1]) * f(x[j], T[j2])
                a = min(1, 0 if k1 == 0 else 1 if k2 == 0 else k1 / k2)
                if random() < a:
                    x[j2], x[j1] = x[j1], x[j2]
                    Nu[j2] += 1
                    Nd[j1] += 1

        # Update temperatures
        Ns = Nu + Nd
        F = Nu / where(Ns > 0, Ns, 1)
        P = cumsum(F)
        total_F = P[-1]
        U = copy(T)
        for i in range(1, K - 1):
            p = total_F * i / K
            k1 = searchsorted(P, p, side='left')
            k2 = 0 if k1 <= 0 else k1 - 1
            k1 = k1
            t = 0 if P[k1] == P[k2] else (p - P[k2]) / (P[k1] - P[k2])
            U[i] = t * T[k1] + (1 - t) * T[k2]
        T = U

    print("FOPTMH Target acceptance rate:", 100 * c[0] / (C * N * S), "%")
    print("FOPTMH Highest Temp. acceptance rate:", 100 * c[-1] / (C * N * S), "%")

    if K < 200:
        print("Final FOPTMH Temperatures:")
        print(T)

    plt.hexbin([x for x, y in z[0]], [y for x, y in z[0]])

def main():
    fig = plt.figure(figsize=(20,10))

    s1 = fig.add_subplot(141)
    sim_mh()
    s2 = fig.add_subplot(142, sharex=s1, sharey=s1)
    sim_pt_mh()
    s3 = fig.add_subplot(143, sharex=s1, sharey=s1)
    sim_fopt_mh()
    plt.axis((0, 1, 0, 1))

    s4 = fig.add_subplot(144)
    xx = linspace(0, 1, 100)
    yy = linspace(0, 1, 100)
    Z = zeros((len(xx), len(yy)))
    for i in range(len(xx)):
        for j in range(len(yy)):
            Z[i, j] = f((xx[i],yy[j]), 0.95)
    plt.contour(xx, yy, Z)

    plt.show()

if __name__ == "__main__":
    main()
