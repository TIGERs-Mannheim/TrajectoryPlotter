from sympy import *

phi = symbols("p")

v = symbols("v")
a = symbols("a")

a_1 = symbols("a1")
a_2 = symbols("a2")

v_10 = symbols("v10")
v_11 = symbols("v11")
v_20 = symbols("v20")
v_21 = symbols("v21")

t_10 = symbols("t10")
t_11 = symbols("t11")
t_20 = symbols("t20")
t_21 = symbols("t21")

s_10 = symbols("s10")
s_11 = symbols("s11")
s_20 = symbols("s20")
s_21 = symbols("s21")

init_printing(use_unicode=True)

if __name__ == '__main__':
    res1 = nonlinsolve(
        [
            v_10+a_1*t_10 - v_11,
        ],
        [
            v_10
        ])
    print(res1)

    res2 = nonlinsolve(
        [
            cos(phi) * a - a_1,
            cos(phi) * v - v_11,
            sin(phi) * a - a_2,
            sin(phi) * v - v_21,
            s_10 + v_10 * t_10 + 0.5 * a_1 * t_10 ** 2 + v_11 * t_11 - s_11,
            s_20 + v_20 * t_20 + 0.5 * a_2 * t_20 ** 2 + v_21 * t_21 - s_21,
            t_10 + t_11 - t_20 - t_21,
            v_10 + a_1 * t_10 - v_11,
            v_20 + a_2 * t_20 - v_21,
        ],
        [
            phi, a_1, a_2, v_11, v_21
        ])
    print(res2)
