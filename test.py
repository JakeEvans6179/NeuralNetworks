import sympy

J, W = sympy.symbols('J, W')

J = 1/W
print(J)

diff = sympy.diff(J, W)
print(diff)

print(diff.subs([(W, 2)]))
