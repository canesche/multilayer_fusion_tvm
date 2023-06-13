import sympy as sym

Ni = sym.Symbol('Ni')
Nj = sym.Symbol('Nj')
Nk = sym.Symbol('Nk')

Ti_0, Tj_0 = sym.symbols('Ti_0 Tj_0')

expr = Ni*Nj*Nk*(1/Tj_0 + 1/Ti_0 + 2/Nk)

der_expr = expr.diff(Ti_0)

der_second_expr = der_expr.diff(Ti_0)

print(der_expr)
print(der_second_expr)

print(sym.diff(expr, Ti_0, 2, Tj_0, 2))