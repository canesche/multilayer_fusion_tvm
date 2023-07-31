import sympy as sym

Ni = 10000
Nj = 10000
Nk = 10000
Nh = 2
Nw = 4

Tk_0=1   
Ti_0=251   
Tj_0=254   
Th_0=2   
Tw_0=4   
#Tk_1=386   
#Ti_1=251   
Tj_1=254   
Th_1=2   
Tw_1=4


'''
Ti_0, Tj_0 = sym.symbols('Ti_0 Tj_0')

expr = Ni*Nj*Nk*(1/Tj_0 + 1/Ti_0 + 2/Nk)

der_expr = expr.diff(Ti_0)

der_second_expr = der_expr.diff(Ti_0)

print(der_expr)
print(der_second_expr)

print(sym.diff(expr, Ti_0, 2, Tj_0, 2))
'''

for Tk_1 in range(1024,8192,1024):
    for Ti_1 in range(1024,8192,1024):
        expr = Nh*Ni*Nj*Nk*Nw*(2*(Th_0 + Ti_1 - 1)*(Tj_0 + Tw_0 - 1)/(Th_0*Ti_1*Tj_0*Tk_1*Tw_0) + (Th_0 + Ti_0 - 1)/(Th_0*Ti_0*Tj_0*Tw_0) + (Tj_0 + Tw_0 - 1)/(Th_0*Ti_0*Tj_0*Tw_0))
        print("%d,%d,%s" %(Tk_1, Ti_1, expr))