
replace_dict = {'^':'**',
                't':'x'}
eq = 'x^2+2*x'

for d,v in replace_dict.items():
    eq = eq.replace(d,v)


# peicewice
if '}' in eq:
    # split ouside
    # split inside
    eq_ls = eq.split(',')