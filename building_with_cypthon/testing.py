from fem import Fem2D

def f(x, y):
    return 2*x*(1-x) + 2*y*(1-y)

def root_function(x, y):
    return x*y*(1-x)*(1-y)

def root_function_deviation_x(x, y):
    return y*(1-y)*(1-2*x)

def root_function_deviation_y(x, y):
    return x*(1-x)*(1-2*y)


print("Processing finite element method with function -Uxx - Uyy = f")
temp = Fem2D()
temp.dirichlet_boundary(fn_f=f, fn_root=root_function,
                          fn_root_dev_x=root_function_deviation_x,
                          fn_root_dev_y=root_function_deviation_y,
                          n_iter=7, square_size=1, r_const=0, p_const=1, plot = False)

temp.error_in_point(0.69, 0.69)
