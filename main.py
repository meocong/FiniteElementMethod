from triangulation import Triangulation
from gaussian import IntergralationGaussian
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
import time
from math import sqrt, exp, sin, cos
import math
from scipy.sparse.linalg import cg, cgs,lsqr, lgmres, qmr, lsmr, bicg, bicgstab
import numpy as np
from pyamg import smoothed_aggregation_solver
import multiprocessing as mp

class Fem2D:
    gauss = IntergralationGaussian()
    triandulation = Triangulation()

    # Multithreading update F
    @staticmethod
    def update_F(args):
        fn_f, triangle = args
        res = []
        for i in range(0, 3):
            if (triangle.vertices[i].on_bound == False):
                res.append((triangle.vertices[i].idx,IntergralationGaussian.computing_intergralation_f_multi_base_function_on_triangle(
                    fn_f,
                    triangle,
                    triangle.vertices[i],
                    triangle.vertices[i - 1],
                    triangle.vertices[i - 2])))
        return res

    def _computing_force_vector(self, list_triangles, n_element, fn_f):
        F = np.zeros(n_element)

        # Processing F with multithreading
        with mp.Pool(processes=10) as pool:
            result = pool.map(self.update_F, [(fn_f, triangle) for triangle in list_triangles])
        for res in result:
            for update in res:
                F[update[0]] += update[1]

        return lil_matrix(F).transpose()

    def check_function_is_constant(self, fun):
        try:
            fun("xxx", "xxx")
            return True
        except:
            return False

    @staticmethod
    def multiprocessing_r_const_u_v(args):
        r_const, triangle = args
        res = []
        for i in range(0, 3):
            if (triangle.vertices[i].on_bound != True):
                res.append((triangle.vertices[i].idx, triangle.vertices[i].idx,
                    r_const * IntergralationGaussian.computing_intergralation_u_multi_v_function_on_triangle_r_const(
                        triangle,
                        1.0 / 12)))

                if (triangle.vertices[i - 1].on_bound != True):
                    res.append((triangle.vertices[i].idx, triangle.vertices[i - 1].idx,
                        r_const * IntergralationGaussian.computing_intergralation_u_multi_v_function_on_triangle_r_const(
                            triangle,
                            1.0 / 24)))
        return res

    @staticmethod
    def multiprocessing_fn_r_u_v(args):
        fn_r, triangle = args
        res = []
        for i in range(0, 3):
            if (triangle.vertices[i].on_bound != True):
                res.append((triangle.vertices[i].idx, triangle.vertices[i].idx,
                    IntergralationGaussian.computing_intergralation_u_multi_u_function_on_triangle_using_element_center(
                        fn_r,
                        triangle,
                        triangle.vertices[i],
                        triangle.vertices[i - 1],
                        triangle.vertices[i - 2])))

                if (triangle.vertices[i - 1].on_bound != True):
                    res.append((triangle.vertices[i].idx, triangle.vertices[i - 1].idx,
                        IntergralationGaussian.computing_intergralation_u_multi_v_function_on_triangle_using_element_center(
                            fn_r,
                            triangle,
                            triangle.vertices[i],
                            triangle.vertices[i - 1],
                            triangle.vertices[i - 2])))
        return res

    @staticmethod
    def multiprocessing_p_const_u_deri_v_deri(args):
        p_const, triangle = args
        res = []
        for i in range(0, 3):
            if (triangle.vertices[i].on_bound != True):
                res.append((triangle.vertices[i].idx, triangle.vertices[i].idx,
                    p_const * IntergralationGaussian.computing_intergralation_u_deri_multi_u_deri_function_on_triangle_p_const(
                        triangle,
                        triangle.vertices[i - 1],
                        triangle.vertices[i - 2])))

                if (triangle.vertices[i - 1].on_bound != True):
                    res.append((triangle.vertices[i].idx, triangle.vertices[i - 1].idx,
                        p_const * IntergralationGaussian.computing_intergralation_u_deri_multi_v_deri_function_on_triangle_p_const(
                            triangle,
                            triangle.vertices[i],
                            triangle.vertices[i - 1],
                            triangle.vertices[i - 2])))

        return res

    @staticmethod
    def multiprocessing_fn_p_u_deri_v_deri(args):
        fn_p, triangle = args
        res = []
        for i in range(0, 3):
            if (triangle.vertices[i].on_bound != True):
                res.append((triangle.vertices[i].idx, triangle.vertices[i].idx,
                    IntergralationGaussian.computing_intergralation_u_deri_multi_u_deri_function_on_triangle_using_element_center(
                        fn_p,
                        triangle,
                        triangle.vertices[i],
                        triangle.vertices[i - 1],
                        triangle.vertices[i - 2])))

                if (triangle.vertices[i - 1].on_bound != True):
                    res.append((triangle.vertices[i].idx, triangle.vertices[i - 1].idx,
                        IntergralationGaussian.computing_intergralation_u_deri_multi_v_deri_function_on_triangle_using_element_center(
                            fn_p,
                            triangle,
                            triangle.vertices[i],
                            triangle.vertices[i - 1],
                            triangle.vertices[i - 2])))

        return res

    def _computing_stiffness(self, list_triangles, n_element, fn_r, fn_p):
        A = lil_matrix((n_element, n_element))

        if (self.check_function_is_constant(fn_r) == True):
            r_const = fn_r("xxx",'xxx')
            if (r_const != 0):
                # Multi processing intergral r_const * u * v
                with mp.Pool(processes=10) as pool:
                    result = pool.map(self.multiprocessing_r_const_u_v, [(r_const, triangle) for triangle in list_triangles])
                for res in result:
                    for update in res:
                        A[update[0], update[1]] += update[2]

                        if (update[0] != update[1]):
                            A[update[1], update[0]] += update[2]
        else:
            # Multi processing intergral fn_r * u * v
            with mp.Pool(processes=10) as pool:
                result = pool.map(self.multiprocessing_fn_r_u_v,
                                  [(fn_r, triangle) for triangle in list_triangles])
            for res in result:
                for update in res:
                    A[update[0], update[1]] += update[2]

                    if (update[0] != update[1]):
                        A[update[1], update[0]] += update[2]

        if (self.check_function_is_constant(fn_p) == True):
            p_const = fn_p("xxx",'xxx')

            if (p_const != 0):
                # Multi processing intergral p_const * u_deri * v_deri
                with mp.Pool(processes=10) as pool:
                    result = pool.map(self.multiprocessing_p_const_u_deri_v_deri, [(p_const, triangle) for triangle in list_triangles])
                for res in result:
                    for update in res:
                        A[update[0], update[1]] += update[2]

                        if (update[0] != update[1]):
                            A[update[1], update[0]] += update[2]

        else:
            # Multi processing intergral fn_p * u_deri * v_deri
            with mp.Pool(processes=10) as pool:
                result = pool.map(self.multiprocessing_fn_p_u_deri_v_deri,
                                  [(fn_p, triangle) for triangle in list_triangles])
            for res in result:
                for update in res:
                    A[update[0], update[1]] += update[2]

                    if (update[0] != update[1]):
                        A[update[1], update[0]] += update[2]

        return A

    def cg_method(self, A, F, max_iter, epsilon):
        A_temp = A.tocsr()
        X = np.zeros(F.shape[0])
        R1 = F.transpose().tocsr()
        P = F.transpose().tocsr()
        norm1 = R1.dot(R1.transpose())[0,0]
        norm2 = 0

        for n_iter in range(1,max_iter+1):
            if (n_iter != 1):
                P = R1 + norm1 * P / norm2

            alpha = norm1 / P.dot(A_temp).dot(P.transpose())[0,0]
            X += alpha * P

            norm2 = norm1
            R1 = R1 - alpha * A_temp.dot(P.transpose()).transpose()
            norm1 = R1.dot(R1.transpose())[0,0]

            if (norm1 < epsilon):
                break

        return np.array(X)[0], n_iter

    def cg_method_optimize(self, A, F, max_iter, epsilon):
        time_iter = 0

        def callback(xk):
            nonlocal time_iter
            time_iter += 1

        temp = A.tocsr()
        ml = smoothed_aggregation_solver(temp)
        M = ml.aspreconditioner()
        return np.array(cg(temp, F.toarray(), callback=callback, maxiter=max_iter, tol=epsilon, M=M)[0]), time_iter

    def _estimated_error_in_l2(self, list_triangles, fn_root, Un):
        error = 0
        for triangle in list_triangles:
            error += self.gauss.estimate_error_on_element_l2_space(triangle, fn_root, Un)

        return sqrt(error)

    @staticmethod
    def _multiprocessing_error(args):
        triangle, fn_root, fn_root_dev_x, fn_root_dev_y, Un = args
        return (IntergralationGaussian.estimate_error_on_element_l2_space(triangle, fn_root, Un),
                IntergralationGaussian.estimate_error_on_element_dev_x(triangle, fn_root_dev_x, Un),
                IntergralationGaussian.estimate_error_on_element_dev_y(triangle, fn_root_dev_y, Un))


    def _estimated_error_in_h10(self, list_triangles, fn_root, fn_root_dev_x, fn_root_dev_y, Un):
        # l2_error = 0
        # for triangle in list_triangles:
        #     l2_error += self.gauss.estimate_error_on_element_l2_space(triangle, fn_root, Un)

        # Multi processing l2_error
        with mp.Pool(processes=5) as pool:
            result = np.array(pool.map(self._multiprocessing_error,
                              [(triangle, fn_root, fn_root_dev_x, fn_root_dev_y, Un) for triangle in list_triangles]))
        l2_error = np.sum(result[:,0])
        fx_error = np.sum(result[:,1])
        fy_error = np.sum(result[:,2])

        # fx_error = 0
        # for triangle in list_triangles:
        #     fx_error += self.gauss.estimate_error_on_element_dev_x(triangle, fn_root_dev_x, Un)

        # fy_error = 0
        # for triangle in list_triangles:
        #     fy_error += self.gauss.estimate_error_on_element_dev_y(triangle, fn_root_dev_y, Un)

        return sqrt(l2_error), sqrt(l2_error + fx_error + fy_error)

    def dirichlet_boundary_rectangle(self, fn_f, fn_root, fn_root_dev_x, fn_root_dev_y, fn_r, fn_p, plot,
                           square_size, adaptive, n_iter, max_element):
        max_element = int(max_element)
        time_start = time.time()

        self.fn_root = fn_root
        self.square_size = square_size
        self.n_iter = n_iter

        print("Deviding triangle element")

        self.list_triangles, list_inner_vertices, list_bound_vertices, dict_segment = \
            self.triandulation.initalize_process_square(adaptive=adaptive,
                                                        square_size=square_size, n_iter=n_iter, plot=plot, fn_f =fn_f)
        n_element = len(self.list_triangles)

        part_time = time.time()
        print("Devided {0} triangle elements in {1:2} seconds".format(n_element, part_time - time_start))
        print("Number of vertices inside     : {0}".format(list_inner_vertices.length))
        print("Number of vertices on boundary: {0}".format(list_bound_vertices.length))

        print("Computing force vector")
        self.F = self._computing_force_vector(self.list_triangles, n_element, fn_f)
        print("Computed force vector in {0} seconds".format(time.time() - part_time))
        part_time = time.time()

        print("Computing stiffness matrix")
        self.A = self._computing_stiffness(self.list_triangles, n_element, fn_r, fn_p)
        print("Computed stiffness matrix in {0:2} seconds".format(time.time() - part_time))
        num_nonzero = len(self.A.nonzero()[0])
        print("Number of nonzero values {0}".format(num_nonzero))
        part_time = time.time()

        print("Solving equations problem by using Preconditioner conjugate gradient method")
        # self.Un, time_iter = self.cg_method(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
        self.Un, time_iter = self.cg_method_optimize(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
        # print(self.Un, time_iter)
        print("Solved PCG in {0:2} seconds with {1} iterations".format(time.time() - part_time, time_iter))


        print("Finished FEM in {0:2} seconds".format(time.time() - time_start))
        part_time = time.time()

        if (adaptive == True):
            step = 0
            print("Initial {0} elements".format(len(self.list_triangles)))
            while (len(self.list_triangles) < max_element):
                step += 1
                print("  Adaptive threshold {0}th step".format(step))

                self.list_triangles, list_inner_vertices, list_bound_vertices, dict_segment = \
                    self.triandulation.process_free_shape_adaptively(
                                                          list_triangles=self.list_triangles,
                                                          vertices_inner=list_inner_vertices,
                                                          vertices_bound=list_bound_vertices,
                                                          dict_segment=dict_segment,
                                                          Un=self.Un,
                                                          plot=plot,fn_f=fn_f,max_element=max_element)
                n_element = len(self.list_triangles)

                part_time = time.time()
                print("  Devided {0} triangle elements in {1:2} seconds".format(n_element, part_time - time_start))
                print("  Number of vertices inside     : {0}".format(list_inner_vertices.length))
                print("  Number of vertices on boundary: {0}".format(list_bound_vertices.length))

                print("  Computing force vector")
                self.F = self._computing_force_vector(self.list_triangles, n_element, fn_f)
                print("  Computed force vector in {0} seconds".format(time.time() - part_time))
                part_time = time.time()

                print("  Computing stiffness matrix")
                self.A = self._computing_stiffness(self.list_triangles, n_element, fn_r, fn_p)
                print("  Computed stiffness matrix in {0:2} seconds".format(time.time() - part_time))
                num_nonzero = len(self.A.nonzero()[0])
                print("  Number of nonzero values {0}".format(num_nonzero))
                part_time = time.time()

                print("  Solving equations problem by using Preconditioner conjugate gradient method")
                # self.Un, time_iter = self.cg_method(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
                self.Un, time_iter = self.cg_method_optimize(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
                print("  Solved PCG in {0:2} seconds with {1} iterations".format(time.time() - part_time, time_iter))

        if (fn_root is not None):
            print("Computing error")
            l2_error, h10_error = self._estimated_error_in_h10(self.list_triangles, fn_root, fn_root_dev_x, fn_root_dev_y, self.Un)
            print("Error in L2 space : {0}".format(l2_error))
            print("Error in H10 space: {0} estimated in {1:2} seconds".format(h10_error, time.time() - part_time))

    def dirichlet_boundary_free_shape(self, fn_f, fn_root, fn_root_dev_x, fn_root_dev_y, fn_r, fn_p, plot,
                           adaptive, n_iter, max_element, shape_dir = None, is_map = False,
                                      map_width = None, map_height = None,
                                      shape = None, option="", max_vertice_add_each_edge=1, max_vertice_added_near_each_vertice=0):
        max_element = int(max_element)
        time_start = time.time()

        self.fn_root = fn_root
        self.n_iter = n_iter

        print("Deviding initial triangle element")

        self.list_triangles, list_inner_vertices, list_bound_vertices, dict_segment = \
            self.triandulation.initalize_process_free_shape(shape_dir=shape_dir, is_map=is_map, map_width=map_width,
                                                  map_height=map_height,
                                                  shape=shape, option=option,
                                                  plot=plot, fn_f =fn_f,
                                                  max_vertice_add_each_edge=max_vertice_add_each_edge,
                                                  max_vertice_added_near_each_vertice=max_vertice_added_near_each_vertice)
        n_element = len(self.list_triangles)

        part_time = time.time()
        print("Devided {0} triangle elements in {1:2} seconds".format(n_element, part_time - time_start))
        print("Number of vertices inside     : {0}".format(list_inner_vertices.length))
        print("Number of vertices on boundary: {0}".format(list_bound_vertices.length))

        print("Computing force vector")
        self.F = self._computing_force_vector(self.list_triangles, n_element, fn_f)
        print("Computed force vector in {0} seconds".format(time.time() - part_time))
        part_time = time.time()

        print("Computing stiffness matrix")
        self.A = self._computing_stiffness(self.list_triangles, n_element, fn_r, fn_p)
        print("Computed stiffness matrix in {0:2} seconds".format(time.time() - part_time))
        num_nonzero = len(self.A.nonzero()[0])
        print("Number of nonzero values {0}".format(num_nonzero))
        part_time = time.time()

        print("Solving equations problem by using Preconditioner conjugate gradient method")
        # self.Un, time_iter = self.cg_method(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
        self.Un, time_iter = self.cg_method_optimize(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
        print("Solved PCG in {0:2} seconds with {1} iterations".format(time.time() - part_time, time_iter))

        if (adaptive == True):
            step = 0
            print("Initial {0} elements".format(len(self.list_triangles)))
            while (len(self.list_triangles) < max_element):
                step += 1
                print("  Adaptive threshold {0}th step".format(step))

                self.list_triangles, list_inner_vertices, list_bound_vertices, dict_segment = \
                    self.triandulation.process_free_shape_adaptively(
                                                          list_triangles=self.list_triangles,
                                                          vertices_inner=list_inner_vertices,
                                                          vertices_bound=list_bound_vertices,
                                                          dict_segment=dict_segment,
                                                          Un=self.Un,
                                                          plot=plot,fn_f=fn_f,max_element=max_element)
                n_element = len(self.list_triangles)

                part_time = time.time()
                print("  Devided {0} triangle elements in {1:2} seconds".format(n_element, part_time - time_start))
                print("  Number of vertices inside     : {0}".format(list_inner_vertices.length))
                print("  Number of vertices on boundary: {0}".format(list_bound_vertices.length))

                print("  Computing force vector")
                self.F = self._computing_force_vector(self.list_triangles, n_element, fn_f)
                print("  Computed force vector in {0} seconds".format(time.time() - part_time))
                part_time = time.time()

                print("  Computing stiffness matrix")
                self.A = self._computing_stiffness(self.list_triangles, n_element, fn_r, fn_p)
                print("  Computed stiffness matrix in {0:2} seconds".format(time.time() - part_time))
                num_nonzero = len(self.A.nonzero()[0])
                print("  Number of nonzero values {0}".format(num_nonzero))
                part_time = time.time()

                print("  Solving equations problem by using Preconditioner conjugate gradient method")
                # self.Un, time_iter = self.cg_method(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
                self.Un, time_iter = self.cg_method_optimize(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
                print("  Solved PCG in {0:2} seconds with {1} iterations".format(time.time() - part_time, time_iter))

        print("Finished FEM in {0:2} seconds".format(time.time() - time_start))
        part_time = time.time()

        if (fn_root is not None):
            print("Computing error")
            l2_error, h10_error = self._estimated_error_in_h10(self.list_triangles, fn_root, fn_root_dev_x, fn_root_dev_y, self.Un)
            print("Error in L2 space : {0}".format(l2_error))
            print("Error in H10 space: {0} estimated in {1:2} seconds".format(h10_error, time.time() - part_time))

    def error_in_point(self, x, y):
        triangle_idx = self.triandulation.find_exactly_element(self.square_size, self.n_iter, x, y)
        if (triangle_idx == None):
            predicted = 0
        else:
            predicted = self.gauss.estimate_point_value_in_triangle(self.list_triangles[triangle_idx], x, y, self.Un)
        # print(triangle.vertices[0].x,triangle.vertices[0].y,triangle.vertices[1].x,triangle.vertices[1].y,triangle.vertices[2].x,triangle.vertices[2].y)

        print("#############################################")
        print("Estimate value:              {0}".format(predicted))

        if (self.fn_root is not None):
            real = self.fn_root(x, y)
            print("Real value    : f({0},{1}) = {2}".format(x, y, real))
            print("Error         :              {0}".format(abs(predicted - real)))


class test1:
    def f(self, x, y):
        return 2*x*(1-x) + 2*y*(1-y)

    def root_function(self, x, y):
        return x*y*(1-x)*(1-y)

    def root_function_deviation_x(self, x, y):
        return y*(1-y)*(1-2*x)

    def root_function_deviation_y(self, x, y):
        return x*(1-x)*(1-2*y)

    def r(self, x, y):
        return 0

    def p(self, x,y):
        return 1

class test2:
    def __init__(self):
        self.a = 10
        self.a2 = self.a**2

    def f(self, x, y):
        # 4*a^2*(1-a*r^2)*e^(-a^2*r^2)
        r2 = (x-0.5)**2 + (y-0.5)**2
        # print("xxx", 4*self.a2*(1-self.a*r2)*exp(-self.a2*r2), x, y, self.a, self.a2, r2)
        return 400*self.a2*(1-self.a*r2)*np.exp(-self.a2*r2)

    def root_function(self, x, y):
        return self.a * np.exp(-self.a * (x-0.5)**2 + (y-0.5)**2)*100

    def root_function_deviation_x(self, x, y):
        return 0

    def root_function_deviation_y(self, x, y):
        return 0

    def r(self, x, y):
        return 0

    def p(self, x,y):
        return 1

class test3:
    def __init__(self):
        self.a = 10
        self.a2 = self.a**2

    def f(self, x, y):
        return 2*math.pi**2 * sin(math.pi * x) * sin(math.pi * y)*exp(-20*(x**2+y**2))

    def root_function(self, x, y):
        return None

    def root_function_deviation_x(self, x, y):
        return None

    def root_function_deviation_y(self, x, y):
        return None

    def r(self, x, y):
        return 0

    def p(self, x,y):
        return 1


############## Finite element method on rectangle with equally devided ###################
# print("Processing finite element method with function -Uxx - Uyy = f")
# temp = Fem2D()
# test = test2()
# temp.dirichlet_boundary_rectangle(fn_f=test.f, fn_root=test.root_function,
#                           fn_root_dev_x=test.root_function_deviation_x,
#                           fn_root_dev_y=test.root_function_deviation_y,
#                           fn_r=test.r, fn_p=test.p, plot = True, square_size=1, adaptive=True,
#                           n_iter=3, max_element = 1024)

# for i in range(0,10):
#     temp.dirichlet_boundary_rectangle(fn_f=test.f, fn_root=test.root_function,
#                                       fn_root_dev_x=test.root_function_deviation_x,
#                                       fn_root_dev_y=test.root_function_deviation_y,
#                                       fn_r=test.r, fn_p=test.p, plot=False, square_size=1, adaptive=False,
#                                       threshold_adaptive=0.5,
#                                       n_iter=i, max_element=1e7)


############## Finite element method on rectangle with free heap adaptive devided ###################
# print("Processing finite element method with function -Uxx - Uyy = f")
# temp = Fem2D()
# test = test1()
# temp.dirichlet_boundary_free_shape(fn_f=test.f, fn_root=test.root_function,
#                           fn_root_dev_x=test.root_function_deviation_x,
#                           fn_root_dev_y=test.root_function_deviation_y,
#                           fn_r=test.r, fn_p=test.p, plot = True, adaptive=True,
#                           n_iter=2, max_element = 2024, shape=[[0,0],[1,0],[1,1],[0,1]],
#                                    is_map=True, map_width=1, map_height=1, option="pq30Da.005", max_vertice_add_each_edge=30,
#                                    max_vertice_added_near_each_vertice=4)

# temp = Fem2D()
# test = test1()
# for i in range(0,10):
#     temp.dirichlet_boundary_free_shape(fn_f=test.f, fn_root=test.root_function,
#                               fn_root_dev_x=test.root_function_deviation_x,
#                               fn_root_dev_y=test.root_function_deviation_y,
#                               fn_r=test.r, fn_p=test.p, plot = False, adaptive=True, threshold_adaptive=None,
#                               n_iter=0, max_element = 4**(i+1), shape=[[0,0],[1,0],[1,1],[0,1]],
#                                        is_map=True, map_width=1, map_height=1, option="pq30a.01D")

############## Finite element method on ha noi map with free heap adaptive devided ###################
print("Processing finite element method with function -Uxx - Uyy = f")
temp = Fem2D()
test = test2()
temp.dirichlet_boundary_free_shape(fn_f=test.f, fn_root=None,
                          fn_root_dev_x=None,
                          fn_root_dev_y=None,
                          fn_r=test.r, fn_p=test.p, plot = True, adaptive=False,
                          n_iter=2, max_element = 1e3, shape_dir="/Users/nguyenviet/project/FiniteElementMethod/testing triangulation/hanoi_polygon.txt",
                                   is_map=True, map_width=8/5, map_height=1, option="pq20Da.00000001")



temp.error_in_point(0.51, 0.52)
