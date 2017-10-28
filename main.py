from triangulation import Triangulation
from gaussian import IntergralationGaussian
from scipy.sparse import lil_matrix
import time
from math import sqrt
from scipy.sparse.linalg import cg, cgs,lsqr, lgmres
import numpy as np

class Fem2D:
    gauss = IntergralationGaussian()
    triandulation = Triangulation()

    def _computing_force_vector(self, list_triangles, n_element, fn_f):
        F = lil_matrix((n_element,1))

        for triangle in list_triangles:
            for i in range(0,3):
                if (triangle.vertices[i].on_bound == False):
                    F[triangle.vertices[i].idx,0] += self.gauss.computing_intergralation_f_multi_base_function_on_triangle(
                                                                                                            fn_f,
                                                                                                            triangle,
                                                                                                            triangle.vertices[i],
                                                                                                            triangle.vertices[i-1],
                                                                                                            triangle.vertices[i-2])
        return F

    def _computing_stiffness(self, list_triangles, n_element, r_const, p_const):
        A = lil_matrix((n_element, n_element))

        if (r_const != 0):
            for triangle in list_triangles:
                for i in range(0, 3):
                    if (triangle.vertices[i].on_bound != True):
                        A[triangle.vertices[i].idx, triangle.vertices[i].idx] += \
                            r_const*self.gauss.computing_intergralation_u_multi_v_function_on_triangle(
                                                                                     triangle,
                                                                                     1/12)

                        if (triangle.vertices[i - 1].on_bound != True):
                            A[triangle.vertices[i].idx, triangle.vertices[i - 1].idx] += \
                                r_const * self.gauss.computing_intergralation_u_multi_v_function_on_triangle(
                                    triangle,
                                    1 / 24)

        if (p_const != 0):
            for triangle in list_triangles:
                for i in range(0, 3):
                    if (triangle.vertices[i].on_bound != True):
                        A[triangle.vertices[i].idx, triangle.vertices[i].idx] += \
                            p_const*self.gauss.computing_intergralation_u_deri_multi_u_deri_function_on_triangle(
                                                                                     triangle,
                                                                                    triangle.vertices[i - 1],
                                                                                    triangle.vertices[i - 2])

                        if (triangle.vertices[i - 1].on_bound != True):
                            temp = \
                                p_const * self.gauss.computing_intergralation_u_deri_multi_v_deri_function_on_triangle(
                                    triangle,
                                    triangle.vertices[i],
                                    triangle.vertices[i - 1],
                                    triangle.vertices[i - 2])

                            A[triangle.vertices[i].idx, triangle.vertices[i - 1].idx] += temp
                            A[triangle.vertices[i - 1].idx, triangle.vertices[i].idx] += temp
        return A

    def cg_method(self, A, F, max_iter, epsilon):
        X = lil_matrix((F.shape[0],1))
        R1 = F.copy()
        P = F.copy()
        norm1 = R1.transpose().dot(R1)[0,0]
        norm2 = 0

        for n_iter in range(1,max_iter+1):
            if (n_iter != 1):
                P = R1 + norm1 * P / norm2

            alpha = norm1 / P.transpose().dot(A).dot(P)[0,0]
            X = X + alpha * P

            norm2 = norm1
            R1 = R1 - alpha * A.dot(P)
            norm1 = R1.transpose().dot(R1)[0,0]

            if (R1.transpose().dot(R1)[0,0] < epsilon):
                break
        return X, n_iter


    def _estimated_error_in_l2(self, list_triangles, fn_root, Un):
        error = 0
        for triangle in list_triangles:
            error += self.gauss.estimate_error_on_element_l2_space(triangle, fn_root, Un)

        return sqrt(error)

    def _estimated_error_in_h10(self, list_triangles, fn_root, fn_root_dev_x, fn_root_dev_y, Un):
        error = 0
        for triangle in list_triangles:
            error += self.gauss.estimate_error_on_element_l2_space(triangle, fn_root, Un)

        for triangle in list_triangles:
                    error += self.gauss.estimate_error_on_element_dev_x(triangle, fn_root_dev_x, Un)

        for triangle in list_triangles:
                    error += self.gauss.estimate_error_on_element_dev_y(triangle, fn_root_dev_y, Un)

        return sqrt(error)

    def dirichlet_boundary(self, fn_f, fn_root, fn_root_dev_x, fn_root_dev_y, n_iter, square_size, r_const, p_const):
        time_start = time.time()

        self.fn_root = fn_root
        self.square_size = square_size
        self.n_iter = n_iter

        print("Deviding triangle element")

        list_triangles, list_inner_vertices, list_bound_vertices = self.triandulation.process_square(square_size, n_iter, plot=False)
        n_element = len(list_triangles)

        part_time = time.time()
        print("Devided {0} triangle elements in {1:2} seconds".format(n_element, part_time - time_start))
        print("Number of vertices inside     : {0}".format(list_inner_vertices.length))
        print("Number of vertices on boundary: {0}".format(list_bound_vertices.length))

        print("Computing force vector")
        self.F = self._computing_force_vector(list_triangles, n_element, fn_f)
        print("Computed force vector in {0} seconds".format(time.time() - part_time))
        part_time = time.time()

        print("Computing stiffness matrix")
        self.A = self._computing_stiffness(list_triangles, n_element, r_const, p_const)
        print("Computed stiffness matrix in {0:2} seconds".format(time.time() - part_time))
        num_nonzero = len(self.A.nonzero()[0])
        print("Number of nonzero values {0}".format(num_nonzero))
        part_time = time.time()


        # print("A", self.A)
        # print("F", self.F)
        #
        # print("Solving equations problem by using Conjugate gradient method")
        # self.Un, time_iter = self.cg_method(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
        # print("Solved CG in {0:2} seconds with {1} iterations".format(time.time() - part_time, time_iter))
        # part_time = time.time()
        # # print(self.Un)

        self.Un = lil_matrix(lgmres(A=self.A,b=np.array([x[0] for x in self.F.toarray()]))[0]).transpose()
        # print(self.Un)

        print("Finished FEM in {0:2} seconds".format(time.time() - time_start))

        l2_error = self._estimated_error_in_l2(list_triangles, fn_root, self.Un)
        print("Error in L2 space: {0} estimated in {1:2} seconds".format(l2_error, time.time() - part_time))
        part_time = time.time()

        l2_error = self._estimated_error_in_h10(list_triangles, fn_root, fn_root_dev_x, fn_root_dev_y, self.Un)
        print("Error in H10 space: {0} estimated in {1:2} seconds".format(l2_error, time.time() - part_time))
        part_time = time.time()

    def error_in_point(self, x, y):
        triangle = self.triandulation.find_exactly_element(self.square_size, self.n_iter, x, y)
        real = self.fn_root(x,y)
        print("#############################################")
        print("Real value    : f({0},{1}) = {2}".format(x,y,real))
        predicted = self.gauss.estimate_point_value_in_triangle(triangle,x,y,self.Un)
        print("Estimate value:              {0}".format(predicted))
        print("Error         :              {0}".format(abs(predicted - real)))


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
                                  n_iter=8, square_size=1, r_const=0, p_const=1)
temp.error_in_point(0.69, 0.69)
