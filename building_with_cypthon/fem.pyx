from triangulation import Triangulation
from gaussian import IntergralationGaussian
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
import time
from math import sqrt
from scipy.sparse.linalg import cg, cgs,lsqr, lgmres, qmr, lsmr, bicg, bicgstab
import numpy as np

class Fem2D:
    gauss = IntergralationGaussian()
    triandulation = Triangulation()

    def _computing_force_vector(self, list_triangles, n_element, fn_f):
        # F = lil_matrix((n_element,1))
        F = np.zeros(n_element)

        for triangle in list_triangles:
            # temp = self.gauss.computing_intergralation_f_multi_base_function_on_a_triangle(fn_f, triangle)
            # count = -1
            for i in range(0,3):
                if (triangle.vertices[i].on_bound == False):
                    # count += 1
                    # F[triangle.vertices[i].idx] += temp[count]
                    F[triangle.vertices[i].idx] += self.gauss.computing_intergralation_f_multi_base_function_on_triangle(
                                                                                                            fn_f,
                                                                                                            triangle,
                                                                                                            triangle.vertices[i],
                                                                                                            triangle.vertices[i-1],
                                                                                                            triangle.vertices[i-2])
        return lil_matrix(F).transpose()

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
        A_temp = A.tocsr()
        X = csr_matrix((F.shape[0],1))
        R1 = F.tocsr()
        P = F.tocsr()
        norm1 = R1.transpose().dot(R1)[0,0]
        norm2 = 0

        for n_iter in range(1,max_iter+1):
            if (n_iter != 1):
                P = R1 + norm1 * P / norm2

            alpha = norm1 / P.transpose().dot(A_temp).dot(P)[0,0]
            X = X + alpha * P

            norm2 = norm1
            R1 = R1 - alpha * A_temp.dot(P)
            norm1 = R1.transpose().dot(R1)[0,0]

            if (norm1 < epsilon):
                break
        return X, n_iter

    def cg_method_optimize(self, A, F, max_iter, epsilon):
        time_iter = 0

        def callback(xk):
            nonlocal time_iter
            time_iter += 1

        return lil_matrix(cg(A.tocsr(), F.toarray(), callback=callback, maxiter=max_iter, tol=epsilon)[0]).transpose(), time_iter

    def _estimated_error_in_l2(self, list_triangles, fn_root, Un):
        error = 0
        for triangle in list_triangles:
            error += self.gauss.estimate_error_on_element_l2_space(triangle, fn_root, Un)

        return sqrt(error)

    def _estimated_error_in_h10(self, list_triangles, fn_root, fn_root_dev_x, fn_root_dev_y, Un):
        l2_error = 0
        for triangle in list_triangles:
            l2_error += self.gauss.estimate_error_on_element_l2_space(triangle, fn_root, Un)

        fx_error = 0
        for triangle in list_triangles:
            fx_error += self.gauss.estimate_error_on_element_dev_x(triangle, fn_root_dev_x, Un)

        fy_error = 0
        for triangle in list_triangles:
            fy_error += self.gauss.estimate_error_on_element_dev_y(triangle, fn_root_dev_y, Un)

        return sqrt(l2_error), sqrt(l2_error + fx_error + fy_error)

    def dirichlet_boundary(self, fn_f, fn_root, fn_root_dev_x, fn_root_dev_y, n_iter, square_size, r_const, p_const, plot = False):
        time_start = time.time()

        self.fn_root = fn_root
        self.square_size = square_size
        self.n_iter = n_iter

        print("Deviding triangle element")

        self.list_triangles, list_inner_vertices, list_bound_vertices = self.triandulation.process_square(square_size, n_iter, plot=plot)
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
        self.A = self._computing_stiffness(self.list_triangles, n_element, r_const, p_const)
        print("Computed stiffness matrix in {0:2} seconds".format(time.time() - part_time))
        num_nonzero = len(self.A.nonzero()[0])
        print("Number of nonzero values {0}".format(num_nonzero))
        part_time = time.time()


        # print("A", self.A)
        # print("F", self.F)

        print("Solving equations problem by using Conjugate gradient method")
        # self.Un, time_iter = self.cg_method(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
        self.Un, time_iter = self.cg_method_optimize(self.A, self.F, max_iter=num_nonzero, epsilon=1e-10)
        print("Solved CG in {0:2} seconds with {1} iterations".format(time.time() - part_time, time_iter))
        part_time = time.time()

        # print(self.Un)

        # print(self.Un)
        # print(self.A.dot(self.Un) - self.F)

        print("Finished FEM in {0:2} seconds".format(time.time() - time_start))
        part_time = time.time()

        print("Computing error")
        l2_error, h10_error = self._estimated_error_in_h10(self.list_triangles, fn_root, fn_root_dev_x, fn_root_dev_y, self.Un)
        print("Error in L2 space : {0}".format(l2_error))
        print("Error in H10 space: {0} estimated in {1:2} seconds".format(h10_error, time.time() - part_time))
        part_time = time.time()

    def error_in_point(self, x, y):
        triangle_idx = self.triandulation.find_exactly_element(self.square_size, self.n_iter, x, y)
        if (triangle_idx == None):
            predicted = 0
        else:
            predicted = self.gauss.estimate_point_value_in_triangle(self.list_triangles[triangle_idx], x, y, self.Un)
        # print(triangle.vertices[0].x,triangle.vertices[0].y,triangle.vertices[1].x,triangle.vertices[1].y,triangle.vertices[2].x,triangle.vertices[2].y)
        real = self.fn_root(x,y)
        print("#############################################")
        print("Real value    : f({0},{1}) = {2}".format(x,y,real))
        print("Estimate value:              {0}".format(predicted))
        print("Error         :              {0}".format(abs(predicted - real)))