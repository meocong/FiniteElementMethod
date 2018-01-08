from math import sqrt
from shape.triangle import Triangle
from vertices import Vertex
import numpy as np

class IntergralationGaussian:
    @staticmethod
    def transfrom_x_y_from_csi_eta(csi, eta, ver1, ver2, ver3):
        return (ver2.x - ver1.x) * csi + (ver3.x - ver1.x) * eta + ver1.x, \
               (ver2.y - ver1.y) * csi + (ver3.y - ver1.y) * eta + ver1.y

    @staticmethod
    def transform_csi_eta_from_x_y(x, y, v1, v2, v3):
        delta = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x)
        return ((v3.y - v1.y) * (x - v1.x) - (v3.x - v1.x) * (y - v1.y)) / delta, \
              -((v2.y - v1.y) * (x - v1.x) - (v2.x - v1.x) * (y - v1.y)) / delta

    @staticmethod
    def dev_csi_x(v1,v2,v3):
        delta = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x)
        return (v3.y - v1.y) / delta

    @staticmethod
    def dev_eta_x(v1,v2,v3):
        delta = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x)
        return -(v2.y - v1.y) / delta

    @staticmethod
    def dev_csi_y(v1,v2,v3):
        delta = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x)
        return - (v3.x - v1.x) / delta

    @staticmethod
    def dev_eta_y(v1,v2,v3):
        delta = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x)
        return (v2.x - v1.x) / delta

    @staticmethod
    def _gaussian(g):
        sqrt15 = sqrt(15)
        return (0.5)* ((9.0/40) * g(1.0/3, 1.0/3)
                        +((155 - sqrt15) / 1200)*g((6 - sqrt15)*(1.0/21), (1.0/21)*(6 - sqrt15))
                        +((155 - sqrt15) / 1200)*g((9 + 2*sqrt15)*(1.0/21), (6 - sqrt15)*(1.0/21))
                        +((155 - sqrt15) / 1200)*g((6 - sqrt15)*(1.0/21), (9 + 2*sqrt15)*(1.0/21))
                        +((155 + sqrt15) / 1200)*g((6 + sqrt15)*(1.0/21), (6 + sqrt15)*(1.0/21))
                        +((155 + sqrt15) / 1200)*g((6 + sqrt15)*(1.0/21), (9 - 2*sqrt15)*(1.0/21))
                        +((155 + sqrt15) / 1200)*g((9 - 2*sqrt15)*(1.0/21), (6 + sqrt15)*(1.0/21)))

    def computing_intergralation_f_on_triangle(self, fn_f, triangle:Triangle):
        def new_func(csi, eta):
            x,y = IntergralationGaussian.transfrom_x_y_from_csi_eta(csi, eta,
                                                                    triangle.vertices[0],
                                                                    triangle.vertices[1],
                                                                    triangle.vertices[2])
            return fn_f(x, y)

        return triangle.area2()*IntergralationGaussian._gaussian(new_func)

    def computing_intergralation_f2_on_triangle(self, fn_f, triangle:Triangle):
        def new_func(csi, eta):
            x,y = IntergralationGaussian.transfrom_x_y_from_csi_eta(csi, eta,
                                                                    triangle.vertices[0],
                                                                    triangle.vertices[1],
                                                                    triangle.vertices[2])
            return fn_f(x, y)**2

        return triangle.area2()*IntergralationGaussian._gaussian(new_func)

    def computing_intergralation_adaptive_error_on_triangle(self, fn_f, triangle:Triangle):
        def new_func(csi, eta):
            x,y = IntergralationGaussian.transfrom_x_y_from_csi_eta(csi, eta,
                                                                    triangle.vertices[0],
                                                                    triangle.vertices[1],
                                                                    triangle.vertices[2])
            return fn_f(x, y)**2

        return (triangle.area2()/2)**(1/2)*IntergralationGaussian._gaussian(new_func)

    @staticmethod
    def computing_intergralation_f_multi_base_function_on_triangle(fn_f, triangle:Triangle, ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):
        def new_func(csi, eta):
            x,y = IntergralationGaussian.transfrom_x_y_from_csi_eta(csi, eta, ver1, ver2, ver3)
            return fn_f(x, y) * (1 - csi - eta)

        return triangle.area2()*IntergralationGaussian._gaussian(new_func)\

    @staticmethod
    def computing_intergralation_f_multi_base_function_on_triangle_using_element_center(fn_f, triangle:Triangle, ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):
        def new_func(csi, eta):
            x,y = IntergralationGaussian.transfrom_x_y_from_csi_eta(csi, eta, ver1, ver2, ver3)
            return fn_f(x, y) * (1 - csi - eta)

        return triangle.area2()*IntergralationGaussian._gaussian(new_func)

    def computing_intergralation_f_multi_base_function_on_a_triangle(self, fn_f, triangle:Triangle):
        def new_func(csi, eta):
            x,y = self.transfrom_x_y_from_csi_eta(csi, eta, triangle.vertices[0], triangle.vertices[1], triangle.vertices[2])
            fn_fxy = fn_f(x, y)
            vector = []
            if (triangle.vertices[0].on_bound == False):
                vector.append(fn_fxy * (1 - csi - eta))
            if (triangle.vertices[1].on_bound == False):
                vector.append(fn_fxy * csi)
            if (triangle.vertices[2].on_bound == False):
                vector.append(fn_fxy * eta)
            return np.array(vector)

        return triangle.area2()*self._gaussian(new_func)

    @staticmethod
    def estimate_error_on_element_l2_space(triangle:Triangle, fn_root, Un):
        v1 = triangle.vertices[0]
        v2 = triangle.vertices[1]
        v3 = triangle.vertices[2]

        def new_func(csi, eta):
            x, y = IntergralationGaussian.transfrom_x_y_from_csi_eta(csi, eta, v1, v2, v3)

            delta = fn_root(x, y)
            if v1.on_bound == False:
                delta -= Un[v1.idx] * (1 - csi - eta)
            if v2.on_bound == False:
                delta -= Un[v2.idx] * csi
            if v3.on_bound == False:
                delta -= Un[v3.idx] * eta
            return delta**2

        return triangle.area2()*IntergralationGaussian._gaussian(new_func)

    @staticmethod
    def estimate_error_on_element_dev_x(triangle:Triangle, fn_dev_x, Un):
        v1 = triangle.vertices[0]
        v2 = triangle.vertices[1]
        v3 = triangle.vertices[2]

        def new_func(csi, eta):
            x, y = IntergralationGaussian.transfrom_x_y_from_csi_eta(csi, eta, v1, v2, v3)

            delta = fn_dev_x(x, y)
            if v1.on_bound == False:
                delta += Un[v1.idx] * (IntergralationGaussian.dev_csi_x(v1,v2,v3) + IntergralationGaussian.dev_eta_x(v1,v2,v3))
            if v2.on_bound == False:
                delta -= Un[v2.idx] * IntergralationGaussian.dev_csi_x(v1,v2,v3)
            if v3.on_bound == False:
                delta -= Un[v3.idx] * IntergralationGaussian.dev_eta_x(v1,v2,v3)
            return delta**2

        return triangle.area2()*IntergralationGaussian._gaussian(new_func)

    @staticmethod
    def estimate_error_on_element_dev_y(triangle:Triangle, fn_dev_y, Un):
        v1 = triangle.vertices[0]
        v2 = triangle.vertices[1]
        v3 = triangle.vertices[2]

        def new_func(csi, eta):
            x, y = IntergralationGaussian.transfrom_x_y_from_csi_eta(csi, eta, v1, v2, v3)

            delta = fn_dev_y(x, y)
            if v1.on_bound == False:
                delta += Un[v1.idx] * (IntergralationGaussian.dev_csi_y(v1,v2,v3) + IntergralationGaussian.dev_eta_y(v1,v2,v3))
            if v2.on_bound == False:
                delta -= Un[v2.idx] * IntergralationGaussian.dev_csi_y(v1,v2,v3)
            if v3.on_bound == False:
                delta -= Un[v3.idx] * IntergralationGaussian.dev_eta_y(v1,v2,v3)
            return delta**2

        return triangle.area2()*IntergralationGaussian._gaussian(new_func)

    def estimate_point_value_in_triangle(self, triangle:Triangle, x, y, Un):
        if (triangle == None):
            return 0
        v1 = triangle.vertices[0]
        v2 = triangle.vertices[1]
        v3 = triangle.vertices[2]

        csi, eta = self.transform_csi_eta_from_x_y(x, y, v1, v2, v3)

        delta = 0
        if v1.on_bound == False:
            delta += Un[v1.idx] * (1-csi-eta)
        if v2.on_bound == False:
            delta += Un[v2.idx] * csi
        if v3.on_bound == False:
            delta += Un[v3.idx] * eta

        return  delta

    @staticmethod
    def computing_intergralation_u_deri_multi_u_deri_function_on_triangle_p_const(triangle:Triangle,
                                                           ver2:Vertex,
                                                           ver3:Vertex):

        return ((ver2.x - ver3.x)**2 + (ver2.y - ver3.y)**2)/(2*triangle.area2())

    @staticmethod
    def computing_intergralation_u_deri_multi_v_deri_function_on_triangle_p_const(triangle:Triangle, ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):

        return ((ver2.y - ver3.y)*(ver3.y - ver1.y) +(ver2.x - ver3.x)*(ver3.x - ver1.x) )/(2*triangle.area2())\

    @staticmethod
    def computing_intergralation_u_deri_multi_u_deri_function_on_triangle_using_element_center(fn_p, triangle:Triangle,
                                                           ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):

        return ((ver2.x - ver3.x)**2 + (ver2.y - ver3.y)**2)/(2*triangle.area2()) \
               * fn_p((ver1.x + ver2.x + ver3.x)/3.0,(ver1.y + ver2.y + ver3.y)/3.0)

    @staticmethod
    def computing_intergralation_u_deri_multi_v_deri_function_on_triangle_using_element_center(fn_p,
                                                                                               triangle:Triangle,
                                                                                                        ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):

        return ((ver2.y - ver3.y)*(ver3.y - ver1.y) +(ver2.x - ver3.x)*(ver3.x - ver1.x) )/(2*triangle.area2()) \
               * fn_p((ver1.x + ver2.x + ver3.x)/3.0,(ver1.y + ver2.y + ver3.y)/3.0)

    @staticmethod
    def computing_intergralation_u_multi_v_function_on_triangle_r_const(triangle:Triangle, alpha):

        return triangle.area2()*alpha

    @staticmethod
    def computing_intergralation_u_multi_u_function_on_triangle(fn_r, triangle:Triangle, ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):

        def new_func(csi, eta):
            x,y = IntergralationGaussian.transfrom_x_y_from_csi_eta(csi, eta, ver1, ver2, ver3)
            return fn_r(x, y) * (1 - csi - eta)**2

        return triangle.area2()*IntergralationGaussian._gaussian(new_func)

    @staticmethod
    def computing_intergralation_u_multi_u_function_on_triangle_using_element_center(fn_r, triangle:Triangle, ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):

        return triangle.area2()/12.0 * fn_r((ver1.x + ver2.x + ver3.x)/3.0,(ver1.y + ver2.y + ver3.y)/3.0)

    @staticmethod
    def computing_intergralation_u_multi_v_function_on_triangle(fn_r, triangle:Triangle, ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):

        def new_func(csi, eta):
            x,y = IntergralationGaussian.transfrom_x_y_from_csi_eta(csi, eta, ver1, ver2, ver3)
            return fn_r(x, y) * (1 - csi - eta) * csi

        return triangle.area2()*IntergralationGaussian._gaussian(new_func)

    @staticmethod
    def computing_intergralation_u_multi_v_function_on_triangle_using_element_center(fn_r, triangle:Triangle, ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):

        return triangle.area2() / 24.0 * fn_r((ver1.x + ver2.x + ver3.x) / 3.0, (ver1.y + ver2.y + ver3.y) / 3.0)
