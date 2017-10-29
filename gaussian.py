from math import sqrt
from triangle import Triangle
from vertices import Vertex

class IntergralationGaussian:
    def transfrom_x_y_from_csi_eta(self, csi, eta, ver1, ver2, ver3):
        return (ver2.x - ver1.x) * csi + (ver3.x - ver1.x) * eta + ver1.x, \
        (ver2.y - ver1.y) * csi + (ver3.y - ver1.y) * eta + ver1.y

    def transform_csi_eta_from_x_y(self, x, y, v1, v2, v3):
        delta = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x)
        return ((v3.y - v1.y) * (x - v1.x) - (v3.x - v1.x) * (y - v1.y)) / delta, \
              -((v2.y - v1.y) * (x - v1.x) - (v2.x - v1.x) * (y - v1.y)) / delta

    def _gaussian(self, g):
        sqrt15 = sqrt(15)
        return (0.5)* ((9.0/40) * g(1.0/3, 1.0/3)
                        +((155 - sqrt15) / 1200)*g((6 - sqrt15)*(1.0/21), (1.0/21)*(6 - sqrt15))
                        +((155 - sqrt15) / 1200)*g((9 + 2*sqrt15)*(1.0/21), (6 - sqrt15)*(1.0/21))
                        +((155 - sqrt15) / 1200)*g((6 - sqrt15)*(1.0/21), (9 + 2*sqrt15)*(1.0/21))
                        +((155 + sqrt15) / 1200)*g((6 + sqrt15)*(1.0/21), (6 + sqrt15)*(1.0/21))
                        +((155 + sqrt15) / 1200)*g((6 + sqrt15)*(1.0/21), (9 - 2*sqrt15)*(1.0/21))
                        +((155 + sqrt15) / 1200)*g((9 - 2*sqrt15)*(1.0/21), (6 + sqrt15)*(1.0/21)))

    def computing_intergralation_f_multi_base_function_on_triangle(self, fn_f, triangle:Triangle, ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):
        def new_func(csi, eta):
            x,y = self.transfrom_x_y_from_csi_eta(csi, eta, ver1, ver2, ver3)
            return fn_f(x, y) * (1 - csi - eta)

        return triangle.area2()*self._gaussian(new_func)

    def estimate_error_on_element_l2_space(self, triangle:Triangle, fn_root, Un):
        v1 = triangle.vertices[0]
        v2 = triangle.vertices[1]
        v3 = triangle.vertices[2]

        def new_func(csi, eta):
            x, y = self.transfrom_x_y_from_csi_eta(csi, eta, v1, v2, v3)

            delta = fn_root(x, y)
            if v1.on_bound == False:
                delta -= Un[v1.idx, 0] * (1 - csi - eta)
            if v2.on_bound == False:
                delta -= Un[v2.idx, 0] * csi
            if v3.on_bound == False:
                delta -= Un[v3.idx, 0] * eta
            return delta**2

        return triangle.area2()*self._gaussian(new_func)

    def estimate_error_on_element_dev_x(self, triangle:Triangle, fn_dev_x, Un):
        v1 = triangle.vertices[0]
        v2 = triangle.vertices[1]
        v3 = triangle.vertices[2]

        def new_func(csi, eta):
            x, y = self.transfrom_x_y_from_csi_eta(csi, eta, v1, v2, v3)

            delta = fn_dev_x(x, y)
            if v1.on_bound == False:
                delta += Un[v1.idx, 0]
            if v2.on_bound == False:
                delta += Un[v2.idx, 0]
            return delta**2

        return triangle.area2()*self._gaussian(new_func)

    def estimate_error_on_element_dev_y(self, triangle:Triangle, fn_dev_y, Un):
        v1 = triangle.vertices[0]
        v2 = triangle.vertices[1]
        v3 = triangle.vertices[2]

        def new_func(csi, eta):
            x, y = self.transfrom_x_y_from_csi_eta(csi, eta, v1, v2, v3)

            delta = fn_dev_y(x, y)
            if v1.on_bound == False:
                delta += Un[v1.idx, 0]
            if v3.on_bound == False:
                delta += Un[v3.idx, 0]
            return delta**2

        return triangle.area2()*self._gaussian(new_func)

    def estimate_point_value_in_triangle(self, triangle:Triangle, x, y, Un):
        if (triangle == None):
            return 0
        v1 = triangle.vertices[0]
        v2 = triangle.vertices[1]
        v3 = triangle.vertices[2]

        csi, eta = self.transform_csi_eta_from_x_y(x, y, v1, v2, v3)

        delta = 0
        if v1.on_bound == False:
            delta += Un[v1.idx,0] * (1-csi-eta)
        if v2.on_bound == False:
            delta += Un[v2.idx,0] * csi
        if v3.on_bound == False:
            delta += Un[v3.idx,0] * eta

        return  delta

    def computing_intergralation_u_deri_multi_u_deri_function_on_triangle(self, triangle:Triangle,
                                                           ver2:Vertex,
                                                           ver3:Vertex):

        return ((ver2.x - ver3.x)**2 + (ver2.y - ver3.y)**2)/(2*triangle.area2())

    def computing_intergralation_u_deri_multi_v_deri_function_on_triangle(self, triangle:Triangle, ver1:Vertex,
                                                           ver2:Vertex,
                                                           ver3:Vertex):

        return ((ver2.y - ver3.y)*(ver3.y - ver1.y) +(ver2.x - ver3.x)*(ver3.x - ver1.x) )/(2*triangle.area2())

    def computing_intergralation_u_multi_v_function_on_triangle(self, triangle:Triangle, alpha):

        return triangle.area2()*alpha
