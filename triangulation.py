from triangle import Triangle
from vertices import Vertex, ListVertices
import matplotlib.pyplot as plt
from gaussian import IntergralationGaussian
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from heap import Heap

class Triangulation:
    gauss = IntergralationGaussian()

    def check_both_in_bound(self, vertex1:Vertex, vertex2:Vertex):
        return not (vertex1.on_bound == False or vertex2.on_bound == False)

    def vertices_append_center_edge(self, vertices_inner:ListVertices, vertices_bound:ListVertices, vertex1:Vertex, vertex2:Vertex):
        if self.check_both_in_bound(vertex1, vertex2):
            return vertices_bound.append_center_egde(vertex1, vertex2)
        else:
            return vertices_inner.append_center_egde(vertex1, vertex2)

    def _triangulate_one_triangle(self, triangle, new_list_triangles, vertices_inner, vertices_bound):
        vertex1 = triangle.vertices[0]
        vertex2 = triangle.vertices[1]
        vertex3 = triangle.vertices[2]

        center_edge12 = self.vertices_append_center_edge(vertices_inner, vertices_bound, vertex1, vertex2)
        center_edge13 = self.vertices_append_center_edge(vertices_inner, vertices_bound, vertex1, vertex3)
        center_edge23 = self.vertices_append_center_edge(vertices_inner, vertices_bound, vertex2, vertex3)

        new_list_triangles.append(Triangle(vertex1, center_edge12, center_edge13))
        new_list_triangles.append(Triangle(vertex2, center_edge23, center_edge12))
        new_list_triangles.append(Triangle(vertex3, center_edge23, center_edge13))
        new_list_triangles.append(Triangle(center_edge12, center_edge13, center_edge23))

    def _triangulation(self, list_triangles, vertices_inner, vertices_bound, n_iter):
        for iter in range(n_iter):
            new_list_triangles = []
            for triangle in list_triangles:
                self._triangulate_one_triangle(triangle, new_list_triangles, vertices_inner, vertices_bound)

            del list_triangles
            list_triangles = new_list_triangles
        return list_triangles

    def _check_element_triangulated(self, triangle, fn_f, threshold_adaptive):
        # print(self.gauss.computing_intergralation_f_on_triangle(fn_f=fn_f,triangle=triangle))
        return self.gauss.computing_intergralation_f2_on_triangle(fn_f=fn_f,triangle=triangle) >= threshold_adaptive

    def _adaptive_triangulation_with_threshold(self, list_triangles, vertices_inner, vertices_bound, threshold_adaptive, max_element, fn_f):
        n = len(list_triangles)
        while n < max_element:
            previous_list_triangles = list_triangles
            list_triangles = []
            first = 0
            last = len(previous_list_triangles)
            n = 0

            while last > first and n + last - first < max_element:
                triangle = previous_list_triangles[first]
                first += 1

                if (self._check_element_triangulated(triangle, fn_f, threshold_adaptive) == True):
                    self._triangulate_one_triangle(triangle, list_triangles, vertices_inner, vertices_bound)
                    n += 4
                else:
                    list_triangles.append(triangle)
                    n += 1
            list_triangles.extend(previous_list_triangles[first:last])
            n = n + last - first

            if (n == last):
                break

        return list_triangles

    def _adaptive_triangulation(self, list_triangles, vertices_inner, vertices_bound, threshold_adaptive, max_element, fn_f):
        heap = Heap(fn_f, max_element)

        for triangle in list_triangles:
            heap.append(triangle)

        while (heap.n < max_element):
            _, triangle = heap.pop()
            self._triangulate_one_triangle(triangle, heap, vertices_inner, vertices_bound)

        return [element[1] for element in heap.list[1:heap.n + 1]]

    def init_triangles(self, square_size):
        list_triangles = []
        vertices_inner = ListVertices(on_bound=False)
        vertices_inner.append(square_size / 2, square_size / 2)

        vertices_bound = ListVertices(on_bound=True)
        vertices_bound.append(0, 0)
        vertices_bound.append(0, square_size)
        vertices_bound.append(square_size, square_size)
        vertices_bound.append(square_size, 0)

        for i in range(0, vertices_bound.length):
            list_triangles.append(Triangle(vertices_bound.list[i], vertices_bound.list[i - 1],
                                           vertices_inner.list[0]))
        return list_triangles, vertices_inner, vertices_bound

    def _get_one_triangle_satify_x_y(self,list_triangles,x,y):
        if (list_triangles[0] != None):
            for idx, triangle in enumerate(list_triangles):
                if triangle.contain(x,y):
                    return [triangle], idx
        return [None], None

    def find_exactly_element(self, square_size, n_iter, x, y):
        list_triangles, vertices_inner, vertices_bound = self.init_triangles(square_size)

        trian_idx = 0
        list_triangles, idx = self._get_one_triangle_satify_x_y(list_triangles, x,y)
        if (list_triangles[0] == None):
            return None
        trian_idx = trian_idx * 4 + idx

        if (n_iter > 0):
            for iter in range(n_iter):
                list_triangles = self._triangulation(list_triangles, vertices_inner, vertices_bound, n_iter = 1)
                list_triangles, idx = self._get_one_triangle_satify_x_y(list_triangles, x, y)
                trian_idx = trian_idx * 4 + idx

        return trian_idx

    def process_square(self, square_size, n_iter, plot=False, adaptive = False, threshold_adaptive = 10, fn_f = None, max_element = 1000000):
        list_triangles, vertices_inner, vertices_bound = self.init_triangles(square_size)

        if (n_iter > 0):
            list_triangles = self._triangulation(list_triangles, vertices_inner, vertices_bound, n_iter = n_iter)

        if (adaptive == True):
            if (threshold_adaptive == None):
                list_triangles = self._adaptive_triangulation(list_triangles, vertices_inner, vertices_bound,
                                             threshold_adaptive, max_element, fn_f)
            else:
                list_triangles = self._adaptive_triangulation_with_threshold(list_triangles, vertices_inner, vertices_bound,
                                                              threshold_adaptive, max_element, fn_f)

        if (plot == True):
            self.plot_triangles(list_triangles, square_size)

        return list_triangles, vertices_inner, vertices_bound

    def plot_triangles(self, list_triangles, square_size):
        left, bottom, width, height = (-0.5, -0.5, square_size+0.5, square_size+0.5)

        plt.figure(1)
        plt.axis('equal')

        for triangle in list_triangles:
            vertex1 = triangle.vertices[0]
            vertex2 = triangle.vertices[1]
            vertex3 = triangle.vertices[2]
            # print([vertex1.x, vertex2.x, vertex3.x], [vertex1.y, vertex2.y, vertex3.y])
            plt.plot([vertex1.x, vertex2.x, vertex3.x, vertex1.x], [vertex1.y, vertex2.y, vertex3.y, vertex1.y], color='r')

        plt.show()
