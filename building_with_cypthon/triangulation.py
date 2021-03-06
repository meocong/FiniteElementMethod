from triangle import Triangle
from vertices import Vertex, ListVertices
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Bbox

class Triangulation:
    def check_both_in_bound(self, vertex1:Vertex, vertex2:Vertex):
        return not (vertex1.on_bound == False or vertex2.on_bound == False)

    def vertices_append_center_edge(self, vertices_inner:ListVertices, vertices_bound:ListVertices, vertex1:Vertex, vertex2:Vertex):
        if self.check_both_in_bound(vertex1, vertex2):
            return vertices_bound.append_center_egde(vertex1, vertex2)
        else:
            return vertices_inner.append_center_egde(vertex1, vertex2)

    def _triangulation(self, list_triangles, vertices_inner, vertices_bound, n_iter):
        for iter in range(n_iter):
            new_list_triangles = []
            for triangle in list_triangles:
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

            del list_triangles
            list_triangles = new_list_triangles
        return list_triangles

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

    def process_square(self, square_size, n_iter, plot=False):
        list_triangles, vertices_inner, vertices_bound = self.init_triangles(square_size)

        if (n_iter > 0):
            list_triangles = self._triangulation(list_triangles, vertices_inner, vertices_bound, n_iter = n_iter)

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
