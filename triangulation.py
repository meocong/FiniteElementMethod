from shape.triangle import Triangle
from vertices import Vertex, ListVertices
import matplotlib.pyplot as plt
from gaussian import IntergralationGaussian
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from heap import Heap
import numpy as np
import matplotlib as mpl
from triangle import triangulate, plot as tplot

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


    def _adaptive_triangulation(self, list_triangles, vertices_inner, vertices_bound, max_element, fn_f, Un, dict_segment):
        def _triangulate_one_triangle_heap(triangle):
            vertex1 = triangle.vertices[0]
            vertex2 = triangle.vertices[1]
            vertex3 = triangle.vertices[2]

            center_edge12 = self.vertices_append_center_edge(vertices_inner, vertices_bound, vertex1, vertex2)
            center_edge13 = self.vertices_append_center_edge(vertices_inner, vertices_bound, vertex1, vertex3)
            center_edge23 = self.vertices_append_center_edge(vertices_inner, vertices_bound, vertex2, vertex3)

            list_triangles.append(Triangle(vertex1, center_edge12, center_edge13))
            list_triangles.append(Triangle(vertex2, center_edge23, center_edge12))
            list_triangles.append(Triangle(vertex3, center_edge23, center_edge13))
            list_triangles.append(Triangle(center_edge12, center_edge13, center_edge23))

        heap = Heap(fn_f, max_element)

        for triangle in list_triangles:
            heap.append(triangle, list_triangles, dict_segment, Un)

        n_element = len(list_triangles)/10
        temp = []
        while (heap.n < max_element and n_element > 0):
            _, triangle = heap.pop()
            temp.append(triangle)
            n_element -= 1
        for triangle in temp:
            _triangulate_one_triangle_heap(triangle)

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

    def initalize_process_square(self, adaptive, square_size, n_iter, plot=False, fn_f = None):
        list_triangles, vertices_inner, vertices_bound = self.init_triangles(square_size)

        if (n_iter > 0):
            list_triangles = self._triangulation(list_triangles, vertices_inner, vertices_bound, n_iter = n_iter)

        if (adaptive == True):
            list_triangles, vertices_inner, vertices_bound, min_x, max_x, min_y, max_y, dict_segment \
                = self.re_triangulation_on_failue(list_triangles, vertices_inner, vertices_bound)
        else:
            dict_segment = {}

        if (plot == True):
            print("Plotting...")
            fig = plt.figure(figsize=plt.figaspect(0.5))

            ax = fig.add_subplot(1, 2, 1)
            self.plot_triangles(ax, list_triangles, vertices_inner, vertices_bound)

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            # self.plot_function_actual(ax, fn_f, 0, 1, 0, 1)
            self.plot_function_adaptive(ax, fn_f, list_triangles, vertices_inner, vertices_bound)
            plt.show()

        return list_triangles, vertices_inner, vertices_bound, dict_segment

    def turn_to_2D(self, long, latt, MAP_WIDTH, MAP_HEIGHT):
        return ((MAP_WIDTH / 360.0) * (180 + long)), MAP_HEIGHT - ((MAP_HEIGHT / 180.0) * (90.0 - latt))

    def _read_shape_from_dir(self, shape_dir, is_map, map_width, map_height):
        vertices = []

        if (is_map == True):
            with open(shape_dir, "r") as file:
                data = file.readlines()
                for i in range(2, len(data)):
                    if (len(data[i]) == 4):
                        break

                    coordinate = data[i][1:len(data[i]) - 1].split("\t")
                    x, y = self.turn_to_2D(float(coordinate[0]), float(coordinate[1]), map_width, map_height)
                    vertices.append([x, y])
        else:
            with open(shape_dir, "r") as file:
                data = file.readlines()
                for i in range(0, len(data)):
                    coordinate = data[i].split("\t")
                    x = float(coordinate[0])
                    y = float(coordinate[1])
                    vertices.append([x, y])

        return vertices

    def init_triangles_in_shape(self, vertices, option, max_vertice_add_each_edge, max_vertice_added_near_each_vertice):
        temp = []
        for idx, vertice in enumerate(vertices):
            if (idx < len(vertices) - 1):
                nex_vertice = vertices[idx + 1]
            else:
                nex_vertice = vertices[0]

            temp.extend([[vertice[0] + (nex_vertice[0] - vertice[0])/max_vertice_add_each_edge * i,
                          vertice[1] + (nex_vertice[1] - vertice[1])/max_vertice_add_each_edge * i]
                         for i in range(0, max_vertice_add_each_edge)])
        vertices = temp

        segments = [[i, i + 1] for i in range(0, len(vertices) - 1)]
        segments.append([len(vertices) - 1, 0])

        polygon = {'vertices': np.array(vertices), 'segments': np.array(segments)}

        print("Free Triangulation...")
        cncfq20adt = triangulate(polygon, option)
        list_triangles = []
        vertices_inner = ListVertices(on_bound=False)
        vertices_bound = ListVertices(on_bound=True)

        last = 0
        A = []
        min_x = 99999999
        max_x = -99999999

        min_y = 99999999
        max_y = -99999999
        for idx, vertice in enumerate(cncfq20adt['vertices']):
            x = vertice[0]
            y = vertice[1]

            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            if (cncfq20adt['vertex_markers'][idx][0] == 0):
                vertices_inner.append(vertice[0], vertice[1])
                A.append(last)
            else:
                vertices_bound.append(vertice[0], vertice[1])
                last += 1
                A.append(last)

        dict_segment = {}
        for i, triangle in enumerate(cncfq20adt['triangles']):
            if (cncfq20adt['vertex_markers'][triangle[0]][0] == 0):
                ver1 = vertices_inner.list[triangle[0] - A[triangle[0]]]
            else:
                ver1 = vertices_bound.list[A[triangle[0]] - 1]

            if (cncfq20adt['vertex_markers'][triangle[1]][0] == 0):
                ver2 = vertices_inner.list[triangle[1] - A[triangle[1]]]
            else:
                ver2 = vertices_bound.list[A[triangle[1]] - 1]

            if (cncfq20adt['vertex_markers'][triangle[2]][0] == 0):
                ver3 = vertices_inner.list[triangle[2] - A[triangle[2]]]
            else:
                ver3 = vertices_bound.list[A[triangle[2]] - 1]

            if (ver1.x - ver2.x) * (ver1.y - ver3.y) - (ver1.y - ver2.y) * (ver1.x - ver3.x) < 0:
                ver1, ver2, ver3 = (ver3, ver2, ver1)

            edge1 = str(ver3.idx) + " " + str(ver1.idx)
            dict_segment[edge1] = i

            edge1 = str(ver1.idx) + " " + str(ver2.idx)
            dict_segment[edge1] = i

            edge1 = str(ver2.idx) + " " + str(ver3.idx)
            dict_segment[edge1] = i

            list_triangles.append(Triangle(ver1, ver2, ver3))

        return list_triangles, vertices_inner, vertices_bound, min_x, max_x, min_y, max_y, dict_segment

    def re_triangulation_on_failue(self, triangles, vertices_inner, vertices_bound):
        vertices = []
        vertex_markers = []
        vertices.extend([[vertice.x, vertice.y] for vertice in vertices_inner.list])
        vertex_markers.extend([[0] for vertice in vertices_inner.list])
        vertices.extend([[vertice.x, vertice.y] for vertice in vertices_bound.list])
        vertex_markers.extend([[1] for vertice in vertices_bound.list])

        segments = []
        idx_triangles = []
        for triangle in triangles:
            vertice1 = triangle.vertices[0]
            vertice2 = triangle.vertices[1]
            vertice3 = triangle.vertices[2]

            if (vertice1.on_bound == True):
                idx1 = vertices_inner.length + vertice1.idx
            else:
                idx1 = vertice1.idx

            if (vertice2.on_bound == True):
                idx2 = vertices_inner.length + vertice2.idx
            else:
                idx2 = vertice2.idx

            if (vertice3.on_bound == True):
                idx3 = vertices_inner.length + vertice3.idx
            else:
                idx3 = vertice3.idx

            segments.append([idx1, idx2])
            segments.append([idx2, idx3])
            segments.append([idx3, idx1])
            idx_triangles.append([idx1, idx2, idx3])

        polygon = {'vertices': np.array(vertices), 'segments': np.array(segments), 'vertex_markers':vertex_markers}
        # polygon = {'vertices': np.array(vertices), 'segments': np.array(segments), 'triangles':idx_triangles}
        print("Free Triangulation...")
        cncfq20adt = triangulate(polygon, "q30D")
        print("Finishing free triangulation")
        list_triangles = []
        vertices_inner = ListVertices(on_bound=False)
        vertices_bound = ListVertices(on_bound=True)

        last = 0
        A = []
        min_x = 99999999
        max_x = -99999999

        min_y = 99999999
        max_y = -99999999
        for idx, vertice in enumerate(cncfq20adt['vertices']):
            x = vertice[0]
            y = vertice[1]

            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            if (cncfq20adt['vertex_markers'][idx][0] == 0):
                vertices_inner.append(vertice[0], vertice[1])
                A.append(last)
            else:
                vertices_bound.append(vertice[0], vertice[1])
                last += 1
                A.append(last)

        dict_segment = {}
        for i, triangle in enumerate(cncfq20adt['triangles']):
            if (cncfq20adt['vertex_markers'][triangle[0]][0] == 0):
                ver1 = vertices_inner.list[triangle[0] - A[triangle[0]]]
            else:
                ver1 = vertices_bound.list[A[triangle[0]] - 1]

            if (cncfq20adt['vertex_markers'][triangle[1]][0] == 0):
                ver2 = vertices_inner.list[triangle[1] - A[triangle[1]]]
            else:
                ver2 = vertices_bound.list[A[triangle[1]] - 1]

            if (cncfq20adt['vertex_markers'][triangle[2]][0] == 0):
                ver3 = vertices_inner.list[triangle[2] - A[triangle[2]]]
            else:
                ver3 = vertices_bound.list[A[triangle[2]] - 1]

            if  (ver1.x - ver2.x) * (ver1.y - ver3.y) - (ver1.y - ver2.y) * (ver1.x - ver3.x) < 0:
                ver1, ver2, ver3 = (ver3, ver2, ver1)

            edge1 = str(ver3.idx) + " " + str(ver1.idx)
            dict_segment[edge1] = i

            edge1 = str(ver1.idx) + " " + str(ver2.idx)
            dict_segment[edge1] = i

            edge1 = str(ver2.idx) + " " + str(ver3.idx)
            dict_segment[edge1] = i

            list_triangles.append(Triangle(ver1, ver2, ver3))

        return list_triangles, vertices_inner, vertices_bound, min_x, max_x, min_y, max_y, dict_segment

    def initalize_process_free_shape(self, shape_dir, is_map, shape, map_width, map_height, option, plot=False,
                           fn_f = None, max_vertice_add_each_edge = 0,
                           max_vertice_added_near_each_vertice=0):
        if (shape_dir is not None):
            shape = self._read_shape_from_dir(shape_dir, is_map, map_width, map_height)

        list_triangles, vertices_inner, vertices_bound, min_x, max_x, min_y, max_y, dict_segment = \
            self.init_triangles_in_shape(shape, option,max_vertice_add_each_edge,max_vertice_added_near_each_vertice)

        if (plot == True):
            print("Plotting...")
            fig = plt.figure(figsize=plt.figaspect(0.5))

            ax = fig.add_subplot(1, 2, 1)
            self.plot_triangles(ax, list_triangles, vertices_inner, vertices_bound)

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            # self.plot_function_actual(ax, fn_f, 0, 1, 0, 1)
            self.plot_function_adaptive(ax, fn_f, list_triangles, vertices_inner, vertices_bound)
            plt.show()
        return list_triangles, vertices_inner, vertices_bound, dict_segment

    def process_free_shape_adaptively(self, list_triangles, vertices_inner, vertices_bound, dict_segment, Un,
                           plot=False, fn_f = None, max_element = 1000000):
        list_triangles = self._adaptive_triangulation(list_triangles, vertices_inner, vertices_bound,
                                      max_element, fn_f, Un, dict_segment)

        list_triangles, vertices_inner, vertices_bound, min_x, max_x, min_y, max_y, dict_segment \
        = self.re_triangulation_on_failue(list_triangles, vertices_inner, vertices_bound)


        if (plot == True):
            print("Plotting...")
            fig = plt.figure(figsize=plt.figaspect(0.5))

            ax = fig.add_subplot(1, 2, 1)
            self.plot_triangles(ax, list_triangles, vertices_inner, vertices_bound)

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            # self.plot_function_actual(ax, fn_f, 0, 1, 0, 1)
            self.plot_function_adaptive(ax, fn_f, list_triangles, vertices_inner, vertices_bound)
            plt.show()

        return list_triangles, vertices_inner, vertices_bound, dict_segment

    def plot_function_actual(self, ax, fn_f, min_x, max_x, min_y, max_y):
        X = np.linspace(min_x, max_x, endpoint=True, num=50)
        Y = np.linspace(min_y, max_y, endpoint=True, num=50)
        X, Y = np.meshgrid(X, Y)
        X, Y = X.flatten(), Y.flatten()
        Z = fn_f(X,Y)

        tri = mtri.Triangulation(X, Y)
        ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, cmap=plt.cm.Spectral)
        # ax.set_zlim(-1, 1)

    def plot_function_adaptive(self, ax, fn_f, list_triangles, vertices_inner, vertices_bound):
        X = [vertex.x for vertex in vertices_inner.list]
        X.extend([vertex.x for vertex in vertices_bound.list])
        X = np.array(X)
        n_inner = len(vertices_inner.list)

        Y = [vertex.y for vertex in vertices_inner.list]
        Y.extend([vertex.y for vertex in vertices_bound.list])
        Y = np.array(Y)

        triangles = [[triangle.vertices[i].idx + n_inner * (triangle.vertices[i].on_bound == True) for i in range(0, 3)]
                     for triangle in list_triangles]
        triangles = np.array(triangles)

        Z = fn_f(X,Y)

        ax.plot_trisurf(X, Y, Z, triangles=triangles, cmap=plt.cm.Spectral)
        # ax.set_zlim(-1, 1)

    def plot_triangles(self, ax, list_triangles, vertices_inner, vertices_bound):
        ax.axis('equal')

        # for triangle in list_triangles:
        #     vertex1 = triangle.vertices[0]
        #     vertex2 = triangle.vertices[1]
        #     vertex3 = triangle.vertices[2]
        #     # print([vertex1.x, vertex2.x, vertex3.x], [vertex1.y, vertex2.y, vertex3.y])
        #     ax.plot([vertex1.x, vertex2.x, vertex3.x, vertex1.x], [vertex1.y, vertex2.y, vertex3.y, vertex1.y], color='r')
        # #
        X = [vertex.x for vertex in vertices_inner.list]
        X.extend([vertex.x for vertex in vertices_bound.list])
        X = np.array(X)
        n_inner = len(vertices_inner.list)

        Y = [vertex.y for vertex in vertices_inner.list]
        Y.extend([vertex.y for vertex in vertices_bound.list])
        Y = np.array(Y)

        triangles = [[triangle.vertices[i].idx + n_inner * (triangle.vertices[i].on_bound == True) for i in range(0,3)]
                     for triangle in list_triangles]
        triangles = np.array(triangles)

        # for triangle in list_triangles:
        #     for i in range(0,3):
        #         print(triangle.vertices[i].x, x[triangle.vertices[i].idx + n_inner * (triangle.vertices[i].on_bound == True)],
        #               triangle.vertices[i].y, y[triangle.vertices[i].idx + n_inner * (triangle.vertices[i].on_bound == True)],
        #               triangle.vertices[i].idx, triangle.vertices[i].idx + n_inner * (triangle.vertices[i].on_bound == True))

        mpl.rcParams['agg.path.chunksize'] = 10000
        ax.triplot(X, Y, triangles, 'r-')
