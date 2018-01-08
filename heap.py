from gaussian import IntergralationGaussian
from shape.triangle import Triangle
from vertices import Vertex, ListVertices

class Heap:
    gauss = IntergralationGaussian()

    def __init__(self, fn_f, max_element):
        self.list = [0] * (max_element + 100)
        self.n = 0
        self.fn_f = fn_f

    def _swap(self, pos1, pos2):
        temp = self.list[pos1]
        self.list[pos1] = self.list[pos2]
        self.list[pos2] = temp

    def _swap_pop(self, pos1):
        self.list[pos1] = self.list[self.n]
        self.n -= 1

    def _up_heap(self, pos):
        while (pos > 1):
            father = int(pos / 2)

            if (self.list[father][0] < self.list[pos][0]):
                self._swap(pos, father)
                pos = father
            else:
                break

    def _down_heap(self, pos):
        while (pos < self.n):
            child = pos * 2
            childright = pos * 2 + 1
            if (childright <= self.n and self.list[child][0] < self.list[childright][0]):
                child = childright

            if (child <= self.n and self.list[child][0] > self.list[pos][0]):
                self._swap(pos, child)
                pos = child
            else:
                break

    def push(self, element):
        self.n += 1
        if (self.n > len(self.list) - 1):
            self.list.append(element)
        else:
            self.list[self.n] = element

        self._up_heap(self.n)

    def computing_gradient_error_on_element(self, triangle:Triangle, list_triangles, dict_segment, Un):
        def get_Un(ver:Vertex):
            if (ver.on_bound == True):
                return 0
            else:
                return Un[ver.idx]

        def get_Grad(element:Triangle):
            gradUvertexX = 0
            gradUvertexY = 0

            ver1 = element.vertices[0]
            ver2 = element.vertices[1]
            ver3 = element.vertices[2]
            gradUvertexX += get_Un(ver1) * (ver2.x - ver3.x)
            gradUvertexY += get_Un(ver1) * (ver2.y - ver3.y)

            ver1 = element.vertices[1]
            ver2 = element.vertices[2]
            ver3 = element.vertices[0]
            gradUvertexX += get_Un(ver1) * (ver2.x - ver3.x)
            gradUvertexY += get_Un(ver1) * (ver2.y - ver3.y)

            ver1 = element.vertices[2]
            ver2 = element.vertices[0]
            ver3 = element.vertices[1]
            gradUvertexX += get_Un(ver1) * (ver2.x - ver3.x)
            gradUvertexY += get_Un(ver1) * (ver2.y - ver3.y)
            return gradUvertexX/element.area2()/8, gradUvertexY/element.area2()/8

        value = 0
        grad_mainX, grad_mainY = get_Grad(triangle)
        for i in range(0,3):
            ver1 = triangle.vertices[i-1]
            ver2 = triangle.vertices[i]

            if (ver1.on_bound != True or ver2.on_bound != True):
                another_grad_mainX, another_grad_mainY = get_Grad(list_triangles[dict_segment[str(ver2.idx) + " " + str(ver1.idx)]])

                value += -(ver2.y - ver1.y) * (grad_mainX - another_grad_mainX) + (ver2.x - ver1.x) * (grad_mainY - another_grad_mainY) / \
                         ((ver2.x - ver1.x)**2 + (ver2.y - ver1.y)**2)**(1/2)
        return (triangle.area2()/2)**(1/4)/2 * value

    def append(self, element:Triangle, list_triangles, dict_segment, Un):
        self.push((self.gauss.computing_intergralation_adaptive_error_on_triangle(fn_f=self.fn_f,triangle=element)
                   + self.computing_gradient_error_on_element(element, list_triangles, dict_segment, Un), element))

    def pop(self):
        if (self.n > 0):
            pop_element = self.list[1]

            self._swap_pop(1)

            if (self.n > 1):
                self._down_heap(1)
            return pop_element
        else:
            return None

    def is_empty(self):
        return self.n > 0