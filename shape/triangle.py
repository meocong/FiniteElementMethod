from vertices import Vertex
from math import sqrt

class Triangle:
    def __init__(self, idx1:Vertex, idx2:Vertex, idx3:Vertex):
        self.vertices = [idx1, idx2, idx3]

    def _distance(self, ver1:Vertex, ver2:Vertex):
        return sqrt((ver1.x - ver2.x)**2 + (ver1.y - ver2.y)**2)

    def area2(self):
        try:
            return self._area
        except:
            a = self._distance(self.vertices[0], self.vertices[1])
            b = self._distance(self.vertices[0], self.vertices[2])
            c = self._distance(self.vertices[1], self.vertices[2])

            p = (a+b+c)/2
            self._area = sqrt(p*(p-a)*(p-b)*(p-c)) * 2
            return self._area


    def sign(self, x, y, v1:Vertex, v2:Vertex):
        return (x - v2.x) * (v1.y - v2.y) - (v1.x - v2.x) * (y - v2.y)

    def contain(self, x, y):
        a = self.sign(x, y, self.vertices[0], self.vertices[1])
        b = self.sign(x, y, self.vertices[1], self.vertices[2])
        c = self.sign(x, y, self.vertices[2], self.vertices[0])

        return a*self.sign(self.vertices[2].x,self.vertices[2].y, self.vertices[0], self.vertices[1]) >= 0 and \
        b*self.sign(self.vertices[0].x,self.vertices[0].y, self.vertices[1], self.vertices[2]) >= 0 and \
        c*self.sign(self.vertices[1].x,self.vertices[1].y, self.vertices[2], self.vertices[0]) >= 0