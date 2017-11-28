class Vertex:
    def __init__(self,x,y,idx,on_bound = False):
        self.x = x
        self.y = y
        self.on_bound = on_bound
        self.idx = idx

class ListVertices:
    def __init__(self, on_bound):
        self.on_bound = on_bound

        self.list = []
        self._check = {}
        self.length = 0

    def append(self, x, y):
        if ((x,y) not in self._check):
            self.length += 1
            self._check[(x,y)] = self.length - 1
            self.list.append(Vertex(x,y,self.length - 1, self.on_bound))
            return self.list[-1]
        return self.list[self._check[(x,y)]]

    def append_center_egde(self, vertex1:Vertex, vertex2:Vertex):
        x = (vertex1.x + vertex2.x)/2
        y = (vertex1.y + vertex2.y)/2

        if ((x,y) not in self._check):
            self.length += 1
            self._check[(x,y)] = self.length - 1
            self.list.append(Vertex(x,y,self.length - 1, self.on_bound))
            return self.list[-1]
        return self.list[self._check[(x,y)]]