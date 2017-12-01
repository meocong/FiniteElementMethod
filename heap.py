from gaussian import IntergralationGaussian

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

    def append(self, element):
        self.push((self.gauss.computing_intergralation_f2_on_triangle(fn_f=self.fn_f,triangle=element), element))

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