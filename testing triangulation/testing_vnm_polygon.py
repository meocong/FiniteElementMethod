import numpy as np
from numpy import array
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, delaunay_plot_2d
from triangle import triangulate, plot as tplot

def turn_to_2D(long, latt, MAP_WIDTH, MAP_HEIGHT):
    return ((MAP_WIDTH / 360.0) * (180 + long)), MAP_HEIGHT - ((MAP_HEIGHT / 180.0) * (90.0 - latt))

vn_polygon = []
MAP_WIDTH = 80
MAP_HEIGHT = 50
vertices = []
with open("hanoi_polygon.txt","r") as file:
    data = file.readlines()
    for i in range(2,len(data)):
        if (len(data[i]) == 4):
            break

        coordinate = data[i][1:len(data[i])-1].split("\t")
        x, y = turn_to_2D(float(coordinate[0]), float(coordinate[1]), MAP_WIDTH, MAP_HEIGHT)
        vertices.append([x,y])

segments = [[i,i+1] for i in range(0, len(vertices) - 1)]
segments.append([len(vertices) - 1, 0])
vn_polygon = {'vertices':np.array(vertices), 'segments':np.array(segments)}

print("Triangulation...")
cncfq20adt = triangulate(vn_polygon, 'pq20D')
print(cncfq20adt)
print("Number of triangles: ", len(cncfq20adt['triangles']))
print("Triangulationed")
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
tplot.plot(ax, **cncfq20adt)
# plt.show()