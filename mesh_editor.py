import copy
import open3d as o3d
from os.path import expanduser
from lib import visualization


def main():
    local_cube = cubeGenerator()
    local_cube.save("test_cube", "")
    visualization.visualize_mesh(local_cube.cube, linewidth = 1.0, show = True)

pass

class cubeGenerator():
    def __init__(self, w = 1.0, h = 1.0 , d = 1.0):
        self.w = w
        self.h = h
        self.d = d
        self.cube = self.generate_cube()

    def generate_cube(self):
        return o3d.geometry.TriangleMesh.create_box(width = self.w,
                                                    height = self.h,
                                                    depth = self.d)

    def save(self, name, path):
        o3d.io.write_triangle_mesh(name + ".ply", self.cube)


if __name__ == "__main__":
    main()
