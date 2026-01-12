import time
import numpy as np
from plyfile import PlyData
import viser

ply = PlyData.read("gaussians/dogs-jump_1_model.ply")
v = ply["vertex"]
points = np.stack([v["x"], v["y"], v["z"]], axis=1)

server = viser.ViserServer()
colors = np.full((points.shape[0], 3), 200, dtype=np.uint8)
server.scene.add_point_cloud("gaussians", points=points, colors=colors, point_size=0.003)
port = server.get_port()
print(f"Open: http://localhost:{port}")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
