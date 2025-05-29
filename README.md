# Scenario Creator App

- GUI App is created using Open3D
- Can select point cloud prototypes source (e.g Person Point cloud from a lidar pcd capture) and target point cloud
- Realtime Visualization of Augmentation Process (with Raycasting and Shadowcasting Experimentation)
- Final data saved in pickle format, CSV or NumPy Format (Can be toggled in GUI).
- Main file under [src/app.py](./src/app.py)
- To understand the Surface Roughness, it is calculated using Eigenvalues of the points. Can be looked in file [src/surface_variation.py](./src/surface_variation.py)


Output :

![Point Cloud Recombination Output Image](./output/Combining%20PointClouds.png)


![Point Cloud Recombination Output Image](./output/Recombining%20PCD%202.png)


![Point Cloud Recombination Output Image](./output/Recombining%20PCD%203.png)


