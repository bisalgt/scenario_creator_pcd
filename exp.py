# ----------------------------------------------------------------------------

# -                        Open3D: www.open3d.org                            -

# ----------------------------------------------------------------------------

# Copyright (c) 2018-2023 www.open3d.org

# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------


import numpy as np

import open3d as o3d

import matplotlib.pyplot as plt

import open3d.visualization.gui as gui

import open3d.visualization.rendering as rendering



# This example displays a point cloud and if you Ctrl-click on a point

# (Cmd-click on macOS) it will show the coordinates of the point.

# This example illustrates:

# - custom mouse handling on SceneWidget

# - getting a the depth value of a point (OpenGL depth)

# - converting from a window point + OpenGL depth to world coordinate

g_i = 0

class ExampleApp:


    def __init__(self):

        # We will create a SceneWidget that fills the entire window, and then

        # a label in the lower left on top of the SceneWidget to display the

        # coordinate.

        app = gui.Application.instance

        self.window = app.create_window("Open3D - GetCoord Example", 1024, 768)

        # Since we want the label on top of the scene, we cannot use a layout,

        # so we need to manually layout the window's children.

        self.window.set_on_layout(self._on_layout)
        # self.window.set_on_resize(self._on_window_resize)

        self.widget3d = gui.SceneWidget()

        self.window.add_child(self.widget3d)

        self.info = gui.Label("")

        self.info.visible = False

        self.window.add_child(self.info)


        # Create a vertical grid layout for the buttons
        num_of_columns = 1
        spacing_betn_items = 10
        margins = gui.Margins(5, 20, 5, 10)
        # self.button_layout = gui.VGrid(num_of_columns, spacing_betn_items)  # 1 column
        self.button_layout = gui.VGrid(num_of_columns, spacing_betn_items, margins)  # 1 column
        print(self.button_layout.preferred_width)
        self.button_layout.background_color = gui.Color(0.5, 0.5, 0.5, 0.6)

        button_layout_width = 200  # Adjust as needed
        # print(self.window.size)
        # print(dir(self.window.size))
        button_layout_height = self.window.size.height
        button_layout_x = self.window.size.width - button_layout_width
        button_layout_y = 0  # Top of the window
        self.button_layout.frame = gui.Rect(button_layout_x, button_layout_y, button_layout_width, button_layout_height)
        

        # Adding to the layout
        self.source_pcd_text = gui.TextEdit()
        self.source_pcd_text.text_value = "only_road_cloud.ply"
        self.button_layout.add_child(self.source_pcd_text)

        self.top_horizontal_grid = gui.Horiz(spacing=2)
        # self.top_horizontal_grid.margins = gui.Margins(2, 2, 2, 2)

        # Create the buttons and add them to the layout
        self.source_pcd_select_btn = gui.Button("Confirm")
        self.source_pcd_select_btn.set_on_clicked(self._on_source_pcd_select_btn_clicked)
        self.top_horizontal_grid.add_child(self.source_pcd_select_btn)
        # self.button_layout.add_child(self.source_pcd_select_btn)


        self.reset_btn = gui.Button("Reset")
        self.reset_btn.set_on_clicked(self._on_reset_btn_clicked)
        
        self.top_horizontal_grid.add_child(self.reset_btn)
        # self.button_layout.add_child(self.reset_btn)


        # self.top_horizontal_grid.add_stretch()
        self.button_layout.add_child(self.top_horizontal_grid)


        self.separator0 = gui.Label("----------------------------------------")
        self.button_layout.add_child(self.separator0)


        # self.group_of_points_select_btn = gui.Button("Select Group of Points")
        # self.group_of_points_select_btn.set_on_clicked(self._on_group_of_points_select_btn_clicked)
        # self.button_layout.add_child(self.group_of_points_select_btn)


        self.group_of_points_chk_box = gui.Checkbox("Select Group of Points")
        self.group_of_points_chk_box.set_on_checked(self._on_group_of_points_chk_box_clicked)
        self.button_layout.add_child(self.group_of_points_chk_box)

        self.separator1 = gui.Label("----------------------------------------")
        self.button_layout.add_child(self.separator1)
        

        self.rect_group_of_points_chk_box = gui.Checkbox("Select RECT Group of Points")
        self.rect_group_of_points_chk_box.set_on_checked(self._on_rect_group_of_points_chk_box_clicked)
        self.button_layout.add_child(self.rect_group_of_points_chk_box)
        self.separator2 = gui.Label("----------------------------------------")
        self.button_layout.add_child(self.separator2)

        
        ### IMP Selected indices used for selecting a part of point cloud for effective processing
        self.selected_pcd_indices = None


        # Step 3: Add Target Cloud
        self.target_pcd_text = gui.TextEdit()
        self.target_pcd_text.text_value = "only_person_cloud.ply"
        self.button_layout.add_child(self.target_pcd_text)
        

        self.target_pcd_load_btn = gui.Button("Load Target Cloud")
        self.target_pcd_load_btn.set_on_clicked(self._on_target_pcd_load_btn_clicked)
        self.button_layout.add_child(self.target_pcd_load_btn)

        self.separator3 = gui.Label("----------------------------------------")
        self.button_layout.add_child(self.separator3)


        self.surface_reconstruct_btn = gui.Button("Reconstruct Surface")
        self.surface_reconstruct_btn.set_on_clicked(self._on_surface_reconstruct_btn_clicked)
        self.button_layout.add_child(self.surface_reconstruct_btn)


        self.show_density_btn = gui.Button("Show Density")
        self.show_density_btn.set_on_clicked(self._on_show_density_btn_clicked)
        self.button_layout.add_child(self.show_density_btn)


        self.filter_density_slider = gui.Slider(gui.Slider.DOUBLE)
        self.filter_density_slider.set_limits(0, 1)
        self.button_layout.add_child(self.filter_density_slider)

        self.filter_density_btn = gui.Button("Filter Density")
        self.filter_density_btn.set_on_clicked(self._on_filter_density_btn_clicked)
        self.button_layout.add_child(self.filter_density_btn)




        self.separator4 = gui.Label("----------------------------------------")
        self.button_layout.add_child(self.separator4)


        # Add the layout to the window
        self.window.add_child(self.button_layout)


        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)


        self.mat = rendering.MaterialRecord()

        self.mat.shader = "defaultUnlit"

        # Point size is in native pixels, but "pixel" means different things to

        # different platforms (macOS, in particular), so multiply by Window scale

        # factor.

        # mat.point_size = 3 * self.window.scaling

        # self.widget3d.scene.add_geometry("Point Cloud", cloud, mat)


        # bounds = self.widget3d.scene.bounding_box

        # center = bounds.get_center()

        # self.widget3d.setup_camera(60, bounds, center)

        # self.widget3d.look_at(center, center - [0, 0, 3], [0, -1, 0])
        self.widget3d.look_at(  [0, 0, 0],  # center
                                [0, 0, 50],  # eye
                                [4, 4, 4])  # Up


        self.widget3d.set_on_mouse(self._on_mouse_widget3d)

    def _on_source_pcd_select_btn_clicked(self):
        print("Button clicked")
        print(self.source_pcd_text.text_value)
        self.mat.point_size = 3 * self.window.scaling
        self.cloud = o3d.io.read_point_cloud(self.source_pcd_text.text_value)

        num_points = len(self.cloud.points)
        colors = np.zeros((num_points, 3))
        self.cloud.colors = o3d.utility.Vector3dVector(colors)

        self.widget3d.scene.add_geometry("source_cloud", self.cloud, self.mat)
        # self.source_pcd_select_btn.enabled = False

    def _on_target_pcd_load_btn_clicked(self):
        ## TODO: Registration of the source and target point clouds
        print("Taget button clicked")
        self.target_cloud = o3d.io.read_point_cloud(self.target_pcd_text.text_value)
        num_points = len(self.target_cloud.points)
        colors = np.zeros((num_points, 3))
        colors[:,1] = 1 # set all points to green
        self.target_cloud.colors = o3d.utility.Vector3dVector(colors)

        self.merged_cloud = self.cloud + self.target_cloud

        if self.widget3d.scene.scene.has_geometry("source_cloud"):
                print("Updating the geometry")
                self.widget3d.scene.scene.remove_geometry("source_cloud")

        self.widget3d.scene.scene.add_geometry("source_cloud", self.merged_cloud, self.mat)
        self.widget3d.force_redraw()

        print("Done loading target cloud")


    def _on_surface_reconstruct_btn_clicked(self):
        print("Surface Reconstruction Button clicked")
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.37, max_nn=6)
        self.cloud.estimate_normals(search_param=search_param)

        print('run Poisson surface reconstruction')
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            self.mesh, self.densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                self.cloud, depth=9)
        print(self.mesh)
        self.mesh.compute_vertex_normals()
        # Paint it gray. Not necessary but the reflection of lighting is hardly perceivable with black surfaces.
        self.mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))

        self.widget3d.scene.scene.add_geometry("mesh", self.mesh, self.mat)


    def _on_show_density_btn_clicked(self):
        print("Show Density Button clicked")

        print('visualize densities')
        self.densities = np.asarray(self.densities)
        density_colors = plt.get_cmap('plasma')(
            (self.densities - self.densities.min()) / (self.densities.max() - self.densities.min()))
        density_colors = density_colors[:, :3]
        density_mesh = o3d.geometry.TriangleMesh()
        density_mesh.vertices = self.mesh.vertices
        density_mesh.triangles = self.mesh.triangles
        density_mesh.triangle_normals = self.mesh.triangle_normals
        density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
        self.widget3d.scene.scene.add_geometry("original_density_mesh", density_mesh, self.mat)


    def _on_filter_density_btn_clicked(self):
        print("Filter Density Button clicked")
        vertices_to_remove = self.densities < np.quantile(self.densities, self.filter_density_slider.double_value)
        self.mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(self.mesh)

        self.mesh.compute_vertex_normals()
        # Paint it gray. Not necessary but the reflection of lighting is hardly perceivable with black surfaces.
        self.mesh.paint_uniform_color(np.array([[0],[0],[1]])) # blue

        self.widget3d.scene.scene.remove_geometry("original_density_mesh")
        self.widget3d.scene.scene.add_geometry("filtered_density_mesh", self.mesh, self.mat)
        self.widget3d.force_redraw()
        print("Done filtering density")




    def _on_reset_btn_clicked(self):
        print("Reset Button clicked")
        self.widget3d.scene.remove_geometry("source_cloud")
        self.source_pcd_text.text_value = ""
        # self.source_pcd_select_btn.enabled = True

    def _on_group_of_points_chk_box_clicked(self, checked):
        print("Group of Points check box clicked")
        print(checked)
        print("___________________________________________")
        # self.source_pcd_select_btn.enabled = False
    
    def _on_rect_group_of_points_chk_box_clicked(self, checked):
        print("RECT Group of Points check box clicked")
        if checked:
            print(checked)
            print("___________________________________________")

            # Convert the point cloud to a numpy array
            points = np.asarray(self.cloud.points)
            colors = np.asarray(self.cloud.colors)

            # Extract the red points
            red_points = points[(colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)]

            if len(red_points) == 0:
                print("No red points found")
                return

            # Compute the 2D bounding box of the red points in the XY plane
            min_xy = np.min(red_points[:, :2], axis=0)
            max_xy = np.max(red_points[:, :2], axis=0)

            # Select all points indices that fall within this bounding box
            selected_indices = (points[:, 0] >= min_xy[0]) & (points[:, 0] <= max_xy[0]) & (points[:, 1] >= min_xy[1]) & (points[:, 1] <= max_xy[1])
            self.selected_pcd_indices = selected_indices # used later for selection of a part of pcd for effective processing


            colors[selected_indices] = [1, 0, 0]  # Change to red

            self.cloud.colors = o3d.utility.Vector3dVector(colors)

            # Create a new point cloud with the selected points
            # selected_cloud = o3d.geometry.PointCloud()
            # selected_cloud.points = o3d.utility.Vector3dVector(selected_points)
            # selected_cloud.colors = o3d.utility.Vector3dVector(colors[(points[:, 0] >= min_xy[0]) & (points[:, 0] <= max_xy[0]) & (points[:, 1] >= min_xy[1]) & (points[:, 1] <= max_xy[1])])

            if self.widget3d.scene.scene.has_geometry("source_cloud"):
                print("Updating the geometry")
                self.widget3d.scene.scene.remove_geometry("source_cloud")

            self.widget3d.scene.scene.add_geometry("source_cloud", self.cloud, self.mat)
            self.widget3d.force_redraw()
        # self.source_pcd_select_btn.enabled = False

    def _on_layout(self, layout_context):
        global g_i
        g_i += 1
        print("---------------------------------------------------------")
        print("Layout Changed! - ", g_i)
        self._on_window_resize(self.window.size.width, self.window.size.height)
        print("---------------------------------------------------------")

        r = self.window.content_rect

        self.widget3d.frame = r

        pref = self.info.calc_preferred_size(layout_context,

                                             gui.Widget.Constraints())

        self.info.frame = gui.Rect(r.x,

                                   r.get_bottom() - pref.height, pref.width,

                                   pref.height)

    def _on_window_resize(self, width, height):
        button_layout_width = 200  # Adjust as needed
        button_layout_height = height
        button_layout_x = width - button_layout_width
        button_layout_y = 0  # Top of the window
        self.button_layout.frame = gui.Rect(button_layout_x, button_layout_y, button_layout_width, button_layout_height)


    def _on_mouse_widget3d(self, event):
        if  event.is_modifier_down(gui.KeyModifier.CTRL):
            if self.group_of_points_chk_box.checked:
                print("Mouse Button Up Event Occured with btn checked")
                print(event.x, event.y) # prints the mouse position. 0,0 is the top left corner of the window
                def depth_callback(depth_image):
                    x = event.x - self.widget3d.frame.x
                    y = event.y - self.widget3d.frame.y
                    # Note that np.asarray() reverses the axes.
                    depth = np.asarray(depth_image)[y, x]
                    if depth == 1.0:
                        print("Clicked on nothing")
                    else:
                        print("Clicked on something")
                        world = self.widget3d.scene.camera.unproject(
                            x, y, depth, self.widget3d.frame.width,
                                self.widget3d.frame.height)

                        text = "({:.3f}, {:.3f}, {:.3f})".format(
                            world[0], world[1], world[2])
                        
                        print("----------------------------------")
                        print(text)
                        print("----------------------------------")

                        distances = np.sum((self.cloud.points - world) ** 2, axis=1)
                        nearest_point_index = np.argmin(distances)
                        print("Nearest Point Index: ", nearest_point_index)

                        # print(self.cloud.colors)
                        # print(np.asarray(self.cloud.colors)[nearest_point_index])
                        print(self.cloud.colors[nearest_point_index])
                        
                        self.cloud.colors[nearest_point_index] = [1, 0, 0]  # Change to red

                        # Redraw the point cloud
                        print(type(self.cloud))
                        # print(type(self.cloud.to_legacy()))
                        print(self.widget3d.scene.scene.UPDATE_COLORS_FLAG)
                        
                        cloud_tensor = o3d.t.geometry.PointCloud().from_legacy(self.cloud)
                        # print(dir(cloud_tensor))
                        # cloud_tensor.points = o3d.core.Tensor(self.cloud.points)
                        # cloud_tensor.colors = o3d.core.Tensor(self.cloud.colors)

                        # self.cloud = cloud_tensor

                        if self.widget3d.scene.scene.has_geometry("source_cloud"):
                            print("Updating the geometry")
                            self.widget3d.scene.scene.remove_geometry("source_cloud")

                        self.widget3d.scene.scene.add_geometry("source_cloud", cloud_tensor, self.mat)
                        self.widget3d.force_redraw()
                        # self.window.set_needs_layout()
                
                self.widget3d.scene.scene.render_to_depth_image(depth_callback)
                return gui.Widget.EventCallbackResult.HANDLED
            elif self.rect_group_of_points_chk_box.checked:
                print("Mouse Button Up Event Occured with btn checked")
                print(event.x, event.y)
                self.mouse_start_point = [event.x, event.y]
            else:
                print("Mouse Button Up Event Occured with btn UNchecked")
            return gui.Widget.EventCallbackResult.IGNORED
        else:
            print("Ignored mouse event!")
            return gui.Widget.EventCallbackResult.IGNORED

    # def _on_mouse_widget3d(self, event):

    #     # We could override BUTTON_DOWN without a modifier, but that would

    #     # interfere with manipulating the scene.

    #     if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(

    #             gui.KeyModifier.CTRL):


    #         def depth_callback(depth_image):

    #             # Coordinates are expressed in absolute coordinates of the

    #             # window, but to dereference the image correctly we need them

    #             # relative to the origin of the widget. Note that even if the

    #             # scene widget is the only thing in the window, if a menubar

    #             # exists it also takes up space in the window (except on macOS).

    #             x = event.x - self.widget3d.frame.x

    #             y = event.y - self.widget3d.frame.y

    #             # Note that np.asarray() reverses the axes.

    #             depth = np.asarray(depth_image)[y, x]


    #             if depth == 1.0:  # clicked on nothing (i.e. the far plane)

    #                 text = ""

    #             else:

    #                 world = self.widget3d.scene.camera.unproject(

    #                     x, y, depth, self.widget3d.frame.width,

    #                     self.widget3d.frame.height)

    #                 text = "({:.3f}, {:.3f}, {:.3f})".format(

    #                     world[0], world[1], world[2])


    #             # This is not called on the main thread, so we need to

    #             # post to the main thread to safely access UI items.

    #             def update_label():

    #                 self.info.text = text

    #                 self.info.visible = (text != "")

    #                 # We are sizing the info label to be exactly the right size,

    #                 # so since the text likely changed width, we need to

    #                 # re-layout to set the new frame.

    #                 self.window.set_needs_layout()


    #             gui.Application.instance.post_to_main_thread(

    #                 self.window, update_label)


    #         self.widget3d.scene.scene.render_to_depth_image(depth_callback)

    #         return gui.Widget.EventCallbackResult.HANDLED

    #     return gui.Widget.EventCallbackResult.IGNORED



def main():

    app = gui.Application.instance

    app.initialize()


    # This example will also work with a triangle mesh, or any 3D object.

    # If you use a triangle mesh you will probably want to set the material

    # shader to "defaultLit" instead of "defaultUnlit".

    pcd_data = o3d.data.DemoICPPointClouds()

    cloud = o3d.io.read_point_cloud("only_road_cloud.ply")

    ex = ExampleApp()


    app.run()



if __name__ == "__main__":

    main()