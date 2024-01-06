# ----------------------------------------------------------------------------

# -                        Open3D: www.open3d.org                            -

# ----------------------------------------------------------------------------

# Copyright (c) 2018-2023 www.open3d.org

# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------


import numpy as np

import open3d as o3d

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


        bounds = self.widget3d.scene.bounding_box

        center = bounds.get_center()

        # self.widget3d.setup_camera(60, bounds, center)

        # self.widget3d.look_at(center, center - [0, 0, 3], [0, -1, 0])
        self.widget3d.look_at(  [0, 0, 0],  # center
                                [0, 0, 50],  # eye
                                [4, 4, 4])  # Up


        # self.widget3d.set_on_mouse(self._on_mouse_widget3d)

    def _on_source_pcd_select_btn_clicked(self):
        print("Button clicked")
        print(self.source_pcd_text.text_value)
        self.mat.point_size = 3 * self.window.scaling
        cloud = o3d.io.read_point_cloud(self.source_pcd_text.text_value)
        self.widget3d.scene.add_geometry("source_cloud", cloud, self.mat)
        # self.source_pcd_select_btn.enabled = False

    def _on_reset_btn_clicked(self):
        print("Reset Button clicked")
        self.widget3d.scene.remove_geometry("source_cloud")
        self.source_pcd_text.text_value = ""
        # self.source_pcd_select_btn.enabled = True

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