import copy

import math

import numpy as np

import open3d as o3d

import matplotlib.pyplot as plt

import open3d.visualization.gui as gui

import open3d.visualization.rendering as rendering



class ScenarioCreatorApp:


    def __init__(self):
        
        # region 0: Variables
        # Variables
        
        
        self.source_cloud = None
        self.target_cloud = None
        self.selected_pcd_indices = None

        
        
        # endregion 0

        app = gui.Application.instance
        self.window = app.create_window("Scenario-Creator App", 1024, 768)
        self.em = self.window.theme.font_size  # Standard Font size of the window (so that UI changes with different OS)
        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)


        # Create a vertical grid layout for the buttons
        spacing_betn_items = 10
        margins = gui.Margins(5, 20, 5, 10)
        self.main_layout = gui.CollapsableVert("Main Layout", spacing_betn_items, margins)
        self.main_layout.background_color = gui.Color(0.5, 0.5, 0.5, 0.5)

        main_layout_width = 400  # Adjust as needed
        main_layout_height = self.window.size.height
        main_layout_x = self.window.size.width - main_layout_width
        main_layout_y = 0  # Top of the window
        self.main_layout.frame = gui.Rect(main_layout_x, main_layout_y, main_layout_width, main_layout_height)
        

        # region 1: POINT CLOUD LOAD AND VISUALIZATION

        spacing_betn_items_in_region = 0.5*self.em
        margins_for_region = gui.Margins(2 * self.em, 0*self.em, 2*self.em, 0*self.em)

        self.rgn1_pcd_load_layout = gui.CollapsableVert("Load Point Cloud", spacing_betn_items_in_region,
                                         margins_for_region)
        rgn1_horiz_row_grid_spacing = 0.1 * self.em
        rgn1_horiz_row_grid_margin = gui.Margins(0.3*self.em, 0*self.em, 0.3*self.em, 0*self.em)

        self.rgn1_horiz_row_1_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1_grid.preferred_height = 2 * self.em
        self.source_pcd_label = gui.Label("Source PCD Filename: ")
        self.source_pcd_text = gui.TextEdit()
        self.source_pcd_text.text_value = "only_person_cloud.ply"

        self.rgn1_horiz_row_1_grid.add_stretch()
        self.rgn1_horiz_row_1_grid.add_child(self.source_pcd_label)
        self.rgn1_horiz_row_1_grid.add_child(self.source_pcd_text)
        self.rgn1_horiz_row_1_grid.add_stretch()

        self.rgn1_horiz_row_2_grid = gui.Horiz()
        self.rgn1_horiz_row_2_grid.preferred_height = 2 * self.em
        self.source_pcd_load_btn = gui.Button(f"Load Source Cloud")
        self.source_pcd_load_btn.set_on_clicked(self._on_source_pcd_load_btn_clicked)
        self.source_pcd_remove_btn = gui.Button(f"Remove Source Cloud")
        self.source_pcd_remove_btn.set_on_clicked(self._on_source_pcd_remove_btn_clicked)

        self.rgn1_horiz_row_2_grid.add_stretch()
        self.rgn1_horiz_row_2_grid.add_child(self.source_pcd_load_btn)
        self.rgn1_horiz_row_2_grid.add_stretch()
        self.rgn1_horiz_row_2_grid.add_child(self.source_pcd_remove_btn)
        self.rgn1_horiz_row_2_grid.add_stretch()


        rgn1_horiz_row_grid_margin = gui.Margins(0.3*self.em, 2*self.em, 0.3*self.em, 0*self.em)
        self.rgn1_horiz_row_3_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_3_grid.preferred_height = 2 * self.em
        self.target_pcd_label = gui.Label("Target PCD Filename: ")
        self.target_pcd_text = gui.TextEdit()
        self.target_pcd_text.text_value = "only_road_cloud.ply"

        self.rgn1_horiz_row_3_grid.add_stretch()
        self.rgn1_horiz_row_3_grid.add_child(self.target_pcd_label)
        self.rgn1_horiz_row_3_grid.add_child(self.target_pcd_text)
        self.rgn1_horiz_row_3_grid.add_stretch()

        self.rgn1_horiz_row_4_grid = gui.Horiz()
        self.rgn1_horiz_row_4_grid.preferred_height = 2 * self.em
        self.target_pcd_load_btn = gui.Button(f"Load Target Cloud")
        self.target_pcd_load_btn.set_on_clicked(self._on_target_pcd_load_btn_clicked)
        self.target_pcd_remove_btn = gui.Button(f"Remove Target Cloud")
        self.target_pcd_remove_btn.set_on_clicked(self._on_target_pcd_remove_btn_clicked)

        self.rgn1_horiz_row_4_grid.add_stretch()
        self.rgn1_horiz_row_4_grid.add_child(self.target_pcd_load_btn)
        self.rgn1_horiz_row_4_grid.add_stretch()
        self.rgn1_horiz_row_4_grid.add_child(self.target_pcd_remove_btn)
        self.rgn1_horiz_row_4_grid.add_stretch()

        self.rgn1_pcd_load_layout.add_child(self.rgn1_horiz_row_1_grid)
        self.rgn1_pcd_load_layout.add_child(self.rgn1_horiz_row_2_grid)
        self.rgn1_pcd_load_layout.add_child(self.rgn1_horiz_row_3_grid)
        self.rgn1_pcd_load_layout.add_child(self.rgn1_horiz_row_4_grid)

        self.main_layout.add_child(self.rgn1_pcd_load_layout)

        # endregion 1

        # region 2: Select Region of Interest

        self.rgn2_roi_select_layout = gui.CollapsableVert("ROI Select", spacing_betn_items_in_region,
                                         margins_for_region)
        # rgn2_horiz_row_grid_spacing = 0.1 * self.em
        # rgn2_horiz_row_grid_margin = gui.Margins(0.3*self.em, 0*self.em, 0.3*self.em, 0*self.em)

        self.rgn2_horiz_row_1_grid = gui.Horiz()
        self.rgn2_horiz_row_1_grid.preferred_height = 2 * self.em
        self.roi_select_boundary_chk_box = gui.Checkbox(f"Select ROI Boundary")
        self.roi_select_boundary_chk_box.set_on_checked(self._on_roi_select_boundary_chk_box_clicked)
        self.roi_select_rect_regn_btn = gui.Button(f"Select RECT ROI")
        self.roi_select_rect_regn_btn.set_on_clicked(self._on_roi_select_rect_regn_btn_clicked)

        self.rgn2_horiz_row_1_grid.add_stretch()
        self.rgn2_horiz_row_1_grid.add_child(self.roi_select_boundary_chk_box)
        self.rgn2_horiz_row_1_grid.add_stretch()
        self.rgn2_horiz_row_1_grid.add_child(self.roi_select_rect_regn_btn)
        self.rgn2_horiz_row_1_grid.add_stretch()


        self.rgn2_horiz_row_2_grid = gui.Horiz()
        self.rgn2_horiz_row_2_grid.preferred_height = 2 * self.em
        self.roi_reset_btn = gui.Button(f"{' '* self.em}   RESET ROI Boundary    {' '* self.em}")
        self.roi_reset_btn.set_on_clicked(self._on_roi_reset_btn_clicked)

        self.rgn2_horiz_row_2_grid.add_stretch()
        self.rgn2_horiz_row_2_grid.add_child(self.roi_reset_btn)
        self.rgn2_horiz_row_2_grid.add_stretch()

        self.rgn2_roi_select_layout.add_child(self.rgn2_horiz_row_1_grid)
        self.rgn2_roi_select_layout.add_child(self.rgn2_horiz_row_2_grid)

        self.main_layout.add_child(self.rgn2_roi_select_layout)

        # endregion 2




        # Add the layout to the window
        self.window.add_child(self.main_layout)
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.mat = rendering.MaterialRecord()
        self.mat.point_size = 3 * self.window.scaling
        self.mat.shader = "defaultUnlit"
        self.widget3d.look_at(  [0, 0, 0],  # center
                                [0, 0, 30],  # eye
                                [4, 4, 4])  # Up
        
        self.widget3d.set_on_mouse(self._on_mouse_widget3d)

    # region member functions

    def _on_window_resize(self, width, height):
        main_layout_width = 25 * self.em  # Adjust as needed
        main_layout_height = height
        main_layout_x = width - main_layout_width
        main_layout_y = 0  # Top of the window
        self.main_layout.frame = gui.Rect(main_layout_x, main_layout_y, main_layout_width, main_layout_height)


    def _on_layout(self, event):
        print("Layout event")
        self._on_window_resize(self.window.size.width, self.window.size.height)
        r = self.window.content_rect
        self.widget3d.frame = r


    def _on_source_pcd_load_btn_clicked(self):
        print("Source PCD Load Button clicked")        
        self.source_cloud = o3d.io.read_point_cloud(self.source_pcd_text.text_value)
        num_points = len(self.source_cloud.points)
        colors = np.zeros((num_points, 3))
        self.source_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.widget3d.scene.add_geometry("source_cloud", self.source_cloud, self.mat)

    
    def _on_source_pcd_remove_btn_clicked(self):
        print("Source PCD Remove Button clicked")
        self.source_cloud = None
        self.widget3d.scene.scene.remove_geometry("source_cloud")


    def _on_target_pcd_load_btn_clicked(self):
        print("Source PCD Load Button clicked")
        self.target_cloud = o3d.io.read_point_cloud(self.target_pcd_text.text_value)
        num_points = len(self.target_cloud.points)
        colors = np.zeros((num_points, 3))
        colors[:,1] = 1 # set all points to green
        self.target_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.widget3d.scene.scene.add_geometry("target_cloud", self.target_cloud, self.mat)

    
    def _on_target_pcd_remove_btn_clicked(self):
        print("Target PCD Remove Button clicked")
        self.target_cloud = None
        self.widget3d.scene.scene.remove_geometry("target_cloud")


    def _on_roi_select_boundary_chk_box_clicked(self, checked):
        print("ROI Select Boundary Chk Box clicked : ", checked)

    def _on_roi_select_rect_regn_btn_clicked(self):  # Maybe later it could be dynamic for both clouds
        print("ROI Select Rectangular Region Button clicked")
        # Convert the point cloud to a numpy array
        points = np.asarray(self.target_cloud.points)
        colors = np.asarray(self.target_cloud.colors)
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
        self.target_cloud.colors = o3d.utility.Vector3dVector(colors)
        if self.widget3d.scene.scene.has_geometry("target_cloud"):
            print("Updating the geometry")
            self.widget3d.scene.scene.remove_geometry("target_cloud")
        self.widget3d.scene.scene.add_geometry("target_cloud", self.target_cloud, self.mat)
        self.widget3d.force_redraw()
    
    def _on_roi_reset_btn_clicked(self):
        print("ROI Reset Button clicked")
        self.widget3d.scene.scene.remove_geometry("target_cloud")
        # Reset the color of the target cloud to green
        num_points = len(self.target_cloud.points)
        colors = np.zeros((num_points, 3))
        colors[:,1] = 1 # set all points to green
        self.target_cloud.colors = o3d.utility.Vector3dVector(colors)
        # Add the target cloud again
        self.widget3d.scene.scene.add_geometry("target_cloud", self.target_cloud, self.mat)

    def _on_mouse_widget3d(self, event):
        if  event.is_modifier_down(gui.KeyModifier.CTRL):
            if self.roi_select_boundary_chk_box.checked:
                print("CTRL/CMD + Mouse DOWN BTN Clicked")
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
                        distances = np.sum((self.target_cloud.points - world) ** 2, axis=1)
                        nearest_point_index = np.argmin(distances)
                        print("Nearest Point Index: ", nearest_point_index)
                        self.target_cloud.colors[nearest_point_index] = [1, 0, 0]  # Change to red                        
                        cloud_tensor = o3d.t.geometry.PointCloud().from_legacy(self.target_cloud)
                        if self.widget3d.scene.scene.has_geometry("target_cloud"):
                            print("Updating the geometry")
                            self.widget3d.scene.scene.remove_geometry("target_cloud")
                        self.widget3d.scene.scene.add_geometry("target_cloud", cloud_tensor, self.mat)
                        self.widget3d.force_redraw()
                self.widget3d.scene.scene.render_to_depth_image(depth_callback)
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED



    # endregion member functions


def main():

    app = gui.Application.instance
    app.initialize()
    ex = ScenarioCreatorApp()
    app.run()



if __name__ == "__main__":

    main()