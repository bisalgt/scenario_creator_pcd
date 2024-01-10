import copy

import math

import numpy as np

import open3d as o3d

import matplotlib.pyplot as plt

import open3d.visualization.gui as gui

import open3d.visualization.rendering as rendering



class ScenarioCreatorApp:


    def __init__(self):

        # Variables
        self.source_cloud = None
        self.target_cloud = None


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

        self.rg1_pcd_load_layout = gui.CollapsableVert("Load Point Cloud", spacing_betn_items_in_region,
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

        self.rg1_pcd_load_layout.add_child(self.rgn1_horiz_row_1_grid)
        self.rg1_pcd_load_layout.add_child(self.rgn1_horiz_row_2_grid)
        self.rg1_pcd_load_layout.add_child(self.rgn1_horiz_row_3_grid)
        self.rg1_pcd_load_layout.add_child(self.rgn1_horiz_row_4_grid)

        self.main_layout.add_child(self.rg1_pcd_load_layout)

        # endregion 1

        # region 2: Select Region of Interest




        # Add the layout to the window
        self.window.add_child(self.main_layout)
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.widget3d.look_at(  [0, 0, 0],  # center
                                [0, 0, 30],  # eye
                                [4, 4, 4])  # Up

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
        print(self.source_pcd_text.text_value)
        self.mat.point_size = 3 * self.window.scaling
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

    # endregion member functions


def main():

    app = gui.Application.instance
    app.initialize()
    ex = ScenarioCreatorApp()
    app.run()



if __name__ == "__main__":

    main()