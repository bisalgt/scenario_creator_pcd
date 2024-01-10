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
        self.centroid_of_reference_roi = None
        self.centroid_of_target_roi = None
        self.source_cloud_transformed = None
        self.reconstructed_source_mesh = None
        self.reconstructed_source_mesh_densities = None

        
        
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

        # region 3: Transformation of Source Cloud to the selected ROI of Target Cloud
        
        self.rgn3_transform_source_layout = gui.CollapsableVert("Transform Source Cloud to Target ROI", spacing_betn_items_in_region,
                                         margins_for_region)


        self.rgn3_horiz_row_1_grid = gui.Horiz()
        self.rgn3_horiz_row_1_grid.preferred_height = 2 * self.em
        self.calculate_centroid_of_reference_roi_btn = gui.Button(f" FindRef.ROICenter ")
        self.calculate_centroid_of_reference_roi_btn.set_on_clicked(self._on_calculate_centroid_of_reference_roi_btn_clicked)
        self.calculate_centroid_of_target_roi_btn = gui.Button(f"FindTarget.ROICenter")
        self.calculate_centroid_of_target_roi_btn.set_on_clicked(self._on_calculate_centroid_of_target_roi_btn_clicked)
        self.rgn3_horiz_row_1_grid.add_stretch()
        self.rgn3_horiz_row_1_grid.add_child(self.calculate_centroid_of_reference_roi_btn)
        self.rgn3_horiz_row_1_grid.add_stretch()
        self.rgn3_horiz_row_1_grid.add_child(self.calculate_centroid_of_target_roi_btn)
        self.rgn3_horiz_row_1_grid.add_stretch()

        self.rgn3_horiz_row_2_grid = gui.Horiz()
        self.rgn3_horiz_row_2_grid.preferred_height = 2 * self.em
        self.transform_source_pcd_to_target_roi = gui.Button(f"Transf. Source Cloud")
        self.transform_source_pcd_to_target_roi.set_on_clicked(self._on_transform_source_pcd_to_target_roi_clicked)
        self.remove_transformed_source_pcd_to_target_roi = gui.Button(f"RemoveTransformSrc")
        self.remove_transformed_source_pcd_to_target_roi.set_on_clicked(self._on_remove_transformed_source_pcd_to_target_roi_clicked)
        self.rgn3_horiz_row_2_grid.add_stretch()
        self.rgn3_horiz_row_2_grid.add_child(self.transform_source_pcd_to_target_roi)
        self.rgn3_horiz_row_2_grid.add_stretch()
        self.rgn3_horiz_row_2_grid.add_child(self.remove_transformed_source_pcd_to_target_roi)
        self.rgn3_horiz_row_2_grid.add_stretch()

        self.rgn3_transform_source_layout.add_child(self.rgn3_horiz_row_1_grid)
        self.rgn3_transform_source_layout.add_child(self.rgn3_horiz_row_2_grid)

        self.main_layout.add_child(self.rgn3_transform_source_layout)



        # endregion 3


        # region 4: Surface Reconstruction

        self.rgn4_surface_reconstruct_layout = gui.CollapsableVert("SurfaceReconstruction & Filter by Densities", spacing_betn_items_in_region, margins_for_region)

        self.rgn4_horiz_row_1_grid = gui.Horiz()
        self.rgn4_horiz_row_1_grid.preferred_height = 2 * self.em
        self.rgn4_radius_label = gui.Label("  R : ")
        self.rgn4_radius_text = gui.TextEdit()
        self.rgn4_radius_text.text_value = "0.37"
        self.rgn4_nearest_neighbors_label = gui.Label("  NN : ")
        self.rgn4_nearest_neighbors_text = gui.TextEdit()
        self.rgn4_nearest_neighbors_text.text_value = "6"
        self.rgn4_reconstruct_surf_depth_label = gui.Label("  D : ")
        self.rgn4_reconstruct_surf_depth_text = gui.TextEdit()
        self.rgn4_reconstruct_surf_depth_text.text_value = "9"


        self.rgn4_horiz_row_1_grid.add_stretch()
        self.rgn4_horiz_row_1_grid.add_child(self.rgn4_radius_label)
        self.rgn4_horiz_row_1_grid.add_child(self.rgn4_radius_text)
        self.rgn4_horiz_row_1_grid.add_stretch()
        self.rgn4_horiz_row_1_grid.add_child(self.rgn4_nearest_neighbors_label)
        self.rgn4_horiz_row_1_grid.add_child(self.rgn4_nearest_neighbors_text)
        self.rgn4_horiz_row_1_grid.add_stretch()
        self.rgn4_horiz_row_1_grid.add_child(self.rgn4_reconstruct_surf_depth_label)
        self.rgn4_horiz_row_1_grid.add_child(self.rgn4_reconstruct_surf_depth_text)

        self.rgn4_horiz_row_2_grid = gui.Horiz()
        self.rgn4_horiz_row_2_grid.preferred_height = 2 * self.em
        self.reconstruct_surface_btn = gui.Button(f"ShowReconstr.Surf.")
        self.reconstruct_surface_btn.toggleable = True
        self.reconstruct_surface_btn.set_on_clicked(self._on_reconstruct_surface_btn_clicked)
        self.calculate_density_mesh_btn = gui.Button(f"ShowDensityMesh")
        self.calculate_density_mesh_btn.toggleable = True
        self.calculate_density_mesh_btn.set_on_clicked(self._on_calculate_density_mesh_btn_clicked)
        self.rgn4_horiz_row_2_grid.add_stretch()
        self.rgn4_horiz_row_2_grid.add_child(self.reconstruct_surface_btn)
        self.rgn4_horiz_row_2_grid.add_stretch()
        self.rgn4_horiz_row_2_grid.add_child(self.calculate_density_mesh_btn)
        self.rgn4_horiz_row_2_grid.add_stretch()

        self.rgn4_horiz_row_3_grid = gui.Horiz()
        self.rgn4_horiz_row_3_grid.preferred_height = 2 * self.em
        self.filter_density_slider = gui.Slider(gui.Slider.DOUBLE)
        self.filter_density_slider.set_limits(0, 1)
        self.filter_density_slider.double_value = 0.5
        self.filter_density_btn = gui.Button(f"Filter Density Mesh")
        self.filter_density_btn.set_on_clicked(self._on_filter_density_btn_clicked)

        self.rgn4_horiz_row_3_grid.add_stretch()
        self.rgn4_horiz_row_3_grid.add_child(self.filter_density_slider)
        self.rgn4_horiz_row_3_grid.add_stretch()


        self.rgn4_horiz_row_4_grid = gui.Horiz()
        self.rgn4_horiz_row_4_grid.preferred_height = 2 * self.em
        self.filter_density_btn = gui.Button(f"Filter Density Mesh")
        self.filter_density_btn.toggleable = True
        self.filter_density_btn.set_on_clicked(self._on_filter_density_btn_clicked)

        self.rgn4_horiz_row_4_grid.add_stretch()
        self.rgn4_horiz_row_4_grid.add_child(self.filter_density_btn)
        self.rgn4_horiz_row_4_grid.add_stretch()


        self.rgn4_surface_reconstruct_layout.add_child(self.rgn4_horiz_row_1_grid)
        self.rgn4_surface_reconstruct_layout.add_child(self.rgn4_horiz_row_2_grid)
        self.rgn4_surface_reconstruct_layout.add_child(self.rgn4_horiz_row_3_grid)
        self.rgn4_surface_reconstruct_layout.add_child(self.rgn4_horiz_row_4_grid)

        self.main_layout.add_child(self.rgn4_surface_reconstruct_layout)

        # endregion 4


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
        
    def _is_pcd_loaded(self):
        if self.source_cloud is None or self.target_cloud is None:
            print("Source or Target PCD is not loaded")
            return False
        else:
            return True

    def check_if_pcd_is_loaded(func):
        def wrapper(self, *args, **kwargs):
            if self.source_cloud is None or self.target_cloud is None:
                print("Source or Target PCD is not loaded")
                return
            else:
                return func(self, *args, **kwargs)
        return wrapper

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
        if self.source_cloud is None:
            print("Source PCD is not loaded to remove")
            return
        print("Source PCD Remove Button clicked")
        self.source_cloud = None
        self.widget3d.scene.scene.remove_geometry("source_cloud")


    def _on_target_pcd_load_btn_clicked(self):
        print("Target PCD Load Button clicked")
        self.target_cloud = o3d.io.read_point_cloud(self.target_pcd_text.text_value)
        num_points = len(self.target_cloud.points)
        colors = np.zeros((num_points, 3))
        colors[:,1] = 1 # set all points to green
        self.target_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.widget3d.scene.scene.add_geometry("target_cloud", self.target_cloud, self.mat)

    
    def _on_target_pcd_remove_btn_clicked(self):
        if self.target_cloud is None:
            print("Target PCD is not loaded to remove")
            return
        print("Target PCD Remove Button clicked")
        self.target_cloud = None
        self.widget3d.scene.scene.remove_geometry("target_cloud")

    def _on_roi_select_boundary_chk_box_clicked(self, checked):
        print("ROI Select Boundary Chk Box clicked : ", checked)
    
    @check_if_pcd_is_loaded
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
    
    @check_if_pcd_is_loaded
    def _on_roi_reset_btn_clicked(self):
        print("ROI Reset Button clicked")
        self.selected_pcd_indices = None
        self.widget3d.scene.scene.remove_geometry("target_cloud")
        # Reset the color of the target cloud to green
        num_points = len(self.target_cloud.points)
        colors = np.zeros((num_points, 3))
        colors[:,1] = 1 # set all points to green
        self.target_cloud.colors = o3d.utility.Vector3dVector(colors)
        # Add the target cloud again
        self.widget3d.scene.scene.add_geometry("target_cloud", self.target_cloud, self.mat)

    @check_if_pcd_is_loaded
    def _calculate_centroid_of_roi(self):
        if not self._is_pcd_loaded():
            return
        if self.selected_pcd_indices is None:
            print("No ROI selected")
            return
        _centroid = np.asarray(self.target_cloud.points)[self.selected_pcd_indices].mean(axis=0)
        return _centroid
    
    @check_if_pcd_is_loaded
    def _on_calculate_centroid_of_reference_roi_btn_clicked(self):
        print("Calculate Centroid of Reference ROI Button clicked")
        self.centroid_of_reference_roi = self._calculate_centroid_of_roi()
        print("Centroid of Reference ROI: ", self.centroid_of_reference_roi)

    @check_if_pcd_is_loaded
    def _on_calculate_centroid_of_target_roi_btn_clicked(self):
        print("Calculate Centroid of Target ROI Button clicked")
        self.centroid_of_target_roi = self._calculate_centroid_of_roi()
        print("Centroid of Target ROI: ", self.centroid_of_target_roi)

    @check_if_pcd_is_loaded
    def _on_transform_source_pcd_to_target_roi_clicked(self):
        print("Transform Source PCD to Target ROI Button clicked")

        if self.centroid_of_reference_roi is None or self.centroid_of_target_roi is None:
            print("Centroids of ROIs are not calculated")
            return
        # Translation part works fine
        translation = self.centroid_of_target_roi - self.centroid_of_reference_roi
        print("Translation: ", translation)
        translation_matrix = np.identity(4)
        translation_matrix[:3, 3] = translation
        self.source_cloud_transformed  = copy.deepcopy(self.source_cloud).transform(translation_matrix)

        # Needs to figure out rotation part
        start_vector = np.array(self.source_cloud.get_center())
        end_vector = np.array(self.source_cloud_transformed.get_center())
        # Calculate the cosine of the angle
        cos_angle = np.dot(start_vector, end_vector) / (np.linalg.norm(start_vector) * np.linalg.norm(end_vector))
        # Calculate the sine of the angle
        sin_angle = np.linalg.norm(np.cross(start_vector, end_vector)) / (np.linalg.norm(start_vector) * np.linalg.norm(end_vector))
        # Calculate the angle in radians
        angle_rad = np.arctan2(sin_angle, cos_angle)
        # If the z-component of the cross product is negative, the angle should be negative
        if np.cross(start_vector, end_vector)[2] < 0:
            print("Negative angle")
            angle_rad = -angle_rad
        else:
            print("Positive angle")
        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)


        print("Angle in degrees: ", angle_deg)
        print("Angle in radians: ", angle_rad)

        # Create the rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        print("Rotation Matrix: ", rotation_matrix)
        self.source_cloud_transformed.rotate(rotation_matrix, center=np.asarray(self.source_cloud_transformed.get_center()))
        self.widget3d.scene.scene.add_geometry("source_cloud_transformed", self.source_cloud_transformed, self.mat)

    @check_if_pcd_is_loaded
    def _on_remove_transformed_source_pcd_to_target_roi_clicked(self):
        print("Remove Transform Source PCD to Target ROI Button clicked")
        self.widget3d.scene.scene.remove_geometry("source_cloud_transformed")
        self.source_cloud_transformed = None
        self.centroid_of_reference_roi = None
        self.centroid_of_target_roi = None
        self._on_roi_reset_btn_clicked()

    
    def _on_reconstruct_surface_btn_clicked(self):
        if self.source_cloud is None:
            self.reconstruct_surface_btn.is_on = False
            print("Source PCD is not loaded")
            return
        print("Reconstruct Surface Button clicked")

        if self.reconstruct_surface_btn.is_on:
            self.reconstruct_surface_btn.text = "RemoveReconstr.Surf."
            print("Reconstruct Surface Button is ON")
            radius = float(self.rgn4_radius_text.text_value)
            nearest_neighbors = int(self.rgn4_nearest_neighbors_text.text_value)
            depth = int(self.rgn4_reconstruct_surf_depth_text.text_value)
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nearest_neighbors)
            source_cloud = copy.deepcopy(self.source_cloud)
            source_cloud.estimate_normals(search_param=search_param)
            print('run Poisson surface reconstruction')
            with o3d.utility.VerbosityContextManager(
                    o3d.utility.VerbosityLevel.Debug) as cm:
                self.reconstructed_source_mesh, self.reconstructed_source_mesh_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    source_cloud, depth=depth)
            print(self.reconstructed_source_mesh)
            self.reconstructed_source_mesh.compute_vertex_normals()
            # Paint it gray. Not necessary but the reflection of lighting is hardly perceivable with black surfaces.
            self.reconstructed_source_mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))

            self.widget3d.scene.scene.add_geometry("reconstructed_source_mesh", self.reconstructed_source_mesh, self.mat)
        else:
            self.reconstruct_surface_btn.text = "ShowReconstr.Surf."
            print("Reconstruct Surface Button is OFF")
            self.widget3d.scene.scene.remove_geometry("reconstructed_source_mesh")
            self.reconstructed_source_mesh = None
            self.reconstructed_source_mesh_densities = None
            if self.widget3d.scene.scene.has_geometry("reconstructed_source_mesh_densities_with_color"):
                self.calculate_density_mesh_btn.is_on = False
                self.calculate_density_mesh_btn.text = "FilterDensityMesh"
                self.widget3d.scene.scene.remove_geometry("reconstructed_source_mesh_densities_with_color")
                self.reconstructed_source_mesh_densities_with_color = None
                self.reconstructed_source_mesh_densities_array = None

                if self.widget3d.scene.scene.has_geometry("reconstructed_source_mesh_filtered_densities_mesh"):
                    self.filter_density_btn.is_on = False
                    self.filter_density_btn.text = "FilterDensityMesh"
                    self.widget3d.scene.scene.remove_geometry("reconstructed_source_mesh_filtered_densities_mesh")
                    self.reconstructed_source_mesh_filtered_densities_mesh = None
                    print("Done removing filtered density")

        self.widget3d.force_redraw()


        


    def _on_calculate_density_mesh_btn_clicked(self):
        print("Calculate Density Mesh Button clicked")
        if self.reconstructed_source_mesh is None or self.reconstructed_source_mesh_densities is None:
            self.calculate_density_mesh_btn.is_on = False
            print("Perform Surface Reconstruction First! Surface Mesh is not available")
            return
        if self.calculate_density_mesh_btn.is_on:
            print("Calculate Density Mesh Button is ON")
            self.calculate_density_mesh_btn.text = "RemoveDensityMesh"
            self.reconstructed_source_mesh_densities_array = copy.deepcopy(np.asarray(self.reconstructed_source_mesh_densities))
            density_colors = plt.get_cmap('plasma')(
                (self.reconstructed_source_mesh_densities_array - self.reconstructed_source_mesh_densities_array.min()) / (self.reconstructed_source_mesh_densities_array.max() - self.reconstructed_source_mesh_densities_array.min()))
            density_colors = density_colors[:, :3]
            self.reconstructed_source_mesh_densities_with_color = o3d.geometry.TriangleMesh()
            self.reconstructed_source_mesh_densities_with_color.vertices = self.reconstructed_source_mesh.vertices
            self.reconstructed_source_mesh_densities_with_color.triangles = self.reconstructed_source_mesh.triangles
            self.reconstructed_source_mesh_densities_with_color.triangle_normals = self.reconstructed_source_mesh.triangle_normals
            self.reconstructed_source_mesh_densities_with_color.vertex_colors = o3d.utility.Vector3dVector(density_colors)
            self.widget3d.scene.scene.add_geometry("reconstructed_source_mesh_densities_with_color", self.reconstructed_source_mesh_densities_with_color, self.mat)
        else:
            print("Calculate Density Mesh Button is OFF")
            self.calculate_density_mesh_btn.text = "ShowDensityMesh"
            self.widget3d.scene.scene.remove_geometry("reconstructed_source_mesh_densities_with_color")
            self.reconstructed_source_mesh_densities_with_color = None
            self.reconstructed_source_mesh_densities_array = None



    def _on_filter_density_btn_clicked(self):
        print("Filter Density Button clicked")
        if self.reconstructed_source_mesh_densities_array is None:
            self.filter_density_btn.is_on = False
            self.filter_density_btn.text = "FilterDensityMesh"
            print("Calculate Density Mesh First!")
            return
        if self.filter_density_btn.is_on:
            print("Filter Density Button clicked ON")
            self.filter_density_btn.text = "RemoveFilteredDensity"
            self.reconstructed_source_mesh_filtered_densities_mesh = copy.deepcopy(self.reconstructed_source_mesh)
            vertices_to_remove = self.reconstructed_source_mesh_densities_array < np.quantile(self.reconstructed_source_mesh_densities_array, self.filter_density_slider.double_value)
            self.reconstructed_source_mesh_filtered_densities_mesh.remove_vertices_by_mask(vertices_to_remove)
            self.reconstructed_source_mesh_filtered_densities_mesh.compute_vertex_normals()
            # Paint it gray. Not necessary but the reflection of lighting is hardly perceivable with black surfaces.
            self.reconstructed_source_mesh_filtered_densities_mesh.paint_uniform_color(np.array([[0],[0],[1]])) # blue
            self.widget3d.scene.scene.add_geometry("reconstructed_source_mesh_filtered_densities_mesh", self.reconstructed_source_mesh_filtered_densities_mesh, self.mat)
            self.widget3d.force_redraw()
            print("Done filtering density")
        else:
            print("Filter Density Button clicked OFF")
            self.filter_density_btn.text = "FilterDensityMesh"
            self.widget3d.scene.scene.remove_geometry("reconstructed_source_mesh_filtered_densities_mesh")
            self.reconstructed_source_mesh_filtered_densities_mesh = None
            self.widget3d.force_redraw()
            print("Done removing filtered density")

    def _on_mouse_widget3d(self, event):
        if  event.is_modifier_down(gui.KeyModifier.CTRL):
            if self.roi_select_boundary_chk_box.checked:
                if not self._is_pcd_loaded():
                    return gui.Widget.EventCallbackResult.IGNORED
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