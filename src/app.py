import os

import copy

import math

import pickle

import numpy as np

import pandas as pd

import open3d as o3d

import matplotlib.pyplot as plt

import open3d.visualization.gui as gui

import open3d.visualization.rendering as rendering


from surface_variation import PointCloudAnalysis



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
        self.reconstructed_source_mesh_densities_array = None
        self.reconstructed_source_mesh_filtered_densities_mesh = None
        self.raycasted_source_cloud = None
        self.final_merged_cloud = None
        self.final_merged_cloud_after_shadow_cast = None
        self.shadowed_cloud = None
        self.shadowed_cloud_indices = None
        self.surface_variation = None
        self.planarity = None
        self.linearity = None
        self.z_value = None
        self.selected_pcd_roi_boundary_indices = []
        self.source_scene_cloud = None
        self.source_scene_object_of_interest_indices = None
        self.selected_pcd_indices_with_obj_indices = None
        self.object_of_interest_color = [0,0,0]
        self.roi_color = [1,0,0]
        self.surrounding_color = [0,1,0]
        self.final_cloud = None
        self.shadow_casted_pcd_using_ray_cast_without_prototype = None
            



        
        
        # endregion 0

        app = gui.Application.instance
        self.window = app.create_window("Scenario-Creator App", 1024, 768)
        self.window.set_on_close(self._on_close)
        self.em = self.window.theme.font_size  # Standard Font size of the window (so that UI changes with different OS)
        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)


        # Create a vertical grid layout for the buttons
        spacing_betn_items = 10
        margins = gui.Margins(5, 20, 5, 10)
        self.main_layout = gui.CollapsableVert("Main Layout", spacing_betn_items, margins)
        self.main_layout.set_is_open(False)
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
        self.rgn1_pcd_load_layout.set_is_open(False)
        rgn1_horiz_row_grid_spacing = 0.1 * self.em
        rgn1_horiz_row_grid_margin = gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em)

        self.rgn1_horiz_row_1_grid = gui.CollapsableVert("Load Source Cloud")
        self.rgn1_horiz_row_1_grid.set_is_open(False)

        self.rgn1_horiz_row_1__subrow_1_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_source_scene_pcd_label = gui.Label("Load Source Scene: ")
        self.rgn1_source_scene_pcd_text = gui.TextEdit()
        self.rgn1_source_scene_pcd_text.text_value = "jan31_semantic_lidar_source_scene.csv"  # we will extract source cloud from the source scene cloud

        self.rgn1_horiz_row_1__subrow_1_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_1_grid.add_child(self.rgn1_source_scene_pcd_label)
        self.rgn1_horiz_row_1__subrow_1_grid.add_child(self.rgn1_source_scene_pcd_text)
        self.rgn1_horiz_row_1__subrow_1_grid.add_stretch()

        self.rgn1_horiz_row_1__subrow_2_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn1_horiz_row_1__subrow_2_grid.preferred_height = 2 * self.em
        self.rgn1_source_scene_pcd_load_btn = gui.Button(f"Load Scene")
        self.rgn1_source_scene_pcd_load_btn.set_on_clicked(self._on_source_scene_pcd_load_btn_clicked)
        self.rgn1_source_scene_pcd_remove_btn = gui.Button(f"Remove Scene")
        self.rgn1_source_scene_pcd_remove_btn.set_on_clicked(self._on_source_scene_pcd_remove_btn_clicked)

        self.rgn1_horiz_row_1__subrow_2_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_2_grid.add_child(self.rgn1_source_scene_pcd_load_btn)
        self.rgn1_horiz_row_1__subrow_2_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_2_grid.add_child(self.rgn1_source_scene_pcd_remove_btn)
        self.rgn1_horiz_row_1__subrow_2_grid.add_stretch()


        self.rgn1_horiz_row_1__subrow_3_grid = gui.CollapsableVert("Parameters for Source Cloud Extraction")
        self.rgn1_horiz_row_1__subrow_3_grid.set_is_open(False)

        self.rgn1_horiz_row_1__subrow_3_r0_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_3_r0_grid.preferred_height = 2 * self.em

        self.rgn1_use_ml_model_to_extract_src_pcd_chk_box = gui.Checkbox(f"UseMLToExtractSourcePCDfromROI")
        self.rgn1_use_ml_model_to_extract_src_pcd_chk_box.set_on_checked(self._on_rgn1_use_ml_model_to_extract_src_pcd_chk_box_checked)

        self.rgn1_horiz_row_1__subrow_3_r0_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r0_grid.add_child(self.rgn1_use_ml_model_to_extract_src_pcd_chk_box)
        self.rgn1_horiz_row_1__subrow_3_r0_grid.add_stretch()

        self.rgn1_horiz_row_1__subrow_3_r1_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_3_r1_grid.preferred_height = 2 * self.em

        self.rgn1_use_labels_to_extract_src_pcd_chk_box = gui.Checkbox(f"UseLabelToExtractSourcePCDfromROI")
        self.rgn1_use_labels_to_extract_src_pcd_chk_box.set_on_checked(self._on_rgn1_use_labels_to_extract_src_pcd_chk_box_checked)

        self.rgn1_horiz_row_1__subrow_3_r1_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r1_grid.add_child(self.rgn1_use_labels_to_extract_src_pcd_chk_box)
        self.rgn1_horiz_row_1__subrow_3_r1_grid.add_stretch()


        self.rgn1_horiz_row_1__subrow_3_r2_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_3_r2_grid.preferred_height = 2 * self.em
        self.rgn1_use_geometric_features_to_extract_src_pcd_chk_box = gui.Checkbox(f"Use GeometricFeatures to Extract Source")
        self.rgn1_use_geometric_features_to_extract_src_pcd_chk_box.set_on_checked(self._on_rgn1_use_geometric_features_to_extract_src_pcd_chk_box_checked)


        self.rgn1_horiz_row_1__subrow_3_r2_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r2_grid.add_child(self.rgn1_use_geometric_features_to_extract_src_pcd_chk_box)
        self.rgn1_horiz_row_1__subrow_3_r2_grid.add_stretch()


        self.rgn1_horiz_row_1__subrow_3_r0a_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_3_r0a_grid.preferred_height = 2 * self.em
        self.rgn1_ml_model_to_extract_src_pcd_label = gui.Label("Model Filename : ")
        self.rgn1_ml_model_to_extract_src_pcd_text = gui.TextEdit()
        self.rgn1_ml_model_to_extract_src_pcd_text.text_value = "trained_model/model_trained_on_features_knn_5.pkl"  # we will extract source cloud from the source scene cloud

        self.rgn1_horiz_row_1__subrow_3_r0a_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r0a_grid.add_child(self.rgn1_ml_model_to_extract_src_pcd_label)
        self.rgn1_horiz_row_1__subrow_3_r0a_grid.add_child(self.rgn1_ml_model_to_extract_src_pcd_text)
        self.rgn1_horiz_row_1__subrow_3_r0a_grid.add_stretch()

        self.rgn1_horiz_row_1__subrow_3_r0a_grid.enabled = False
        self.rgn1_ml_model_to_extract_src_pcd_text.enabled = False


        self.rgn1_horiz_row_1__subrow_3_r1a_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_3_r1a_grid.preferred_height = 2 * self.em
        self.rgn1_labels_to_extract_src_pcd_label = gui.Label("Label to Extract on ROI: ")
        self.rgn1_labels_to_extract_src_pcd_text = gui.TextEdit()
        self.rgn1_labels_to_extract_src_pcd_text.text_value = "580"  # we will extract source cloud from the source scene cloud

        self.rgn1_horiz_row_1__subrow_3_r1a_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r1a_grid.add_child(self.rgn1_labels_to_extract_src_pcd_label)
        self.rgn1_horiz_row_1__subrow_3_r1a_grid.add_child(self.rgn1_labels_to_extract_src_pcd_text)
        self.rgn1_horiz_row_1__subrow_3_r1a_grid.add_stretch()

        self.rgn1_horiz_row_1__subrow_3_r1a_grid.enabled = False
        self.rgn1_labels_to_extract_src_pcd_text.enabled = False


        self.rgn1_horiz_row_1__subrow_3_r2a_grid = gui.VGrid(1, spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)

        self.rgn1_horiz_row_1__subrow_3_r2aa_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_3_r2aa_grid.preferred_height = 2 * self.em

        self.rgn1_features_to_extract_src_pcd_label = gui.Label("Features to Extract on ROI: ")

        self.rgn1_horiz_row_1__subrow_3_r2aa_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r2aa_grid.add_child(self.rgn1_features_to_extract_src_pcd_label)
        self.rgn1_horiz_row_1__subrow_3_r2aa_grid.add_stretch()

        self.rgn1_horiz_row_1__subrow_3_r2ab_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_3_r2ab_grid.preferred_height = 2 * self.em

        self.rgn1_surface_variation_chk_box = gui.Checkbox(f"Surface Variation")
        self.rgn1_surface_variation_chk_box.set_on_checked(self._on_rgn1_surface_variation_chk_box_checked)

        # Mininum Threshold = Minimum Possible Value. Anything above or equal the value will be considered
        # Maximum Threshold = Maximum Possible Value. Anything below or equal the value will be considered

        self.rgn1_surface_variation_threshold_label = gui.Label("MinThres[0-1]: ")
        self.rgn1_surface_variation_threshold_text = gui.TextEdit()
        self.rgn1_surface_variation_threshold_text.text_value = "0.00000001" # This value was giving better results with nn= 30, for the taken dataset to extract person. Need to remove the ground plane also.

        self.rgn1_horiz_row_1__subrow_3_r2ab_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r2ab_grid.add_child(self.rgn1_surface_variation_chk_box)
        self.rgn1_horiz_row_1__subrow_3_r2ab_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r2ab_grid.add_child(self.rgn1_surface_variation_threshold_label)
        self.rgn1_horiz_row_1__subrow_3_r2ab_grid.add_child(self.rgn1_surface_variation_threshold_text)
        self.rgn1_horiz_row_1__subrow_3_r2ab_grid.add_stretch()
        
        self.rgn1_horiz_row_1__subrow_3_r2ac_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_3_r2ac_grid.preferred_height = 2 * self.em

        self.rgn1_planarity_chk_box = gui.Checkbox(f"Planarity")
        self.rgn1_planarity_chk_box.set_on_checked(self._on_rgn1_planarity_chk_box_checked)

        self.rgn1_planarity_threshold_label = gui.Label("MaxThres[0-1]: ")
        self.rgn1_planarity_threshold_text = gui.TextEdit()
        self.rgn1_planarity_threshold_text.text_value = "0.9"

        self.rgn1_horiz_row_1__subrow_3_r2ac_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r2ac_grid.add_child(self.rgn1_planarity_chk_box)
        self.rgn1_horiz_row_1__subrow_3_r2ac_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r2ac_grid.add_child(self.rgn1_planarity_threshold_label)
        self.rgn1_horiz_row_1__subrow_3_r2ac_grid.add_child(self.rgn1_planarity_threshold_text)
        self.rgn1_horiz_row_1__subrow_3_r2ac_grid.add_stretch()

        self.rgn1_horiz_row_1__subrow_3_r2ad_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_3_r2ad_grid.preferred_height = 2 * self.em

        self.rgn1_linearity_chk_box = gui.Checkbox(f"Linearity")
        self.rgn1_linearity_chk_box.set_on_checked(self._on_rgn1_linearity_chk_box_checked)

        self.rgn1_linearity_threshold_label = gui.Label("MaxThres[0-1]: ")
        self.rgn1_linearity_threshold_text = gui.TextEdit()
        self.rgn1_linearity_threshold_text.text_value = "0.9"

        self.rgn1_horiz_row_1__subrow_3_r2ad_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r2ad_grid.add_child(self.rgn1_linearity_chk_box)
        self.rgn1_horiz_row_1__subrow_3_r2ad_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r2ad_grid.add_child(self.rgn1_linearity_threshold_label)
        self.rgn1_horiz_row_1__subrow_3_r2ad_grid.add_child(self.rgn1_linearity_threshold_text)
        self.rgn1_horiz_row_1__subrow_3_r2ad_grid.add_stretch()


        self.rgn1_horiz_row_1__subrow_3_r2ae_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_3_r2ae_grid.preferred_height = 2 * self.em

        self.rgn1_z_value_chk_box = gui.Checkbox(f"Z-Value")
        self.rgn1_z_value_chk_box.set_on_checked(self._on_rgn1_z_value_chk_box_checked)

        self.rgn1_z_value_threshold_label = gui.Label("MinThres: ")
        self.rgn1_z_value_threshold_text = gui.TextEdit()
        self.rgn1_z_value_threshold_text.text_value = "-2.4"

        self.rgn1_horiz_row_1__subrow_3_r2ae_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r2ae_grid.add_child(self.rgn1_z_value_chk_box)
        self.rgn1_horiz_row_1__subrow_3_r2ae_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_3_r2ae_grid.add_child(self.rgn1_z_value_threshold_label)
        self.rgn1_horiz_row_1__subrow_3_r2ae_grid.add_child(self.rgn1_z_value_threshold_text)
        self.rgn1_horiz_row_1__subrow_3_r2ae_grid.add_stretch()

        
        self.rgn1_horiz_row_1__subrow_3_r2a_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r2aa_grid)
        self.rgn1_horiz_row_1__subrow_3_r2a_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r2ab_grid)
        self.rgn1_horiz_row_1__subrow_3_r2a_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r2ac_grid)
        self.rgn1_horiz_row_1__subrow_3_r2a_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r2ad_grid)
        self.rgn1_horiz_row_1__subrow_3_r2a_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r2ae_grid)


        self.rgn1_horiz_row_1__subrow_3_r2a_grid.enabled = False
        self.rgn1_horiz_row_1__subrow_3_r2aa_grid.enabled = False
        self.rgn1_horiz_row_1__subrow_3_r2ab_grid.enabled = False
        self.rgn1_horiz_row_1__subrow_3_r2ac_grid.enabled = False
        self.rgn1_horiz_row_1__subrow_3_r2ad_grid.enabled = False
        self.rgn1_surface_variation_chk_box.enabled = False
        self.rgn1_surface_variation_threshold_text.enabled = False
        self.rgn1_planarity_chk_box.enabled = False
        self.rgn1_planarity_threshold_text.enabled = False
        self.rgn1_linearity_chk_box.enabled = False
        self.rgn1_linearity_threshold_text.enabled = False
        self.rgn1_z_value_chk_box.enabled = False
        self.rgn1_z_value_threshold_text.enabled = False
        

        self.rgn1_horiz_row_1__subrow_3_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r0_grid)
        self.rgn1_horiz_row_1__subrow_3_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r0a_grid)
        self.rgn1_horiz_row_1__subrow_3_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r1_grid)
        self.rgn1_horiz_row_1__subrow_3_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r1a_grid)
        self.rgn1_horiz_row_1__subrow_3_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r2_grid)
        self.rgn1_horiz_row_1__subrow_3_grid.add_child(self.rgn1_horiz_row_1__subrow_3_r2a_grid)


        self.rgn1_horiz_row_1__subrow_4_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_1__subrow_4_grid.preferred_height = 2 * self.em

        
        self.rgn1_extract_src_pcd_btn = gui.Button(f"ShowExtractedResult")
        self.rgn1_extract_src_pcd_btn.set_on_clicked(self._on_extract_src_pcd_btn_clicked)
        self.rgn1_extract_src_pcd_btn.toggleable = True
        self.rgn1_finalize_extracted_src_pcd_btn = gui.Button(f"FinalizeExtractedResult")
        self.rgn1_finalize_extracted_src_pcd_btn.set_on_clicked(self._on_finalize_extracted_src_pcd_btn_clicked)
        # self.rgn1_finalize_extracted_src_pcd_btn.toggleable = True


        self.rgn1_horiz_row_1__subrow_4_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_4_grid.add_child(self.rgn1_extract_src_pcd_btn)
        self.rgn1_horiz_row_1__subrow_4_grid.add_stretch()
        self.rgn1_horiz_row_1__subrow_4_grid.add_child(self.rgn1_finalize_extracted_src_pcd_btn)
        self.rgn1_horiz_row_1__subrow_4_grid.add_stretch()


        self.rgn1_horiz_row_1_grid.add_child(self.rgn1_horiz_row_1__subrow_1_grid)
        self.rgn1_horiz_row_1_grid.add_child(self.rgn1_horiz_row_1__subrow_2_grid)
        self.rgn1_horiz_row_1_grid.add_child(self.rgn1_horiz_row_1__subrow_3_grid)
        self.rgn1_horiz_row_1_grid.add_child(self.rgn1_horiz_row_1__subrow_4_grid)




        rgn1_horiz_row_grid_margin = gui.Margins(0.3*self.em, 2*self.em, 0.3*self.em, 0*self.em)
        self.rgn1_horiz_row_3_grid = gui.Horiz(spacing=rgn1_horiz_row_grid_spacing, margins=rgn1_horiz_row_grid_margin)
        self.rgn1_horiz_row_3_grid.preferred_height = 2 * self.em
        self.target_pcd_label = gui.Label("Target PCD Filename: ")
        self.target_pcd_text = gui.TextEdit()
        self.target_pcd_text.text_value = "jan31_semantic_lidar_target_scene_without_person.ply"

        self.rgn1_horiz_row_3_grid.add_stretch()
        self.rgn1_horiz_row_3_grid.add_child(self.target_pcd_label)
        self.rgn1_horiz_row_3_grid.add_child(self.target_pcd_text)
        self.rgn1_horiz_row_3_grid.add_stretch()

        self.rgn1_horiz_row_4_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
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
        # self.rgn1_pcd_load_layout.add_child(self.rgn1_horiz_row_2_grid)
        self.rgn1_pcd_load_layout.add_child(self.rgn1_horiz_row_3_grid)
        self.rgn1_pcd_load_layout.add_child(self.rgn1_horiz_row_4_grid)

        self.main_layout.add_child(self.rgn1_pcd_load_layout)

        # endregion 1

        # region 2: Select Region of Interest

        self.rgn2_roi_select_layout = gui.CollapsableVert("ROI Select", spacing_betn_items_in_region,
                                         margins_for_region)
        self.rgn2_roi_select_layout.set_is_open(False)
        # rgn2_horiz_row_grid_spacing = 0.1 * self.em
        # rgn2_horiz_row_grid_margin = gui.Margins(0.3*self.em, 0*self.em, 0.3*self.em, 0*self.em)

        self.rgn2_horiz_row_1_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
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


        self.rgn2_horiz_row_2_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
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
        self.rgn3_transform_source_layout.set_is_open(False)

        self.rgn3_horiz_row_1_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
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

        self.rgn3_horiz_row_2_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn3_horiz_row_2_grid.preferred_height = 2 * self.em
        self.transform_source_pcd_to_target_roi = gui.Button(f"Transf.SourceCloud")
        self.transform_source_pcd_to_target_roi.set_on_clicked(self._on_transform_source_pcd_to_target_roi_clicked)
        self.transform_source_pcd_to_target_roi.toggleable = True
        self.finalize_transformed_source_pcd_to_target_roi = gui.Button(f"FinalizeTransform")
        self.finalize_transformed_source_pcd_to_target_roi.set_on_clicked(self._on_finalize_transformed_source_pcd_to_target_roi_clicked)
        self.rgn3_horiz_row_2_grid.add_stretch()
        self.rgn3_horiz_row_2_grid.add_child(self.transform_source_pcd_to_target_roi)
        self.rgn3_horiz_row_2_grid.add_stretch()
        self.rgn3_horiz_row_2_grid.add_child(self.finalize_transformed_source_pcd_to_target_roi)
        self.rgn3_horiz_row_2_grid.add_stretch()

        # self.rgn3_transform_source_layout.add_child(self.rgn3_horiz_row_1_grid)
        self.rgn3_transform_source_layout.add_child(self.rgn3_horiz_row_2_grid)

        self.main_layout.add_child(self.rgn3_transform_source_layout)



        # endregion 3


        # region 4: Surface Reconstruction

        self.rgn4_surface_reconstruct_layout = gui.CollapsableVert("SurfaceReconstruction & Filter by Densities", spacing_betn_items_in_region, margins_for_region)
        self.rgn4_surface_reconstruct_layout.set_is_open(False)

        self.rgn4_horiz_row_1_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn4_horiz_row_1_grid.preferred_height = 2 * self.em
        self.rgn4_radius_label = gui.Label("  R : ")
        self.rgn4_radius_text = gui.TextEdit()
        self.rgn4_radius_text.text_value = "0.1"
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

        self.rgn4_horiz_row_2_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
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

        self.rgn4_horiz_row_3_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn4_horiz_row_3_grid.preferred_height = 2 * self.em
        self.filter_density_label = gui.Label("Filter Density : ")
        self.filter_density_slider = gui.Slider(gui.Slider.DOUBLE)
        self.filter_density_slider.set_limits(0, 1)
        self.filter_density_slider.double_value = 0.5

        self.rgn4_horiz_row_3_grid.add_stretch()
        self.rgn4_horiz_row_3_grid.add_child(self.filter_density_label)
        self.rgn4_horiz_row_3_grid.add_child(self.filter_density_slider)
        self.rgn4_horiz_row_3_grid.add_stretch()


        self.rgn4_horiz_row_4_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn4_horiz_row_4_grid.preferred_height = 2 * self.em
        self.filter_density_btn = gui.Button(f"Filter Density Mesh")
        self.filter_density_btn.toggleable = True
        self.filter_density_btn.set_on_clicked(self._on_filter_density_btn_clicked)

        self.rgn4_horiz_row_4_grid.add_stretch()
        self.rgn4_horiz_row_4_grid.add_child(self.filter_density_btn)
        self.rgn4_horiz_row_4_grid.add_stretch()


        self.rgn4_surface_reconstruct_layout.add_child(self.rgn4_horiz_row_1_grid)
        self.rgn4_surface_reconstruct_layout.add_child(self.rgn4_horiz_row_2_grid)
        # self.rgn4_surface_reconstruct_layout.add_child(self.rgn4_horiz_row_3_grid)
        self.rgn4_surface_reconstruct_layout.add_child(self.rgn4_horiz_row_4_grid)

        self.main_layout.add_child(self.rgn4_surface_reconstruct_layout)

        # endregion 4


        # region 5 : Ray Casting and Visualization
        self.rgn5_raycast_layout = gui.CollapsableVert("Raycasting", spacing_betn_items_in_region, margins_for_region)
        self.rgn5_raycast_layout.set_is_open(False)

        self.rgn5_horiz_row_1_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn5_horiz_row_1_grid.preferred_height = 2 * self.em

        self.filter_rays_label = gui.Label("Filter Rays : ")

        self.filter_rays_slider = gui.Slider(gui.Slider.DOUBLE)
        self.filter_rays_slider.set_limits(0, 1)
        self.filter_rays_slider.double_value = 0.561

        self.rgn5_horiz_row_1_grid.add_stretch()
        self.rgn5_horiz_row_1_grid.add_child(self.filter_rays_label)
        self.rgn5_horiz_row_1_grid.add_stretch()
        self.rgn5_horiz_row_1_grid.add_child(self.filter_rays_slider)
        self.rgn5_horiz_row_1_grid.add_stretch()

        self.rgn5_horiz_row_2_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn5_horiz_row_2_grid.preferred_height = 2 * self.em
        self.show_rays_btn = gui.Button(f"Show Rays")
        self.show_rays_btn.toggleable = True
        self.show_rays_btn.set_on_clicked(self._on_show_rays_btn_clicked)
        self.show_raycasted_pcd_btn = gui.Button(f"Show RayCasted PCD")
        self.show_raycasted_pcd_btn.toggleable = True
        self.show_raycasted_pcd_btn.set_on_clicked(self._on_show_raycasted_pcd_btn_clicked)
        self.rgn5_horiz_row_2_grid.add_stretch()
        self.rgn5_horiz_row_2_grid.add_child(self.show_rays_btn)
        self.rgn5_horiz_row_2_grid.add_stretch()
        self.rgn5_horiz_row_2_grid.add_child(self.show_raycasted_pcd_btn)
        self.rgn5_horiz_row_2_grid.add_stretch()




        self.rgn5_raycast_layout.add_child(self.rgn5_horiz_row_1_grid)
        self.rgn5_raycast_layout.add_child(self.rgn5_horiz_row_2_grid)


        self.main_layout.add_child(self.rgn5_raycast_layout)

        # endregion 5



        # region 6 : Shadow Casting
        self.rgn6_shadow_casting_layout = gui.CollapsableVert("Shadow Casting", spacing_betn_items_in_region, margins_for_region)
        self.rgn6_shadow_casting_layout.set_is_open(False)

        # add a button to show shadow casted by raycasted method
        # show_correct_shadow_casting_btn refers to the shadow casted during raycasting, this method is better than HPR algorithm
        self.rgn6_horiz_row_0_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn6_horiz_row_0_grid.preferred_height = 2 * self.em

        self.show_correct_shadow_casting_btn = gui.Button(f"ShowShadowCastedByRaycasting")
        self.show_correct_shadow_casting_btn.toggleable = True
        self.show_correct_shadow_casting_btn.set_on_clicked(self._on_show_correct_shadow_casting_btn_clicked)

        self.rgn6_horiz_row_0_grid.add_stretch()
        self.rgn6_horiz_row_0_grid.add_child(self.show_correct_shadow_casting_btn)
        self.rgn6_horiz_row_0_grid.add_stretch()

        self.rgn6_horiz_row_1_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn6_horiz_row_1_grid.preferred_height = 2 * self.em
        self.rgn6_radius_label = gui.Label("  Radius : ")
        self.rgn6_radius_text = gui.TextEdit()
        self.rgn6_radius_text.text_value = "300"
        
        self.rgn6_horiz_row_1_grid.add_stretch()
        self.rgn6_horiz_row_1_grid.add_child(self.rgn6_radius_label)
        self.rgn6_horiz_row_1_grid.add_child(self.rgn6_radius_text)
        self.rgn6_horiz_row_1_grid.add_stretch()


        self.rgn6_horiz_row_2_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn6_horiz_row_2_grid.preferred_height = 2 * self.em

        self.show_shadow_casting_btn = gui.Button(f"ShowShadowCasted")
        self.show_shadow_casting_btn.toggleable = True
        self.show_shadow_casting_btn.set_on_clicked(self._on_show_shadow_casting_btn_clicked)

        self.finalize_shadow_casting_btn = gui.Button(f"FinalizeShadowCasting")
        self.finalize_shadow_casting_btn.set_on_clicked(self._on_finalize_shadow_casting_btn_clicked)

        self.rgn6_horiz_row_2_grid.add_stretch()
        self.rgn6_horiz_row_2_grid.add_child(self.show_shadow_casting_btn)
        self.rgn6_horiz_row_2_grid.add_stretch()
        self.rgn6_horiz_row_2_grid.add_child(self.finalize_shadow_casting_btn)
        self.rgn6_horiz_row_2_grid.add_stretch()


        self.rgn6_shadow_casting_layout.add_child(self.rgn6_horiz_row_0_grid)
        # self.rgn6_shadow_casting_layout.add_child(self.rgn6_horiz_row_1_grid)
        # self.rgn6_shadow_casting_layout.add_child(self.rgn6_horiz_row_2_grid)


        self.main_layout.add_child(self.rgn6_shadow_casting_layout)


        # endregion 6


        # region 7 : Show/ Hide Visualizations (DONOT Reset any Variables, just remove from the visualization geometry)

        self.rgn7_show_hide_layout = gui.CollapsableVert("Show/Hide Visualizations", spacing_betn_items_in_region, margins_for_region)
        self.rgn7_show_hide_layout.set_is_open(False)
        # self.rgn7_horiz_row_1_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        # self.rgn7_horiz_row_1_grid.preferred_height = 2 * self.em

        
        self.rgn7_vert_grid_col_1 = gui.Vert(spacing=0.2*self.em, margins=gui.Margins(0.4*self.em, 0.4*self.em, 0.4*self.em, 0.4*self.em))
        self.rgn7_show_source_pcd_chk_box = gui.Checkbox(f"Source Cloud")
        self.rgn7_show_source_pcd_chk_box.set_on_checked(self._on_rgn7_show_source_pcd_chk_box_clicked)
        self.rgn7_show_target_pcd_chk_box = gui.Checkbox(f"Target Cloud")
        self.rgn7_show_target_pcd_chk_box.set_on_checked(self._on_rgn7_show_target_pcd_chk_box_clicked)

        # self.rgn7_horiz_row_1_grid.add_stretch()
        # self.rgn7_horiz_row_1_grid.add_child(self.rgn7_show_source_pcd_chk_box)
        # self.rgn7_horiz_row_1_grid.add_stretch()
        # self.rgn7_horiz_row_1_grid.add_child(self.rgn7_show_target_pcd_chk_box)
        # self.rgn7_horiz_row_1_grid.add_stretch()


        # self.rgn7_horiz_row_2_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        # self.rgn7_horiz_row_2_grid.preferred_height = 2 * self.em
        self.rgn7_show_recostructed_surface_chk_box = gui.Checkbox(f"Reconstruced Surface")
        self.rgn7_show_recostructed_surface_chk_box.set_on_checked(self._on_rgn7_show_recostructed_surface_chk_box_clicked)
        self.rgn7_show_reconst_density_mesh_chk_box = gui.Checkbox(f"Reconstructed Density Mesh")
        self.rgn7_show_reconst_density_mesh_chk_box.set_on_checked(self._on_rgn7_show_reconst_density_mesh_chk_box_clicked)

        # self.rgn7_horiz_row_2_grid.add_stretch()
        # self.rgn7_horiz_row_2_grid.add_child(self.rgn7_show_recostructed_surface_chk_box)
        # self.rgn7_horiz_row_2_grid.add_stretch()
        # self.rgn7_horiz_row_2_grid.add_child(self.rgn7_show_reconst_density_mesh_chk_box)
        # self.rgn7_horiz_row_2_grid.add_stretch()

        self.rgn7_vert_grid_col_1.add_child(self.rgn7_show_source_pcd_chk_box)
        self.rgn7_vert_grid_col_1.add_child(self.rgn7_show_target_pcd_chk_box)
        self.rgn7_vert_grid_col_1.add_child(self.rgn7_show_recostructed_surface_chk_box)
        self.rgn7_vert_grid_col_1.add_child(self.rgn7_show_reconst_density_mesh_chk_box)


        # self.rgn7_horiz_row_3_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        # self.rgn7_horiz_row_3_grid.preferred_height = 2 * self.em

        # self.rgn7_vert_grid_col_2 = gui.Vert(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))


        self.rgn7_show_filtered_density_mesh_chk_box = gui.Checkbox(f"Filtered Density Mesh")
        self.rgn7_show_filtered_density_mesh_chk_box.set_on_checked(self._on_rgn7_show_filtered_density_mesh_chk_box_clicked)
        self.rgn7_show_directed_rays_chk_box = gui.Checkbox(f"Rays")
        self.rgn7_show_directed_rays_chk_box.set_on_checked(self._on_rgn7_show_directed_rays_chk_box_clicked)

        # self.rgn7_horiz_row_3_grid.add_stretch()
        # self.rgn7_horiz_row_3_grid.add_child(self.rgn7_show_filtered_density_mesh_chk_box)
        # self.rgn7_horiz_row_3_grid.add_stretch()
        # self.rgn7_horiz_row_3_grid.add_child(self.rgn7_show_directed_rays_chk_box)
        # self.rgn7_horiz_row_3_grid.add_stretch()
        

        # self.rgn7_horiz_row_4_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        # self.rgn7_horiz_row_4_grid.preferred_height = 2 * self.em
        self.rgn7_show_raycasted_source_pcd_chk_box = gui.Checkbox(f"RayCasted Point Cloud")
        self.rgn7_show_raycasted_source_pcd_chk_box.set_on_checked(self._on_rgn7_show_raycasted_source_pcd_chk_box_clicked)
        self.rgn7_show_casted_shadow_chk_box = gui.Checkbox(f"Shadow Casted by HPR")
        self.rgn7_show_casted_shadow_chk_box.set_on_checked(self._on_rgn7_show_casted_shadow_chk_box_clicked)
        self.rgn7_show_correct_casted_shadow_chk_box = gui.Checkbox(f"Shadow Casted by RayCasting")
        self.rgn7_show_correct_casted_shadow_chk_box.set_on_checked(self._on_rgn7_show_correct_casted_shadow_chk_box_clicked)

        # self.rgn7_horiz_row_4_grid.add_stretch()
        # self.rgn7_horiz_row_4_grid.add_child(self.rgn7_show_raycasted_source_pcd_chk_box)
        # self.rgn7_horiz_row_4_grid.add_stretch()
        # self.rgn7_horiz_row_4_grid.add_child(self.rgn7_show_casted_shadow_chk_box)
        # self.rgn7_horiz_row_4_grid.add_stretch()

        self.rgn7_vert_grid_col_1.add_child(self.rgn7_show_filtered_density_mesh_chk_box)
        self.rgn7_vert_grid_col_1.add_child(self.rgn7_show_directed_rays_chk_box)
        self.rgn7_vert_grid_col_1.add_child(self.rgn7_show_raycasted_source_pcd_chk_box)
        # self.rgn7_vert_grid_col_1.add_child(self.rgn7_show_casted_shadow_chk_box) # Update for GUI 2.0
        self.rgn7_vert_grid_col_1.add_child(self.rgn7_show_correct_casted_shadow_chk_box)


        # self.rgn7_show_hide_layout.add_child(self.rgn7_horiz_row_1_grid)
        # self.rgn7_show_hide_layout.add_child(self.rgn7_horiz_row_2_grid)
        # self.rgn7_show_hide_layout.add_child(self.rgn7_horiz_row_3_grid)
        # self.rgn7_show_hide_layout.add_child(self.rgn7_horiz_row_4_grid)

        self.rgn7_show_hide_layout.add_child(self.rgn7_vert_grid_col_1)
        # self.rgn7_show_hide_layout.add_child(self.rgn7_vert_grid_col_2)

        self.main_layout.add_child(self.rgn7_show_hide_layout)

        # endregion 7
        
        # region 8 : Save Final Merged Point Cloud and Reset All Variables

        self.rgn8_save_final_merged_pcd_layout = gui.CollapsableVert("Save Final Merged Point Cloud", spacing_betn_items_in_region, gui.Margins(1*self.em, 0.3*self.em, 1*self.em, 2*self.em))
        self.rgn8_save_final_merged_pcd_layout.set_is_open(False)
        self.rgn8_horiz_row_1_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn8_horiz_row_1_grid.preferred_height = 2 * self.em
        self.rgn8_save_final_merged_pcd_label = gui.Label("Filename : ")
        self.rgn8_save_final_merged_pcd_text = gui.TextEdit()
        self.rgn8_save_final_merged_pcd_text.text_value = "final_merged_pcd.ply"
        
        self.rgn8_horiz_row_1_grid.add_stretch()
        self.rgn8_horiz_row_1_grid.add_child(self.rgn8_save_final_merged_pcd_label)
        self.rgn8_horiz_row_1_grid.add_child(self.rgn8_save_final_merged_pcd_text)
        self.rgn8_horiz_row_1_grid.add_stretch()


        self.rgn8_horiz_row_2_grid = gui.Horiz(spacing=0.05*self.em, margins=gui.Margins(0.3*self.em, 0.3*self.em, 0.3*self.em, 0.3*self.em))
        self.rgn8_horiz_row_2_grid.preferred_height = 2 * self.em
        self.rgn8_save_final_merged_pcd_btn = gui.Button(f"Save Final Merged PCD")
        self.rgn8_save_final_merged_pcd_btn.set_on_clicked(self._on_rgn8_save_final_merged_pcd_btn_clicked)
        self.rgn8_reset_all_variables_btn = gui.Button(f"Reset All")
        self.rgn8_reset_all_variables_btn.set_on_clicked(self._on_rgn8_reset_all_variables_btn_clicked)

        self.rgn8_horiz_row_2_grid.add_stretch()
        self.rgn8_horiz_row_2_grid.add_child(self.rgn8_save_final_merged_pcd_btn)
        self.rgn8_horiz_row_2_grid.add_stretch()
        self.rgn8_horiz_row_2_grid.add_child(self.rgn8_reset_all_variables_btn)
        self.rgn8_horiz_row_2_grid.add_stretch()


        self.rgn8_save_final_merged_pcd_layout.add_child(self.rgn8_horiz_row_1_grid)
        self.rgn8_save_final_merged_pcd_layout.add_child(self.rgn8_horiz_row_2_grid)

        self.main_layout.add_child(self.rgn8_save_final_merged_pcd_layout)

        self.main_layout.add_child(gui.Label(" "))

        # endregion 8



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
    

    def update_show_hide_checkboxes(self):
        print("update_show_hide_checkboxes")
        if self.widget3d.scene.scene.has_geometry("source_cloud"):
            if self.widget3d.scene.scene.geometry_is_visible("source_cloud"):
                self.rgn7_show_source_pcd_chk_box.checked = True
            else:
                self.rgn7_show_source_pcd_chk_box.checked = False
        else:
            self.rgn7_show_source_pcd_chk_box.checked = False
        if self.widget3d.scene.scene.has_geometry("target_cloud"):
            if self.widget3d.scene.scene.geometry_is_visible("target_cloud"):
                self.rgn7_show_target_pcd_chk_box.checked = True
            else:
                self.rgn7_show_target_pcd_chk_box.checked = False
        else:
            self.rgn7_show_target_pcd_chk_box.checked = False
        if self.widget3d.scene.scene.has_geometry("reconstructed_source_mesh"):
            if self.widget3d.scene.scene.geometry_is_visible("reconstructed_source_mesh"):
                self.rgn7_show_recostructed_surface_chk_box.checked = True
            else:
                self.rgn7_show_recostructed_surface_chk_box.checked = False
        else:
            self.rgn7_show_recostructed_surface_chk_box.checked = False
        if self.widget3d.scene.scene.has_geometry("reconstructed_source_mesh_densities_with_color"):
            if self.widget3d.scene.scene.geometry_is_visible("reconstructed_source_mesh_densities_with_color"):
                self.rgn7_show_reconst_density_mesh_chk_box.checked = True
            else:
                self.rgn7_show_reconst_density_mesh_chk_box.checked = False
        else:
            self.rgn7_show_reconst_density_mesh_chk_box.checked = False
        if self.widget3d.scene.scene.has_geometry("reconstructed_source_mesh_filtered_densities_mesh"):
            if self.widget3d.scene.scene.geometry_is_visible("reconstructed_source_mesh_filtered_densities_mesh"):
                self.rgn7_show_filtered_density_mesh_chk_box.checked = True
            else:
                self.rgn7_show_filtered_density_mesh_chk_box.checked = False
        else:
            self.rgn7_show_filtered_density_mesh_chk_box.checked = False
        if self.widget3d.scene.scene.has_geometry("directed_rays"):
            if self.widget3d.scene.scene.geometry_is_visible("directed_rays"):
                self.rgn7_show_directed_rays_chk_box.checked = True
            else:
                self.rgn7_show_directed_rays_chk_box.checked = False
        else:
            self.rgn7_show_directed_rays_chk_box.checked = False
        if self.widget3d.scene.scene.has_geometry("raycasted_source_cloud"):
            if self.widget3d.scene.scene.geometry_is_visible("raycasted_source_cloud"):
                self.rgn7_show_raycasted_source_pcd_chk_box.checked = True
            else:
                self.rgn7_show_raycasted_source_pcd_chk_box.checked = False
        else:
            self.rgn7_show_raycasted_source_pcd_chk_box.checked = False
        if self.widget3d.scene.scene.has_geometry("final_merged_cloud_after_shadow_cast"):
            print("final_merged_cloud_after_shadow_cast exists")
            if self.widget3d.scene.scene.geometry_is_visible("final_merged_cloud_after_shadow_cast"):
                self.rgn7_show_casted_shadow_chk_box.checked = True
            else:
                self.rgn7_show_casted_shadow_chk_box.checked = False
        else:
            self.rgn7_show_casted_shadow_chk_box.checked = False
        if self.widget3d.scene.scene.has_geometry("shadow_cast_using_raycast_method"):
            print("shadow_cast_using_raycast_method exists")
            if self.widget3d.scene.scene.geometry_is_visible("shadow_cast_using_raycast_method"):
                self.rgn7_show_correct_casted_shadow_chk_box.checked = True
            else:
                self.rgn7_show_correct_casted_shadow_chk_box.checked = False
        else:
            self.rgn7_show_correct_casted_shadow_chk_box.checked = False


            

        
        
        

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

    def _on_source_scene_pcd_load_btn_clicked(self):
        print("Source Scene PCD Load Button clicked")
        if self.rgn1_source_scene_pcd_text.text_value == "":
            print("Source Scene PCD Text is empty")
            return
        if self.rgn1_source_scene_pcd_text.text_value[-4:] != ".csv":
            print("Source Scene PCD file is not a .csv file. Only CSV file with four columns (x, y, z, label) is supported")
            return
        if not os.path.exists(self.rgn1_source_scene_pcd_text.text_value):
            print("Source Scene PCD file doesnot exist")
            return
        try:
            df_source_scene = pd.read_csv(self.rgn1_source_scene_pcd_text.text_value, sep=",", names=["x", "y", "z", "label"], skiprows=1)
        except Exception as e:
            print("Error in reading the Source Scene PCD file : ", e)
            print("Source Scene PCD file should be a CSV file with four columns (x, y, z, label)")
            return
        self.source_scene_cloud = o3d.geometry.PointCloud()
        ar = df_source_scene[["x", "y", "z"]].to_numpy()
        print(ar)
        print(ar.shape)
        self.source_scene_cloud.points = o3d.utility.Vector3dVector(df_source_scene[["x", "y", "z"]].to_numpy())
        num_points = len(self.source_scene_cloud.points)
        self.source_scene_labels = df_source_scene["label"].to_numpy()
        colors = np.zeros((num_points, 3))
        # make all points green
        colors[:, 1] = 1.0
        # Color the points whose labels are not 0 as black. 0 labels for stationery object achieved from carla
        self.source_scene_object_of_interest_indices = np.where(self.source_scene_labels != 0)[0] # indices of the points whose labels are not 0, i.e. person or other non-stationary objects
        colors[self.source_scene_object_of_interest_indices] = [0, 0, 0]
        self.source_scene_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.widget3d.scene.scene.add_geometry("source_scene_cloud", self.source_scene_cloud, self.mat)
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

    def _on_source_scene_pcd_remove_btn_clicked(self):
        print("Source Scene PCD Remove Button clicked")
        self.source_scene_cloud = None
        self.widget3d.scene.scene.remove_geometry("source_scene_cloud")
        # reset indices
        self.source_scene_object_of_interest_indices = None
        self.source_scene_labels = None
        self.selected_pcd_indices = None
        self.selected_pcd_roi_boundary_indices = []
        self.selected_pcd_indices_with_obj_indices = None
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

    def _on_rgn1_use_ml_model_to_extract_src_pcd_chk_box_checked(self, checked):
        print("Use ML Model to Extract Source PCD Chk Box checked : ", checked)
        if checked:
            self.rgn1_horiz_row_1__subrow_3_r0a_grid.enabled = True
            self.rgn1_ml_model_to_extract_src_pcd_text.enabled = True
        else:
            self.rgn1_horiz_row_1__subrow_3_r0a_grid.enabled = False
            self.rgn1_ml_model_to_extract_src_pcd_text.enabled = False
    
    def _on_rgn1_use_labels_to_extract_src_pcd_chk_box_checked(self, checked):
        print("Use Labels to Extract Source PCD Chk Box checked : ", checked)
        if checked:
            self.rgn1_horiz_row_1__subrow_3_r1a_grid.enabled = True
            self.rgn1_labels_to_extract_src_pcd_text.enabled = True

        else:
            self.rgn1_horiz_row_1__subrow_3_r1a_grid.enabled = False
            self.rgn1_labels_to_extract_src_pcd_text.enabled = False


    def _on_rgn1_use_geometric_features_to_extract_src_pcd_chk_box_checked(self, checked):
        print("Use Geometric Features to Extract Source PCD Chk Box checked : ", checked)
        if checked:
            self.rgn1_horiz_row_1__subrow_3_r2a_grid.enabled = True
            self.rgn1_horiz_row_1__subrow_3_r2aa_grid.enabled = True
            self.rgn1_horiz_row_1__subrow_3_r2ab_grid.enabled = True
            self.rgn1_horiz_row_1__subrow_3_r2ac_grid.enabled = True
            self.rgn1_horiz_row_1__subrow_3_r2ad_grid.enabled = True
            self.rgn1_surface_variation_chk_box.enabled = True
            self.rgn1_surface_variation_threshold_text.enabled = True
            self.rgn1_planarity_chk_box.enabled = True
            self.rgn1_planarity_threshold_text.enabled = True
            self.rgn1_linearity_chk_box.enabled = True
            self.rgn1_linearity_threshold_text.enabled = True
            self.rgn1_z_value_chk_box.enabled = True
            self.rgn1_z_value_threshold_text.enabled = True

        else:
            self.rgn1_horiz_row_1__subrow_3_r2a_grid.enabled = False
            self.rgn1_horiz_row_1__subrow_3_r2aa_grid.enabled = False
            self.rgn1_horiz_row_1__subrow_3_r2ab_grid.enabled = False
            self.rgn1_horiz_row_1__subrow_3_r2ac_grid.enabled = False
            self.rgn1_horiz_row_1__subrow_3_r2ad_grid.enabled = False
            self.rgn1_surface_variation_chk_box.enabled = False
            self.rgn1_surface_variation_threshold_text.enabled = False
            self.rgn1_planarity_chk_box.enabled = False
            self.rgn1_planarity_threshold_text.enabled = False
            self.rgn1_linearity_chk_box.enabled = False
            self.rgn1_linearity_threshold_text.enabled = False
            self.rgn1_z_value_chk_box.enabled = False
            self.rgn1_z_value_threshold_text.enabled = False





    def _on_rgn1_surface_variation_chk_box_checked(self, checked):
        print("Surface Variation Chk Box checked : ", checked)
    
    def _on_rgn1_planarity_chk_box_checked(self, checked):
        print("Planarity Chk Box checked : ", checked)

    def _on_rgn1_linearity_chk_box_checked(self, checked): 
        print("Linearity Chk Box checked : ", checked)
    
    def _on_rgn1_z_value_chk_box_checked(self, checked):
        print("Z Value Chk Box checked : ", checked)

    def _on_extract_src_pcd_btn_clicked(self):
        print("Extract Source PCD Button clicked")
        if self.source_scene_cloud is None:
            self.rgn1_extract_src_pcd_btn.is_on = False
            self.rgn1_extract_src_pcd_btn.text = "ExtractSourcePCD"
            print("Source Scene PCD is not loaded")
            return
        # if not self.rgn1_use_labels_to_extract_src_pcd_chk_box.checked:
            
        #     print("Please check the checkbox of selecting by labels")
        #     print("For now we only check with labels to extract source pcd. ")
        #     return
        if self.rgn1_extract_src_pcd_btn.is_on:
            if  self.selected_pcd_indices_with_obj_indices is None or len(self.selected_pcd_indices_with_obj_indices) == 0:
                    print("No points are selected to extract object.")
                    self.rgn1_extract_src_pcd_btn.is_on = False
                    self.rgn1_extract_src_pcd_btn.text = "ExtractSourcePCD"
                    return
            self.rgn1_extract_src_pcd_btn.text = "RemoveExtracedSource"

            if self.rgn1_use_ml_model_to_extract_src_pcd_chk_box.checked:
                print("Using ML Model to Extract Source PCD")
                model_filename = self.rgn1_ml_model_to_extract_src_pcd_text.text_value
                if model_filename == "":
                    print("Model filename is empty")
                    self.rgn1_extract_src_pcd_btn.is_on = False
                    self.rgn1_extract_src_pcd_btn.text = "ExtractSourcePCD"
                    return
                if not os.path.exists(model_filename):
                    print("Model file doesnot exist")
                    self.rgn1_extract_src_pcd_btn.is_on = False
                    self.rgn1_extract_src_pcd_btn.text = "ExtractSourcePCD"
                    return
                if model_filename[-4:] != ".pkl":
                    print("Model file is not a .pkl file")
                    self.rgn1_extract_src_pcd_btn.is_on = False
                    self.rgn1_extract_src_pcd_btn.text = "ExtractSourcePCD"
                    return
                # Calculating feature of ROI and Predicting using the pre-trained model
                roi_source_cloud = self.source_scene_cloud.select_by_index(self.selected_pcd_indices_with_obj_indices)

                knn_number = int(self.rgn1_ml_model_to_extract_src_pcd_text.text_value.split("_")[-1].split(".")[0])
                print("Nearest Neighbour Number : ", knn_number)
                search_tree = o3d.geometry.KDTreeSearchParamKNN(knn_number)
                pcd_obj = PointCloudAnalysis(roi_source_cloud, is_point_type_open3d=True, search_tree=search_tree)
                normalized_surface_variation = pcd_obj.get_normalized_surface_variation()
                normalized_planarity = pcd_obj.get_normalized_planarity()
                normalized_linearity = pcd_obj.get_normalized_linearity()
                df_roi_source_scene_features = pd.DataFrame({"surface_variation": normalized_surface_variation, "planarity": normalized_planarity, "linearity": normalized_linearity})
                # print(df_roi_source_scene_features.head())
                # print(df_roi_source_scene_features.describe())

                # Loading the pre-trained model
                model_filename = self.rgn1_ml_model_to_extract_src_pcd_text.text_value
                with open(model_filename, "rb") as f:
                    classifier_model = pickle.load(f)
                
                # Predicting the labels
                X_features = df_roi_source_scene_features[["surface_variation", "planarity", "linearity"]]
                predicted_labels = classifier_model.predict(X_features)

                # Filtering the ROI based on the predicted labels
                filtered_roi_indices = np.where(predicted_labels == 1)[0]
                self.source_cloud = roi_source_cloud.select_by_index(filtered_roi_indices)
                self.source_cloud.paint_uniform_color([0, 0, 0])
                roi_ground_indices = np.where(predicted_labels == 0)[0]
                source_scene_roi_ground_indices = self.selected_pcd_indices_with_obj_indices[roi_ground_indices]
                self.selected_pcd_indices = source_scene_roi_ground_indices
                print("===============================================================================")
                print("Length of selected pcd indices : ", len(self.selected_pcd_indices))
                # print("Length of roi_ground_indices : ", len(roi_ground_indices))
                print("Length of filtered_roi_indices : ", len(filtered_roi_indices))
                print("Length of total indices : ", len(self.selected_pcd_indices_with_obj_indices))
                print("===============================================================================")

                # ---------------------------------------------------------------------------------------------
                # roi_selected_indices = df_roi_source_scene_features[df_roi_source_scene_features["select_row"] == 1]["index_array"].to_numpy()
                # source_scene_cloud_indices = self.selected_pcd_indices_with_obj_indices[roi_selected_indices]
                # # print("=====================================")
                # # print(df_roi_source_scene_features[df_roi_source_scene_features["select_row"] == 0])
                # # print("=====================================")
                # roi_ground_indices = df_roi_source_scene_features[df_roi_source_scene_features["select_row"] == 0]["index_array"].to_numpy()
                # source_scene_roi_ground_indices = self.selected_pcd_indices_with_obj_indices[roi_ground_indices]
                # # self.selected_pcd_indices = df_roi_source_scene_features[df_roi_source_scene_features["select_row"] == 0]["index_array"].to_numpy() # ROI selected indices of ground
                # self.source_cloud = self.source_scene_cloud.select_by_index(source_scene_cloud_indices)
                # self.source_cloud.paint_uniform_color([0, 0, 0])
                # self.selected_pcd_indices = source_scene_roi_ground_indices
                # self.selected_pcd_indices_with_obj_indices = source_scene_roi_ground_indices
                # ---------------------------------------------------------------------------------------------

                

            elif self.rgn1_use_labels_to_extract_src_pcd_chk_box.checked:
                # if self.rgn1_extract_src_pcd_btn.is_on:
                print("Extract Source PCD Button is already on")
                # self.rgn1_extract_src_pcd_btn.text = "RemoveExtracedSource"
                indx_to_filter = int(self.rgn1_labels_to_extract_src_pcd_text.text_value)
                labels_roi_array = self.source_scene_labels[self.selected_pcd_indices_with_obj_indices]

                filtered_roi_indices = np.where(labels_roi_array == indx_to_filter)[0]

                roi_source_cloud = self.source_scene_cloud.select_by_index(self.selected_pcd_indices_with_obj_indices)
                self.source_cloud = roi_source_cloud.select_by_index(filtered_roi_indices)
                self.source_cloud.paint_uniform_color([0, 0, 0])
                # source_cloud_indices = np.where((np.asarray(roi_source_cloud.colors) == self.object_of_interest_color).all(axis=1))[0]
                # self.source_cloud = roi_source_cloud.select_by_index(source_cloud_indices)
                #     if self.widget3d.scene.scene.has_geometry("source_scene_cloud"):
                #         print("Updating the geometry")
                #         self.widget3d.scene.scene.show_geometry("source_scene_cloud", show=False)
                #     self.widget3d.scene.scene.add_geometry("source_cloud", self.source_cloud, self.mat)
                #     self.widget3d.force_redraw()
                # else:
                    # self.rgn1_extract_src_pcd_btn.text = "ExtractSourcePCD"
                    # self.source_cloud = None
                    # self.widget3d.scene.scene.remove_geometry("source_cloud")
                    # self.widget3d.scene.scene.show_geometry("source_scene_cloud", show=True)
                    # self.widget3d.force_redraw()
            elif self.rgn1_use_geometric_features_to_extract_src_pcd_chk_box.checked:
                print("Using Geometric Features to Extract Source PCD")
                # self.rgn1_extract_src_pcd_btn.text = "RemoveExtracedSource"
                # print(dir(self.source_scene_cloud))
                if len(self.selected_pcd_indices_with_obj_indices) == 0 or self.selected_pcd_indices_with_obj_indices is None:
                    print("No points are selected to calculate the geometric features.")
                    return
                roi_source_scene_cloud = self.source_scene_cloud.select_by_index(self.selected_pcd_indices_with_obj_indices)
                search_tree = o3d.geometry.KDTreeSearchParamKNN(5)
                pcd_obj = PointCloudAnalysis(roi_source_scene_cloud, is_point_type_open3d=True, search_tree=search_tree)
                normalized_surface_variation = pcd_obj.get_normalized_surface_variation()
                normalized_planarity = pcd_obj.get_normalized_planarity()
                normalized_linearity = pcd_obj.get_normalized_linearity()
                z_values = np.asarray(roi_source_scene_cloud.points)[:, 2]
                index_array = np.arange(len(z_values))
                select_row = np.zeros(len(z_values)) # Initially all rows are not selected. All is zeros. The row whose value is one is selected
                df_roi_source_scene_features = pd.DataFrame({"surface_variation": normalized_surface_variation, "planarity": normalized_planarity, "linearity": normalized_linearity, "z_values": z_values, "index_array": index_array, "select_row": select_row})
                # print(df_roi_source_scene_features.head())
                # print(df_roi_source_scene_features.describe())
                # print(dir(roi_source_scene_cloud))
                if self.rgn1_surface_variation_chk_box.checked:
                    print("Using Surface Variation for filtering PCD")
                    # for NN = 10, 0.0055 seems to be a better minimum threshold to seperate the person from the selected roi
                    min_surface_variation_threshold = float(self.rgn1_surface_variation_threshold_text.text_value)
                    # df_roi_source_scene_features = df_roi_source_scene_features[df_roi_source_scene_features["surface_variation"] >= min_surface_variation_threshold]
                    df_roi_source_scene_features["select_row"] = np.where(df_roi_source_scene_features["surface_variation"] >= min_surface_variation_threshold, 1, 0)
                    # print(df_roi_source_scene_features["select_row"].unique())
                    
                if self.rgn1_planarity_chk_box.checked:
                    print("Using Planarity to Extract Source PCD")
                    max_planarity_threshold = float(self.rgn1_planarity_threshold_text.text_value)
                    # df_roi_source_scene_features = df_roi_source_scene_features[df_roi_source_scene_features["planarity"] <= max_planarity_threshold]
                    df_roi_source_scene_features["select_row"] = np.where(df_roi_source_scene_features["planarity"] <= max_planarity_threshold, 1, 0)
                
                if self.rgn1_linearity_chk_box.checked:
                    print("Using Linearity to Extract Source PCD")
                    max_linearity_threshold = float(self.rgn1_linearity_threshold_text.text_value)
                    # df_roi_source_scene_features = df_roi_source_scene_features[df_roi_source_scene_features["linearity"] <= max_linearity_threshold]
                    df_roi_source_scene_features["select_row"] = np.where(df_roi_source_scene_features["linearity"] <= max_linearity_threshold, 1, 0)
                
                if self.rgn1_z_value_chk_box.checked:
                    print("Using Z Value to Extract Source PCD")
                    min_z_value_threshold = float(self.rgn1_z_value_threshold_text.text_value)
                    print(min_z_value_threshold)
                    df_roi_source_scene_features["select_row"] = np.where(df_roi_source_scene_features["z_values"] >= min_z_value_threshold, 1, 0)
                    # df_roi_source_scene_features["select_row"] = 1

                # print("=====================================")
                # print(df_roi_source_scene_features.head())
                # df_roi_source_scene_features.to_csv("df_roi_source_scene_features.csv")
                roi_selected_indices = df_roi_source_scene_features[df_roi_source_scene_features["select_row"] == 1]["index_array"].to_numpy()
                source_scene_cloud_indices = self.selected_pcd_indices_with_obj_indices[roi_selected_indices]
                # print("=====================================")
                # print(df_roi_source_scene_features[df_roi_source_scene_features["select_row"] == 0])
                # print("=====================================")
                roi_ground_indices = df_roi_source_scene_features[df_roi_source_scene_features["select_row"] == 0]["index_array"].to_numpy()
                source_scene_roi_ground_indices = self.selected_pcd_indices_with_obj_indices[roi_ground_indices]
                # self.selected_pcd_indices = df_roi_source_scene_features[df_roi_source_scene_features["select_row"] == 0]["index_array"].to_numpy() # ROI selected indices of ground
                self.source_cloud = self.source_scene_cloud.select_by_index(source_scene_cloud_indices)
                self.source_cloud.paint_uniform_color([0, 0, 0])
                self.selected_pcd_indices = source_scene_roi_ground_indices
                self.selected_pcd_indices_with_obj_indices = source_scene_roi_ground_indices
                # print(df_roi_source_scene_features.describe())
                # print(roi_selected_indices)
                # print(len(roi_selected_indices))
                if len(roi_selected_indices) == 0:
                    print("No points are selected. Please check the thresholds")
                    return
                # print(self.source_cloud)
                # print("=====================================")
                


            else:
                self.rgn1_extract_src_pcd_btn.is_on = False
                self.rgn1_extract_src_pcd_btn.text = "ExtractSourcePCD"
                print("Please check the checkbox of selecting by labels or geometric features")
                return
            
            # Saving the source cloud
            o3d.io.write_point_cloud("feb02/source_cloud.ply", self.source_cloud)
            
            if self.widget3d.scene.scene.has_geometry("source_scene_cloud") and self.source_cloud is not None:
                print("Updating the geometry")
                self.widget3d.scene.scene.show_geometry("source_scene_cloud", show=False)
                self.widget3d.scene.scene.add_geometry("source_cloud", self.source_cloud, self.mat)
            
        else:
            print("Extract Source PCD Button is OFF")
            self.rgn1_extract_src_pcd_btn.text = "ExtractSourcePCD"
            self.source_cloud = None
            self._on_roi_select_rect_regn_btn_clicked()
            self.widget3d.scene.scene.remove_geometry("source_cloud")
            self.widget3d.scene.scene.show_geometry("source_scene_cloud", show=True)
      
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

    def _on_finalize_extracted_src_pcd_btn_clicked(self):
        print("Finalize Extracted Source PCD Button clicked")
        if self.source_scene_cloud is None:
            print("Source Scene PCD is not loaded")
            return
        if self.source_cloud is None:
            print("Source PCD is not loaded")
            return
        print("Finalize Extracted Source PCD Button clicked")
        print("This is irreversible process. If you need to  extract source pcd again, you need to reload the source scene pcd")
        self._on_calculate_centroid_of_reference_roi_btn_clicked() # calculate the centroid of the reference roi when we finalize the transformation of the source_cloud
        self.reference_of_source_scene_cloud = copy.deepcopy(self.source_scene_cloud)
        self.source_scene_cloud = None
        self.selected_pcd_roi_boundary_indices = []
        self.selected_pcd_indices_with_obj_indices = None
        self.selected_pcd_indices = None
        self.rgn1_extract_src_pcd_btn.is_on = False
        self.rgn1_extract_src_pcd_btn.text = "ExtractSourcePCD"
        self.widget3d.scene.scene.remove_geometry("source_scene_cloud")
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()
        



    def _on_source_pcd_load_btn_clicked(self):
        print("Source PCD Load Button clicked")        
        self.source_cloud = o3d.io.read_point_cloud(self.source_pcd_text.text_value)
        num_points = len(self.source_cloud.points)
        colors = np.zeros((num_points, 3))
        self.source_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.widget3d.scene.scene.add_geometry("source_cloud", self.source_cloud, self.mat)
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

    
    def _on_source_pcd_remove_btn_clicked(self):
        if self.source_cloud is None:
            print("Source PCD is not loaded to remove")
            return
        print("Source PCD Remove Button clicked")
        self.source_cloud = None
        self.widget3d.scene.scene.remove_geometry("source_cloud")
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()


    def _on_target_pcd_load_btn_clicked(self):
        print("Target PCD Load Button clicked")
        self.target_cloud = o3d.io.read_point_cloud(self.target_pcd_text.text_value)
        num_points = len(self.target_cloud.points)
        colors = np.zeros((num_points, 3))
        colors[:,1] = 1 # set all points to green
        self.target_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.widget3d.scene.scene.add_geometry("target_cloud", self.target_cloud, self.mat)
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

    
    def _on_target_pcd_remove_btn_clicked(self):
        if self.target_cloud is None:
            print("Target PCD is not loaded to remove")
            return
        print("Target PCD Remove Button clicked")
        self.target_cloud = None
        self.widget3d.scene.scene.remove_geometry("target_cloud")
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

    def _on_roi_select_boundary_chk_box_clicked(self, checked):
        print("ROI Select Boundary Chk Box clicked : ", checked)
    
    
    def func_to_track_shadowcasting(self, reset_roi_clicked=False):
        #"used in the evaluation process by selecting the equivalent region of interest in the source scene cloud and shadowcasted target region"
        if not hasattr(self, "reference_of_source_scene_cloud"):
            return
        
        print("func_to_track_shadowcasting")
        
        ref_points = np.asarray(self.reference_of_source_scene_cloud.points)
        ref_colors = np.asarray(self.reference_of_source_scene_cloud.colors)
        
        ref_object_of_interest = np.where((ref_colors == [0, 0, 0]).all(axis=1))[0]
        
        ref_selected_indices = (ref_points[:, 0] >= self.min_xy[0]) & (ref_points[:, 0] <= self.max_xy[0]) & (ref_points[:, 1] >= self.min_xy[1]) & (ref_points[:, 1] <= self.max_xy[1])
        ref_colors[ref_selected_indices] = [1, 0, 0]
        ref_colors[ref_object_of_interest] = [0, 0, 0]

        roi_mask = np.where((ref_colors == [1, 0, 0]).all(axis=1))[0]

        roi_points = ref_points[roi_mask]
        roi_colors = ref_colors[roi_mask]

        roi_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(roi_points))
        roi_cloud.colors = o3d.utility.Vector3dVector(roi_colors)

        o3d.io.write_point_cloud("feb02/ground_truth_casted_shadow_on_roi.ply", roi_cloud)

        # Assign the new colors to the point cloud
        self.reference_of_source_scene_cloud.colors = o3d.utility.Vector3dVector(ref_colors)

        o3d.io.write_point_cloud("feb02/source_scene.ply", self.reference_of_source_scene_cloud)
        print("Saved new ref source scene with color")
        # returning color to original color i.e. green
        ref_colors[ref_selected_indices] = [0, 1, 0]  # Change to green
        ref_colors[ref_object_of_interest] = [0, 0, 0]
        self.reference_of_source_scene_cloud.colors = o3d.utility.Vector3dVector(ref_colors)
        
        # end of this part


    # @check_if_pcd_is_loaded
    def _on_roi_select_rect_regn_btn_clicked(self):  # Maybe later it could be dynamic for both clouds
        print("ROI Select Rectangular Region Button clicked")
        
        # Convert the point cloud to a numpy array
        if self.source_scene_cloud is not None:
            print("Operating on Source Scene Cloud")
            cloud_to_operate = self.source_scene_cloud
            geometry_name = "source_scene_cloud"
        elif self.target_cloud is not None:
            print("Operating on Target Cloud")
            cloud_to_operate = self.target_cloud
            geometry_name = "target_cloud"
        else:
            print("No source scene cloud or target cloud is loaded")
            return

        if len(self.selected_pcd_roi_boundary_indices) < 2:
            print("Select at least two points to form a rectangular region")
            return

        points = np.asarray(cloud_to_operate.points)
        colors = np.asarray(cloud_to_operate.colors)
        # Extract the red points
        # red_points = points[(colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)]
        selected_points = points[np.asarray(self.selected_pcd_roi_boundary_indices)]
        if len(selected_points) == 0:
            print("No red points found")
            return
        # Compute the 2D bounding box of the red points in the XY plane
        min_xy = np.min(selected_points[:, :2], axis=0)
        max_xy = np.max(selected_points[:, :2], axis=0)
        self.min_xy = min_xy
        self.max_xy = max_xy

        # Select all points indices that fall within this bounding box
        selected_indices = (points[:, 0] >= min_xy[0]) & (points[:, 0] <= max_xy[0]) & (points[:, 1] >= min_xy[1]) & (points[:, 1] <= max_xy[1])
        print(selected_indices)
        self.selected_pcd_indices_with_obj_indices = np.where(selected_indices)[0]
        if self.source_scene_cloud is not None:
            if self.source_scene_object_of_interest_indices is not None or len(self.source_scene_object_of_interest_indices) > 0: 
                selected_indices[self.source_scene_object_of_interest_indices] = False
        # selected_indices[self.source_scene_object_of_interest_indices] = False
        print("Selected Indices from ROI ============= : ", selected_indices)
        print("length of true values : ", len(np.where(selected_indices)[0]))
        self.selected_pcd_indices = selected_indices # used later for selection of a part of pcd for effective processing
        colors[selected_indices] = [1, 0, 0]  # Change to red
        cloud_to_operate.colors = o3d.utility.Vector3dVector(colors)
        if self.widget3d.scene.scene.has_geometry(geometry_name):
            print("Updating the geometry")
            self.widget3d.scene.scene.remove_geometry(geometry_name)
        self.widget3d.scene.scene.add_geometry(geometry_name, cloud_to_operate, self.mat)
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()
    
    # @check_if_pcd_is_loaded
    def _on_roi_reset_btn_clicked(self):
        print("ROI Reset Button clicked")
        if self.source_scene_cloud is not None:
            cloud_to_operate = self.source_scene_cloud
            geometry_name = "source_scene_cloud"

            # print(dir(self.source_scene_cloud.colors))
            # print(self.source_scene_cloud.colors)
        elif self.target_cloud is not None:
            cloud_to_operate = self.target_cloud
            geometry_name = "target_cloud"
        else:
            print("No source scene cloud or target cloud is loaded")
            return
        self.selected_pcd_indices = None
        self.selected_pcd_roi_boundary_indices = []
        self.widget3d.scene.scene.remove_geometry(geometry_name)
        # Reset the color of the target cloud to green
        num_points = len(cloud_to_operate.points)
        cloud_to_operate_colors = np.asarray(cloud_to_operate.colors)

        red_indices = np.all(cloud_to_operate_colors == [1, 0, 0], axis=1)
        cloud_to_operate_colors[red_indices] = [0, 1, 0]

        # colors = np.zeros((num_points, 3))
        # colors[:,1] = 1 # set all points to green
        cloud_to_operate.colors = o3d.utility.Vector3dVector(cloud_to_operate_colors)
        # Add the target cloud again
        self.widget3d.scene.scene.add_geometry(geometry_name, cloud_to_operate, self.mat)
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

    # @check_if_pcd_is_loaded
    def _calculate_centroid_of_roi(self, is_source_scene_cloud=False):
        print("Calculate Centroid of ROI")
        print("is_source_scene_cloud : ", is_source_scene_cloud)
        print(self.source_scene_cloud is None)
        print(self.target_cloud is None)
        # print(len(self.selected_pcd_indices))
        if is_source_scene_cloud:
            if self.source_scene_cloud is None:
                print("Source Scene Cloud is not loaded")
                return
            cloud_to_operate = self.source_scene_cloud
        else:
            if self.target_cloud is None:
                print("Target Cloud is not loaded")
                return
            cloud_to_operate = self.target_cloud
        if self.selected_pcd_indices is None:
            print("No ROI selected")
            return
        # print the points and selected_pcd_indices to check why the centroid is not correct/ NaN
        _centroid = np.asarray(cloud_to_operate.points)[self.selected_pcd_indices].mean(axis=0)
        return _centroid
    
    # @check_if_pcd_is_loaded
    def _on_calculate_centroid_of_reference_roi_btn_clicked(self):
        print("Calculate Centroid of Reference ROI Button clicked")
        self.centroid_of_reference_roi = self._calculate_centroid_of_roi(is_source_scene_cloud=True)
        print("Centroid of Reference ROI: ", self.centroid_of_reference_roi)

    # @check_if_pcd_is_loaded
    def _on_calculate_centroid_of_target_roi_btn_clicked(self):
        print("Calculate Centroid of Target ROI Button clicked")
        self.centroid_of_target_roi = self._calculate_centroid_of_roi(is_source_scene_cloud=False)
        print("Centroid of Target ROI: ", self.centroid_of_target_roi)

    @check_if_pcd_is_loaded
    def _on_transform_source_pcd_to_target_roi_clicked(self):
        print("Transform Source PCD to Target ROI Button clicked")
        self._on_calculate_centroid_of_target_roi_btn_clicked()
        if self.centroid_of_reference_roi is None or self.centroid_of_target_roi is None:
            self.transform_source_pcd_to_target_roi.is_on = False
            print("Centroids of ROIs are not calculated")
            return
        if self.transform_source_pcd_to_target_roi.is_on:
        # Translation part works fine
            self.transform_source_pcd_to_target_roi.text = "RemoveTransformedSource"
            translation = self.centroid_of_target_roi - self.centroid_of_reference_roi
            print("Translation: ", translation)
            translation_matrix = np.identity(4)
            translation_matrix[:3, 3] = translation
            self.source_cloud_before_transform = copy.deepcopy(self.source_cloud)
            self.source_cloud  = copy.deepcopy(self.source_cloud_before_transform).transform(translation_matrix)

            # Needs to figure out rotation part
            start_vector = np.array(self.source_cloud_before_transform.get_center())
            end_vector = np.array(self.source_cloud.get_center())
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
            self.source_cloud.rotate(rotation_matrix, center=np.asarray(self.source_cloud.get_center()))
            if self.widget3d.scene.scene.has_geometry("source_cloud"):
                print("Updating the geometry")
                self.widget3d.scene.scene.remove_geometry("source_cloud")
            self.widget3d.scene.scene.add_geometry("source_cloud", self.source_cloud, self.mat)
            self.widget3d.force_redraw()
        else:
            self.transform_source_pcd_to_target_roi.text = "TransformSource"
            self.centroid_of_target_roi = None
            self.selected_pcd_indices = None
            self.selected_pcd_roi_boundary_indices = []
            self.selected_pcd_indices_with_obj_indices = None
            self.source_cloud = copy.deepcopy(self.source_cloud_before_transform)
            if self.widget3d.scene.scene.has_geometry("source_cloud"):
                print("Updating the geometry")
                self.widget3d.scene.scene.remove_geometry("source_cloud")
            self._on_roi_reset_btn_clicked()
            self.widget3d.scene.scene.add_geometry("source_cloud", self.source_cloud, self.mat)
            self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

    @check_if_pcd_is_loaded
    def _on_finalize_transformed_source_pcd_to_target_roi_clicked(self):
        print("Finalize Transform Source PCD to Target ROI Button clicked")
        self.centroid_of_reference_roi = self._calculate_centroid_of_roi()
        self.transform_source_pcd_to_target_roi.is_on = False
        self.transform_source_pcd_to_target_roi.text = "TransformSource"
        self.centroid_of_target_roi = None
        self.selected_pcd_indices = None
        self.selected_pcd_roi_boundary_indices = []
        self.selected_pcd_indices_with_obj_indices = None
        self.source_cloud_before_transform = None
        self.source_cloud_transformed = None
        self.centroid_of_target_roi = None
        self._on_roi_reset_btn_clicked()
        self.update_show_hide_checkboxes()
        # self.widget3d.scene.scene.remove_geometry("source_cloud_transformed")


    
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
            source_cloud.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
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

                    if self.widget3d.scene.scene.has_geometry("raycasted_source_cloud"):
                        self.show_raycasted_pcd_btn.is_on = False
                        self.show_raycasted_pcd_btn.text = "ShowRayCastedPCD"
                        self.widget3d.scene.scene.remove_geometry("raycasted_source_cloud")
                        self.raycasted_source_cloud = None
                        print("Done removing raycasted source cloud")

        self.update_show_hide_checkboxes()
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
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()



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
            # vertices_to_remove = self.reconstructed_source_mesh_densities_array < np.quantile(self.reconstructed_source_mesh_densities_array, self.filter_density_slider.double_value)
            # self.reconstructed_source_mesh_filtered_densities_mesh.remove_vertices_by_mask(vertices_to_remove)
            # self.reconstructed_source_mesh_filtered_densities_mesh.compute_vertex_normals()
            
            # ===================================================================================================
            # Filtering the mesh based on the vertex values and removing traingles from triangle mesh whose all vertices do not contain original point of point cloud
            # ===================================================================================================
            
            original_points = np.asarray(self.source_cloud.points)

            # Get vertices and triangles
            vertices = np.asarray(self.reconstructed_source_mesh_filtered_densities_mesh.vertices)
            triangles = np.asarray(self.reconstructed_source_mesh_filtered_densities_mesh.triangles)

            # Mark triangles for removal
            count_list = []
            triangles_to_remove = []
            for i, triangle in enumerate(triangles):
                vertices_in_triangle = set(triangle)
                count = 0
                for vertex in vertices_in_triangle:
                    if any(np.isclose(vertices[vertex], original_points, atol=1e-1).all(axis=1)):
                        count += 1
                count_list.append(count)
                if count < 3:
                    triangles_to_remove.append(i)

            # Remove triangles that don't contain at least two original point cloud points
            self.reconstructed_source_mesh_filtered_densities_mesh.remove_triangles_by_index(triangles_to_remove)
            self.reconstructed_source_mesh_filtered_densities_mesh.remove_unreferenced_vertices()




            # ===================================================================================================
            # ===================================================================================================


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
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

    
    def _on_show_rays_btn_clicked(self):
        print("Show Rays Button clicked")
        if self.selected_pcd_indices is None:
            self.show_rays_btn.is_on = False
            self.show_rays_btn.text = "Show Rays"
            print("No points Indices selected for Rays")
            return
        if self.show_rays_btn.is_on:
            print("Show Rays Button is ON")
            self.show_rays_btn.text = "Remove Rays"
            rays_select_ratio = self.filter_rays_slider.double_value
            complete_rays = np.asarray(self.target_cloud.points)[self.selected_pcd_indices]
            random_rays_count = int(rays_select_ratio * complete_rays.shape[0])
            selected_rows_indices = np.random.choice(complete_rays.shape[0], size=random_rays_count, replace=False)
            # Extract the subset of the array based on the selected indices
            randomly_selected_rays = complete_rays[selected_rows_indices, :]
            origin = np.array([[0, 0, 0]])
            randomly_selected_rays_with_origin = np.concatenate((origin, randomly_selected_rays), axis=0)
            lines = [[0, i] for i in range(1, len(randomly_selected_rays_with_origin))]
            self.line_set = o3d.geometry.LineSet()
            self.line_set.points = o3d.utility.Vector3dVector(randomly_selected_rays_with_origin)
            self.line_set.lines = o3d.utility.Vector2iVector(lines)
            self.line_set.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 0] for i in range(len(lines))]))
            self.widget3d.scene.scene.add_geometry("directed_rays", self.line_set, self.mat)
        else:
            print("Show Rays Button is OFF")
            self.show_rays_btn.text = "Show Rays"
            self.widget3d.scene.scene.remove_geometry("directed_rays")
            self.line_set = None
        self.update_show_hide_checkboxes()
        self.widget3d.force_redraw()


    def _on_show_raycasted_pcd_btn_clicked(self):
        print("Show Raycasted PCD Button clicked")
        if self.reconstructed_source_mesh_filtered_densities_mesh is None or self.selected_pcd_indices is None:
            self.show_raycasted_pcd_btn.is_on = False
            self.show_raycasted_pcd_btn.text = "ShowRayCastedPCD"
            print("No filtered density mesh or No selected pcd region")
            return
        if self.show_raycasted_pcd_btn.is_on:
            print("Show Raycasted PCD Button is ON")
            self.show_raycasted_pcd_btn.text = "RemoveRayCastedPCD"
            complete_rays_direction = np.asarray(self.target_cloud.points)[self.selected_pcd_indices]
            rays_select_ratio = self.filter_rays_slider.double_value
            random_rays_count = int(rays_select_ratio * complete_rays_direction.shape[0])
            randomly_selected_raycasted_rays = complete_rays_direction[np.random.choice(complete_rays_direction.shape[0], size=random_rays_count, replace=False), :]
            ray_origin = np.zeros((len(randomly_selected_raycasted_rays), 3))
            rays_array = np.concatenate((ray_origin, randomly_selected_raycasted_rays), axis=1)

            # Create a Scene and add the triangle mesh
            mesh_new = o3d.t.geometry.TriangleMesh.from_legacy(self.reconstructed_source_mesh_filtered_densities_mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh_new)

            # Creating Rays for casting using the rays array created earlier
            rays = o3d.core.Tensor(rays_array,dtype=o3d.core.Dtype.Float32)
            # Casting rays on the scene
            ans = scene.cast_rays(rays)

            # ===================================================================================================
            # ===================================================================================================
            # for the calculation of shadow casting
            t_hits_all = ans["t_hit"].numpy()
            not_hits_indices = np.where(t_hits_all == np.inf)[0]
            # rays_not_hitting_the_surface = rays_array[not_hits_indices]
            points_not_hitting_the_surface = randomly_selected_raycasted_rays[not_hits_indices]
            pcd_in_roi_where_rays_not_intersect_with_surface = o3d.geometry.PointCloud()
            pcd_in_roi_where_rays_not_intersect_with_surface.points = o3d.utility.Vector3dVector(points_not_hitting_the_surface)
            pcd_in_roi_where_rays_not_intersect_with_surface.paint_uniform_color([1, 0, 0]) # red

            roi_int_indices = np.where(self.selected_pcd_indices)[0]
            pcd_of_target_cloud_without_roi = self.target_cloud.select_by_index(roi_int_indices, invert=True)
            pcd_of_target_cloud_without_roi.paint_uniform_color([0, 1, 0]) # green
            
            
            # pcd_target_ohne_roi = o3d.geometry.PointCloud()
            # mask = np.isin(np.asarray(self.target_cloud.points), self.selected_pcd_indices, invert=True)
            # pcd_target_ohne_roi.points = o3d.utility.Vector3dVector(np.asarray(self.target_cloud.points)[mask])
            self.shadow_casted_pcd_using_ray_cast_without_prototype = pcd_in_roi_where_rays_not_intersect_with_surface + pcd_of_target_cloud_without_roi
            # self.shadow_casted_pcd_using_ray_cast_without_prototype.paint_uniform_color([0, 1, 0]) # green

            if self.widget3d.scene.scene.has_geometry("shadow_cast_using_raycast_method"):
                self.widget3d.scene.scene.remove_geometry("shadow_cast_using_raycast_method")
            self.widget3d.scene.scene.add_geometry("shadow_cast_using_raycast_method", self.shadow_casted_pcd_using_ray_cast_without_prototype, self.mat)
            self.widget3d.scene.scene.show_geometry("shadow_cast_using_raycast_method", show=False)
            
            # ===================================================================================================

            # self.widget3d.scene.scene.add_geometry("shadow_casted_by_raycasting_process", shadow_casted_pcd, self.mat)
            print("length of target cloud : ", len(self.target_cloud.points))
            print("len of pcd indices : ", len(self.selected_pcd_indices))
            print("selected pcd indices : ", self.selected_pcd_indices)
            print("len of pcd of target cloud without roi : ", len(pcd_of_target_cloud_without_roi.points))

            o3d.io.write_point_cloud("feb02/shadow_casted_by_raycasting_process_without_prototype.ply", self.shadow_casted_pcd_using_ray_cast_without_prototype)
            o3d.io.write_point_cloud("feb02/remaining_region_after_roi.ply", pcd_of_target_cloud_without_roi)
            o3d.io.write_point_cloud("feb02/pcd_in_roi_where_rays_not_intersect_with_surface.ply", pcd_in_roi_where_rays_not_intersect_with_surface)
            # o3d.io.write_point_cloud("feb02/pcd_target_ohne_roi.ply", pcd_target_ohne_roi)

            # End of new way for shadow casting
            # ===================================================================================================
            # ===================================================================================================


            # ===================================================================================================
            # Use of Barycentric Coordinates to find the raycasted point cloud
            # ===================================================================================================

            # mesh_new is the collection of triangle mesh
            # index_of_intersected_triangles = np.where(ans["primitive_ids"].numpy() != scene.INVALID_ID)[0]
            # intersected_triangles = ans["primitive_ids"].numpy()[index_of_intersected_triangles]
            # vertex_indx_of_intersected_triangles = mesh_new.triangle.indices.numpy()[intersected_triangles]
            # vertices_of_intersected_triangles = mesh_new.vertex.positions.numpy()[vertex_indx_of_intersected_triangles]
            # barycentric_coordinates_of_intersected_triangles = ans["primitive_uvs"].numpy()[index_of_intersected_triangles]
            # barycentric_coordinates_of_intersected_triangles = np.concatenate([barycentric_coordinates_of_intersected_triangles, 1- barycentric_coordinates_of_intersected_triangles.sum(axis=1, keepdims=True)], axis=1)
            
            # raycaste_points_from_barycentric_coordinates = np.einsum('ijk,ij->ik',vertices_of_intersected_triangles, barycentric_coordinates_of_intersected_triangles)

            # A = vertices_of_intersected_triangles
            # B = barycentric_coordinates_of_intersected_triangles
            # P = np.zeros((len(intersected_triangles), 3))
            # for i in range(len(intersected_triangles)):
            #     for j in range(3):
            #         print(B[i, j], A[i, j])
            #         P[i] += B[i, j] * A[i, j]


            # o3d.io.write_point_cloud("feb02/raycasted_point_cloud_using_barycentric_coordinates.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raycaste_points_from_barycentric_coordinates)))
            # o3d.io.write_point_cloud("feb02/raycasted_point_cloud_using_barycentric_coordinates_matrix_mul.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)))


            # ===================================================================================================
            # ===================================================================================================



            # ===================================================================================================
            # Use of distance between the point to find the actual hit point
            # ===================================================================================================

            index_of_intersected_triangles = np.where(ans["primitive_ids"].numpy() != scene.INVALID_ID)[0]
            intersected_rays = rays_array[index_of_intersected_triangles]
            t_hit = ans["t_hit"].numpy()[index_of_intersected_triangles]

            df = pd.DataFrame(intersected_rays, columns=["x0", "y0", "z0", "x1", "y1", "z1" ])
            distance = ((df["x0"] - df["x1"])**2 + (df["y0"] - df["y1"])**2 + (df["z0"] - df["z1"])**2)**(1/2)
            df["distance"] = distance
            df["t_hit"] = t_hit
            df["x"] = df["x0"] + df["t_hit"] * (df["x1"] - df["x0"])
            df["y"] = df["y0"] + df["t_hit"] * (df["y1"] - df["y0"])
            df["z"] = df["z0"] + df["t_hit"] * (df["z1"] - df["z0"])
            raycasted_point_cloud_using_t_hit = df[["x", "y", "z"]].values
            o3d.io.write_point_cloud("feb02/raycasted_point_cloud_using_t_hit.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raycasted_point_cloud_using_t_hit)))



            # Using t_hit distance to find the raycasted point works perfectly!







            # ===================================================================================================
            # ===================================================================================================

            # ===================================================================================================
            # Use of Centroid method to find the coordinates of the intersected points
            # ===================================================================================================
            # for the calculation of raycasted point
            # triangle_ids = [i.numpy() for i in ans["primitive_ids"] if i != scene.INVALID_ID] # list of all the triangles through which rays intersected
            # triangle_indices = [i.numpy() for i in mesh_new.triangle.indices] # list of all the triangles in the mesh
            # vertex_list = [i.numpy() for i in mesh_new.vertex.positions]
            # all_intersected_points = np.zeros((len(triangle_ids), 3)) # Creating an empty array
            # for indx, triangle_id in enumerate(triangle_ids): # List of Triangles through which rays intersected
            #     triangle_vertex = triangle_indices[triangle_id] # Gives the list of vertex
            #     vertex0 = vertex_list[triangle_vertex[0]] # Gives the corresponding vertex value in x,y,z
            #     vertex1 = vertex_list[triangle_vertex[1]] # Gives the corresponding vertex value in x,y,z
            #     vertex2 = vertex_list[triangle_vertex[2]] # Gives the corresponding vertex value in x,y,z
            #     all_intersected_points[indx] = [(vertex0[i] + vertex1[i] + vertex2[i])/3 for i in range(3)] # Computing the centroid for now
            # print(all_intersected_points.shape)

            # ===================================================================================================
            # ===================================================================================================



            self.raycasted_source_cloud = o3d.geometry.PointCloud()
            self.raycasted_source_cloud.points = o3d.utility.Vector3dVector(raycasted_point_cloud_using_t_hit)
            self.raycasted_source_cloud.paint_uniform_color(np.array([[1],[0],[0]])) # red

            # Saving the raycasted source cloud
            o3d.io.write_point_cloud("feb02/raycasted_source_cloud.ply", self.raycasted_source_cloud)

            self.widget3d.scene.scene.add_geometry("raycasted_source_cloud", self.raycasted_source_cloud, self.mat)
        else:
            print("Show Raycasted PCD Button is OFF")
            self.show_raycasted_pcd_btn.text = "ShowRayCastedPCD"
            self.widget3d.scene.scene.remove_geometry("raycasted_source_cloud")
            self.raycasted_source_cloud = None
            print("Done removing raycasted source cloud")
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

    def _on_show_correct_shadow_casting_btn_clicked(self):
        print("Show Shadow Casting by Raycast Button clicked")
        if self.shadow_casted_pcd_using_ray_cast_without_prototype is None:
            self.show_correct_shadow_casting_btn.is_on = False
            self.show_correct_shadow_casting_btn.text = "Show ShadowCasted by Raycast"
            print("No shadow_casted_pcd_using_ray_cast_without_prototype")
            return
        if self.show_correct_shadow_casting_btn.is_on:
            self.show_correct_shadow_casting_btn.text = "Remove Casted Shadow from Raycast"
            print("Show Shadow Casting New Button is ON")
            self.final_merged_cloud_after_shadow_cast = self.shadow_casted_pcd_using_ray_cast_without_prototype + self.raycasted_source_cloud
            self.widget3d.scene.scene.show_geometry("target_cloud", show=False)
            # self.widget3d.scene.scene.add_geometry("shadow_cast_using_raycast_method", self.shadow_casted_pcd_using_ray_cast_without_prototype, self.mat)
            self.widget3d.scene.scene.show_geometry("shadow_cast_using_raycast_method", show=True)
            self.update_show_hide_checkboxes()

        else:
            self.show_correct_shadow_casting_btn.text = "Show ShadowCasted by Raycast"
            print("Show Shadow Casting new Button is OFF")
            self.widget3d.scene.scene.show_geometry("target_cloud", show=True)
            self.widget3d.scene.scene.show_geometry("shadow_cast_using_raycast_method", show=False)
            self.final_merged_cloud_after_shadow_cast = None
            # self.shadowed_cloud = None
            # self.final_merged_cloud_without_shadow_indices = None
            print("Done removing casted shadow")
        
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()
        # self.func_to_track_shadowcasting()
            

    @check_if_pcd_is_loaded
    def _on_show_shadow_casting_btn_clicked(self):
        print("Show Shadow Casting Button clicked")
        if self.raycasted_source_cloud is None or self.target_cloud is None or self.raycasted_source_cloud is None:
            self.show_shadow_casting_btn.is_on = False
            self.show_shadow_casting_btn.text = "ShowShadowCasted"
            print("No raycasted source cloud or target cloud")
            return
        if self.show_shadow_casting_btn.is_on:
            self.show_shadow_casting_btn.text = "Remove Casted Shadow"
            print("Show Shadow Casting Button is ON")

            self.target_cloud_subset_to_shadow_cast = self.target_cloud.select_by_index(self.selected_pcd_indices_with_obj_indices)
            self.target_cloud_subset_not_to_shadow_cast = self.target_cloud.select_by_index(self.selected_pcd_indices_with_obj_indices, invert=True)

            self.raycasted_source_cloud.paint_uniform_color([0, 0, 0]) # black

            self.final_merged_cloud_subset_to_shadow_cast = self.target_cloud_subset_to_shadow_cast + self.raycasted_source_cloud
            # self._on_roi_reset_btn_clicked()
            # self.final_merged_cloud = self.target_cloud + self.raycasted_source_cloud

            # merged_cloud_original_indices = np.concatenate((np.asarray(self.target_cloud.points), np.asarray(self.raycasted_source_cloud.points)), axis=0)

            # Hidden Point Removal Used for Shadowcasting
            diameter = np.linalg.norm(
                np.asarray(self.final_merged_cloud_subset_to_shadow_cast.get_max_bound()) - np.asarray(self.final_merged_cloud_subset_to_shadow_cast.get_min_bound()))
            print("Define parameters used for hidden_point_removal")
            camera = o3d.core.Tensor([0, 0, 0], o3d.core.float32)
            radius = diameter * int(self.rgn6_radius_text.text_value)
            print("Get all points that are visible from given view point")
            self.final_merged_cloud_subset_to_shadow_cast = o3d.t.geometry.PointCloud.from_legacy(self.final_merged_cloud_subset_to_shadow_cast)
            _, self.final_merged_cloud_subset_after_shadow_cast_indices = self.final_merged_cloud_subset_to_shadow_cast.hidden_point_removal(camera, radius)
            
            self.final_merged_cloud_subset_after_shadow_cast = self.final_merged_cloud_subset_to_shadow_cast.select_by_index(self.final_merged_cloud_subset_after_shadow_cast_indices)
            self.shadowed_cloud = self.final_merged_cloud_subset_to_shadow_cast.select_by_index(self.final_merged_cloud_subset_after_shadow_cast_indices, invert=True)
            



            self.final_merged_cloud_after_shadow_cast = self.final_merged_cloud_subset_after_shadow_cast + o3d.t.geometry.PointCloud.from_legacy(self.target_cloud_subset_not_to_shadow_cast)
            self.widget3d.scene.scene.show_geometry("target_cloud", show=False)
            self.widget3d.scene.scene.add_geometry("final_merged_cloud_after_shadow_cast", self.final_merged_cloud_after_shadow_cast, self.mat)
            self.final_merged_cloud_after_shadow_cast = self.final_merged_cloud_after_shadow_cast.to_legacy()
            self.update_show_hide_checkboxes()
        else:
            self.show_shadow_casting_btn.text = "Show Shadow"
            print("Show Shadow Casting Button is OFF")
            self.widget3d.scene.scene.show_geometry("target_cloud", show=True)
            self.widget3d.scene.scene.remove_geometry("final_merged_cloud_after_shadow_cast")
            self.final_merged_cloud_after_shadow_cast = None
            self.shadowed_cloud = None
            self.final_merged_cloud_without_shadow_indices = None
            print("Done removing casted shadow")
        
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()
        self.func_to_track_shadowcasting()

    @check_if_pcd_is_loaded
    def _on_finalize_shadow_casting_btn_clicked(self):
        print("Finalize Shadow Casting Button clicked")
        self.show_shadow_casting_btn.is_on = False
        self.show_shadow_casting_btn.text = "Show Shadow"
        self.show_correct_shadow_casting_btn.is_on = False
        self.show_correct_shadow_casting_btn.text = "Show ShadowCasted by Raycast"

        final_cloud = self.final_merged_cloud_after_shadow_cast
        # o3d.io.write_point_cloud("final_cloud_1.ply", final_cloud)
        # original_src_pcd_indices_to_remove = np.where((np.asarray(final_cloud.colors) == self.object_of_interest_color).all(axis=1))[0]
        # self.final_cloud = final_cloud.select_by_index(original_src_pcd_indices_to_remove, invert=True)
        # o3d.io.write_point_cloud("final_cloud_2.ply", self.final_cloud)
        self.final_cloud = final_cloud

        shadow_cast_by_hpr = False

        if self.widget3d.scene.scene.has_geometry("final_merged_cloud_after_shadow_cast"):
            self.widget3d.scene.scene.remove_geometry("final_merged_cloud_after_shadow_cast")
            self.widget3d.scene.scene.add_geometry("final_merged_cloud_after_shadow_cast", self.final_cloud, self.mat)
            shadow_cast_by_hpr = True

        if self.widget3d.scene.scene.has_geometry("shadow_cast_using_raycast_method"):
            self.widget3d.scene.scene.remove_geometry("shadow_cast_using_raycast_method")
            self.widget3d.scene.scene.add_geometry("shadow_cast_using_raycast_method", self.shadow_casted_pcd_using_ray_cast_without_prototype, self.mat)
            shadow_cast_by_hpr = False

        self.widget3d.scene.scene.show_geometry("target_cloud", show=False)
        self.widget3d.scene.scene.show_geometry("source_cloud", show=False)
        if shadow_cast_by_hpr:
            self.widget3d.scene.scene.show_geometry("final_merged_cloud_after_shadow_cast", show=True)
        else:
            self.widget3d.scene.scene.show_geometry("shadow_cast_using_raycast_method", show=True)
        self.widget3d.scene.scene.show_geometry("raycasted_source_cloud", show=False)
        self.widget3d.scene.scene.show_geometry("directed_rays", show=False)
        self.widget3d.scene.scene.show_geometry("reconstructed_source_mesh_filtered_densities_mesh", show=False)
        self.widget3d.scene.scene.show_geometry("reconstructed_source_mesh_densities_with_color", show=False)
        self.widget3d.scene.scene.show_geometry("reconstructed_source_mesh", show=False)
        self.widget3d.force_redraw()
        self.update_show_hide_checkboxes()

        

    def _on_rgn7_show_source_pcd_chk_box_clicked(self, checked):
        print("Show Source PCD Chk Box clicked : ", checked)
        if checked:
            if self.widget3d.scene.scene.geometry_is_visible("source_cloud"):
                print("Source PCD is already added")
            else:
                self.widget3d.scene.scene.show_geometry("source_cloud", show=True)
        else:
            if self.widget3d.scene.scene.geometry_is_visible("source_cloud"):
                self.widget3d.scene.scene.show_geometry("source_cloud", show=False)
            else:
                print("Source PCD is not available to remove")
            


    def _on_rgn7_show_target_pcd_chk_box_clicked(self, checked):
        print("Show Target PCD Chk Box clicked : ", checked)
        if checked:
            if self.widget3d.scene.scene.geometry_is_visible("target_cloud"):
                print("Target PCD is already added")
            else:
                self.widget3d.scene.scene.show_geometry("target_cloud", show=True)
        else:
            if self.widget3d.scene.scene.geometry_is_visible("target_cloud"):
                self.widget3d.scene.scene.show_geometry("target_cloud", show=False)
            else:
                print("Target PCD is not available to remove")

    def _on_rgn7_show_recostructed_surface_chk_box_clicked(self, checked):
        print("Show Reconstructed Source Surface Chk Box clicked : ", checked)
        if checked:
            if self.widget3d.scene.scene.geometry_is_visible("reconstructed_source_mesh"):
                print("Reconstructed Source Surface is already added")
            else:
                self.widget3d.scene.scene.show_geometry("reconstructed_source_mesh", show=True)
        else:
            if self.widget3d.scene.scene.geometry_is_visible("reconstructed_source_mesh"):
                self.widget3d.scene.scene.show_geometry("reconstructed_source_mesh", show=False)
            else:
                print("Reconstructed Source Surface is not available to remove")

    
    def _on_rgn7_show_reconst_density_mesh_chk_box_clicked(self, checked):
        print("Show Reconstructed Source Surface Densities Chk Box clicked : ", checked)
        if checked:
            if self.widget3d.scene.scene.geometry_is_visible("reconstructed_source_mesh_densities_with_color"):
                print("Reconstructed Source Surface Densities is already added")
            else:
                self.widget3d.scene.scene.show_geometry("reconstructed_source_mesh_densities_with_color", show=True)
        else:
            if self.widget3d.scene.scene.geometry_is_visible("reconstructed_source_mesh_densities_with_color"):
                self.widget3d.scene.scene.show_geometry("reconstructed_source_mesh_densities_with_color", show=False)
            else:
                print("Reconstructed Source Surface Densities is not available to remove")
    
    
    def _on_rgn7_show_filtered_density_mesh_chk_box_clicked(self, checked):
        print("Show Reconstructed Filtered Source Surface Densities Chk Box clicked : ", checked)
        if checked:
            if self.widget3d.scene.scene.geometry_is_visible("reconstructed_source_mesh_filtered_densities_mesh"):
                print("Reconstructed Filtered Source Surface Densities is already added")
            else:
                self.widget3d.scene.scene.show_geometry("reconstructed_source_mesh_filtered_densities_mesh", show=True)
        else:
            if self.widget3d.scene.scene.geometry_is_visible("reconstructed_source_mesh_filtered_densities_mesh"):
                self.widget3d.scene.scene.show_geometry("reconstructed_source_mesh_filtered_densities_mesh", show=False)
            else:
                print("Reconstructed Filtered Source Surface Densities is not available to remove")
    

    def _on_rgn7_show_directed_rays_chk_box_clicked(self, checked):
        print("Show Rays Chk Box clicked : ", checked)
        if checked:
            if self.widget3d.scene.scene.geometry_is_visible("directed_rays"):
                print("Rays is already added")
            else:
                self.widget3d.scene.scene.show_geometry("directed_rays", show=True)
        else:
            if self.widget3d.scene.scene.geometry_is_visible("directed_rays"):
                self.widget3d.scene.scene.show_geometry("directed_rays", show=False)
            else:
                print("Rays is not available to remove")


    def _on_rgn7_show_raycasted_source_pcd_chk_box_clicked(self, checked):
        print("Show Raycasted PointCloud Chk Box clicked : ", checked)
        if checked:
            if self.widget3d.scene.scene.geometry_is_visible("raycasted_source_cloud"):
                print("Raycasted PointCloud is already added")
            else:
                self.widget3d.scene.scene.show_geometry("raycasted_source_cloud", show=True)
        else:
            if self.widget3d.scene.scene.geometry_is_visible("raycasted_source_cloud"):
                self.widget3d.scene.scene.show_geometry("raycasted_source_cloud", show=False)
            else:
                print("Raycasted PointCloud is not available to remove")


    def _on_rgn7_show_casted_shadow_chk_box_clicked(self, checked):
        print("Show ShadowCasted PointCloud Chk Box clicked : ", checked)
        if checked:
            if self.widget3d.scene.scene.geometry_is_visible("final_merged_cloud_after_shadow_cast"):
                print("ShadowCasted PointCloud is already added")
            else:
                self.widget3d.scene.scene.show_geometry("final_merged_cloud_after_shadow_cast", show=True)
        else:
            if self.widget3d.scene.scene.geometry_is_visible("final_merged_cloud_after_shadow_cast"):
                self.widget3d.scene.scene.show_geometry("final_merged_cloud_after_shadow_cast", show=False)
            else:
                print("ShadowCasted PointCloud is not available to remove")
    
    def _on_rgn7_show_correct_casted_shadow_chk_box_clicked(self, checked):
        print("Show Correct ShadowCasted PointCloud Chk Box clicked : ", checked)
        if checked:
            if self.widget3d.scene.scene.geometry_is_visible("shadow_cast_using_raycast_method"):
                print("Correct ShadowCasted PointCloud is already added")
            else:
                self.widget3d.scene.scene.show_geometry("shadow_cast_using_raycast_method", show=True)
        else:
            if self.widget3d.scene.scene.geometry_is_visible("shadow_cast_using_raycast_method"):
                self.widget3d.scene.scene.show_geometry("shadow_cast_using_raycast_method", show=False)
            else:
                print("Correct ShadowCasted PointCloud is not available to remove")

    def _on_rgn8_save_final_merged_pcd_btn_clicked(self):
        print("Save Final Merged PCD Button clicked")
        # o3d.io.write_point_cloud("source_scene_with_colors_test.ply", self.source_scene_cloud)
        if self.final_merged_cloud_after_shadow_cast is None:
            print("Final Merged PCD is not available to save")
            return
        else:
            if self.rgn8_save_final_merged_pcd_text.text_value == "":
                print("Please enter a valid file name")
                return
            elif self.rgn8_save_final_merged_pcd_text.text_value[-4:] != ".ply":
                print("Please enter a valid file name with .ply extension")
                return
            self.final_cloud = self.final_merged_cloud_after_shadow_cast
            final_cloud_points = np.asarray(self.final_cloud.points)
            final_colors = np.asarray(self.final_cloud.colors)

            roi_colors = np.where((final_colors == [1, 0, 0]).all(axis=1))[0]

            roi_points = final_cloud_points[roi_colors]
            roi_colors = final_colors[roi_colors]

            roi_cloud = o3d.geometry.PointCloud()
            roi_cloud.points = o3d.utility.Vector3dVector(roi_points)
            roi_cloud.colors = o3d.utility.Vector3dVector(roi_colors)

            o3d.io.write_point_cloud("feb02/"+"predicted_casted_shadow_on_roi.ply", roi_cloud)

            

            o3d.io.write_point_cloud("feb02/"+self.rgn8_save_final_merged_pcd_text.text_value, self.final_merged_cloud_after_shadow_cast)
            print("Final Merged PCD saved successfully")

    def _on_rgn8_reset_all_variables_btn_clicked(self):
        print("Reset All Variables Button clicked")
        
        self.source_cloud = None
        self.target_cloud = None
        self.reconstructed_source_mesh = None
        self.reconstructed_source_mesh_densities = None
        self.reconstructed_source_mesh_densities_array = None
        self.reconstructed_source_mesh_densities_with_color = None
        self.reconstructed_source_mesh_filtered_densities_mesh = None
        self.raycasted_source_cloud = None
        self.final_merged_cloud_after_shadow_cast = None
        self.centroid_of_reference_roi = None
        self.centroid_of_target_roi = None
        self.selected_pcd_indices = None
        self.line_set = None
        self.shadowed_cloud = None
        self.shadowed_cloud_indices = None
        self.final_merged_cloud_after_shadow_cast = None
        self.final_merged_cloud_without_shadow_indices = None
        self.final_cloud = None
        self.selected_pcd_indices_with_obj_indices = None
        self.selected_pcd_roi_boundary_indices = []
        self.source_cloud_before_transform = None
        self.selected_pcd_indices = None
        self.raycasted_source_cloud = None

        self.source_scene_cloud = None

        self.shadow_casted_pcd_using_ray_cast_without_prototype = None



        # open3d gui geometry reset
        if self.widget3d.scene.scene.has_geometry("source_scene_cloud"):
            self.widget3d.scene.scene.remove_geometry("source_scene_cloud")
        if self.widget3d.scene.scene.has_geometry("source_cloud"):
            self.widget3d.scene.scene.remove_geometry("source_cloud")
        if self.widget3d.scene.scene.has_geometry("target_cloud"):
            self.widget3d.scene.scene.remove_geometry("target_cloud")
        if self.widget3d.scene.scene.has_geometry("source_cloud_transformed"):
            self.widget3d.scene.scene.remove_geometry("source_cloud_transformed")
        if self.widget3d.scene.scene.has_geometry("reconstructed_source_mesh"):
            self.widget3d.scene.scene.remove_geometry("reconstructed_source_mesh")
        if self.widget3d.scene.scene.has_geometry("reconstructed_source_mesh_densities_with_color"):
            self.widget3d.scene.scene.remove_geometry("reconstructed_source_mesh_densities_with_color")
        if self.widget3d.scene.scene.has_geometry("reconstructed_source_mesh_filtered_densities_mesh"):
            self.widget3d.scene.scene.remove_geometry("reconstructed_source_mesh_filtered_densities_mesh")
        if self.widget3d.scene.scene.has_geometry("directed_rays"):
            self.widget3d.scene.scene.remove_geometry("directed_rays")
        if self.widget3d.scene.scene.has_geometry("raycasted_source_cloud"):
            self.widget3d.scene.scene.remove_geometry("raycasted_source_cloud")
        if self.widget3d.scene.scene.has_geometry("final_merged_cloud_after_shadow_cast"):
            self.widget3d.scene.scene.remove_geometry("final_merged_cloud_after_shadow_cast")
        if self.widget3d.scene.scene.has_geometry("shadow_cast_using_raycast_method"):
            self.widget3d.scene.scene.remove_geometry("shadow_cast_using_raycast_method")
        

        # GUI Elements
        self.rgn1_source_scene_pcd_text.text_value = "jan31_semantic_lidar_source_scene.csv"
        self.rgn1_use_labels_to_extract_src_pcd_chk_box.checked = False
        self.rgn1_use_geometric_features_to_extract_src_pcd_chk_box.checked = False
        self.rgn1_labels_to_extract_src_pcd_text.enabled = False
        self.rgn1_surface_variation_chk_box.checked = False
        self.rgn1_surface_variation_chk_box.enabled = False
        self.rgn1_surface_variation_threshold_text.enabled = False
        self.rgn1_planarity_chk_box.checked = False
        self.rgn1_planarity_chk_box.enabled = False
        self.rgn1_planarity_threshold_text.enabled = False
        self.rgn1_linearity_chk_box.checked = False
        self.rgn1_linearity_chk_box.enabled = False
        self.rgn1_linearity_threshold_text.enabled = False
        self.rgn1_z_value_chk_box.checked = False
        self.rgn1_z_value_chk_box.enabled = False
        self.rgn1_z_value_threshold_text.enabled = False
        self.rgn1_extract_src_pcd_btn.is_on = False
        self.rgn1_extract_src_pcd_btn.text = "Extract Source PCD"
        self.rgn1_finalize_extracted_src_pcd_btn.is_on = False
        self.rgn1_finalize_extracted_src_pcd_btn.text = "FinalizeExtractedResult"
        self.roi_select_boundary_chk_box.checked = False
        self.reconstruct_surface_btn.is_on = False
        self.reconstruct_surface_btn.text = "ShowReconstr.Surf."
        self.calculate_density_mesh_btn.is_on = False
        self.calculate_density_mesh_btn.text = "ShowDensityMesh"
        self.filter_density_btn.is_on = False
        self.filter_density_btn.text = "FilterDensityMesh"
        self.filter_rays_slider.double_value = 0.5
        self.show_rays_btn.is_on = False
        self.show_rays_btn.text = "Show Rays"
        self.show_raycasted_pcd_btn.is_on = False
        self.show_raycasted_pcd_btn.text = "ShowRayCastedPCD"
        self.show_shadow_casting_btn.is_on = False
        self.show_shadow_casting_btn.text = "ShowShadowCasted"

        self.rgn7_show_source_pcd_chk_box.checked = False
        self.rgn7_show_target_pcd_chk_box.checked = False
        self.rgn7_show_recostructed_surface_chk_box.checked = False
        self.rgn7_show_reconst_density_mesh_chk_box.checked = False
        self.rgn7_show_filtered_density_mesh_chk_box.checked = False
        self.rgn7_show_directed_rays_chk_box.checked = False
        self.rgn7_show_raycasted_source_pcd_chk_box.checked = False
        self.rgn7_show_casted_shadow_chk_box.checked = False




        
        self.widget3d.force_redraw()
        


    





    def _on_mouse_widget3d(self, event):
        """
        Selects a region of interest on the target point cloud or source scene point cloud (complete source scene).

        CTRL/CMD + roi_select_boundary_chk_box
        The selected region will be color in red. The red color will be used later to select the rectangular region of interest.
        """
        # print("Mouse event")
        if  event.is_modifier_down(gui.KeyModifier.CTRL):
            if self.roi_select_boundary_chk_box.checked:
                if self.target_cloud is None and self.source_scene_cloud is None:
                    return gui.Widget.EventCallbackResult.IGNORED
                # print("CTRL/CMD + Mouse DOWN BTN Clicked")
                # print(event.x, event.y) # prints the mouse position. 0,0 is the top left corner of the window
                def depth_callback(depth_image):
                    x = event.x - self.widget3d.frame.x
                    y = event.y - self.widget3d.frame.y
                    # Note that np.asarray() reverses the axes.
                    depth = np.asarray(depth_image)[y, x]
                    if depth == 1.0:
                        pass
                        # print("Clicked on nothing")
                    else:
                        # print("Clicked on something")
                        world = self.widget3d.scene.camera.unproject(
                            x, y, depth, self.widget3d.frame.width,
                                self.widget3d.frame.height)
                        text = "({:.3f}, {:.3f}, {:.3f})".format(
                            world[0], world[1], world[2])
                        if self.source_scene_cloud is not None:
                            cloud_to_operate = self.source_scene_cloud
                            geometry_name = "source_scene_cloud"
                        elif self.target_cloud is not None:
                            cloud_to_operate = self.target_cloud
                            geometry_name = "target_cloud"
                        else:
                            return gui.Widget.EventCallbackResult.IGNORED
                        distances = np.sum((cloud_to_operate.points - world) ** 2, axis=1)
                        nearest_point_index = np.argmin(distances)
                        # print("Nearest Point Index: ", nearest_point_index)
                        # print("Nearest Point: ", cloud_to_operate.points[nearest_point_index])
                        # print("Nearest Point Color: ", cloud_to_operate.colors[nearest_point_index])
                        # print(dir(cloud_to_operate.colors))
                        if np.array_equal(cloud_to_operate.colors[nearest_point_index],[0, 1, 0]): # if the point is green
                            cloud_to_operate.colors[nearest_point_index] = [1, 0, 0]  # Change to red                        
                            self.selected_pcd_roi_boundary_indices.append(nearest_point_index)
                        cloud_tensor = o3d.t.geometry.PointCloud().from_legacy(cloud_to_operate)
                        if self.widget3d.scene.scene.has_geometry(geometry_name):
                            self.widget3d.scene.scene.remove_geometry(geometry_name)
                        self.widget3d.scene.scene.add_geometry(geometry_name, cloud_tensor, self.mat)
                        self.widget3d.force_redraw()
                self.widget3d.scene.scene.render_to_depth_image(depth_callback)
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    
    def _on_close(self):
        print("Closing the app after resetting all the variables")
        # print(self.__dict__)
        for attr in self.__dict__:
            # print("Setting ", attr, " to None")
            if attr != "window":
                setattr(self, attr, None)
        return True # return True to allow the window to continue closing


    # endregion member functions


def main():

    app = gui.Application.instance
    app.initialize()
    ex = ScenarioCreatorApp()
    app.run()



if __name__ == "__main__":

    main()