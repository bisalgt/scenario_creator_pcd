import open3d as o3d
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns



class PointCloudAnalysis:
    """
    Class to analyse pointcloud using open3d library.
    
    List of calculated features: surface_variation, planarity, sphericity, linearity, omnivariance, anisotropy, eigenentropy.
    Each features are calculated, visualized after normalizing.
    Instance is initialized using points which is a pandas dataframe.
    """
    
    def __init__(self, points, is_point_type_open3d=False, search_tree=None):
        if not is_point_type_open3d:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
        else:
            self.pcd = points
        self.search_tree = search_tree
        if self.search_tree is None:
            self.search_tree = o3d.geometry.KDTreeSearchParamKNN(50)
        self._compute_covariances_mtx()
        self._compute_eigenvalues()
        
    def _compute_covariances_mtx(self):
        self.pcd.estimate_covariances(self.search_tree)
        self._covariances_mtx = np.asarray(self.pcd.covariances)
        
    def _compute_eigenvalues(self):
        self._compute_covariances_mtx()
        self._eigenvalues, self._eigenvectors = np.linalg.eig(self._covariances_mtx)
    
    @property
    def covariances_mtx(self):
        return self._covariances_mtx
        
    @property
    def eigenvalues(self):
        return self._eigenvalues
        
    @staticmethod
    def get_normalized_feature(feature):
        # To make the value in between [0, 1]
        normalized_feature = (feature - feature.min())/(feature.max() - feature.min())
        return normalized_feature
    
    @staticmethod
    def _sort_1array_ascending(arr):
        return sorted(arr)
    

    
    @staticmethod
    def _compute_1array_surface_variation(arr):
        return arr.min()/(arr.sum())
    
    def compute_surface_variation(self):
        self._surface_variation = np.apply_along_axis(self._compute_1array_surface_variation, 1, self._eigenvalues)
        return self._surface_variation
    
    def get_normalized_surface_variation(self):
        self.compute_surface_variation()
        normalized_surface_variation = self.get_normalized_feature(self._surface_variation)
        return normalized_surface_variation
    
    
    @staticmethod
    def _compute_1array_planarity(arr):
        lambda_min, lambda_mid, lambda_max = sorted(arr)
        return ((lambda_mid - lambda_min) / lambda_max)

    
    def compute_planarity(self):
        self._planarity = np.apply_along_axis(self._compute_1array_planarity, 1, self._eigenvalues)
        return self._planarity

    def get_normalized_planarity(self):
        self.compute_planarity()
        normalized_planarity = self.get_normalized_feature(self._planarity)
        return normalized_planarity
    
    
    @staticmethod
    def _compute_1array_sphericity(arr):
        lambda_min, lambda_mid, lambda_max = sorted(arr)
        return (lambda_min / lambda_max)

    
    def compute_sphericity(self):
        self._sphericity = np.apply_along_axis(self._compute_1array_sphericity, 1, self._eigenvalues)
        return self._sphericity

    def get_normalized_sphericity(self):
        self.compute_sphericity()
        normalized_sphericity = self.get_normalized_feature(self._sphericity)
        return normalized_sphericity
    
    
    @staticmethod
    def _compute_1array_linearity(arr):
        lambda_min, lambda_mid, lambda_max = sorted(arr)
        return ((lambda_max - lambda_mid) / lambda_max)

    
    def compute_linearity(self):
        self._linearity = np.apply_along_axis(self._compute_1array_linearity, 1, self._eigenvalues)
        return self._linearity

    def get_normalized_linearity(self):
        self.compute_linearity()
        normalized_linearity = self.get_normalized_feature(self._linearity)
        return normalized_linearity
    
    
    @staticmethod
    def _compute_1array_omnivariance(arr):
        lambda_min, lambda_mid, lambda_max = sorted(arr)
        return (lambda_max * lambda_mid * lambda_min)**(1/3)

    
    def compute_omnivariance(self):
        self._omnivariance = np.apply_along_axis(self._compute_1array_omnivariance, 1, self._eigenvalues)
        return self._omnivariance

    def get_normalized_omnivariance(self):
        self.compute_omnivariance()
        normalized_omnivariance = self.get_normalized_feature(self._omnivariance)
        return normalized_omnivariance
    
    
    @staticmethod
    def _compute_1array_anisotropy(arr):
        lambda_min, lambda_mid, lambda_max = sorted(arr)
        return ((lambda_max - lambda_min) / lambda_max)

    
    def compute_anisotropy(self):
        self._anisotropy = np.apply_along_axis(self._compute_1array_anisotropy, 1, self._eigenvalues)
        return self._anisotropy

    def get_normalized_anisotropy(self):
        self.compute_anisotropy()
        normalized_anisotropy = self.get_normalized_feature(self._anisotropy)
        return normalized_anisotropy
    
    
    @staticmethod
    def _compute_1array_eigenentropy(arr):
        lambda_min, lambda_mid, lambda_max = sorted(arr)
        return -((lambda_max * math.log(lambda_max)) +(lambda_mid * math.log(lambda_mid)) + (lambda_min * math.log(lambda_min)))
    
    def compute_eigenentropy(self):
        self._eigenentropy = np.apply_along_axis(self._compute_1array_eigenentropy, 1, self._eigenvalues)
        return self._eigenentropy

    def get_normalized_eigenentropy(self):
        self.compute_eigenentropy()
        normalized_eigenentropy = self.get_normalized_feature(self._eigenentropy)
        return normalized_eigenentropy
    
    
    def compute_sum_of_eigenvalues(self):
        self._sum_of_eigenvalues = np.sum(self._eigenvalues, axis=1)
        return self._sum_of_eigenvalues
    
    def get_normalized_sum_of_eigenvalues(self):
        self.compute_sum_of_eigenvalues()
        normalized_sum_of_eigenvalues = self.get_normalized_feature(self._sum_of_eigenvalues)
        return normalized_sum_of_eigenvalues
    
    
    @staticmethod
    def get_rgb_color_from_intensity(normalized_intensity):
        # Normalized Intensity is used
        # Creating Color stack
        r_color = np.zeros(normalized_intensity.shape)
        g_color = normalized_intensity
        b_color = np.zeros(normalized_intensity.shape)
        # Creating 3d points using np.stack
        intensity_3d = np.stack((r_color, g_color, b_color), axis=-1)
        return intensity_3d

    @staticmethod
    def _calculate_bins_for_histogram(_data):
        """
        Bin calculated using Freedman-Diaconis rule for the histogram plot
        """
        iqr = np.percentile(_data, 75) - np.percentile(_data, 25)
        bin_width = 2 * iqr / np.power(len(_data), 1/3)
        num_bins = int((np.max(_data) - np.min(_data)) / bin_width)
        return num_bins

    def plot_histogram_with_gaussian_distribution(self, _data):
        """
        Plotting 4 Rows of the given data.
        """
        _data = np.asarray(sorted(_data))
        bins = self._calculate_bins_for_histogram(_data)
        # Calculate mean and standard deviation of the data
        mean, standard_deviation = np.mean(_data), np.std(_data)
        # Calculate pdf of data
        y_pdf = stats.norm.pdf(_data, loc=mean, scale=standard_deviation)
        # Allocating 4 rows for 4 subplots of figure with shared x-axis.
        figure, axes = plt.subplots(4, sharex=True)
        figure.set_size_inches(15,15)
        # Plot Histogram with Data vs Frequency of items in each bins.
        axes[0].set_title("Histogram plot of data vs frequency of items in each bins.")
        axes[0].hist(_data, bins=bins)
        # Plot Histogram with Data vs Probability Distribution; calculated by density=True
        axes[1].set_title("Histogram plot of data vs probability density of items in each bins.")
        axes[1].hist(_data, bins=bins, density=True)
        # Plot Histogram plot with Normal Distribution curve
        axes[2].set_title("Histogram plot with normal distribution.")
        axes[2].hist(_data, bins=bins, density=True)
        axes[2].plot(_data, y_pdf, 'r')
        # Plot Normal Distribution Curve
        axes[3].set_title("Normal Distribution Curve.")
        axes[3].plot(_data, y_pdf)
   
    def display_surface_variation(self):
        normalized_surface_variation = self.get_normalized_surface_variation()
        intensity_3d = self.get_rgb_color_from_intensity(normalized_surface_variation)
        self.pcd.colors = o3d.utility.Vector3dVector(intensity_3d)
        o3d.visualization.draw_geometries([self.pcd])
        
    def display_planarity(self):
        normalized_planarity = self.get_normalized_planarity()
        intensity_3d = self.get_rgb_color_from_intensity(normalized_planarity)
        self.pcd.colors = o3d.utility.Vector3dVector(intensity_3d)
        o3d.visualization.draw_geometries([self.pcd])
        
    def display_sphericity(self):
        normalized_sphericity = self.get_normalized_sphericity()
        intensity_3d = self.get_rgb_color_from_intensity(normalized_sphericity)
        self.pcd.colors = o3d.utility.Vector3dVector(intensity_3d)
        o3d.visualization.draw_geometries([self.pcd])
        
    def display_linearity(self):
        normalized_linearity = self.get_normalized_linearity()
        intensity_3d = self.get_rgb_color_from_intensity(normalized_linearity)
        self.pcd.colors = o3d.utility.Vector3dVector(intensity_3d)
        o3d.visualization.draw_geometries([self.pcd])
    
    def display_omnivariance(self):
        normalized_omnivariance = self.get_normalized_omnivariance()
        intensity_3d = self.get_rgb_color_from_intensity(normalized_omnivariance)
        self.pcd.colors = o3d.utility.Vector3dVector(intensity_3d)
        o3d.visualization.draw_geometries([self.pcd])
    
    def display_anisotropy(self):
        normalized_anisotropy = self.get_normalized_anisotropy()
        intensity_3d = self.get_rgb_color_from_intensity(normalized_anisotropy)
        self.pcd.colors = o3d.utility.Vector3dVector(intensity_3d)
        o3d.visualization.draw_geometries([self.pcd])
    
    def display_eigenentropy(self):
        normalized_eigenentropy = self.get_normalized_eigenentropy()
        intensity_3d = self.get_rgb_color_from_intensity(normalized_eigenentropy)
        self.pcd.colors = o3d.utility.Vector3dVector(intensity_3d)
        o3d.visualization.draw_geometries([self.pcd])
    
    def display_sum_of_eigenvalues(self):
        normalized_sum_of_eigenvalues = self.get_normalized_sum_of_eigenvalues()
        intensity_3d = self.get_rgb_color_from_intensity(normalized_sum_of_eigenvalues)
        self.pcd.colors = o3d.utility.Vector3dVector(intensity_3d)
        o3d.visualization.draw_geometries([self.pcd])
    
    
    def display_custom_rgb(self, rgb_array):
        self.pcd.colors = o3d.utility.Vector3dVector(rgb_array)
        o3d.visualization.draw_geometries([self.pcd])

    
    def display_hist_from_seaborn(self, feature_array):
        try:
            number_of_bins = self._calculate_bins_for_histogram(feature_array)
            sns.displot(feature_array, bins=number_of_bins, kde=True)
        except Exception as e:
            print(f"Exception Occured,{e}, plot may not be consistent")
            sns.displot(feature_array, kde=True)
        

    
def calculate_bins_for_histogram(_data):
    """
    Bin calculated using Freedman-Diaconis rule for the histogram plot
    """
    iqr = np.percentile(_data, 75) - np.percentile(_data, 25)
    bin_width = 2 * iqr / np.power(len(_data), 1/3)
    num_bins = int((np.max(_data) - np.min(_data)) / bin_width)
    return num_bins


