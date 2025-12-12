# services/terrain_classification.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import requests
import logging
import os

logger = logging.getLogger(__name__)

class TerrainClassifier:
    def __init__(self):
        self.model = None
        self.terrain_classes = [
            'urban', 'rural', 'forest', 'water', 'desert', 
            'mountain', 'agricultural', 'coastal', 'savanna'
        ]
        self.load_model()
        logger.info("TerrainClassifier initialized")
    
    def load_model(self):
        """Load or create terrain classification model"""
        try:
            # In production, you'd load a pre-trained model
            # For now, we'll use a simple color/texture based approach
            self.model = "color_texture_based"
            logger.info("Using color/texture based terrain classification")
        except Exception as e:
            logger.warning(f"Could not load ML model, using fallback: {e}")
            self.model = "fallback"
    
    def classify_terrain(self, image_path):
        """
        Classify terrain type from image using multiple methods
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image"}
            
            results = {
                'primary_method': 'color_texture_analysis',
                'predictions': {},
                'dominant_terrain': '',
                'confidence': 0,
                'features': {}
            }
            
            # Method 1: Color histogram analysis
            color_analysis = self._analyze_colors(img)
            
            # Method 2: Texture analysis
            texture_analysis = self._analyze_texture(img)
            
            # Method 3: Edge density analysis
            edge_analysis = self._analyze_edges(img)
            
            # Method 4: Vegetation index (for rural/forest detection)
            vegetation_analysis = self._analyze_vegetation(img)
            
            # Combine all analyses
            combined_predictions = self._combine_analyses(
                color_analysis, texture_analysis, edge_analysis, vegetation_analysis
            )
            
            results['predictions'] = combined_predictions
            results['dominant_terrain'] = max(combined_predictions, key=combined_predictions.get)
            results['confidence'] = combined_predictions[results['dominant_terrain']]
            results['features'] = {
                'color_dominance': color_analysis,
                'texture_complexity': texture_analysis.get('complexity', 0),
                'edge_density': edge_analysis.get('density', 0),
                'vegetation_index': vegetation_analysis.get('index', 0)
            }
            
            logger.info(f"🌄 Terrain classification: {results['dominant_terrain']} ({results['confidence']:.1f}% confidence)")
            
            return results
            
        except Exception as e:
            logger.error(f"Terrain classification failed: {e}")
            return {"error": f"Terrain classification failed: {str(e)}"}
    
    def _analyze_colors(self, img):
        """Analyze color distribution for terrain clues"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist / h_hist.sum()
        s_hist = s_hist / s_hist.sum()
        v_hist = v_hist / v_hist.sum()
        
        # Analyze color characteristics
        avg_saturation = np.mean(s_hist * np.arange(256))
        avg_value = np.mean(v_hist * np.arange(256))
        
        # Color-based terrain predictions
        predictions = {}
        
        # Urban areas often have gray tones (low saturation)
        if avg_saturation < 50:
            predictions['urban'] = min(80, (50 - avg_saturation) * 2)
        
        # Rural/agricultural areas often have green tones
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_percentage = np.sum(green_mask > 0) / (img.shape[0] * img.shape[1])
        if green_percentage > 0.3:
            predictions['agricultural'] = min(90, green_percentage * 150)
            predictions['forest'] = min(70, green_percentage * 120)
        
        # Desert areas have yellow/brown tones
        desert_mask = cv2.inRange(hsv, (20, 30, 50), (35, 200, 255))
        desert_percentage = np.sum(desert_mask > 0) / (img.shape[0] * img.shape[1])
        if desert_percentage > 0.2:
            predictions['desert'] = min(85, desert_percentage * 200)
            predictions['savanna'] = min(60, desert_percentage * 150)
        
        # Water bodies have blue tones
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        blue_percentage = np.sum(blue_mask > 0) / (img.shape[0] * img.shape[1])
        if blue_percentage > 0.1:
            predictions['water'] = min(95, blue_percentage * 300)
            predictions['coastal'] = min(70, blue_percentage * 200)
        
        return predictions
    
    def _analyze_texture(self, img):
        """Analyze texture patterns for terrain classification"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate GLCM (Gray Level Co-occurrence Matrix) features
        # Simplified texture analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        texture_complexity = np.std(gradient_magnitude)
        
        # Texture-based predictions
        predictions = {}
        
        # Urban areas often have complex textures (buildings, roads)
        if texture_complexity > 50:
            predictions['urban'] = min(80, texture_complexity / 100 * 80)
        
        # Rural areas often have medium texture complexity
        elif texture_complexity > 20:
            predictions['rural'] = min(70, texture_complexity / 50 * 70)
            predictions['agricultural'] = min(60, texture_complexity / 50 * 60)
        
        # Water and desert have low texture complexity
        else:
            predictions['water'] = min(90, (50 - texture_complexity) * 2)
            predictions['desert'] = min(70, (50 - texture_complexity) * 1.5)
        
        return {'complexity': texture_complexity, 'predictions': predictions}
    
    def _analyze_edges(self, img):
        """Analyze edge density for man-made vs natural terrain"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        
        predictions = {}
        
        # High edge density suggests urban areas
        if edge_density > 0.1:
            predictions['urban'] = min(90, edge_density * 500)
        
        # Medium edge density suggests rural areas
        elif edge_density > 0.05:
            predictions['rural'] = min(70, edge_density * 700)
        
        # Low edge density suggests natural areas
        else:
            predictions['forest'] = min(80, (0.1 - edge_density) * 800)
            predictions['water'] = min(90, (0.1 - edge_density) * 900)
        
        return {'density': edge_density, 'predictions': predictions}
    
    def _analyze_vegetation(self, img):
        """Calculate vegetation index for rural/forest detection"""
        # Convert to LAB color space for better vegetation detection
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Enhanced vegetation index (simplified)
        b, g, r = cv2.split(img)
        
        # Avoid division by zero
        r = r.astype(float) + 1
        g = g.astype(float) + 1
        
        # Simple vegetation index
        vegetation_index = np.mean((g - r) / (g + r))
        
        predictions = {}
        
        if vegetation_index > 0.1:
            predictions['forest'] = min(90, vegetation_index * 400)
            predictions['agricultural'] = min(80, vegetation_index * 300)
            predictions['rural'] = min(70, vegetation_index * 250)
        elif vegetation_index > 0:
            predictions['savanna'] = min(60, vegetation_index * 200)
        
        return {'index': vegetation_index, 'predictions': predictions}
    
    def _combine_analyses(self, color, texture, edge, vegetation):
        """Combine all analysis methods"""
        combined = {}
        
        # Combine predictions from all methods
        all_predictions = [
            color,
            texture.get('predictions', {}),
            edge.get('predictions', {}),
            vegetation.get('predictions', {})
        ]
        
        for predictions in all_predictions:
            for terrain, confidence in predictions.items():
                if terrain in combined:
                    combined[terrain] = (combined[terrain] + confidence) / 2
                else:
                    combined[terrain] = confidence
        
        # Normalize to 0-100 scale
        if combined:
            max_conf = max(combined.values())
            if max_conf > 0:
                for terrain in combined:
                    combined[terrain] = (combined[terrain] / max_conf) * 100
        
        return combined
    
    def get_terrain_constraints(self, terrain_type):
        """Get geographic constraints for terrain type (Nigeria-specific)"""
        nigeria_terrain_constraints = {
            'urban': {
                'regions': ['Lagos', 'Abuja', 'Kano', 'Ibadan', 'Port Harcourt'],
                'elevation': 'low',
                'water_bodies': 'rare'
            },
            'rural': {
                'regions': ['all'],
                'elevation': 'varied',
                'water_bodies': 'common'
            },
            'forest': {
                'regions': ['Cross River', 'Ondo', 'Delta', 'Ogun'],
                'elevation': 'varied',
                'water_bodies': 'common'
            },
            'agricultural': {
                'regions': ['Benue', 'Kano', 'Kaduna', 'Plateau'],
                'elevation': 'varied',
                'water_bodies': 'common'
            },
            'savanna': {
                'regions': ['Northern Nigeria', 'Sokoto', 'Katsina', 'Bornu'],
                'elevation': 'low',
                'water_bodies': 'rare'
            },
            'coastal': {
                'regions': ['Lagos', 'Delta', 'Rivers', 'Bayelsa'],
                'elevation': 'very low',
                'water_bodies': 'abundant'
            }
        }
        
        return nigeria_terrain_constraints.get(terrain_type, {})