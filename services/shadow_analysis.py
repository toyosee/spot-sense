# services/shadow_analysis.py - Enhanced Shadow Analysis
import cv2
import numpy as np
import math
from datetime import datetime
from pysolar.solar import get_azimuth, get_altitude
import pytz
import logging
import json

logger = logging.getLogger(__name__)

class ShadowAnalyzer:
    def __init__(self):
        self.DEFAULT_TZ = pytz.timezone('Africa/Lagos')
        logger.info("ShadowAnalyzer initialized")
    
    def analyze_image_shadows(self, image_path, capture_time=None):
        """
        Comprehensive shadow analysis with validation and confidence scoring
        
        Args:
            image_path (str): Path to the image file
            capture_time (datetime): Image capture time
            
        Returns:
            dict: Shadow analysis results with confidence scores
        """
        if capture_time is None:
            capture_time = datetime.now(self.DEFAULT_TZ)
            logger.warning("Using current time as fallback for capture time")
        
        logger.info(f"Starting shadow analysis for: {image_path}")
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                error_msg = "Could not load image"
                logger.error(error_msg)
                return {"error": error_msg, "reason": "File not found or corrupted"}
            
            # Comprehensive shadow analysis with multiple methods
            shadow_analysis = self._comprehensive_shadow_analysis(img)
            
            if not shadow_analysis['has_shadows']:
                logger.warning(f"No shadows detected: {shadow_analysis['reason']}")
                return {
                    "error": "No reliable shadows detected", 
                    "reason": shadow_analysis['reason'],
                    "confidence": 0,
                    "has_shadows": False
                }
            
            # Calculate solar azimuth from shadow direction
            shadow_angle = shadow_analysis['shadow_angle']
            solar_azimuth = (shadow_angle + 180) % 360
            
            logger.info(f"Shadow analysis successful: angle={shadow_angle:.2f}°, azimuth={solar_azimuth:.2f}°")
            
            return {
                'shadow_angle': shadow_angle,
                'solar_azimuth': solar_azimuth,
                'capture_time': capture_time,
                'confidence': shadow_analysis['confidence'],
                'shadow_quality': shadow_analysis['shadow_quality'],
                'has_shadows': True,
                'analysis_methods': shadow_analysis['methods_used'],
                'details': {
                    'shadow_consistency': shadow_analysis.get('consistency_score', 0),
                    'image_quality': shadow_analysis.get('image_quality', 'unknown')
                }
            }
            
        except Exception as e:
            error_msg = f"Shadow analysis failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "reason": "Unexpected analysis error"}
    
    def _comprehensive_shadow_analysis(self, img):
        """
        Enhanced shadow detection using multiple validation methods
        
        Returns:
            dict: Comprehensive shadow analysis results
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        results = {
            'has_shadows': False,
            'reason': 'No shadows detected',
            'confidence': 0,
            'shadow_quality': 'poor',
            'shadow_angle': None,
            'methods_used': [],
            'consistency_score': 0
        }
        
        # Method 1: Enhanced Hough Line Transform
        hough_results = self._detect_lines_hough_enhanced(gray)
        
        # Method 2: Gradient-based analysis
        gradient_results = self._analyze_gradients_enhanced(gray)
        
        # Method 3: Shadow region detection
        region_results = self._detect_shadow_regions(img)
        
        # Method 4: Edge-based analysis
        edge_results = self._analyze_edges_enhanced(gray)
        
        # Combine results from all methods
        all_angles = []
        method_weights = []
        method_confidences = []
        
        if hough_results['has_lines']:
            all_angles.extend(hough_results['angles'])
            method_weights.extend([0.3] * len(hough_results['angles']))
            method_confidences.append(hough_results['confidence'])
            results['methods_used'].append('hough_lines')
            logger.debug(f"Hough lines: {len(hough_results['angles'])} lines found")
        
        if gradient_results['has_gradients']:
            all_angles.extend(gradient_results['angles'])
            method_weights.extend([0.25] * len(gradient_results['angles']))
            method_confidences.append(gradient_results['confidence'])
            results['methods_used'].append('gradient_analysis')
            logger.debug(f"Gradient analysis: {len(gradient_results['angles'])} angles found")
        
        if region_results['has_shadows']:
            all_angles.extend(region_results['angles'])
            method_weights.extend([0.35] * len(region_results['angles']))
            method_confidences.append(region_results['confidence'])
            results['methods_used'].append('shadow_regions')
            logger.debug(f"Shadow regions: {len(region_results['angles'])} regions found")
        
        if edge_results['has_edges']:
            all_angles.extend(edge_results['angles'])
            method_weights.extend([0.1] * len(edge_results['angles']))
            method_confidences.append(edge_results['confidence'])
            results['methods_used'].append('edge_analysis')
            logger.debug(f"Edge analysis: {len(edge_results['angles'])} edges found")
        
        if not all_angles:
            results['reason'] = 'No shadow features detected by any method'
            return results
        
        # Calculate weighted average angle using circular statistics
        angles_rad = np.radians(all_angles)
        weights = np.array(method_weights[:len(angles_rad)])  # Ensure same length
        
        # Weighted circular mean
        weighted_sin = np.average(np.sin(angles_rad), weights=weights)
        weighted_cos = np.average(np.cos(angles_rad), weights=weights)
        
        mean_angle = np.degrees(np.arctan2(weighted_sin, weighted_cos))
        if mean_angle < 0:
            mean_angle += 360
        
        # Calculate consistency score
        consistency = self._calculate_angle_consistency(all_angles, mean_angle)
        
        # Overall confidence based on method confidences and consistency
        avg_method_confidence = np.mean(method_confidences) if method_confidences else 0
        overall_confidence = min(100, avg_method_confidence * consistency * 100)
        
        # Determine shadow quality
        if overall_confidence > 70:
            shadow_quality = 'excellent'
        elif overall_confidence > 50:
            shadow_quality = 'good'
        elif overall_confidence > 30:
            shadow_quality = 'fair'
        else:
            shadow_quality = 'poor'
        
        results.update({
            'has_shadows': True,
            'shadow_angle': mean_angle,
            'confidence': overall_confidence,
            'shadow_quality': shadow_quality,
            'consistency_score': consistency,
            'reason': f'Shadows detected with {shadow_quality} quality',
            'detected_angles_count': len(all_angles),
            'method_count': len(results['methods_used'])
        })
        
        logger.info(f"Shadow analysis complete: quality={shadow_quality}, confidence={overall_confidence:.1f}%")
        
        return results
    
    def _detect_lines_hough_enhanced(self, gray):
        """Enhanced line detection with better filtering"""
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Hough Line Transform with adaptive parameters
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=30, 
                               minLineLength=40, 
                               maxLineGap=15)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Filter by length and angle
                if length > 50:  # Minimum line length
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    # Normalize angle to 0-360
                    if angle < 0:
                        angle += 360
                    
                    # Filter out near-vertical lines (likely not shadows)
                    if not (80 <= angle <= 100) and not (260 <= angle <= 280):
                        angles.append(angle)
        
        confidence = min(100, len(angles) * 5)  # More lines = higher confidence
        
        return {
            'has_lines': len(angles) > 0,
            'angles': angles,
            'confidence': confidence,
            'line_count': len(angles)
        }
    
    def _analyze_gradients_enhanced(self, gray):
        """Enhanced gradient analysis for shadow direction"""
        # Calculate gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orientation = np.arctan2(sobely, sobelx)
        
        # Filter significant gradients
        threshold = np.percentile(magnitude, 75)  # Higher percentile for better quality
        mask = magnitude > threshold
        
        if not np.any(mask):
            return {'has_gradients': False, 'angles': [], 'confidence': 0}
        
        angles = np.degrees(orientation[mask])
        # Normalize angles
        angles = [(angle + 360) % 360 for angle in angles]
        
        # Filter angles (remove near-vertical)
        filtered_angles = [angle for angle in angles 
                          if not (80 <= angle <= 100) and not (260 <= angle <= 280)]
        
        confidence = min(100, len(filtered_angles) / 100 * 80)  # Scale confidence
        
        return {
            'has_gradients': len(filtered_angles) > 0,
            'angles': filtered_angles,
            'confidence': confidence,
            'gradient_count': len(filtered_angles)
        }
    
    def _detect_shadow_regions(self, img):
        """Detect shadow regions using color and intensity analysis"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        
        # Shadow characteristics: low value, low saturation
        low_sat = cv2.inRange(hsv, (0, 0, 0), (180, 80, 160))
        low_val = cv2.inRange(hsv, (0, 0, 0), (180, 255, 120))
        
        shadow_mask = cv2.bitwise_or(low_sat, low_val)
        
        # Remove small noise
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'has_shadows': False, 'angles': [], 'confidence': 0}
        
        # Analyze large shadow regions
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
        
        if not large_contours:
            return {'has_shadows': False, 'angles': [], 'confidence': 0}
        
        angles = []
        for cnt in large_contours[:5]:  # Analyze top 5 largest shadows
            if len(cnt) > 5:  # Need enough points for ellipse fitting
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    angle = ellipse[2]  # Ellipse angle
                    angles.append(angle)
                except:
                    continue
        
        confidence = min(100, len(angles) * 20)  # Based on number of regions
        
        return {
            'has_shadows': len(angles) > 0,
            'angles': angles,
            'confidence': confidence,
            'region_count': len(angles)
        }
    
    def _analyze_edges_enhanced(self, gray):
        """Enhanced edge analysis for shadow boundaries"""
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 70, 200)
        
        combined_edges = cv2.bitwise_or(edges1, edges2)
        combined_edges = cv2.bitwise_or(combined_edges, edges3)
        
        # Find lines in combined edges
        lines = cv2.HoughLinesP(combined_edges, 1, np.pi/180, 
                               threshold=25, 
                               minLineLength=30, 
                               maxLineGap=20)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if length > 40:
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    if angle < 0:
                        angle += 360
                    
                    # Filter angles
                    if not (75 <= angle <= 105) and not (255 <= angle <= 285):
                        angles.append(angle)
        
        confidence = min(100, len(angles) * 4)
        
        return {
            'has_edges': len(angles) > 0,
            'angles': angles,
            'confidence': confidence,
            'edge_line_count': len(angles)
        }
    
    def _calculate_angle_consistency(self, angles, mean_angle):
        """Calculate how consistent the detected angles are"""
        if not angles:
            return 0
        
        # Calculate circular variance
        angles_rad = np.radians(angles)
        mean_angle_rad = np.radians(mean_angle)
        
        # Circular variance: 1 - mean resultant length
        sin_sum = np.sum(np.sin(angles_rad - mean_angle_rad))
        cos_sum = np.sum(np.cos(angles_rad - mean_angle_rad))
        
        resultant_length = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
        circular_variance = 1 - resultant_length
        
        # Convert to consistency score (0-1)
        consistency = 1 - circular_variance
        
        return max(0, min(1, consistency))
    
    def _calculate_image_quality(self, img):
        """Calculate image quality metrics for confidence adjustment"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Sharpness (variance of Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness (mean intensity)
        brightness = np.mean(gray)
        
        # Normalize metrics
        contrast_score = min(1, contrast / 50)
        sharpness_score = min(1, laplacian_var / 100)
        brightness_score = 1 - abs(brightness - 127) / 127  # Ideal around 127
        
        # Combined quality score
        quality_score = (contrast_score + sharpness_score + brightness_score) / 3
        
        return {
            'overall': quality_score,
            'contrast': contrast_score,
            'sharpness': sharpness_score,
            'brightness': brightness_score
        }
    
    def validate_shadow_detection(self, img, shadow_angle, confidence):
        """Validate shadow detection results"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Basic validation checks
        checks = {
            'sufficient_contrast': np.std(gray) > 20,
            'reasonable_angle': 0 <= shadow_angle <= 360,
            'min_confidence': confidence > 20,
            'image_size': gray.shape[0] * gray.shape[1] > 10000  # At least 100x100
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        validation_score = passed_checks / total_checks
        
        return {
            'is_valid': validation_score > 0.5,
            'validation_score': validation_score,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'details': checks
        }

# Utility function for standalone testing
def analyze_image_shadows_standalone(image_path):
    """Standalone function for testing shadow analysis"""
    analyzer = ShadowAnalyzer()
    return analyzer.analyze_image_shadows(image_path)

if __name__ == "__main__":
    # Test the shadow analyzer
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = analyze_image_shadows_standalone(image_path)
        print("Shadow Analysis Results:")
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Usage: python shadow_analysis.py <image_path>")