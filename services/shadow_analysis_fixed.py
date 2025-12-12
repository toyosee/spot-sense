# services/shadow_analysis_fixed.py
import cv2
import numpy as np
import math
from datetime import datetime
from pysolar.solar import get_azimuth, get_altitude
import pytz
import logging

from services.location_diagnostics import LocationDiagnostics

logger = logging.getLogger(__name__)

class ShadowAnalyzerFixed:
    def __init__(self):
        self.DEFAULT_TZ = pytz.timezone('Africa/Lagos')
        logger.info("ShadowAnalyzerFixed initialized")
    
    def analyze_image_shadows(self, image_path, capture_time=None):
        """Fixed shadow analysis with validation"""
        if capture_time is None:
            capture_time = datetime.now(self.DEFAULT_TZ)
        
        logger.info(f"🔄 Starting shadow analysis for: {image_path}")
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image", "reason": "File not found or corrupted"}
            
            # Get image dimensions
            height, width = img.shape[:2]
            logger.info(f"📐 Image size: {width}x{height}")
            
            # Enhanced shadow detection
            shadow_analysis = self._robust_shadow_detection(img)
            
            if not shadow_analysis['has_shadows']:
                logger.warning(f"❌ No shadows detected: {shadow_analysis['reason']}")
                return {
                    "error": "No reliable shadows detected", 
                    "reason": shadow_analysis['reason'],
                    "confidence": 0,
                    "has_shadows": False
                }
            
            shadow_angle = shadow_analysis['shadow_angle']
            solar_azimuth = (shadow_angle + 180) % 360
            
            logger.info(f"✅ Shadow analysis successful:")
            logger.info(f"   🌑 Detected shadow angle: {shadow_angle:.2f}°")
            logger.info(f"   ☀️ Calculated solar azimuth: {solar_azimuth:.2f}°")
            logger.info(f"   📊 Confidence: {shadow_analysis['confidence']:.1f}%")
            logger.info(f"   🎯 Quality: {shadow_analysis['shadow_quality']}")
            
            return {
                'shadow_angle': shadow_angle,
                'solar_azimuth': solar_azimuth,
                'capture_time': capture_time,
                'confidence': shadow_analysis['confidence'],
                'shadow_quality': shadow_analysis['shadow_quality'],
                'has_shadows': True,
                'analysis_methods': shadow_analysis['methods_used'],
                'image_characteristics': {
                    'size': f"{width}x{height}",
                    'detected_features': shadow_analysis['detected_angles_count']
                }
            }
            
        except Exception as e:
            error_msg = f"Shadow analysis failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "reason": "Unexpected analysis error"}
    
    def _robust_shadow_detection(self, img):
        """More robust shadow detection focusing on Nigeria-specific conditions"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Nigeria-specific: Expect brighter images with strong shadows
        avg_brightness = np.mean(gray)
        logger.info(f"💡 Average brightness: {avg_brightness:.1f}")
        
        # Adjust detection parameters based on image characteristics
        if avg_brightness < 80:
            logger.warning("🌙 Image appears dark - may affect shadow detection")
        
        results = {
            'has_shadows': False,
            'reason': 'No shadows detected',
            'confidence': 0,
            'shadow_quality': 'poor',
            'shadow_angle': None,
            'methods_used': [],
            'detected_angles_count': 0
        }
        
        # Method 1: Primary shadow detection using edge-based approach
        primary_angles = self._detect_primary_shadows(gray)
        
        # Method 2: Secondary validation using texture
        secondary_angles = self._validate_with_texture(gray)
        
        # Method 3: Structural analysis for confirmation
        structural_angles = self._structural_analysis(img)
        
        all_angles = primary_angles + secondary_angles + structural_angles
        
        if not all_angles:
            results['reason'] = 'No shadow features detected by any method'
            return results
        
        # Use circular statistics with outlier removal
        filtered_angles = self._remove_angle_outliers(all_angles)
        
        if not filtered_angles:
            results['reason'] = 'All detected angles were outliers'
            return results
        
        # Calculate robust mean
        mean_angle = self._circular_mean(filtered_angles)
        
        # Calculate consistency
        consistency = self._calculate_consistency(filtered_angles, mean_angle)
        
        # Confidence calculation
        base_confidence = min(80, len(filtered_angles) * 3)  # Cap at 80% for detection-based
        final_confidence = base_confidence * consistency
        
        # Quality assessment
        if final_confidence > 60:
            quality = 'good'
        elif final_confidence > 40:
            quality = 'fair'
        else:
            quality = 'poor'
        
        results.update({
            'has_shadows': True,
            'shadow_angle': mean_angle,
            'confidence': final_confidence,
            'shadow_quality': quality,
            'methods_used': ['primary', 'texture', 'structural'],
            'detected_angles_count': len(filtered_angles),
            'consistency_score': consistency,
            'reason': f'Shadows detected with {quality} quality'
        })
        
        logger.info(f"📊 Final shadow analysis: {len(filtered_angles)} angles, consistency: {consistency:.2f}")
        
        return results
    
    def _detect_primary_shadows(self, gray):
        """Primary shadow detection focusing on clear edge features"""
        # Enhanced preprocessing for Nigerian conditions
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)
        
        # Adaptive edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Detect lines with Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=40,  # Higher threshold for cleaner lines
                               minLineLength=60, 
                               maxLineGap=15)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Only consider substantial lines
                if length > 80:
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    # Normalize to 0-360
                    if angle < 0:
                        angle += 360
                    
                    # Filter for reasonable shadow angles (not vertical/horizontal)
                    if not self._is_vertical_or_horizontal(angle):
                        angles.append(angle)
        
        logger.info(f"📏 Primary detection: {len(angles)} shadow lines found")
        return angles
    
    def _validate_with_texture(self, gray):
        """Validate shadows using texture analysis"""
        # Calculate gradient magnitude and orientation
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orientation = np.arctan2(sobely, sobelx)
        
        # Focus on strong gradients (likely shadow boundaries)
        threshold = np.percentile(magnitude, 70)
        strong_gradients = magnitude > threshold
        
        # Get angles from strong gradients
        strong_angles = orientation[strong_gradients]
        
        # Convert to degrees and filter
        angles_deg = np.degrees(strong_angles)
        filtered_angles = []
        
        for angle in angles_deg:
            # Normalize to 0-360
            normalized_angle = angle % 360
            if not self._is_vertical_or_horizontal(normalized_angle):
                filtered_angles.append(normalized_angle)
        
        logger.info(f"🎨 Texture validation: {len(filtered_angles)} gradient angles")
        return filtered_angles
    
    def _structural_analysis(self, img):
        """Structural analysis for shadow regions"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Shadow regions typically have lower value and saturation
        shadow_mask = cv2.inRange(hsv, (0, 0, 0), (180, 80, 120))
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angles = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:  # Only substantial regions
                if len(cnt) > 5:
                    try:
                        ellipse = cv2.fitEllipse(cnt)
                        angle = ellipse[2]
                        if not self._is_vertical_or_horizontal(angle):
                            angles.append(angle)
                    except:
                        continue
        
        logger.info(f"🏗️ Structural analysis: {len(angles)} region angles")
        return angles
    
    def _is_vertical_or_horizontal(self, angle, tolerance=15):
        """Check if angle is near vertical or horizontal"""
        angle = angle % 180  # Normalize to 0-180 for line direction
        return (angle < tolerance or angle > 180 - tolerance or 
                abs(angle - 90) < tolerance)
    
    def _remove_angle_outliers(self, angles, max_deviation=45):
        """Remove angles that deviate too much from the main cluster"""
        if not angles:
            return []
        
        # Convert to circular data
        angles_rad = np.radians(angles)
        
        # Calculate circular mean
        mean_angle = np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
        
        # Calculate deviations
        deviations = []
        for angle in angles_rad:
            diff = abs(angle - mean_angle)
            diff = min(diff, 2*np.pi - diff)  # Circular difference
            deviations.append(np.degrees(diff))
        
        # Filter angles within tolerance
        filtered = [angles[i] for i, dev in enumerate(deviations) if dev <= max_deviation]
        
        logger.info(f"📊 Outlier removal: {len(angles)} → {len(filtered)} angles")
        return filtered
    
    def _circular_mean(self, angles):
        """Calculate circular mean of angles"""
        angles_rad = np.radians(angles)
        mean_angle = np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
        return np.degrees(mean_angle) % 360
    
    def _calculate_consistency(self, angles, mean_angle):
        """Calculate how consistent the angles are"""
        if len(angles) < 2:
            return 0.5
        
        angles_rad = np.radians(angles)
        mean_rad = np.radians(mean_angle)
        
        # Circular variance
        sin_sum = np.sum(np.sin(angles_rad - mean_rad))
        cos_sum = np.sum(np.cos(angles_rad - mean_rad))
        resultant_length = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
        
        return resultant_length  # 1 = perfect consistency, 0 = no consistency
