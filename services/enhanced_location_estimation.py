# services/enhanced_location_estimation.py
import numpy as np
from pysolar.solar import get_azimuth
import pytz
from datetime import datetime
import logging
from services.terrain_classification import TerrainClassifier
from services.map_visualization import MapVisualizer

logger = logging.getLogger(__name__)

class EnhancedLocationEstimator:
    def __init__(self):
        self.africa_tz = pytz.timezone('Africa/Lagos')
        self.terrain_classifier = TerrainClassifier()
        self.map_visualizer = MapVisualizer()
        self.nigerian_regions = self._load_nigerian_regions()
        logger.info("EnhancedLocationEstimator initialized")
    
    def estimate_location_with_terrain(self, image_path, shadow_data, capture_time=None, actual_location=None):
        """
        Enhanced location estimation using terrain classification
        """
        if capture_time is None:
            capture_time = datetime.now(self.africa_tz)
        
        logger.info("🎯 Starting enhanced location estimation with terrain analysis")
        
        # Step 1: Classify terrain
        terrain_results = self.terrain_classifier.classify_terrain(image_path)
        
        # Step 2: Get terrain-constrained location estimates
        location_estimates = self._get_terrain_constrained_estimates(
            shadow_data, capture_time, terrain_results
        )
        
        # Step 3: Create visualization
        map_file = self._create_visualization(
            location_estimates, actual_location, terrain_results
        )
        
        # Step 4: Return comprehensive results
        return {
            'terrain_analysis': terrain_results,
            'location_estimates': location_estimates,
            'visualization': map_file,
            'recommendations': self._generate_recommendations(terrain_results, location_estimates)
        }
    
    def _get_terrain_constrained_estimates(self, shadow_data, capture_time, terrain_results):
        """Get location estimates constrained by terrain type"""
        target_azimuth = shadow_data['solar_azimuth']
        shadow_confidence = shadow_data.get('confidence', 50)
        terrain_type = terrain_results.get('dominant_terrain', 'unknown')
        
        logger.info(f"🌄 Using terrain constraints: {terrain_type}")
        
        # Get terrain-appropriate regions
        terrain_constraints = self.terrain_classifier.get_terrain_constraints(terrain_type)
        constrained_regions = terrain_constraints.get('regions', ['all'])
        
        estimates = []
        
        # Search in terrain-appropriate regions
        for region in constrained_regions:
            region_estimates = self._search_in_region(
                region, target_azimuth, capture_time, shadow_confidence
            )
            estimates.extend(region_estimates)
        
        # Apply terrain-based confidence adjustments
        estimates = self._apply_terrain_confidence(estimates, terrain_type, terrain_results)
        
        return sorted(estimates, key=lambda x: x[1], reverse=True)  # Sort by confidence
    
    def _search_in_region(self, region, target_azimuth, capture_time, shadow_confidence):
        """Search for locations in specific region"""
        estimates = []
        
        if region == 'all' or region == 'Northern Nigeria':
            # Search northern regions
            estimates.extend(self._search_northern_nigeria(target_azimuth, capture_time, shadow_confidence))
        
        if region == 'all' or region in ['Lagos', 'Delta', 'Rivers', 'Bayelsa']:
            # Search coastal regions
            estimates.extend(self._search_coastal_nigeria(target_azimuth, capture_time, shadow_confidence))
        
        if region == 'all' or region in ['Benue', 'Plateau', 'Kaduna']:
            # Search central regions
            estimates.extend(self._search_central_nigeria(target_azimuth, capture_time, shadow_confidence))
        
        if region == 'all' or region in ['Cross River', 'Ondo', 'Ogun']:
            # Search forest regions
            estimates.extend(self._search_forest_regions(target_azimuth, capture_time, shadow_confidence))
        
        return estimates
    
    def _search_northern_nigeria(self, target_azimuth, capture_time, shadow_confidence):
        """Search in Northern Nigeria regions"""
        northern_cities = {
            'Kano': (12.002, 8.592),
            'Kaduna': (10.523, 7.440),
            'Sokoto': (13.007, 5.247),
            'Maiduguri': (11.833, 13.150),
            'Katsina': (12.999, 7.600),
            'Zaria': (11.072, 7.710)
        }
        
        return self._search_cities(northern_cities, target_azimuth, capture_time, shadow_confidence, 'Northern Nigeria')
    
    def _search_coastal_nigeria(self, target_azimuth, capture_time, shadow_confidence):
        """Search in Coastal Nigeria regions"""
        coastal_cities = {
            'Lagos': (6.524, 3.379),
            'Port Harcourt': (4.815, 7.050),
            'Calabar': (4.952, 8.322),
            'Warri': (5.517, 5.750),
            'Yenagoa': (4.927, 6.267)
        }
        
        return self._search_cities(coastal_cities, target_azimuth, capture_time, shadow_confidence, 'Coastal')
    
    def _search_central_nigeria(self, target_azimuth, capture_time, shadow_confidence):
        """Search in Central Nigeria regions"""
        central_cities = {
            'Abuja': (9.076, 7.398),
            'Jos': (9.917, 8.900),
            'Minna': (9.614, 6.547),
            'Lokoja': (7.802, 6.743)
        }
        
        return self._search_cities(central_cities, target_azimuth, capture_time, shadow_confidence, 'Central')
    
    def _search_forest_regions(self, target_azimuth, capture_time, shadow_confidence):
        """Search in Forest regions"""
        forest_cities = {
            'Benin City': (6.335, 5.627),
            'Akure': (7.257, 5.205),
            'Owo': (7.183, 5.583),
            'Ibadan': (7.378, 3.947)
        }
        
        return self._search_cities(forest_cities, target_azimuth, capture_time, shadow_confidence, 'Forest')
    
    def _search_cities(self, cities, target_azimuth, capture_time, shadow_confidence, region):
        """Search specific cities for location matches"""
        estimates = []
        
        for city, (lat, lon) in cities.items():
            try:
                calc_azimuth = get_azimuth(lat, lon, capture_time)
                diff = self._angular_difference(calc_azimuth, target_azimuth)
                
                # Base confidence from azimuth match
                base_confidence = max(0, 100 - (diff / 180 * 100))
                
                # Apply shadow confidence
                final_confidence = base_confidence * (shadow_confidence / 100)
                
                if final_confidence > 20:  # Only include reasonable matches
                    estimates.append((
                        (lat, lon),
                        final_confidence,
                        {
                            'estimation_method': 'city_match',
                            'azimuth_error': diff,
                            'region': region,
                            'matched_city': city,
                            'reliability': 'High' if final_confidence > 70 else 'Medium' if final_confidence > 50 else 'Low'
                        }
                    ))
                    
            except Exception as e:
                logger.debug(f"Failed to calculate azimuth for {city}: {e}")
        
        return estimates
    
    def _apply_terrain_confidence(self, estimates, terrain_type, terrain_results):
        """Apply terrain-based confidence adjustments"""
        terrain_confidence = terrain_results.get('confidence', 50)
        
        adjusted_estimates = []
        for (coords, confidence, details) in estimates:
            region = details.get('region', '')
            
            # Boost confidence if region matches terrain
            if self._region_matches_terrain(region, terrain_type):
                confidence *= 1.2  # 20% boost
                details['terrain_match'] = 'Excellent'
            elif self._region_compatible_with_terrain(region, terrain_type):
                confidence *= 1.1  # 10% boost
                details['terrain_match'] = 'Good'
            else:
                confidence *= 0.8  # 20% penalty
                details['terrain_match'] = 'Poor'
            
            # Cap confidence at 100%
            confidence = min(100, confidence)
            
            adjusted_estimates.append((coords, confidence, details))
        
        return adjusted_estimates
    
    def _region_matches_terrain(self, region, terrain_type):
        """Check if region perfectly matches terrain type"""
        perfect_matches = {
            'urban': ['Lagos', 'Abuja', 'Kano', 'Port Harcourt'],
            'coastal': ['Coastal'],
            'forest': ['Forest'],
            'savanna': ['Northern Nigeria'],
            'agricultural': ['Central', 'Northern Nigeria']
        }
        
        return region in perfect_matches.get(terrain_type, [])
    
    def _region_compatible_with_terrain(self, region, terrain_type):
        """Check if region is compatible with terrain type"""
        compatible_matches = {
            'urban': ['Central'],
            'rural': ['all'],
            'forest': ['Central'],
            'agricultural': ['all'],
            'savanna': ['Central']
        }
        
        return region in compatible_matches.get(terrain_type, []) or 'all' in compatible_matches.get(terrain_type, [])
    
    def _create_visualization(self, location_estimates, actual_location, terrain_results):
        """Create comprehensive visualization"""
        try:
            map_obj = self.map_visualizer.create_location_map(
                location_estimates, 
                actual_location,
                terrain_results.get('dominant_terrain')
            )
            
            if map_obj:
                filename = f"enhanced_location_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                return self.map_visualizer.save_map(map_obj, filename)
        
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
        
        return None
    
    def _generate_recommendations(self, terrain_results, location_estimates):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        terrain_type = terrain_results.get('dominant_terrain', 'unknown')
        terrain_confidence = terrain_results.get('confidence', 0)
        
        if terrain_confidence > 70:
            recommendations.append(f"High confidence in {terrain_type} terrain classification")
        elif terrain_confidence > 50:
            recommendations.append(f"Moderate confidence in {terrain_type} terrain")
        else:
            recommendations.append("Low terrain classification confidence - consider manual verification")
        
        # Analyze location estimates
        if location_estimates:
            best_estimate = location_estimates[0]
            best_confidence = best_estimate[1]
            
            if best_confidence > 70:
                recommendations.append("High confidence location estimate obtained")
            elif best_confidence > 50:
                recommendations.append("Moderate confidence location estimate")
            else:
                recommendations.append("Low confidence in location estimates - consider alternative methods")
            
            # Check for consistency
            if len(location_estimates) > 1:
                confidence_range = location_estimates[0][1] - location_estimates[-1][1]
                if confidence_range < 20:
                    recommendations.append("Multiple locations with similar confidence - manual review recommended")
        
        return recommendations
    
    def _angular_difference(self, angle1, angle2):
        """Calculate minimum difference between two angles"""
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)
    
    def _load_nigerian_regions(self):
        """Load Nigerian regional data"""
        return {
            'Northern Nigeria': {'min_lat': 10, 'max_lat': 14, 'min_lon': 3, 'max_lon': 15},
            'Central Nigeria': {'min_lat': 7, 'max_lat': 10, 'min_lon': 4, 'max_lon': 10},
            'Coastal Nigeria': {'min_lat': 4, 'max_lat': 7, 'min_lon': 2, 'max_lon': 9},
            'Forest Zones': {'min_lat': 5, 'max_lat': 8, 'min_lon': 4, 'max_lon': 9}
        }