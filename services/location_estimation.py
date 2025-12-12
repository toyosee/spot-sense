# services/location_estimation.py - Enhanced Location Estimation
import numpy as np
from pysolar.solar import get_azimuth
import pytz
from datetime import datetime
import logging
import math

logger = logging.getLogger(__name__)

class LocationEstimator:
    def __init__(self):
        self.possible_locations = self._load_possible_locations()
        logger.info("LocationEstimator initialized")
    
    def estimate_from_shadows(self, shadow_data, capture_time, search_center=None):
        """
        Enhanced location estimation using multiple strategies
        
        Args:
            shadow_data (dict): Shadow analysis results
            capture_time (datetime): Image capture time
            search_center (tuple): (lat, lon) to search around
            
        Returns:
            tuple: (coordinates, confidence, details) or (None, 0, {}) if no match found
        """
        logger.info(f"Starting location estimation with shadow confidence: {shadow_data.get('confidence', 'unknown')}")
        
        if search_center is None:
            search_center = (0, 0)  # Default to equator/prime meridian
            logger.warning("Using default search center (0, 0)")
        
        center_lat, center_lon = search_center
        
        # Validate shadow data
        validation_result = self._validate_shadow_data(shadow_data)
        if not validation_result['is_valid']:
            logger.warning(f"Shadow data validation failed: {validation_result['reason']}")
            return None, 0, {'error': 'Invalid shadow data', 'details': validation_result}
        
        target_azimuth = shadow_data['solar_azimuth']
        shadow_confidence = shadow_data.get('confidence', 50)
        shadow_quality = shadow_data.get('shadow_quality', 'unknown')
        
        logger.info(f"Target solar azimuth: {target_azimuth:.2f}°, Shadow quality: {shadow_quality}")
        
        # Strategy 1: Local grid search around device location
        local_result = self._local_grid_search(
            target_azimuth, 
            capture_time, 
            center_lat, 
            center_lon,
            shadow_confidence,
            shadow_quality
        )
        
        # Strategy 2: Global likely locations
        global_result = self._global_location_search(
            target_azimuth,
            capture_time,
            shadow_confidence
        )
        
        # Strategy 3: Continent-based search
        continent_result = self._continent_based_search(
            target_azimuth,
            capture_time,
            center_lat,
            center_lon,
            shadow_confidence
        )
        
        # Strategy 4: Adaptive regional search
        regional_result = self._adaptive_regional_search(
            target_azimuth,
            capture_time,
            center_lat,
            center_lon,
            shadow_confidence
        )
        
        # Combine results with weighted confidence
        best_result = self._combine_results(
            [local_result, global_result, continent_result, regional_result],
            shadow_confidence,
            shadow_quality
        )
        
        if best_result:
            logger.info(f"Location estimation successful: {best_result['coordinates']} with {best_result['confidence']:.1f}% confidence")
            details = {
                'estimation_method': best_result['method'],
                'azimuth_error': best_result.get('azimuth_error', 0),
                'search_radius_km': best_result.get('search_radius_km', 0),
                'reliability': best_result.get('reliability', 'Unknown'),
                'all_methods_evaluated': [r.get('method', 'unknown') for r in [local_result, global_result, continent_result, regional_result] if r],
                'shadow_quality_used': shadow_quality
            }
            return best_result['coordinates'], best_result['confidence'], details
        else:
            logger.warning("No suitable location found from any estimation method")
            return None, 0, {'error': 'No location found', 'methods_tried': 4}
    
    def _validate_shadow_data(self, shadow_data):
        """Comprehensive shadow data validation"""
        required_fields = ['solar_azimuth', 'shadow_angle']
        
        for field in required_fields:
            if field not in shadow_data:
                return {
                    'is_valid': False,
                    'reason': f'Missing required field: {field}',
                    'missing_field': field
                }
        
        # Validate azimuth range
        azimuth = shadow_data['solar_azimuth']
        if not (0 <= azimuth <= 360):
            return {
                'is_valid': False,
                'reason': f'Invalid solar azimuth: {azimuth}',
                'invalid_value': azimuth
            }
        
        # Validate shadow angle range
        shadow_angle = shadow_data['shadow_angle']
        if not (0 <= shadow_angle <= 360):
            return {
                'is_valid': False,
                'reason': f'Invalid shadow angle: {shadow_angle}',
                'invalid_value': shadow_angle
            }
        
        # Check confidence if available
        confidence = shadow_data.get('confidence', 0)
        if confidence < 5:  # Very low confidence threshold
            return {
                'is_valid': False,
                'reason': f'Shadow confidence too low: {confidence}%',
                'low_confidence': confidence
            }
        
        # Check if shadows were actually detected
        if not shadow_data.get('has_shadows', True):
            return {
                'is_valid': False,
                'reason': 'No shadows detected in image',
                'no_shadows': True
            }
            
        return {
            'is_valid': True,
            'reason': 'All validation checks passed',
            'confidence': confidence,
            'shadow_quality': shadow_data.get('shadow_quality', 'unknown')
        }
    
    def _local_grid_search(self, target_azimuth, capture_time, center_lat, center_lon, shadow_confidence, shadow_quality):
        """Enhanced local grid search with adaptive parameters"""
        logger.debug(f"Starting local grid search around ({center_lat:.4f}, {center_lon:.4f})")
        
        best_match = None
        min_diff = float('inf')
        
        # Adaptive parameters based on shadow quality and confidence
        search_params = self._get_search_parameters(shadow_confidence, shadow_quality)
        search_radius = search_params['radius']
        step = search_params['step']
        max_points = search_params['max_points']
        
        logger.debug(f"Search parameters: radius={search_radius}°, step={step}°, max_points={max_points}")
        
        search_points = []
        
        # Generate search grid with bounds checking
        lat_range = np.arange(
            max(center_lat - search_radius, -85),
            min(center_lat + search_radius, 85),
            step
        )
        lon_range = np.arange(
            max(center_lon - search_radius, -175),
            min(center_lon + search_radius, 175),
            step
        )
        
        for lat in lat_range:
            for lon in lon_range:
                # Skip invalid coordinates
                if not self._is_valid_coordinate(lat, lon):
                    continue
                
                # Skip unlikely ocean areas for better performance
                if self._is_likely_ocean(lat, lon):
                    continue
                    
                search_points.append((lat, lon))
        
        # Limit search points for performance
        if len(search_points) > max_points:
            # Strategic sampling: prefer points closer to center
            distances = [abs(lat - center_lat) + abs(lon - center_lon) for lat, lon in search_points]
            sorted_indices = np.argsort(distances)
            search_points = [search_points[i] for i in sorted_indices[:max_points]]
            logger.debug(f"Limited search points from {len(search_points)} to {max_points}")
        
        # Search through points
        points_evaluated = 0
        for lat, lon in search_points:
            try:
                calc_azimuth = get_azimuth(lat, lon, capture_time)
                diff = self._angular_difference(calc_azimuth, target_azimuth)
                
                if diff < min_diff:
                    min_diff = diff
                    best_match = (lat, lon)
                    
                points_evaluated += 1
                    
            except Exception as e:
                logger.debug(f"Skipping coordinate ({lat:.4f}, {lon:.4f}): {e}")
                continue
        
        if best_match:
            base_confidence = max(0, 100 - (min_diff / 180 * 100))
            
            # Adjust confidence based on shadow quality and search parameters
            quality_multiplier = self._get_quality_multiplier(shadow_quality)
            adjusted_confidence = min(100, base_confidence * (shadow_confidence / 100) * quality_multiplier)
            
            logger.debug(f"Local search: best match {best_match} with error {min_diff:.2f}°, confidence {adjusted_confidence:.1f}%")
            
            return {
                'coordinates': best_match,
                'confidence': adjusted_confidence,
                'azimuth_error': min_diff,
                'method': 'local_grid_search',
                'search_radius_km': search_radius * 111,
                'points_evaluated': points_evaluated,
                'quality_multiplier': quality_multiplier
            }
        
        logger.debug("Local search: no suitable match found")
        return None
    
    def _global_location_search(self, target_azimuth, capture_time, shadow_confidence):
        """Enhanced global search with better city selection"""
        logger.debug("Starting global location search")
        
        best_match = None
        min_diff = float('inf')
        best_city = None
        
        # Enhanced city database with regional coverage
        major_cities = self._get_global_cities()
        
        for city_data in major_cities:
            lat, lon, weight, region, city_name = city_data
            
            try:
                calc_azimuth = get_azimuth(lat, lon, capture_time)
                diff = self._angular_difference(calc_azimuth, target_azimuth)
                
                # Apply weighting based on population and regional importance
                weighted_diff = diff * (1.3 - weight)  # Lower diff for more important cities
                
                if weighted_diff < min_diff:
                    min_diff = weighted_diff
                    best_match = (lat, lon)
                    best_city = city_name
                    
            except Exception as e:
                logger.debug(f"Skipping city {city_name}: {e}")
                continue
        
        if best_match:
            base_confidence = max(0, 100 - (min_diff / 180 * 100))
            # Global search has inherent uncertainty
            adjusted_confidence = base_confidence * 0.7 * (shadow_confidence / 100)
            
            logger.debug(f"Global search: best match {best_city} with error {min_diff:.2f}°, confidence {adjusted_confidence:.1f}%")
            
            return {
                'coordinates': best_match,
                'confidence': adjusted_confidence,
                'azimuth_error': min_diff,
                'method': 'global_city_search',
                'matched_city': best_city,
                'base_confidence': base_confidence
            }
        
        return None
    
    def _continent_based_search(self, target_azimuth, capture_time, center_lat, center_lon, shadow_confidence):
        """Enhanced continent-based search"""
        logger.debug("Starting continent-based search")
        
        # Determine likely regions from device location
        likely_regions = self._get_likely_regions(center_lat, center_lon)
        
        best_match = None
        min_diff = float('inf')
        best_region = None
        
        for region_name, region_centers in likely_regions.items():
            for center_name, (lat, lon) in region_centers.items():
                try:
                    calc_azimuth = get_azimuth(lat, lon, capture_time)
                    diff = self._angular_difference(calc_azimuth, target_azimuth)
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_match = (lat, lon)
                        best_region = f"{region_name}_{center_name}"
                        
                except Exception as e:
                    logger.debug(f"Skipping region {region_name}_{center_name}: {e}")
                    continue
        
        if best_match:
            base_confidence = max(0, 100 - (min_diff / 180 * 100))
            adjusted_confidence = base_confidence * 0.6 * (shadow_confidence / 100)
            
            logger.debug(f"Continent search: best region {best_region} with error {min_diff:.2f}°")
            
            return {
                'coordinates': best_match,
                'confidence': adjusted_confidence,
                'azimuth_error': min_diff,
                'method': 'continent_based_search',
                'matched_region': best_region
            }
        
        return None
    
    def _adaptive_regional_search(self, target_azimuth, capture_time, center_lat, center_lon, shadow_confidence):
        """Adaptive search that adjusts based on initial results"""
        logger.debug("Starting adaptive regional search")
        
        # First, try a quick local search
        quick_result = self._local_grid_search(
            target_azimuth, capture_time, center_lat, center_lon, 
            shadow_confidence, 'unknown'  # Don't use quality for quick search
        )
        
        if quick_result and quick_result['confidence'] > 40:
            # Good local result found
            quick_result['method'] = 'adaptive_quick_local'
            return quick_result
        
        # If local search failed, expand to regional centers
        regional_centers = self._get_regional_centers(center_lat, center_lon)
        
        best_match = None
        min_diff = float('inf')
        
        for region_name, (lat, lon) in regional_centers.items():
            try:
                calc_azimuth = get_azimuth(lat, lon, capture_time)
                diff = self._angular_difference(calc_azimuth, target_azimuth)
                
                if diff < min_diff:
                    min_diff = diff
                    best_match = (lat, lon)
                    
            except Exception:
                continue
        
        if best_match:
            base_confidence = max(0, 100 - (min_diff / 180 * 100))
            adjusted_confidence = base_confidence * 0.65 * (shadow_confidence / 100)
            
            return {
                'coordinates': best_match,
                'confidence': adjusted_confidence,
                'azimuth_error': min_diff,
                'method': 'adaptive_regional_search',
                'regional_approach': 'expanded_search'
            }
        
        return None
    
    def _get_search_parameters(self, shadow_confidence, shadow_quality):
        """Get adaptive search parameters based on data quality"""
        # Base parameters on confidence
        if shadow_confidence > 75:
            radius = 1.5
            step = 0.1
            max_points = 2000
        elif shadow_confidence > 50:
            radius = 2.5
            step = 0.2
            max_points = 1500
        elif shadow_confidence > 25:
            radius = 4.0
            step = 0.3
            max_points = 1000
        else:
            radius = 6.0
            step = 0.5
            max_points = 800
        
        # Adjust based on shadow quality
        quality_adjustments = {
            'excellent': 1.0,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.4,
            'unknown': 0.7
        }
        
        adjustment = quality_adjustments.get(shadow_quality, 0.7)
        radius *= adjustment
        step = max(0.05, step * adjustment)  # Don't go below 0.05 degree step
        
        return {
            'radius': radius,
            'step': step,
            'max_points': int(max_points * adjustment)
        }
    
    def _get_quality_multiplier(self, shadow_quality):
        """Get confidence multiplier based on shadow quality"""
        multipliers = {
            'excellent': 1.2,
            'good': 1.0,
            'fair': 0.8,
            'poor': 0.5,
            'unknown': 0.7
        }
        return multipliers.get(shadow_quality, 0.7)
    
    def _get_global_cities(self):
        """Enhanced global city database"""
        return [
            # Format: (lat, lon, weight, region, name)
            # North America
            (40.7128, -74.0060, 0.95, 'north_america', 'New York'),
            (34.0522, -118.2437, 0.90, 'north_america', 'Los Angeles'),
            (41.8781, -87.6298, 0.85, 'north_america', 'Chicago'),
            (29.7604, -95.3698, 0.80, 'north_america', 'Houston'),
            
            # Europe
            (51.5074, -0.1278, 0.95, 'europe', 'London'),
            (48.8566, 2.3522, 0.90, 'europe', 'Paris'),
            (52.5200, 13.4050, 0.90, 'europe', 'Berlin'),
            (41.9028, 12.4964, 0.85, 'europe', 'Rome'),
            
            # Asia
            (35.6762, 139.6503, 0.95, 'asia', 'Tokyo'),
            (31.2304, 121.4737, 0.92, 'asia', 'Shanghai'),
            (39.9042, 116.4074, 0.92, 'asia', 'Beijing'),
            (28.6139, 77.2090, 0.90, 'asia', 'Delhi'),
            
            # Other regions
            (-33.8688, 151.2093, 0.85, 'oceania', 'Sydney'),
            (-23.5505, -46.6333, 0.85, 'south_america', 'Sao Paulo'),
            (30.0444, 31.2357, 0.80, 'africa', 'Cairo'),
            (-26.2041, 28.0473, 0.75, 'africa', 'Johannesburg'),
        ]
    
    def _get_likely_regions(self, lat, lon):
        """Get likely regions based on coordinates"""
        regions = {}
        
        if -125 <= lon <= -65 and 25 <= lat <= 50:  # North America
            regions['north_america'] = {
                'center': (39.8283, -98.5795),
                'east': (40.7128, -74.0060),
                'west': (34.0522, -118.2437),
                'south': (29.7604, -95.3698)
            }
        elif -10 <= lon <= 40 and 35 <= lat <= 60:  # Europe
            regions['europe'] = {
                'center': (50.1109, 8.6821),
                'west': (51.5074, -0.1278),
                'east': (52.5200, 13.4050),
                'south': (41.9028, 12.4964)
            }
        elif 70 <= lon <= 140 and 10 <= lat <= 55:  # Asia
            regions['asia'] = {
                'center': (35.6762, 139.6503),
                'north': (39.9042, 116.4074),
                'south': (28.6139, 77.2090),
                'east': (31.2304, 121.4737)
            }
        else:
            # Global fallback
            regions['global'] = {
                'na': (40.7128, -74.0060),
                'eu': (51.5074, -0.1278),
                'as': (35.6762, 139.6503),
                'sa': (-23.5505, -46.6333)
            }
        
        return regions
    
    def _get_regional_centers(self, lat, lon):
        """Get regional centers for adaptive search"""
        # Simple regional division
        return {
            'north_america_east': (40.7128, -74.0060),
            'north_america_west': (34.0522, -118.2437),
            'europe_west': (51.5074, -0.1278),
            'europe_east': (52.5200, 13.4050),
            'asia_east': (35.6762, 139.6503),
            'asia_south': (28.6139, 77.2090),
            'south_america': (-23.5505, -46.6333),
            'africa': (30.0444, 31.2357),
            'oceania': (-33.8688, 151.2093)
        }
    
    def _combine_results(self, results, shadow_confidence, shadow_quality):
        """Enhanced result combination with quality weighting"""
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return None
        
        # Apply quality-based weighting
        for result in valid_results:
            method = result['method']
            
            # Method-specific base weights
            method_weights = {
                'local_grid_search': 1.0,
                'adaptive_quick_local': 1.2,
                'adaptive_regional_search': 0.9,
                'global_city_search': 0.7,
                'continent_based_search': 0.6
            }
            
            base_weight = method_weights.get(method, 0.8)
            quality_multiplier = self._get_quality_multiplier(shadow_quality)
            
            # Adjust confidence with method weight and quality
            result['weighted_confidence'] = result['confidence'] * base_weight * quality_multiplier
        
        # Select best result
        best_result = max(valid_results, key=lambda x: x['weighted_confidence'])
        
        # Final confidence adjustment
        final_confidence = min(100, best_result['weighted_confidence'])
        
        # Reliability rating
        if final_confidence > 75:
            reliability = 'High'
        elif final_confidence > 50:
            reliability = 'Medium'
        elif final_confidence > 25:
            reliability = 'Low'
        else:
            reliability = 'Very Low'
        
        best_result['confidence'] = final_confidence
        best_result['reliability'] = reliability
        best_result['combined_from'] = len(valid_results)
        
        logger.info(f"Combined {len(valid_results)} results, selected {best_result['method']} with {final_confidence:.1f}% confidence")
        
        return best_result
    
    def _angular_difference(self, angle1, angle2):
        """Calculate minimum difference between two angles (0-180 degrees)"""
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)
    
    def _is_valid_coordinate(self, lat, lon):
        """Validate coordinate validity"""
        return (-90 <= lat <= 90) and (-180 <= lon <= 180)
    
    def _is_likely_ocean(self, lat, lon):
        """Simple ocean detection (placeholder for more sophisticated method)"""
        # Very basic implementation - in production, use proper geospatial data
        major_land_areas = [
            # North America
            ((-168, 15), (-52, 72)),
            # South America
            ((-82, -56), (-34, 13)),
            # Europe
            ((-25, 36), (40, 72)),
            # Asia
            ((25, -10), (180, 72)),
            # Africa
            ((-18, -35), (52, 38)),
            # Australia
            ((112, -44), (154, -10))
        ]
        
        for (lon_min, lat_min), (lon_max, lat_max) in major_land_areas:
            if (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max):
                return False
        
        return True
    
    def _load_possible_locations(self):
        """Load database of possible locations"""
        return [
            (40.7128, -74.0060, 1.0),
            (51.5074, -0.1278, 1.0),
            (35.6762, 139.6503, 1.0),
        ]

# Utility functions
def validate_coordinates(lat, lon):
    """Validate latitude and longitude values"""
    return (-90 <= lat <= 90) and (-180 <= lon <= 180)

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in kilometers"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

if __name__ == "__main__":
    # Test the enhanced location estimator
    estimator = LocationEstimator()
    
    # Test data
    test_shadow_data = {
        'solar_azimuth': 180,
        'shadow_angle': 0,
        'confidence': 75,
        'shadow_quality': 'good',
        'has_shadows': True
    }
    
    test_time = datetime.now(pytz.UTC)
    
    coords, confidence, details = estimator.estimate_from_shadows(
        test_shadow_data, test_time, (40.0, -74.0)
    )
    
    if coords:
        print(f"📍 Estimated location: {coords}")
        print(f"🎯 Confidence: {confidence:.1f}%")
        print(f"🔧 Method: {details.get('estimation_method', 'unknown')}")
        print(f"📊 Reliability: {details.get('reliability', 'unknown')}")
    else:
        print("❌ Could not estimate location")
        print(f"Details: {details}")