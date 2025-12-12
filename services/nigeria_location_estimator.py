# services/nigeria_location_estimator.py
import numpy as np
from pysolar.solar import get_azimuth
import pytz
from datetime import datetime
import logging
import math

logger = logging.getLogger(__name__)

class NigeriaLocationEstimator:
    def __init__(self):
        self.africa_tz = pytz.timezone('Africa/Lagos')
        self.nigerian_regions = self._load_nigerian_regions()
        logger.info("NigeriaLocationEstimator initialized")
    
    def estimate_from_shadows(self, shadow_data, capture_time, search_center=None):
        """Nigeria-focused location estimation"""
        logger.info("🇳🇬 Using Nigeria-focused location estimation")
        
        # Default to central Nigeria if no search center
        if search_center is None:
            search_center = (9.0820, 8.6753)  # Central Nigeria
            logger.info(f"Using default search center: {search_center}")
        
        # Validate shadow data
        if not self._validate_for_nigeria(shadow_data, capture_time):
            return None, 0, {'error': 'Invalid data for Nigeria estimation'}
        
        target_azimuth = shadow_data['solar_azimuth']
        shadow_confidence = shadow_data.get('confidence', 50)
        
        logger.info(f"🎯 Target solar azimuth: {target_azimuth:.2f}°")
        
        # Strategy 1: Nigeria-specific grid search
        nigeria_result = self._nigeria_grid_search(target_azimuth, capture_time, search_center, shadow_confidence)
        
        # Strategy 2: Nigerian state capitals
        states_result = self._nigerian_states_search(target_azimuth, capture_time, shadow_confidence)
        
        # Strategy 3: Regional centers
        regional_result = self._regional_centers_search(target_azimuth, capture_time, shadow_confidence)
        
        # Combine results
        best_result = self._select_best_nigerian_result(
            [nigeria_result, states_result, regional_result], 
            shadow_confidence
        )
        
        if best_result:
            logger.info(f"✅ Location estimated: {best_result['coordinates']} with {best_result['confidence']:.1f}% confidence")
            details = {
                'estimation_method': best_result['method'],
                'azimuth_error': best_result.get('azimuth_error', 0),
                'region': best_result.get('region', 'Unknown'),
                'reliability': best_result.get('reliability', 'Unknown')
            }
            return best_result['coordinates'], best_result['confidence'], details
        
        logger.warning("❌ No suitable location found in Nigeria")
        return None, 0, {'error': 'No location found in Nigeria'}
    
    def _validate_for_nigeria(self, shadow_data, capture_time):
        """Validate if the data makes sense for Nigeria"""
        if 'solar_azimuth' not in shadow_data:
            return False
        
        azimuth = shadow_data['solar_azimuth']
        
        # Check if solar azimuth is reasonable for Nigeria
        # Nigeria is between 4°N and 14°N latitude
        # Solar azimuth should generally be between 60° and 300° for daytime
        if not (60 <= azimuth <= 300):
            logger.warning(f"⚠️ Unusual solar azimuth for Nigeria: {azimuth:.1f}°")
            return False
        
        # Check capture time
        if capture_time.tzinfo is None:
            capture_time = self.africa_tz.localize(capture_time)
        
        # Ensure it's daytime in Nigeria
        hour = capture_time.hour
        if not (6 <= hour <= 18):  # Rough daytime hours
            logger.warning(f"⚠️ Unusual capture time for Nigeria: {hour}:00")
        
        return True
    
    def _nigeria_grid_search(self, target_azimuth, capture_time, search_center, shadow_confidence):
        """Search within Nigeria's boundaries"""
        logger.info("🔍 Searching within Nigeria boundaries")
        
        # Nigeria boundaries (approximate)
        nigeria_bounds = {
            'min_lat': 4.0, 'max_lat': 14.0,
            'min_lon': 2.7, 'max_lon': 14.7
        }
        
        best_match = None
        min_diff = float('inf')
        
        # Adaptive grid resolution based on confidence
        if shadow_confidence > 70:
            lat_step, lon_step = 0.2, 0.2
        elif shadow_confidence > 40:
            lat_step, lon_step = 0.5, 0.5
        else:
            lat_step, lon_step = 1.0, 1.0
        
        points_evaluated = 0
        
        for lat in np.arange(nigeria_bounds['min_lat'], nigeria_bounds['max_lat'], lat_step):
            for lon in np.arange(nigeria_bounds['min_lon'], nigeria_bounds['max_lon'], lon_step):
                try:
                    calc_azimuth = get_azimuth(lat, lon, capture_time)
                    diff = self._angular_difference(calc_azimuth, target_azimuth)
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_match = (lat, lon)
                    
                    points_evaluated += 1
                except:
                    continue
        
        if best_match:
            confidence = max(0, 100 - (min_diff / 180 * 100))
            # Adjust for Nigeria-specific confidence
            nigeria_confidence = confidence * (shadow_confidence / 100) * 0.9
            
            # Determine region
            region = self._get_nigerian_region(best_match[0], best_match[1])
            
            logger.info(f"📍 Nigeria grid: {best_match} in {region}, error: {min_diff:.2f}°")
            
            return {
                'coordinates': best_match,
                'confidence': nigeria_confidence,
                'azimuth_error': min_diff,
                'method': 'nigeria_grid_search',
                'region': region,
                'points_evaluated': points_evaluated
            }
        
        return None
    
    def _nigerian_states_search(self, target_azimuth, capture_time, shadow_confidence):
        """Search Nigerian state capitals"""
        logger.info("🏛️ Searching Nigerian state capitals")
        
        state_capitals = {
            'Kano': (12.002, 8.592),
            'Lagos': (6.524, 3.379),
            'Abuja': (9.076, 7.398),
            'Kaduna': (10.523, 7.440),
            'Port Harcourt': (4.815, 7.050),
            'Ibadan': (7.378, 3.947),
            'Ondo': (7.093, 4.835),
            'Maiduguri': (11.833, 13.150),
            'Enugu': (6.452, 7.510),
            'Sokoto': (13.007, 5.247),
            'Calabar': (4.952, 8.322),
            'Benin City': (6.335, 5.627),
            'Jos': (9.917, 8.900),
            'Yola': (9.203, 12.495)
        }
        
        best_match = None
        min_diff = float('inf')
        best_city = None
        
        for city, (lat, lon) in state_capitals.items():
            try:
                calc_azimuth = get_azimuth(lat, lon, capture_time)
                diff = self._angular_difference(calc_azimuth, target_azimuth)
                
                if diff < min_diff:
                    min_diff = diff
                    best_match = (lat, lon)
                    best_city = city
            except:
                continue
        
        if best_match:
            confidence = max(0, 100 - (min_diff / 180 * 100))
            state_confidence = confidence * 0.8 * (shadow_confidence / 100)
            
            logger.info(f"🏛️ State capital match: {best_city}, error: {min_diff:.2f}°")
            
            return {
                'coordinates': best_match,
                'confidence': state_confidence,
                'azimuth_error': min_diff,
                'method': 'nigerian_states_search',
                'matched_city': best_city,
                'region': self._get_nigerian_region(best_match[0], best_match[1])
            }
        
        return None
    
    def _regional_centers_search(self, target_azimuth, capture_time, shadow_confidence):
        """Search regional centers within Nigeria"""
        regional_centers = {
            'north_west': (12.000, 5.000),
            'north_east': (11.000, 13.000),
            'north_central': (9.500, 8.000),
            'south_west': (7.000, 4.000),
            'south_east': (5.500, 7.500),
            'south_south': (5.000, 6.000)
        }
        
        best_match = None
        min_diff = float('inf')
        best_region = None
        
        for region, (lat, lon) in regional_centers.items():
            try:
                calc_azimuth = get_azimuth(lat, lon, capture_time)
                diff = self._angular_difference(calc_azimuth, target_azimuth)
                
                if diff < min_diff:
                    min_diff = diff
                    best_match = (lat, lon)
                    best_region = region
            except:
                continue
        
        if best_match:
            confidence = max(0, 100 - (min_diff / 180 * 100))
            regional_confidence = confidence * 0.7 * (shadow_confidence / 100)
            
            return {
                'coordinates': best_match,
                'confidence': regional_confidence,
                'azimuth_error': min_diff,
                'method': 'regional_centers_search',
                'region': best_region
            }
        
        return None
    
    def _select_best_nigerian_result(self, results, shadow_confidence):
        """Select the best result from Nigerian searches"""
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return None
        
        # Prefer results with smaller azimuth errors
        for result in valid_results:
            error = result['azimuth_error']
            # Heavy penalty for large errors
            if error > 45:
                result['confidence'] *= 0.3
            elif error > 30:
                result['confidence'] *= 0.6
            elif error > 15:
                result['confidence'] *= 0.8
        
        best_result = max(valid_results, key=lambda x: x['confidence'])
        
        # Final reliability assessment
        if best_result['confidence'] > 70:
            best_result['reliability'] = 'High'
        elif best_result['confidence'] > 50:
            best_result['reliability'] = 'Medium'
        else:
            best_result['reliability'] = 'Low'
        
        logger.info(f"🏆 Best Nigerian result: {best_result['method']} with {best_result['confidence']:.1f}% confidence")
        
        return best_result
    
    def _get_nigerian_region(self, lat, lon):
        """Determine Nigerian region from coordinates"""
        if lat > 10:
            if lon < 9:
                return 'North West'
            else:
                return 'North East'
        elif lat > 7:
            return 'North Central'
        elif lon < 6:
            return 'South West'
        elif lon < 8:
            return 'South South'
        else:
            return 'South East'
    
    def _angular_difference(self, angle1, angle2):
        """Calculate minimum difference between two angles"""
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)
    
    def _load_nigerian_regions(self):
        """Load Nigerian regional data"""
        return {
            'North West': {'min_lat': 11, 'max_lat': 14, 'min_lon': 3, 'max_lon': 9},
            'North East': {'min_lat': 10, 'max_lat': 14, 'min_lon': 9, 'max_lon': 15},
            'North Central': {'min_lat': 7, 'max_lat': 10, 'min_lon': 4, 'max_lon': 10},
            'South West': {'min_lat': 4, 'max_lat': 7, 'min_lon': 2, 'max_lon': 6},
            'South East': {'min_lat': 4, 'max_lat': 7, 'min_lon': 7, 'max_lon': 10},
            'South South': {'min_lat': 4, 'max_lat': 7, 'min_lon': 6, 'max_lon': 8}
        }