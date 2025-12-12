# services/location_diagnostics.py
import logging
from pysolar.solar import get_azimuth, get_altitude
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

class LocationDiagnostics:
    def __init__(self):
        self.africa_tz = pytz.timezone('Africa/Lagos')
    
    def diagnose_shadow_analysis(self, image_path, actual_lat, actual_lon, capture_time=None):
        """Diagnose why shadow analysis is failing"""
        if capture_time is None:
            capture_time = datetime.now(self.africa_tz)
        
        logger.info(f"🔍 DIAGNOSING SHADOW ANALYSIS")
        logger.info(f"Actual location: Lat: {actual_lat}, Lon: {actual_lon}")
        logger.info(f"Capture time: {capture_time}")
        
        # Calculate what the solar azimuth SHOULD be at the actual location
        try:
            actual_azimuth = get_azimuth(actual_lat, actual_lon, capture_time)
            actual_altitude = get_altitude(actual_lat, actual_lon, capture_time)
            
            logger.info(f"☀️ ACTUAL solar azimuth at location: {actual_azimuth:.2f}°")
            logger.info(f"📏 ACTUAL solar altitude: {actual_altitude:.2f}°")
            
            # What shadow angle would this produce?
            expected_shadow_angle = (actual_azimuth - 180) % 360
            logger.info(f"🌑 EXPECTED shadow angle: {expected_shadow_angle:.2f}°")
            
            return {
                'actual_azimuth': actual_azimuth,
                'actual_altitude': actual_altitude,
                'expected_shadow_angle': expected_shadow_angle,
                'is_daytime': actual_altitude > 0
            }
            
        except Exception as e:
            logger.error(f"Diagnostics failed: {e}")
            return None
    
    def test_nigerian_locations(self, capture_time):
        """Test solar positions across Nigeria"""
        nigerian_cities = {
            'Kano': (12.002, 8.592),
            'Lagos': (6.524, 3.379),
            'Abuja': (9.076, 7.398),
            'Port Harcourt': (4.815, 7.050),
            'Ibadan': (7.378, 3.947),
            'Ondo': (7.093, 4.835),
            'Maiduguri': (11.833, 13.150),
            'Kaduna': (10.523, 7.440)
        }
        
        logger.info("🇳🇬 TESTING NIGERIAN LOCATIONS")
        results = {}
        
        for city, (lat, lon) in nigerian_cities.items():
            try:
                azimuth = get_azimuth(lat, lon, capture_time)
                altitude = get_altitude(lat, lon, capture_time)
                results[city] = {
                    'azimuth': azimuth,
                    'altitude': altitude,
                    'shadow_angle': (azimuth - 180) % 360
                }
                logger.info(f"📍 {city}: Azimuth {azimuth:6.1f}° | Altitude {altitude:5.1f}° | Shadow {(azimuth-180)%360:6.1f}°")
            except Exception as e:
                logger.error(f"Failed for {city}: {e}")

        return results
    