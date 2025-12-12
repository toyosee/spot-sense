# services/reverse_geocoding.py - Enhanced with Detailed Location Information
import requests
import os
from dotenv import load_dotenv
import logging

load_dotenv()

GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY", "sample_key")
logger = logging.getLogger(__name__)

def reverse_geocode(lat, lon, detailed=True):
    """
    Enhanced reverse geocoding with detailed administrative information
    
    Args:
        lat (float): Latitude
        lon (float): Longitude  
        detailed (bool): Whether to return detailed address information
    
    Returns:
        dict or str: Detailed location information or formatted address
    """
    url = f"https://api.geoapify.com/v1/geocode/reverse?lat={lat}&lon={lon}&apiKey={GEOAPIFY_API_KEY}"
    
    try:
        logger.info(f"Reverse geocoding coordinates: ({lat:.6f}, {lon:.6f})")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('features'):
            logger.warning("No location features found in geocoding response")
            return "Unknown location" if not detailed else _create_unknown_location()
        
        properties = data['features'][0]['properties']
        
        if detailed:
            # Return structured detailed location information
            location_info = _extract_detailed_location(properties, lat, lon)
            logger.info(f"Reverse geocoding successful: {location_info.get('formatted', 'Unknown')}")
            return location_info
        else:
            # Return simple formatted address
            formatted = _create_formatted_address(properties)
            logger.info(f"Simple geocoding result: {formatted}")
            return formatted
            
    except requests.exceptions.Timeout:
        error_msg = "Reverse geocoding timeout - service unavailable"
        logger.error(error_msg)
        return error_msg if not detailed else _create_error_location(error_msg, lat, lon)
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(error_msg)
        return error_msg if not detailed else _create_error_location(error_msg, lat, lon)
    except ValueError as e:
        error_msg = f"Data parsing error: {str(e)}"
        logger.error(error_msg)
        return error_msg if not detailed else _create_error_location(error_msg, lat, lon)
    except Exception as e:
        error_msg = f"Reverse geocoding failed: {str(e)}"
        logger.error(error_msg)
        return error_msg if not detailed else _create_error_location(error_msg, lat, lon)

def _extract_detailed_location(properties, lat, lon):
    """
    Extract comprehensive location details from API response
    
    Returns structured location information with administrative hierarchy
    """
    # Basic address components
    address_info = {
        'coordinates': {
            'latitude': lat,
            'longitude': lon,
            'formatted': f"Lat: {lat:.6f}, Lon: {lon:.6f}"
        },
        'formatted': properties.get('formatted', ''),
        'name': properties.get('name', ''),
        'street': properties.get('street', ''),
        'house_number': properties.get('housenumber', ''),
        'postcode': properties.get('postcode', ''),
        
        # Administrative hierarchy
        'neighborhood': properties.get('suburb', '') or properties.get('neighbourhood', ''),
        'city': properties.get('city', ''),
        'county': properties.get('county', ''),
        'state': properties.get('state', ''),
        'state_code': properties.get('state_code', ''),
        'region': properties.get('region', ''),
        
        # Local government information
        'local_government': properties.get('city') or properties.get('county', ''),
        'district': properties.get('district', ''),
        'municipality': properties.get('municipality', ''),
        
        # Country information
        'country': properties.get('country', ''),
        'country_code': properties.get('country_code', ''),
        
        # Additional context
        'continent': properties.get('continent', ''),
        'timezone': properties.get('timezone', {}).get('name', '') if isinstance(properties.get('timezone'), dict) else '',
        
        # Raw properties for debugging
        'raw_properties': {k: v for k, v in properties.items() if k not in ['timezone']}
    }
    
    # Clean up empty values
    address_info = {k: v for k, v in address_info.items() if v not in [None, '']}
    
    # Create hierarchical display
    address_info['hierarchical_display'] = _create_hierarchical_display(address_info)
    
    # Create simplified display for UI
    address_info['simplified_display'] = _create_simplified_display(address_info)
    
    return address_info

def _create_hierarchical_display(location_info):
    """Create a hierarchical display of the location"""
    hierarchy = []
    
    # Street level
    if location_info.get('house_number') and location_info.get('street'):
        hierarchy.append(f"{location_info['house_number']} {location_info['street']}")
    elif location_info.get('street'):
        hierarchy.append(location_info['street'])
    
    # Neighborhood level
    if location_info.get('neighborhood'):
        hierarchy.append(location_info['neighborhood'])
    
    # City/Local Government level
    if location_info.get('city'):
        hierarchy.append(location_info['city'])
    elif location_info.get('local_government'):
        hierarchy.append(location_info['local_government'])
    
    # State level
    if location_info.get('state'):
        if location_info.get('state_code'):
            hierarchy.append(f"{location_info['state']} ({location_info['state_code']})")
        else:
            hierarchy.append(location_info['state'])
    
    # Country level
    if location_info.get('country'):
        hierarchy.append(location_info['country'])
    
    return " → ".join(hierarchy) if hierarchy else "Location details unavailable"

def _create_simplified_display(location_info):
    """Create a simplified display for compact UI representation"""
    parts = []
    
    # City or Local Government
    if location_info.get('city'):
        parts.append(location_info['city'])
    elif location_info.get('local_government'):
        parts.append(location_info['local_government'])
    
    # State
    if location_info.get('state'):
        parts.append(location_info['state'])
    
    # Country
    if location_info.get('country'):
        parts.append(location_info['country'])
    
    return ", ".join(parts) if parts else "Unknown Location"

def _create_formatted_address(properties):
    """Create a formatted address from properties"""
    formatted = properties.get('formatted', 'Unknown location')
    
    # Fallback: build address from components
    if formatted == 'Unknown location':
        address_parts = [
            properties.get('street', ''),
            properties.get('city', ''),
            properties.get('state', ''),
            properties.get('country', '')
        ]
        formatted = ', '.join(filter(None, address_parts)) or 'Unknown location'
    
    return formatted

def _create_unknown_location():
    """Create a structured unknown location response"""
    return {
        'coordinates': {'latitude': 0, 'longitude': 0, 'formatted': 'Unknown'},
        'formatted': 'Unknown location',
        'hierarchical_display': 'Location details unavailable',
        'simplified_display': 'Unknown Location',
        'error': 'No location information available'
    }

def _create_error_location(error_msg, lat, lon):
    """Create a structured error location response"""
    return {
        'coordinates': {
            'latitude': lat, 
            'longitude': lon, 
            'formatted': f"Lat: {lat:.6f}, Lon: {lon:.6f}"
        },
        'formatted': 'Geocoding service error',
        'hierarchical_display': 'Service unavailable',
        'simplified_display': 'Service Error',
        'error': error_msg
    }

def get_location_summary(lat, lon):
    """
    Get a concise location summary for display in the UI
    
    Returns a simplified structure with key location information
    """
    detailed_info = reverse_geocode(lat, lon, detailed=True)
    
    if isinstance(detailed_info, str):
        # Handle error cases
        return {
            'display_name': detailed_info,
            'coordinates': f"Lat: {lat:.6f}, Lon: {lon:.6f}",
            'administrative_areas': ['Unknown'],
            'country': 'Unknown'
        }
    
    # Extract key administrative areas
    admin_areas = []
    if detailed_info.get('city'):
        admin_areas.append(f"City: {detailed_info['city']}")
    if detailed_info.get('local_government'):
        admin_areas.append(f"Local Govt: {detailed_info['local_government']}")
    if detailed_info.get('county'):
        admin_areas.append(f"County: {detailed_info['county']}")
    if detailed_info.get('state'):
        admin_areas.append(f"State: {detailed_info['state']}")
    if detailed_info.get('region'):
        admin_areas.append(f"Region: {detailed_info['region']}")
    
    return {
        'display_name': detailed_info.get('simplified_display', 'Unknown Location'),
        'coordinates': detailed_info['coordinates']['formatted'],
        'full_address': detailed_info.get('formatted', ''),
        'administrative_areas': admin_areas if admin_areas else ['No administrative data'],
        'country': detailed_info.get('country', 'Unknown'),
        'hierarchical_path': detailed_info.get('hierarchical_display', ''),
        'raw_data': detailed_info  # Include full data for debugging
    }

def batch_reverse_geocode(coordinates_list, detailed=False):
    """
    Reverse geocode multiple coordinates at once
    
    Args:
        coordinates_list: List of (lat, lon) tuples
        detailed (bool): Whether to return detailed information
    
    Returns:
        List of location results
    """
    logger.info(f"Batch geocoding {len(coordinates_list)} coordinates")
    results = []
    
    for i, (lat, lon) in enumerate(coordinates_list):
        try:
            result = reverse_geocode(lat, lon, detailed)
            results.append({
                'coordinates': (lat, lon),
                'location': result,
                'success': True
            })
            logger.debug(f"Batch geocoding {i+1}/{len(coordinates_list)} completed")
        except Exception as e:
            error_msg = f"Failed to geocode ({lat:.6f}, {lon:.6f}): {str(e)}"
            logger.error(error_msg)
            results.append({
                'coordinates': (lat, lon),
                'location': error_msg if not detailed else _create_error_location(error_msg, lat, lon),
                'success': False,
                'error': str(e)
            })
    
    logger.info(f"Batch geocoding completed: {len([r for r in results if r['success']])}/{len(results)} successful")
    return results

# Test function
def test_reverse_geocoding():
    """Test the reverse geocoding with known locations"""
    test_coordinates = [
        (40.7128, -74.0060),   # New York City
        (51.5074, -0.1278),    # London
        (35.6762, 139.6503),   # Tokyo
        (6.5244, 3.3792),      # Lagos
    ]
    
    print("Testing Reverse Geocoding Service")
    print("=" * 50)
    
    for lat, lon in test_coordinates:
        print(f"\n📍 Coordinates: {lat:.6f}, {lon:.6f}")
        
        # Test detailed geocoding
        detailed_result = reverse_geocode(lat, lon, detailed=True)
        
        if isinstance(detailed_result, dict):
            print(f"📋 Formatted: {detailed_result.get('formatted', 'N/A')}")
            print(f"🏛️  Hierarchical: {detailed_result.get('hierarchical_display', 'N/A')}")
            print(f"🏙️  City: {detailed_result.get('city', 'N/A')}")
            print(f"🏛️  Local Government: {detailed_result.get('local_government', 'N/A')}")
            print(f"🗺️  State: {detailed_result.get('state', 'N/A')}")
            print(f"🌍 Country: {detailed_result.get('country', 'N/A')}")
        else:
            print(f"❌ Error: {detailed_result}")
        
        print("-" * 30)
        
        # Test summary
        summary = get_location_summary(lat, lon)
        print(f"📊 Summary: {summary['display_name']}")
        print(f"📍 Coordinates: {summary['coordinates']}")
        print(f"🏛️  Administrative Areas: {', '.join(summary['administrative_areas'])}")

if __name__ == "__main__":
    test_reverse_geocoding()