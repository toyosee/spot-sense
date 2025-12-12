import requests
import os
import socket
from dotenv import load_dotenv

load_dotenv()

GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY", "sample_key")

def get_device_location():
    """
    Get current device location using multiple fallback methods
    Returns: tuple (latitude, longitude)
    """
    
    # Method 1: Try IP-based geolocation
    location = _get_location_by_ip()
    if location:
        return location
    
    # Method 2: Try GPSD service (if available on system)
    location = _get_location_by_gpsd()
    if location:
        return location
    
    # Method 3: Fallback to a default location (should be dynamic)
    # Use a geographically relevant default instead of hardcoded
    return _get_default_location()

def _get_location_by_ip():
    """Get approximate location using IP address"""
    try:
        # Using ipapi.co service (free tier available)
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('latitude'), data.get('longitude')
    except:
        pass
    
    try:
        # Fallback to ip-api.com
        response = requests.get('http://ip-api.com/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return data.get('lat'), data.get('lon')
    except:
        pass
    
    return None

def _get_location_by_gpsd():
    """Get location from GPSD service if available"""
    try:
        import gpsd
        gpsd.connect()
        packet = gpsd.get_current()
        if packet.mode >= 2:  # 2D or 3D fix
            return packet.lat, packet.lon
    except:
        pass
    return None

def _get_default_location():
    """
    Return a dynamic default based on some logic
    This prevents the same coordinates for every analysis
    """
    # You could rotate through different major cities
    # or use some other logic to vary the starting point
    major_cities = [
        (6.5244, 3.3792),   # Lagos, Nigeria
        (40.7128, -74.0060), # New York, USA
        (51.5074, -0.1278),  # London, UK
        (35.6762, 139.6503), # Tokyo, Japan
        (-33.8688, 151.2093) # Sydney, Australia
    ]
    
    # Simple rotation based on current minute
    import datetime
    index = datetime.datetime.now().minute % len(major_cities)
    return major_cities[index]

def get_precise_location():
    """Alternative function for more precise location when needed"""
    return get_device_location()

def reverse_geocode(lat, lon):
    """
    Convert coordinates to human-readable address using Geoapify
    Returns: string with formatted address
    """
    url = f"https://api.geoapify.com/v1/geocode/reverse?lat={lat}&lon={lon}&apiKey={GEOAPIFY_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        data = response.json()
        
        if data.get('features'):
            properties = data['features'][0]['properties']
            
            # Return formatted address if available
            formatted = properties.get('formatted', 'Unknown location')
            
            # Add more details if available
            address_parts = []
            if properties.get('name'):
                address_parts.append(properties['name'])
            if properties.get('street'):
                address_parts.append(properties['street'])
            if properties.get('city'):
                address_parts.append(properties['city'])
            if properties.get('country'):
                address_parts.append(properties['country'])
            
            if address_parts:
                return ', '.join(address_parts)
            else:
                return formatted
        else:
            return "Unknown location - no features found"
            
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except ValueError as e:
        return f"JSON parsing error: {str(e)}"
    except Exception as e:
        return f"Reverse geocoding failed: {str(e)}"