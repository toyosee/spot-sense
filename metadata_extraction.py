import exifread
from PIL import Image, ExifTags
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TZ = pytz.timezone(os.getenv("DEFAULT_TIME_ZONE", "Africa/Lagos"))

def extract_image_metadata(file_path, ext):
    """
    Extract comprehensive metadata from image files
    Supports: JPEG, TIFF, PNG, and other common formats
    """
    metadata = {}
    
    try:
        if ext.lower() in ['.jpg', '.jpeg', '.tiff', '.tif']:
            metadata = _extract_exif_metadata(file_path)
        elif ext.lower() == '.png':
            metadata = _extract_png_metadata(file_path)
        elif ext.lower() in ['.heic', '.heif']:
            metadata = _extract_heic_metadata(file_path)
        else:
            metadata = {"Error": f"Unsupported file format: {ext}"}
            
    except Exception as e:
        metadata = {"Error": f"Metadata extraction failed: {str(e)}"}
    
    # Add basic file info
    metadata.update(_get_basic_file_info(file_path))
    
    return metadata

def _extract_exif_metadata(file_path):
    """Extract EXIF metadata from JPEG/TIFF files"""
    metadata = {}
    
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        
        if not tags:
            return {"Note": "No EXIF metadata found"}
        
        # GPS Data
        gps_data = _extract_gps_data(tags)
        if gps_data:
            metadata['GPS Coordinates'] = gps_data
            metadata['GPS Details'] = _get_detailed_gps(tags)
        
        # Date and Time
        datetime_info = _extract_datetime(tags)
        if datetime_info:
            metadata['Capture Time'] = datetime_info
        
        # Camera Information
        camera_info = _extract_camera_info(tags)
        if camera_info:
            metadata['Camera Info'] = camera_info
        
        # Image Properties
        image_props = _extract_image_properties(tags)
        if image_props:
            metadata['Image Properties'] = image_props
        
        # All other EXIF tags (for debugging)
        other_tags = {}
        for tag, value in tags.items():
            if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
                if 'GPS' not in tag and 'Image' not in tag and 'EXIF' not in tag:
                    other_tags[tag] = str(value)
        
        if other_tags:
            metadata['Other Metadata'] = other_tags
            
    except Exception as e:
        metadata = {"Error": f"EXIF extraction error: {str(e)}"}
    
    return metadata

def _extract_gps_data(tags):
    """Extract and convert GPS coordinates to decimal format"""
    try:
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            lat = _convert_to_degrees(tags['GPS GPSLatitude'].values)
            lon = _convert_to_degrees(tags['GPS GPSLongitude'].values)
            
            # Apply hemisphere directions
            if 'GPS GPSLatitudeRef' in tags:
                lat_ref = str(tags['GPS GPSLatitudeRef'])
                if lat_ref.upper() == 'S':
                    lat = -lat
            
            if 'GPS GPSLongitudeRef' in tags:
                lon_ref = str(tags['GPS GPSLongitudeRef'])
                if lon_ref.upper() == 'W':
                    lon = -lon
            
            return f"Lat: {lat:.6f}, Lon: {lon:.6f}"
        
        return None
        
    except Exception as e:
        return f"GPS parsing error: {str(e)}"

def _get_detailed_gps(tags):
    """Extract detailed GPS information"""
    gps_details = {}
    
    gps_mapping = {
        'GPS GPSLatitude': 'Latitude',
        'GPS GPSLongitude': 'Longitude', 
        'GPS GPSLatitudeRef': 'Latitude Reference',
        'GPS GPSLongitudeRef': 'Longitude Reference',
        'GPS GPSAltitude': 'Altitude',
        'GPS GPSAltitudeRef': 'Altitude Reference',
        'GPS GPSTimeStamp': 'GPS Time',
        'GPS GPSDate': 'GPS Date',
        'GPS GPSProcessingMethod': 'Processing Method',
        'GPS GPSDOP': 'Dilution of Precision'
    }
    
    for exif_tag, friendly_name in gps_mapping.items():
        if exif_tag in tags:
            gps_details[friendly_name] = str(tags[exif_tag])
    
    return gps_details if gps_details else None

def _extract_datetime(tags):
    """Extract date and time information"""
    datetime_str = None
    
    # Try different datetime tags in order of preference
    datetime_tags = [
        'EXIF DateTimeOriginal',
        'EXIF DateTimeDigitized', 
        'Image DateTime',
        'EXIF SubSecTimeOriginal',
        'EXIF GPSTimeStamp'
    ]
    
    for tag in datetime_tags:
        if tag in tags:
            try:
                datetime_str = str(tags[tag])
                # Convert to datetime object
                if ' ' in datetime_str and ':' in datetime_str:
                    dt = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
                    localized_dt = DEFAULT_TZ.localize(dt)
                    return localized_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            except:
                continue
    
    return datetime_str if datetime_str else "Unknown"

def _extract_camera_info(tags):
    """Extract camera and lens information"""
    camera_info = {}
    
    camera_mapping = {
        'Image Make': 'Camera Make',
        'Image Model': 'Camera Model',
        'EXIF LensModel': 'Lens Model',
        'EXIF FNumber': 'Aperture',
        'EXIF FocalLength': 'Focal Length',
        'EXIF ExposureTime': 'Exposure Time',
        'EXIF ISOSpeedRatings': 'ISO',
        'EXIF Flash': 'Flash',
        'EXIF WhiteBalance': 'White Balance',
        'EXIF MeteringMode': 'Metering Mode'
    }
    
    for exif_tag, friendly_name in camera_mapping.items():
        if exif_tag in tags:
            value = str(tags[exif_tag])
            # Clean up common values
            if exif_tag == 'EXIF FocalLength' and '/' in value:
                try:
                    value = f"{eval(value):.1f}mm"
                except:
                    pass
            elif exif_tag == 'EXIF ExposureTime' and '/' in value:
                try:
                    value = f"1/{int(1/eval(value))}s" if eval(value) < 1 else f"{eval(value)}s"
                except:
                    pass
            
            camera_info[friendly_name] = value
    
    return camera_info if camera_info else None

def _extract_image_properties(tags):
    """Extract basic image properties"""
    image_props = {}
    
    property_mapping = {
        'EXIF ExifImageWidth': 'Image Width',
        'EXIF ExifImageLength': 'Image Height',
        'Image ImageWidth': 'Width',
        'Image ImageLength': 'Height',
        'Image Orientation': 'Orientation',
        'Image XResolution': 'X Resolution',
        'Image YResolution': 'Y Resolution',
        'EXIF ColorSpace': 'Color Space',
        'EXIF ResolutionUnit': 'Resolution Unit'
    }
    
    for exif_tag, friendly_name in property_mapping.items():
        if exif_tag in tags:
            image_props[friendly_name] = str(tags[exif_tag])
    
    return image_props if image_props else None

def _extract_png_metadata(file_path):
    """Extract metadata from PNG files"""
    metadata = {"Note": "PNG files have limited EXIF metadata support"}
    
    try:
        with Image.open(file_path) as img:
            # PNG can have some metadata in text chunks
            if hasattr(img, 'text'):
                text_data = {}
                for key, value in img.text.items():
                    if key and value:
                        text_data[key] = value
                
                if text_data:
                    metadata['Text Chunks'] = text_data
            
            # Basic image info
            metadata['Image Size'] = f"{img.width} x {img.height}"
            metadata['Format'] = img.format
            metadata['Mode'] = img.mode
            
    except Exception as e:
        metadata["Error"] = f"PNG metadata extraction failed: {str(e)}"
    
    return metadata

def _extract_heic_metadata(file_path):
    """Extract metadata from HEIC/HEIF files"""
    metadata = {"Note": "HEIC/HEIF support requires pillow-heif library"}
    
    try:
        # Try to use pillow-heif if available
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
            
            with Image.open(file_path) as img:
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
                    # Convert to readable format
                    readable_exif = {}
                    for tag_id, value in exif_data.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        readable_exif[tag] = str(value)
                    
                    metadata['EXIF Data'] = readable_exif
                
                metadata['Image Size'] = f"{img.width} x {img.height}"
                
        except ImportError:
            metadata['Error'] = "Install pillow-heif for HEIC support: pip install pillow-heif"
            
    except Exception as e:
        metadata["Error"] = f"HEIC metadata extraction failed: {str(e)}"
    
    return metadata

def _get_basic_file_info(file_path):
    """Get basic file information"""
    try:
        stat_info = os.stat(file_path)
        file_size = stat_info.st_size
        
        # Convert file size to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if file_size < 1024.0:
                break
            file_size /= 1024.0
        
        return {
            'File Name': os.path.basename(file_path),
            'File Size': f"{file_size:.2f} {unit}",
            'File Format': os.path.splitext(file_path)[1].upper(),
            'Last Modified': datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        }
    except:
        return {}

def _convert_to_degrees(value):
    """Convert EXIF GPS coordinates to decimal degrees"""
    try:
        if hasattr(value, 'values'):
            # exifread format
            d, m, s = value.values
        else:
            # PIL format
            d, m, s = value
        
        # Handle different data types
        d = float(d) if not isinstance(d, exifread.utils.Ratio) else float(d.num) / float(d.den)
        m = float(m) if not isinstance(m, exifread.utils.Ratio) else float(m.num) / float(m.den)
        s = float(s) if not isinstance(s, exifread.utils.Ratio) else float(s.num) / float(s.den)
        
        return d + (m / 60.0) + (s / 3600.0)
        
    except Exception as e:
        raise ValueError(f"Invalid GPS coordinate format: {value}")

def extract_video_metadata(file_path):
    """
    Extract metadata from video files
    Requires: ffmpeg-python
    """
    try:
        import ffmpeg
        
        probe = ffmpeg.probe(file_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        
        metadata = {
            'File Format': probe['format']['format_name'].upper(),
            'Duration': f"{float(probe['format']['duration']):.2f} seconds",
            'File Size': f"{int(probe['format']['size']) / (1024*1024):.2f} MB",
            'Bit Rate': f"{int(probe['format']['bit_rate']) / 1000:.2f} kbps"
        }
        
        if video_stream:
            video_info = {
                'Codec': video_stream['codec_name'],
                'Resolution': f"{video_stream['width']} x {video_stream['height']}",
                'Frame Rate': video_stream.get('r_frame_rate', 'Unknown'),
                'Pixel Format': video_stream.get('pix_fmt', 'Unknown')
            }
            metadata['Video Stream'] = video_info
        
        if audio_stream:
            audio_info = {
                'Codec': audio_stream['codec_name'],
                'Sample Rate': f"{audio_stream.get('sample_rate', 'Unknown')} Hz",
                'Channels': audio_stream.get('channels', 'Unknown'),
                'Channel Layout': audio_stream.get('channel_layout', 'Unknown')
            }
            metadata['Audio Stream'] = audio_info
        
        return metadata
        
    except ImportError:
        return {"Error": "ffmpeg-python not installed. Install with: pip install ffmpeg-python"}
    except Exception as e:
        return {"Error": f"Video metadata extraction failed: {str(e)}"}

# Utility function for quick GPS extraction
def extract_gps_coordinates(file_path):
    """Quickly extract just GPS coordinates from an image"""
    metadata = extract_image_metadata(file_path, os.path.splitext(file_path)[1])
    
    if 'GPS Coordinates' in metadata:
        return metadata['GPS Coordinates']
    elif 'Error' in metadata:
        return metadata['Error']
    else:
        return "No GPS data found"