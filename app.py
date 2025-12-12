# app.py - Enhanced Main Application with Detailed Location Information
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from datetime import datetime
import pytz
import threading
import logging
import json

from services.location_service import get_device_location
from services.reverse_geocoding import reverse_geocode, get_location_summary
# from services.shadow_analysis import ShadowAnalyzer
# from services.location_estimation import LocationEstimator

# In your app.py, add these imports and updates:

from services.enhanced_location_estimation import EnhancedLocationEstimator as LocationEstimator
from services.terrain_classification import TerrainClassifier
from services.shadow_analysis_fixed import ShadowAnalyzerFixed as ShadowAnalyzer
# from services.nigeria_location_estimator import NigeriaLocationEstimator as LocationEstimator
from metadata_extraction import extract_image_metadata, extract_video_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeolocationApp:
    def __init__(self, root):
        self.root = root
        self.shadow_analyzer = ShadowAnalyzer()
        self.location_estimator = LocationEstimator()
        self.current_file_path = None
        self.analysis_in_progress = False
        self.current_device_location = None
        self.setup_ui()
        logger.info("Geolocation Analyzer initialized")
    
    def setup_ui(self):
        """Setup the main user interface"""
        self.root.title("🌍 Advanced Geolocation Analyzer v3.0")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)
        
        # Configure styles
        self.setup_styles()
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self.setup_header(main_frame)
        
        # Controls
        self.setup_controls(main_frame)
        
        # Content area
        self.setup_content_area(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Arial', 18, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Error.TLabel', foreground='red')
    
    def setup_header(self, parent):
        """Setup application header"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            header_frame, 
            text="🌍 Advanced Geolocation Analyzer", 
            style='Header.TLabel'
        ).pack(side=tk.LEFT)
        
        # Version info
        ttk.Label(
            header_frame, 
            text="v3.0", 
            foreground='gray'
        ).pack(side=tk.RIGHT)
    
    def setup_controls(self, parent):
        """Setup control buttons"""
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Button frame
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(side=tk.LEFT)
        
        buttons = [
            ("📁 Upload Image/Video", self.handle_file, 20),
            ("🔄 Re-analyze", self.reanalyze_file, 15),
            ("📊 Methods Info", self.show_methods_info, 12),
            ("🧹 Clear", self.clear_display, 10),
            ("💾 Export Results", self.export_results, 12),
        ]
        
        for text, command, width in buttons:
            ttk.Button(
                button_frame, 
                text=text, 
                command=command, 
                width=width
            ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(controls_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))
    
    def setup_content_area(self, parent):
        """Setup main content area with tabs"""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Image display
        self.setup_image_panel(content_frame)
        
        # Right panel - Analysis results
        self.setup_analysis_panel(content_frame)
    
    def setup_image_panel(self, parent):
        """Setup image display panel"""
        left_frame = ttk.LabelFrame(parent, text="🖼️ Image Preview", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = ttk.Label(
            left_frame, 
            text="No image loaded\n\n📂 Drag and drop or click 'Upload'", 
            background='white', 
            anchor=tk.CENTER, 
            justify=tk.CENTER,
            font=('Arial', 11)
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Image info label
        self.image_info = ttk.Label(left_frame, text="", foreground='gray')
        self.image_info.pack(fill=tk.X, pady=(5, 0))
        
        # Setup drag and drop
        self.setup_drag_drop()
    
    def setup_analysis_panel(self, parent):
        """Setup analysis results panel with tabs"""
        right_frame = ttk.LabelFrame(parent, text="📊 Analysis Results", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Setup tabs
        self.setup_summary_tab()
        self.setup_metadata_tab()
        self.setup_analysis_tab()
        self.setup_location_tab()
        self.setup_debug_tab()
    
    def setup_summary_tab(self):
        """Setup summary tab"""
        self.summary_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.summary_frame, text="📋 Summary")
        
        self.summary_text = tk.Text(
            self.summary_frame, 
            height=20, 
            width=60, 
            wrap=tk.WORD, 
            font=('Arial', 10),
            relief=tk.FLAT,
            bg='#f8f9fa'
        )
        
        summary_scrollbar = ttk.Scrollbar(self.summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_metadata_tab(self):
        """Setup metadata tab"""
        self.metadata_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.metadata_frame, text="📄 Metadata")
        
        self.metadata_text = tk.Text(
            self.metadata_frame, 
            height=20, 
            width=60, 
            wrap=tk.WORD,
            font=('Courier New', 9)
        )
        
        metadata_scrollbar = ttk.Scrollbar(self.metadata_frame, orient=tk.VERTICAL, command=self.metadata_text.yview)
        self.metadata_text.configure(yscrollcommand=metadata_scrollbar.set)
        
        self.metadata_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        metadata_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_analysis_tab(self):
        """Setup analysis tab"""
        self.analysis_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.analysis_frame, text="🔍 Analysis")
        
        self.analysis_text = tk.Text(
            self.analysis_frame, 
            height=20, 
            width=60, 
            wrap=tk.WORD,
            font=('Arial', 10)
        )
        
        analysis_scrollbar = ttk.Scrollbar(self.analysis_frame, orient=tk.VERTICAL, command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=analysis_scrollbar.set)
        
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        analysis_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_location_tab(self):
        """Setup location tab"""
        self.location_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.location_frame, text="📍 Location")
        
        self.location_text = tk.Text(
            self.location_frame, 
            height=20, 
            width=60, 
            wrap=tk.WORD,
            font=('Arial', 10)
        )
        
        location_scrollbar = ttk.Scrollbar(self.location_frame, orient=tk.VERTICAL, command=self.location_text.yview)
        self.location_text.configure(yscrollcommand=location_scrollbar.set)
        
        self.location_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        location_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_debug_tab(self):
        """Setup debug tab for technical details"""
        self.debug_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.debug_frame, text="🐛 Debug")
        
        self.debug_text = tk.Text(
            self.debug_frame, 
            height=20, 
            width=60, 
            wrap=tk.WORD,
            font=('Courier New', 8)
        )
        
        debug_scrollbar = ttk.Scrollbar(self.debug_frame, orient=tk.VERTICAL, command=self.debug_text.yview)
        self.debug_text.configure(yscrollcommand=debug_scrollbar.set)
        
        self.debug_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        debug_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_status_bar(self, parent):
        """Setup status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("🟢 Ready to analyze images and videos")
        
        status_bar = ttk.Label(
            parent, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            padding=(10, 5),
            background='#e9ecef'
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))
    
    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        def handle_drop(event):
            if self.analysis_in_progress:
                messagebox.showwarning("Analysis in Progress", "Please wait for current analysis to complete.")
                return
                
            file_path = event.data.strip('{}')
            if os.path.isfile(file_path):
                self.current_file_path = file_path
                self.analyze_file(file_path)
        
        # Make the image label accept drops
        try:
            self.image_label.drop_target_register('DND_Files')
            self.image_label.dnd_bind('<<Drop>>', handle_drop)
        except Exception as e:
            logger.warning(f"Drag and drop not supported: {e}")

    def handle_file(self):
        """Handle file selection"""
        if self.analysis_in_progress:
            messagebox.showwarning("Analysis in Progress", "Please wait for current analysis to complete.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image or Video File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp"),
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.current_file_path = file_path
        self.analyze_file(file_path)
    
    def reanalyze_file(self):
        """Re-analyze current file"""
        if self.current_file_path and os.path.exists(self.current_file_path):
            self.analyze_file(self.current_file_path)
        else:
            messagebox.showwarning("No File", "Please select a file first.")
    
    def analyze_file(self, file_path):
        """Start analysis in a separate thread"""
        self.analysis_in_progress = True
        self.progress.start()
        self.status_var.set(f"🟡 Analyzing: {os.path.basename(file_path)}...")
        
        # Clear previous results
        self.clear_display()
        
        # Start analysis in thread
        thread = threading.Thread(target=self._perform_analysis, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def _perform_analysis(self, file_path):
        """Perform analysis in background thread"""
        try:
            logger.info(f"Starting analysis for: {file_path}")
            
            # Get device location ONCE for consistency
            self.current_device_location = get_device_location()
            logger.info(f"Device location: {self.current_device_location}")
            
            ext = os.path.splitext(file_path)[1].lower()
            results = {}
            
            if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp']:
                results = self.analyze_image(file_path)
            elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv']:
                results = self.analyze_video(file_path)
            else:
                self._show_error("Unsupported file format")
                return
            
            # Update UI in main thread
            self.root.after(0, self._display_results, results, file_path)
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            self._show_error(error_msg)
        finally:
            self.analysis_in_progress = False
            self.root.after(0, self._analysis_complete)
    
    def analyze_image(self, file_path):
        """Comprehensive image analysis with multiple geolocation methods"""
        logger.info(f"Analyzing image: {file_path}")
        
        results = {
            'metadata': {},
            'analysis': {},
            'location': {},
            'geolocation_methods': [],
            'best_estimate': {},
            'debug_info': {}
        }
        
        try:
            # 1. Extract metadata
            ext = os.path.splitext(file_path)[1].lower()
            metadata = extract_image_metadata(file_path, ext)
            results['metadata'] = metadata
            results['debug_info']['metadata_extraction'] = 'success'
            
            # 2. Display image in main thread
            self.root.after(0, self.display_image, file_path)
            
            # 3. Device location with enhanced geocoding
            if self.current_device_location:
                device_lat, device_lon = self.current_device_location
                device_location_info = get_location_summary(device_lat, device_lon)
                
                results['location']['device'] = {
                    'coordinates': f"Lat: {device_lat:.6f}, Lon: {device_lon:.6f}",
                    'address': device_location_info,
                    'source': 'Current Device Location',
                    'confidence': 'High (known location)',
                    'reliability': 'High'
                }
                results['geolocation_methods'].append('device_location')
                results['debug_info']['device_location'] = 'success'
            else:
                results['debug_info']['device_location'] = 'failed'
            
            # 4. EXIF GPS (most reliable) with enhanced geocoding
            if 'GPS Coordinates' in metadata and 'Error' not in metadata.get('GPS Coordinates', ''):
                gps_coords = metadata['GPS Coordinates']
                
                try:
                    if 'Lat:' in gps_coords and 'Lon:' in gps_coords:
                        parts = gps_coords.split(',')
                        lat = float(parts[0].replace('Lat:', '').strip())
                        lon = float(parts[1].replace('Lon:', '').strip())
                        gps_location_info = get_location_summary(lat, lon)
                        
                        results['location']['image_gps'] = {
                            'coordinates': gps_coords,
                            'address': gps_location_info,
                            'source': 'EXIF Metadata from Image',
                            'confidence': 'Very High (direct from camera)',
                            'reliability': 'Very High'
                        }
                        results['debug_info']['gps_geocoding'] = 'success'
                except Exception as e:
                    results['location']['image_gps'] = {
                        'coordinates': gps_coords,
                        'address': f"Address lookup failed: {str(e)}",
                        'source': 'EXIF Metadata from Image',
                        'confidence': 'Very High (direct from camera)',
                        'reliability': 'Very High'
                    }
                    results['debug_info']['gps_geocoding'] = f'failed: {e}'
                
                results['geolocation_methods'].append('exif_gps')
                results['debug_info']['exif_gps'] = 'success'
            else:
                results['debug_info']['exif_gps'] = 'not_available'
            
            # 5. Shadow Analysis (when no GPS) with enhanced geocoding
            if 'exif_gps' not in results['geolocation_methods'] and self.current_device_location:
                shadow_results = self.perform_shadow_analysis(file_path, metadata)
                results['analysis']['shadow'] = shadow_results
                results['debug_info']['shadow_analysis'] = shadow_results.get('status', 'unknown')
                
                if shadow_results.get('status') == 'success':
                    estimated_coords_str = shadow_results['estimated_coordinates']
                    # Extract coordinates from string for geocoding
                    try:
                        if 'Lat:' in estimated_coords_str and 'Lon:' in estimated_coords_str:
                            parts = estimated_coords_str.split(',')
                            lat = float(parts[0].replace('Lat:', '').strip())
                            lon = float(parts[1].replace('Lon:', '').strip())
                            shadow_location_info = get_location_summary(lat, lon)
                            
                            results['location']['shadow_estimated'] = {
                                'coordinates': estimated_coords_str,
                                'address': shadow_location_info,
                                'confidence': shadow_results.get('confidence', '0%'),
                                'source': 'Shadow Analysis',
                                'reliability': 'Medium',
                                'shadow_quality': shadow_results.get('shadow_quality', 'unknown'),
                                'details': f"Based on {shadow_results.get('shadow_direction', 'unknown')} shadow direction"
                            }
                            results['geolocation_methods'].append('shadow_analysis')
                    except Exception as e:
                        logger.error(f"Failed to parse shadow coordinates: {e}")
                        results['location']['shadow_estimated'] = {
                            'coordinates': estimated_coords_str,
                            'address': 'Failed to parse coordinates for geocoding',
                            'confidence': shadow_results.get('confidence', '0%'),
                            'source': 'Shadow Analysis',
                            'reliability': 'Medium'
                        }
            
            # 6. Determine best location estimate
            results['best_estimate'] = self._determine_best_location(results['location'])
            results['debug_info']['best_estimate'] = results['best_estimate'].get('source', 'unknown')
            
        except Exception as e:
            error_msg = f"Image analysis error: {str(e)}"
            logger.error(error_msg)
            results['debug_info']['overall_analysis'] = f'failed: {e}'
            results['analysis']['error'] = error_msg
        
        return results
    
    def perform_shadow_analysis(self, file_path, metadata):
        """Perform shadow-based geolocation"""
        try:
            # Get capture time from metadata
            capture_time = self._extract_capture_time(metadata)
            logger.info(f"Using capture time: {capture_time}")
            
            # Analyze shadows
            shadow_data = self.shadow_analyzer.analyze_image_shadows(file_path, capture_time)
            
            if 'error' in shadow_data:
                return {
                    'status': 'failed',
                    'message': f"Shadow analysis failed: {shadow_data['error']}",
                    'reason': shadow_data.get('reason', 'Unknown error')
                }
            
            # Estimate location using enhanced estimator
            if self.current_device_location:
                estimated_coords, confidence, details = self.location_estimator.estimate_from_shadows(
                    shadow_data, capture_time, self.current_device_location
                )
            else:
                return {
                    'status': 'failed',
                    'message': "Device location not available for shadow analysis"
                }
            
            if estimated_coords:
                estimated_address = reverse_geocode(estimated_coords[0], estimated_coords[1], detailed=True)
                
                return {
                    'status': 'success',
                    'shadow_direction': f"{shadow_data['shadow_angle']:.2f}°",
                    'solar_azimuth': f"{shadow_data['solar_azimuth']:.2f}°",
                    'capture_time': capture_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    'estimated_coordinates': f"Lat: {estimated_coords[0]:.6f}, Lon: {estimated_coords[1]:.6f}",
                    'estimated_address': estimated_address,
                    'confidence': f"{confidence:.2f}%",
                    'shadow_quality': shadow_data.get('shadow_quality', 'unknown'),
                    'search_center': f"Lat: {self.current_device_location[0]:.6f}, Lon: {self.current_device_location[1]:.6f}",
                    'estimation_details': details,
                    'has_shadows': shadow_data.get('has_shadows', False)
                }
            else:
                return {
                    'status': 'failed',
                    'message': "Could not estimate location from shadows",
                    'reason': "No suitable location found matching shadow analysis"
                }
            
        except Exception as e:
            error_msg = f"Shadow analysis error: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg
            }
    
    def analyze_video(self, file_path):
        """Analyze video file"""
        results = {
            'metadata': {},
            'analysis': {},
            'location': {},
            'geolocation_methods': [],
            'best_estimate': {},
            'debug_info': {}
        }
        
        try:
            # Extract video metadata
            metadata = extract_video_metadata(file_path)
            results['metadata'] = metadata
            results['debug_info']['metadata_extraction'] = 'success'
            
            # Device location with enhanced geocoding
            if self.current_device_location:
                device_lat, device_lon = self.current_device_location
                device_location_info = get_location_summary(device_lat, device_lon)
                
                results['location']['device'] = {
                    'coordinates': f"Lat: {device_lat:.6f}, Lon: {device_lon:.6f}",
                    'address': device_location_info,
                    'source': 'Current Device Location',
                    'confidence': 'High'
                }
                results['geolocation_methods'].append('device_location')
            
            results['analysis']['video'] = {
                'status': 'Basic analysis complete',
                'note': 'Advanced video analysis (frame extraction, shadow analysis) not yet implemented',
                'suggestions': [
                    'Video geolocation requires frame-by-frame analysis',
                    'Consider extracting key frames as images for analysis',
                    'Future versions will include video-specific geolocation'
                ]
            }
            
            results['best_estimate'] = self._determine_best_location(results['location'])
            
        except Exception as e:
            error_msg = f"Video analysis failed: {str(e)}"
            logger.error(error_msg)
            results['metadata'] = {'Error': error_msg}
            results['debug_info']['video_analysis'] = f'failed: {e}'
        
        return results
    
    def _extract_capture_time(self, metadata):
        """Extract capture time from metadata"""
        capture_time_str = metadata.get('Capture Time', 'Unknown')
        
        if isinstance(capture_time_str, str) and capture_time_str != "Unknown":
            try:
                # Try different datetime formats
                for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S']:
                    try:
                        capture_time = datetime.strptime(capture_time_str, fmt)
                        return pytz.timezone('Africa/Lagos').localize(capture_time)
                    except ValueError:
                        continue
            except:
                pass
        
        # Fallback to current time
        logger.warning("Using current time as fallback for capture time")
        return datetime.now(pytz.timezone('Africa/Lagos'))
    
    def _determine_best_location(self, location_data):
        """Determine the most reliable location estimate"""
        priority_order = [
            'image_gps',      # Most reliable - direct from camera
            'device',         # Known device location
            'shadow_estimated', # Estimated from shadows
        ]
        
        for source in priority_order:
            if source in location_data:
                reliability = self._get_reliability_rating(source)
                return {
                    'source': source,
                    'data': location_data[source],
                    'reliability': reliability,
                    'recommendation': self._get_recommendation(source, reliability)
                }
        
        return {
            'source': 'unknown',
            'data': {'coordinates': 'Unknown', 'address': 'Cannot determine location'},
            'reliability': 'None',
            'recommendation': 'No reliable location data found. Try images with GPS metadata or clear shadows.'
        }
    
    def _get_reliability_rating(self, source):
        """Get reliability rating for each method"""
        reliability = {
            'image_gps': 'Very High',
            'device': 'High', 
            'shadow_estimated': 'Medium',
            'unknown': 'None'
        }
        return reliability.get(source, 'Unknown')
    
    def _get_recommendation(self, source, reliability):
        """Get recommendation based on location source and reliability"""
        recommendations = {
            'image_gps': '✅ Excellent! Direct GPS data from camera.',
            'device': '✅ Good! Using current device location as reference.',
            'shadow_estimated': '🟡 Fair! Estimated from shadow analysis.',
            'unknown': '🔴 Unable to determine location.'
        }
        return recommendations.get(source, 'Unknown reliability')
    
    def display_image(self, path):
        """Display image with proper aspect ratio"""
        try:
            img = Image.open(path)
            
            # Calculate thumbnail size while maintaining aspect ratio
            max_size = (400, 400)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
            
            # Update info text
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            info_text = f"{os.path.basename(path)}\n{img.size[0]} x {img.size[1]} pixels | {file_size:.2f} MB"
            self.image_info.configure(text=info_text)
            
        except Exception as e:
            error_msg = f"Error loading image: {str(e)}"
            logger.error(error_msg)
            self.image_label.configure(
                text=error_msg,
                image=''
            )
            self.image_label.image = None
            self.image_info.configure(text="")
    
    def _display_results(self, results, file_path):
        """Display results in the UI"""
        try:
            # Update all tabs
            self._update_summary_tab(results, file_path)
            self._update_metadata_tab(results, file_path)
            self._update_analysis_tab(results)
            self._update_location_tab(results)
            self._update_debug_tab(results)
            
            # Switch to summary tab by default
            self.notebook.select(0)
            
        except Exception as e:
            error_msg = f"Error displaying results: {str(e)}"
            logger.error(error_msg)
            self._show_error(error_msg)
    
    def _update_summary_tab(self, results, file_path):
        """Update summary tab with key information"""
        self.summary_text.delete("1.0", tk.END)
        
        # Header
        self.summary_text.insert(tk.END, f"📄 File: {os.path.basename(file_path)}\n", "header")
        self.summary_text.insert(tk.END, "="*60 + "\n\n")
        
        # Best estimate
        best_estimate = results.get('best_estimate', {})
        if best_estimate.get('source') != 'unknown':
            self.summary_text.insert(tk.END, "🎯 BEST LOCATION ESTIMATE\n", "subheader")
            self.summary_text.insert(tk.END, "─"*40 + "\n")
            
            estimate_data = best_estimate['data']
            address_info = estimate_data.get('address', {})
            
            if isinstance(address_info, dict):
                # Enhanced location display with administrative details
                self.summary_text.insert(tk.END, f"📍 {address_info.get('display_name', 'Unknown Location')}\n\n")
                
                # Administrative hierarchy
                if address_info.get('hierarchical_path'):
                    self.summary_text.insert(tk.END, f"🏛️ {address_info['hierarchical_path']}\n\n")
                
                # Administrative areas
                admin_areas = address_info.get('administrative_areas', [])
                if admin_areas:
                    self.summary_text.insert(tk.END, "🗺️ Administrative Areas:\n")
                    for area in admin_areas[:4]:  # Show top 4 areas
                        self.summary_text.insert(tk.END, f"   • {area}\n")
                    self.summary_text.insert(tk.END, "\n")
                
                # Country
                if address_info.get('country'):
                    self.summary_text.insert(tk.END, f"🌍 Country: {address_info['country']}\n")
            else:
                # Fallback for string addresses
                self.summary_text.insert(tk.END, f"📍 {address_info}\n\n")
            
            # Coordinates and source
            self.summary_text.insert(tk.END, f"📐 {estimate_data.get('coordinates', 'Unknown coordinates')}\n")
            self.summary_text.insert(tk.END, f"📊 Reliability: {best_estimate['reliability']}\n")
            self.summary_text.insert(tk.END, f"🔧 Method: {estimate_data.get('source', 'Unknown')}\n")
            
            if 'confidence' in estimate_data:
                self.summary_text.insert(tk.END, f"🎯 Confidence: {estimate_data['confidence']}\n")
            
            self.summary_text.insert(tk.END, f"\n💡 {best_estimate.get('recommendation', '')}\n")
        else:
            self.summary_text.insert(tk.END, "❌ UNABLE TO DETERMINE LOCATION\n\n", "error")
            self.summary_text.insert(tk.END, "Possible reasons:\n")
            self.summary_text.insert(tk.END, "• No GPS metadata in image\n")
            self.summary_text.insert(tk.END, "• No clear shadows for analysis\n")
            self.summary_text.insert(tk.END, "• Image taken indoors or at night\n")
            self.summary_text.insert(tk.END, "• Limited visual features for analysis\n")
        
        # Methods used
        self.summary_text.insert(tk.END, "\n" + "="*60 + "\n")
        methods = results.get('geolocation_methods', ['None'])
        self.summary_text.insert(tk.END, f"🔍 Methods Used: {', '.join(methods)}\n")
        
        # Configure text tags for styling
        self.summary_text.tag_configure("header", font=('Arial', 12, 'bold'))
        self.summary_text.tag_configure("subheader", font=('Arial', 11, 'bold'))
        self.summary_text.tag_configure("error", foreground='red')
    
    def _update_metadata_tab(self, results, file_path):
        """Update metadata tab"""
        self.metadata_text.delete("1.0", tk.END)
        self.metadata_text.insert(tk.END, f"File: {os.path.basename(file_path)}\n")
        self.metadata_text.insert(tk.END, "="*50 + "\n\n")
        
        if 'metadata' in results:
            for key, value in results['metadata'].items():
                self.metadata_text.insert(tk.END, f"• {key}:\n  {value}\n\n")
    
    def _update_analysis_tab(self, results):
        """Update analysis tab"""
        self.analysis_text.delete("1.0", tk.END)
        if 'analysis' in results:
            for category, analysis_data in results['analysis'].items():
                self.analysis_text.insert(tk.END, f"📊 {category.upper()} ANALYSIS:\n")
                self.analysis_text.insert(tk.END, "="*40 + "\n")
                
                if isinstance(analysis_data, dict):
                    for key, value in analysis_data.items():
                        if key not in ['details', 'estimation_details']:
                            self.analysis_text.insert(tk.END, f"  {key}: {value}\n")
                else:
                    self.analysis_text.insert(tk.END, f"  {analysis_data}\n")
                
                self.analysis_text.insert(tk.END, "\n")
    
    def _update_location_tab(self, results):
        """Update location tab with enhanced administrative information"""
        self.location_text.delete("1.0", tk.END)
        self.location_text.insert(tk.END, "📍 LOCATION ANALYSIS SUMMARY\n")
        self.location_text.insert(tk.END, "="*50 + "\n\n")
        
        if 'location' in results:
            location_sources = {
                'device': '📱 Device Location',
                'image_gps': '🖼️ Image GPS (EXIF)',
                'shadow_estimated': '🌅 Shadow Analysis Estimate'
            }
            
            for loc_type, display_name in location_sources.items():
                if loc_type in results['location']:
                    loc_data = results['location'][loc_type]
                    self.location_text.insert(tk.END, f"{display_name}:\n")
                    self.location_text.insert(tk.END, "─"*35 + "\n")
                    
                    if isinstance(loc_data, dict):
                        # Enhanced location display
                        address_info = loc_data.get('address', {})
                        
                        if isinstance(address_info, dict):
                            # Detailed address information
                            self.location_text.insert(tk.END, f"  📍 Coordinates: {loc_data.get('coordinates', 'Unknown')}\n")
                            self.location_text.insert(tk.END, f"  🏙️ Display Name: {address_info.get('display_name', 'Unknown')}\n")
                            
                            # Administrative hierarchy
                            if address_info.get('hierarchical_path'):
                                self.location_text.insert(tk.END, f"  🏛️ Hierarchy: {address_info['hierarchical_path']}\n")
                            
                            # Administrative areas
                            admin_areas = address_info.get('administrative_areas', [])
                            if admin_areas:
                                self.location_text.insert(tk.END, f"  🗺️ Administrative Areas:\n")
                                for area in admin_areas:
                                    self.location_text.insert(tk.END, f"     • {area}\n")
                            
                            # Full address
                            if address_info.get('full_address'):
                                self.location_text.insert(tk.END, f"  📋 Full Address: {address_info['full_address']}\n")
                            
                            # Country
                            if address_info.get('country'):
                                self.location_text.insert(tk.END, f"  🌍 Country: {address_info['country']}\n")
                        else:
                            # Fallback for string addresses
                            self.location_text.insert(tk.END, f"  📍 Address: {address_info}\n")
                        
                        # Source and confidence
                        self.location_text.insert(tk.END, f"  🔧 Source: {loc_data.get('source', 'Unknown')}\n")
                        if 'confidence' in loc_data:
                            self.location_text.insert(tk.END, f"  🎯 Confidence: {loc_data['confidence']}\n")
                        if 'reliability' in loc_data:
                            self.location_text.insert(tk.END, f"  📊 Reliability: {loc_data['reliability']}\n")
                            
                    else:
                        self.location_text.insert(tk.END, f"  {loc_data}\n")
                    
                    self.location_text.insert(tk.END, "\n")
    
    def _update_debug_tab(self, results):
        """Update debug tab with technical details"""
        self.debug_text.delete("1.0", tk.END)
        self.debug_text.insert(tk.END, "🐛 DEBUG INFORMATION\n")
        self.debug_text.insert(tk.END, "="*50 + "\n\n")
        
        if 'debug_info' in results:
            for key, value in results['debug_info'].items():
                self.debug_text.insert(tk.END, f"{key}: {value}\n")
    
    def show_methods_info(self):
        """Show information about geolocation methods"""
        info_text = """
🌍 Geolocation Methods Used:

1. 🖼️ EXIF GPS Data (Most Reliable)
   - Direct GPS coordinates from camera
   - Accuracy: Very High
   - Availability: When enabled in camera

2. 📱 Device Location
   - Current device GPS/IP location
   - Accuracy: High to Medium
   - Used as reference point

3. 🌅 Shadow Analysis
   - Analyzes shadow direction and length
   - Estimates solar position and location
   - Accuracy: Medium to Low
   - Works best with clear shadows and known time

Confidence Levels:
✅ High: Direct measurement
🟡 Medium: Estimated with good confidence  
🟠 Low: Estimated with limited confidence
🔴 Unknown: Cannot determine

Tips for better results:
• Use images with GPS metadata enabled
• Choose images with clear, distinct shadows
• Outdoor daytime images work best
• Ensure accurate capture time in metadata
        """
        messagebox.showinfo("Geolocation Methods", info_text)
    
    def export_results(self):
        """Export analysis results to JSON file"""
        if not self.current_file_path:
            messagebox.showwarning("No Data", "No analysis results to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # This would export current results - for now just a placeholder
                messagebox.showinfo("Export", "Export functionality will be implemented in next version")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def clear_display(self):
        """Clear all displays"""
        self.image_label.configure(
            text="No image loaded\n\n📂 Drag and drop or click 'Upload'", 
            image=''
        )
        self.image_label.image = None
        self.image_info.configure(text="")
        
        # Clear all text widgets
        for text_widget in [self.summary_text, self.metadata_text, self.analysis_text, 
                          self.location_text, self.debug_text]:
            text_widget.delete("1.0", tk.END)
        
        self.current_device_location = None
        self.status_var.set("🟢 Ready to analyze images and videos")
    
    def _analysis_complete(self):
        """Called when analysis is complete"""
        self.progress.stop()
        self.status_var.set("🟢 Analysis complete")
    
    def _show_error(self, message):
        """Show error message in UI"""
        self.root.after(0, lambda: messagebox.showerror("Error", message))
        self.root.after(0, self._analysis_complete)

def main():
    """Main application entry point"""
    try:
        root = tk.Tk()
        app = GeolocationApp(root)
        
        # Center the window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        root.mainloop()
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")

if __name__ == "__main__":
    main()