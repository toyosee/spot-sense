# services/map_visualization.py
import folium
import branca
import requests
import json
import logging
import os
from folium import plugins
import numpy as np

logger = logging.getLogger(__name__)

class MapVisualizer:
    def __init__(self):
        self.nigeria_center = [9.0820, 8.6753]  # Central Nigeria
        self.nigerian_states = self._load_nigerian_states()
        logger.info("MapVisualizer initialized")
    
    def create_location_map(self, estimated_locations, actual_location=None, terrain_type=None):
        """
        Create an interactive map showing estimated locations
        """
        try:
            # Create base map centered on Nigeria
            m = folium.Map(
                location=self.nigeria_center,
                zoom_start=6,
                tiles='OpenStreetMap'
            )
            
            # Add Nigerian state boundaries
            self._add_state_boundaries(m)
            
            # Add estimated locations
            self._add_estimated_locations(m, estimated_locations)
            
            # Add actual location if known (for testing)
            if actual_location:
                self._add_actual_location(m, actual_location)
            
            # Add terrain constraints if available
            if terrain_type:
                self._add_terrain_constraints(m, terrain_type)
            
            # Add search area visualization
            self._add_search_areas(m, estimated_locations)
            
            # Add legend
            self._add_legend(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Map creation failed: {e}")
            return None
    
    def _load_nigerian_states(self):
        """Load Nigerian state boundaries (simplified)"""
        # In production, you'd load GeoJSON data
        # For now, using approximate centers
        return {
            'Lagos': [6.5244, 3.3792],
            'Kano': [12.0023, 8.5920],
            'Abuja': [9.0765, 7.3986],
            'Rivers': [4.8156, 7.0500],
            'Oyo': [7.3775, 3.9470],
            'Kaduna': [10.5264, 7.4382],
            'Benue': [7.7326, 8.5391],
            'Bornu': [11.8333, 13.1500],
            'Delta': [5.5325, 5.8987],
            'Ondo': [7.0932, 4.8354],
            'Sokoto': [13.0070, 5.2476],
            'Plateau': [9.8965, 8.8583]
        }
    
    def _add_state_boundaries(self, map_obj):
        """Add Nigerian state boundaries to map"""
        for state, center in self.nigerian_states.items():
            folium.CircleMarker(
                location=center,
                radius=8,
                popup=f"<b>{state}</b>",
                tooltip=state,
                color='blue',
                fillColor='lightblue',
                fillOpacity=0.5
            ).add_to(map_obj)
    
    def _add_estimated_locations(self, map_obj, estimated_locations):
        """Add estimated locations to map with confidence indicators"""
        colors = ['red', 'orange', 'green', 'purple', 'darkred']
        
        for i, (location_data, confidence, details) in enumerate(estimated_locations):
            if location_data:
                lat, lon = location_data
                
                # Color based on confidence
                color = 'red'
                if confidence > 70:
                    color = 'green'
                elif confidence > 50:
                    color = 'orange'
                elif confidence > 30:
                    color = 'yellow'
                
                # Size based on confidence
                radius = max(8, min(20, confidence / 5))
                
                popup_text = f"""
                <b>Estimated Location {i+1}</b><br>
                <b>Coordinates:</b> {lat:.4f}, {lon:.4f}<br>
                <b>Confidence:</b> {confidence:.1f}%<br>
                <b>Method:</b> {details.get('estimation_method', 'Unknown')}<br>
                <b>Region:</b> {details.get('region', 'Unknown')}<br>
                <b>Reliability:</b> {details.get('reliability', 'Unknown')}
                """
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    popup=popup_text,
                    tooltip=f"Estimate {i+1}: {confidence:.1f}%",
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(map_obj)
    
    def _add_actual_location(self, map_obj, actual_location):
        """Add actual location marker (for testing/validation)"""
        lat, lon = actual_location
        folium.Marker(
            location=[lat, lon],
            popup="<b>ACTUAL LOCATION</b>",
            tooltip="Actual Photo Location",
            icon=folium.Icon(color='black', icon='info-sign', prefix='fa')
        ).add_to(map_obj)
        
        # Add a circle around actual location
        folium.Circle(
            location=[lat, lon],
            radius=50000,  # 50km radius
            popup="50km radius around actual location",
            color='black',
            fillColor='black',
            fillOpacity=0.1,
            weight=2
        ).add_to(map_obj)
    
    def _add_terrain_constraints(self, map_obj, terrain_type):
        """Add terrain-appropriate regions to map"""
        terrain_regions = {
            'urban': ['Lagos', 'Abuja', 'Kano', 'Port Harcourt', 'Ibadan'],
            'forest': ['Cross River', 'Ondo', 'Delta', 'Ogun'],
            'agricultural': ['Benue', 'Kano', 'Kaduna', 'Plateau'],
            'savanna': ['Sokoto', 'Katsina', 'Bornu', 'Kano'],
            'coastal': ['Lagos', 'Delta', 'Rivers', 'Bayelsa']
        }
        
        regions = terrain_regions.get(terrain_type, [])
        for region in regions:
            if region in self.nigerian_states:
                center = self.nigerian_states[region]
                folium.Circle(
                    location=center,
                    radius=80000,  # 80km radius
                    popup=f"<b>{region}</b><br>Typical for {terrain_type} terrain",
                    color='green',
                    fillColor='green',
                    fillOpacity=0.2,
                    weight=1
                ).add_to(map_obj)
    
    def _add_search_areas(self, map_obj, estimated_locations):
        """Add search areas around estimated locations"""
        for i, (location_data, confidence, details) in enumerate(estimated_locations):
            if location_data and confidence > 30:
                lat, lon = location_data
                
                # Search radius based on confidence
                radius = max(20000, 100000 - (confidence * 1000))  # 20km to 100km
                
                folium.Circle(
                    location=[lat, lon],
                    radius=radius,
                    popup=f"Search Area {i+1}<br>Radius: {radius/1000:.0f}km",
                    color='blue',
                    fillColor='blue',
                    fillOpacity=0.1,
                    weight=1
                ).add_to(map_obj)
    
    def _add_legend(self, map_obj):
        """Add legend to the map"""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; height: 220px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Map Legend</h4>
        <p><i style="background: green; width: 15px; height: 15px; display: inline-block;"></i> High Confidence (>70%)</p>
        <p><i style="background: orange; width: 15px; height: 15px; display: inline-block;"></i> Medium Confidence (50-70%)</p>
        <p><i style="background: red; width: 15px; height: 15px; display: inline-block;"></i> Low Confidence (<50%)</p>
        <p><i style="background: black; width: 15px; height: 15px; display: inline-block;"></i> Actual Location</p>
        <p><i style="background: blue; width: 15px; height: 15px; display: inline-block;"></i> Nigerian States</p>
        <p><i style="background: lightgreen; width: 15px; height: 15px; display: inline-block;"></i> Terrain Regions</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def save_map(self, map_obj, filename="location_analysis.html"):
        """Save map to HTML file"""
        try:
            map_obj.save(filename)
            logger.info(f"Map saved as {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save map: {e}")
            return None