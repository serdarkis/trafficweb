from flask import Flask, jsonify, request
import pandas as pd
import pickle
import traceback
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
import osmnx as ox

app = Flask(__name__)

# Global değişkenler
traffic_data = None
detector_data = None
edges = None
nodes = None
detector_tree = None
detector_points = None

def load_data():
    global traffic_data, detector_data, edges, nodes, detector_tree, detector_points
    
    print("Veriler yükleniyor...")
    try:
        # Trafik verilerini yükle
        traffic_data = pd.read_csv("filtered_augsburg_new.csv", sep=',', 
                                  dtype={
                                      'day': str,
                                      'interval': int,
                                      'detid': str,
                                      'flow': float,
                                      'occ': float,
                                      'speed': float,
                                      'city': str
                                  })
        print("Trafik verileri yüklendi.")

        # Dedektör verilerini yükle
        detector_data = pd.read_csv("filtered_augsburg_detectors.csv", header=None, delimiter=',', encoding='MacRoman')
        detector_data.columns = ['detid', 'citycode', 'length', 'pos', 'fclass', 'road', 'limit', 'city', 'lanes', 'long_scaled', 'lat_scaled']
        detector_data = detector_data.dropna(subset=['long_scaled', 'lat_scaled'])
        detector_data['long'] = detector_data['long_scaled'].astype(float)
        detector_data['lat'] = detector_data['lat_scaled'].astype(float)
        print("Dedektör verileri yüklendi.")

        # Yol ağını indir
        print("\nYol ağı indiriliyor...")
        G = ox.graph_from_place('Augsburg, Germany', network_type='drive')
        G = ox.project_graph(G)
        nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
        edges = edges.reset_index()
        
        # Koordinatları WGS84'e dönüştür
        edges = edges.to_crs(epsg=4326)
        nodes = nodes.to_crs(epsg=4326)
        print(f"Yol ağı indirildi ve WGS84'e dönüştürüldü. Toplam {len(edges)} yol ve {len(nodes)} node.")

        # Dedektör noktalarını GeoDataFrame'e çevir
        detector_points = gpd.GeoDataFrame(
            detector_data,
            geometry=[Point(xy) for xy in zip(detector_data['long'], detector_data['lat'])],
            crs="EPSG:4326"
        )

        # KDTree oluştur (hızlı en yakın nokta araması için)
        detector_coords = np.array([[p.x, p.y] for p in detector_points.geometry])
        detector_tree = KDTree(detector_coords)
        print("KDTree oluşturuldu.")

    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        raise

# Verileri yükle
load_data()

@app.route('/')
def index():
    return app.send_static_file('augsburg_traffic_map.html')

@app.route('/get_traffic_data')
def get_traffic_data():
    try:
        date = request.args.get('date')
        time_str = request.args.get('time')
        
        # Saati sayısal değere dönüştür
        hours, minutes = map(int, time_str.split(':'))
        time = hours * 4 + minutes // 15
        
        print(f"\nTrafik verileri isteniyor - Tarih: {date}, Saat: {time_str}")
        
        # Seçilen tarih ve saat için trafik verilerini al
        current_traffic = traffic_data[
            (traffic_data['day'] == date) & 
            (traffic_data['interval'] == time)
        ]
        
        # Dedektör verilerini hazırla
        detectors = []
        for _, row in detector_data.iterrows():
            det_traffic = current_traffic[current_traffic['detid'] == row['detid']]
            if not det_traffic.empty:
                traffic = det_traffic.iloc[0]
                detector = {
                    'id': row['detid'],
                    'lat': float(row['lat']),
                    'lon': float(row['long']),
                    'flow': float(traffic['flow']) if pd.notnull(traffic['flow']) else 0,
                    'occ': float(traffic['occ']) if pd.notnull(traffic['occ']) else 0,
                    'speed': float(traffic['speed']) if pd.notnull(traffic['speed']) else 0
                }
                detectors.append(detector)
        
        # Yol verilerini hazırla
        roads = []
        for idx, edge in edges.iterrows():
            if edge.geometry is not None:
                try:
                    # Yolun başlangıç ve bitiş noktaları
                    start_point = Point(edge.geometry.coords[0])
                    end_point = Point(edge.geometry.coords[-1])
                    
                    # En yakın dedektörleri bul
                    start_idx = detector_tree.query([start_point.x, start_point.y])[1]
                    end_idx = detector_tree.query([end_point.x, end_point.y])[1]
                    
                    start_detector = detector_data.iloc[start_idx]
                    end_detector = detector_data.iloc[end_idx]
                    
                    # Trafik verilerini al
                    start_traffic = current_traffic[current_traffic['detid'] == start_detector['detid']]
                    end_traffic = current_traffic[current_traffic['detid'] == end_detector['detid']]
                    
                    # Trafik değerlerini hesapla
                    flow = 0
                    occ = 0
                    speed = 0
                    count = 0
                    
                    if not start_traffic.empty:
                        start_row = start_traffic.iloc[0]
                        flow += float(start_row['flow']) if pd.notnull(start_row['flow']) else 0
                        occ += float(start_row['occ']) if pd.notnull(start_row['occ']) else 0
                        speed += float(start_row['speed']) if pd.notnull(start_row['speed']) else 0
                        count += 1
                    
                    if not end_traffic.empty:
                        end_row = end_traffic.iloc[0]
                        flow += float(end_row['flow']) if pd.notnull(end_row['flow']) else 0
                        occ += float(end_row['occ']) if pd.notnull(end_row['occ']) else 0
                        speed += float(end_row['speed']) if pd.notnull(end_row['speed']) else 0
                        count += 1
                    
                    if count > 0:
                        flow /= count
                        occ /= count
                        speed /= count
                    
                    # Koordinatları doğru formatta al (WGS84 - [lat, lon])
                    coords = []
                    for lon, lat in edge.geometry.coords:
                        coords.append([float(lat), float(lon)])  # Leaflet.js için [lat, lon] formatı
                    
                    # Debug için koordinat sayısını yazdır
                    print(f"Yol {idx}: {len(coords)} koordinat")
                    if len(coords) > 0:
                        print(f"İlk koordinat: {coords[0]}")
                    
                    road = {
                        'coords': coords,
                        'flow': flow,
                        'occ': occ,
                        'speed': speed
                    }
                    roads.append(road)
                    
                except Exception as e:
                    print(f"Yol {idx} işlenirken hata: {str(e)}")
                    continue
        
        print(f"Toplam {len(roads)} yol hazırlandı")
        
        response_data = {
            'detectors': detectors,
            'roads': roads
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/get_date_time_options')
def get_date_time_options():
    try:
        unique_dates = sorted(traffic_data['day'].unique())
        unique_times = sorted(traffic_data['interval'].unique())
        
        formatted_times = []
        for time in unique_times:
            hours = time // 4
            minutes = (time % 4) * 15
            formatted_time = f"{hours:02d}:{minutes:02d}"
            formatted_times.append(formatted_time)
        
        formatted_dates = [str(date) for date in unique_dates]
        
        return jsonify({
            'dates': formatted_dates,
            'times': formatted_times
        })
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 