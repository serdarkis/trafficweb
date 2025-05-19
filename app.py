from flask import Flask, jsonify, request
import pandas as pd
import pickle
import traceback
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
import osmnx as ox
from datetime import datetime
# Removed Prophet as per user request for interval-based average data
# from prophet import Prophet
import json
import os
import subprocess
import re

app = Flask(__name__)

# Global variables
# Removed traffic_models as we are loading interval files now
# traffic_models = {}
edges = None
nodes = None
# Removed detector_tree, detector_points, detector_data as they are part of road_network_data now
# detector_tree = None
# detector_points = None
# detector_data = None
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic_models")

# Global variable to store the loaded road network enhanced data
road_network_data = None


# Helper function to load the road network enhanced data once
def load_road_network_enhanced():
    """Loads the enhanced road network data from a pickle file."""
    network_path = os.path.join(MODELS_DIR, 'road_network_enhanced.pkl')
    if os.path.exists(network_path):
        try:
            with open(network_path, 'rb') as f:
                data = pickle.load(f)
                # Store required parts in global variables for easier access
                global edges #, nodes, detector_tree, detector_points, detector_data # Only edges is needed globally now for prepare_road_data
                edges = data.get('edges')
                # nodes = data.get('nodes')
                # detector_tree = data.get('detector_tree')
                # detector_points = data.get('detector_points')
                # detector_data = data.get('detector_data')
                print("Yol ağı başarıyla yüklendi.")
                return data # Return the full data dictionary in case other parts are needed later
        except Exception as e:
            print(f"Hata: road_network_enhanced.pkl yüklenirken hata oluştu: {str(e)}")
            return None
    else:
        print(f"Hata: road_network_enhanced.pkl dosyası bulunamadı: {network_path}")
        return None

# Helper function for comprehensive data type conversion (useful for JSON jsonify)
def convert_numpy_integers(obj):
    """Recursively converts numpy integer types to standard Python integers and floats."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
         return float(obj)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_integers(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_integers(value) for key, value in obj.items()}
    else:
        return obj

# Helper function to convert seconds to HH:MM format
def convert_seconds_to_time(seconds):
    """Saniyeyi HH:MM formatına çevirir (300 → '00:05')"""
    try:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        # secs = seconds % 60 # Not needed for HH:MM format
        return f"{hours:02d}:{minutes:02d}"  # 300 → "00:05", 3600 → "01:00"
    except: # Handle potential errors gracefully
        # print(f"Uyarı: Geçersiz saniye değeri: {seconds}") # Avoid excessive printing
        return "N/A"

# Load road network data once at the start of the application
road_network_data = load_road_network_enhanced()

# load_models function is likely obsolete with interval-based data files, keeping it for reference but it won't be used
# def load_models():
#     """Kaydedilmiş modelleri ve yol ağını diskten yükle"""
#     # This function is likely no longer needed with interval-based data files
#     # Keeping it for now but it can be removed if not used elsewhere
#     global traffic_models, edges, nodes, detector_tree, detector_points, detector_data
    
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     MODELS_DIR = os.path.join(current_dir, "traffic_models")
    
#     if not os.path.exists(MODELS_DIR):
#         print(f"Hata: {MODELS_DIR} dizini bulunamadı.")
#         return False
    
#     try:
#         # Yol ağını yükle
#         network_path = os.path.join(MODELS_DIR, 'road_network_enhanced.pkl')
#         if os.path.exists(network_path):
#             print(f"Yol ağı dosyası bulundu: {network_path}")
#             with open(network_path, 'rb') as f:
#                 network_data = pickle.load(f)
#                 # Based on the new structure, we might only need edges and nodes here
#                 edges = network_data.get('edges')
#                 nodes = network_data.get('nodes')
#                 # The following might not be needed for the new approach
#                 detector_tree = network_data.get('detector_tree')
#                 detector_points = network_data.get('detector_points')
#                 detector_data = network_data.get('detector_data')
#             print("Yol ağı başarıyla yüklendi.")
#         else:
#             print(f"Hata: {network_path} dosyası bulunamadı.")
#             return False
#     except Exception as e:
#         print(f"Yol ağı yüklenirken hata oluştu: {str(e)}")
    
#     # Modelleri yükle (this part is also likely obsolete with interval files)
#     # model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("model_") and f.endswith(".pkl")]
#     # if not model_files:
#     #     print("Hata: Model dosyaları bulunamadı.")
#     #     return False
#     # for model_file in model_files:
#     #     detid = model_file.replace("model_", "").replace(".pkl", "")
#     #     model_path = os.path.join(MODELS_DIR, model_file)
#     #     try:
#     #         with open(model_path, 'rb') as f:
#     #             traffic_models[detid] = pickle.load(f)
#     #     except Exception as e:
#     #         print(f"Model yüklenirken hata oluştu ({model_file}): {str(e)}")
#     #         continue
    
#     # print(f"{len(traffic_models)} model başarıyla yüklendi.")
#     # return len(traffic_models) > 0
#     # Returning True if road network loaded, as per-detector models are gone
#     return edges is not None and nodes is not None

def check_and_create_models():
    """Modellerin (ham veri dosyaları) varlığını kontrol et ve gerekirse oluştur"""
    global road_network_data  # Declare global usage once at the start
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Yol ağı dosyasının tam yolu
    road_network_path = os.path.join(MODELS_DIR, 'road_network_enhanced.pkl')
    print(f"Yol ağı dosyası kontrol ediliyor: {road_network_path}")

    # Yol ağı dosyasının varlığını kontrol et
    road_network_file_exists = os.path.exists(road_network_path)
    print(f"Yol ağı dosyası mevcut mu? {'Evet' if road_network_file_exists else 'Hayır'}")

    # Ham veri dosyalarının varlığını kontrol et (artık map_interval_ yerine raw_data_interval_)
    raw_data_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("raw_data_interval_") and f.endswith(".pkl")]
    raw_data_files_exist = len(raw_data_files) > 0
    print(f"Ham veri dosyaları mevcut mu? {'Evet' if raw_data_files_exist else 'Hayır'}")

    # Eğer her iki dosya da mevcutsa, yol ağı verisini yükle
    if raw_data_files_exist and road_network_file_exists:
        print("Ham veri dosyaları ve yol ağı bulundu.")
        if road_network_data is None:
            road_network_data = load_road_network_enhanced()
        return road_network_data is not None

    # Eğer eksik dosya varsa, oluşturma işlemi başlat
    print("Ham veri dosyaları veya yol ağı bulunamadı. Oluşturuluyor...")

    # Yol ağını oluştur (eğer yoksa)
    if not road_network_file_exists:
        print("Yol ağı oluşturuluyor...")
        try:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'create_road_network.py')
            print(f"Çalıştırılan komut: python {script_path}")
            
            # Script'i çalıştır ve çıktıyı göster
            result = subprocess.run(
                ['python', script_path],
                check=True,
                capture_output=True,
                text=True,
                shell=True  # Windows için shell=True ekleyin
            )
            print("create_road_network.py Çıktısı:")
            print(result.stdout)
            if result.stderr:
                print("Hata Çıktısı:")
                print(result.stderr)

            # Oluşturulan dosyayı yükle
            road_network_data = load_road_network_enhanced()
            if road_network_data is None:
                print("Hata: Yol ağı oluşturuldu ancak yüklenemedi.")
                return False
        except Exception as e:
            print(f"Yol ağı oluşturma hatası: {str(e)}")
            return False

    # Ham veri dosyalarını oluştur (eğer yoksa)
    if not raw_data_files_exist:
        print("Ham veri dosyaları oluşturuluyor...")
        try:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'create_traffic_models.py')
            print(f"Çalıştırılan komut: python {script_path}")
            
            # Script'i çalıştır ve çıktıyı göster
            result = subprocess.run(
                ['python', script_path],
                check=True,
                capture_output=True,
                text=True,
                shell=True  # Windows için shell=True ekleyin
            )
            print("create_traffic_models.py Çıktısı:")
            print(result.stdout)
            if result.stderr:
                print("Hata Çıktısı:")
                print(result.stderr)

            # Oluşturulan dosyaları kontrol et (artık raw_data_interval_)
            raw_data_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("raw_data_interval_") and f.endswith(".pkl")]
            if not raw_data_files:
                print("Hata: Ham veri dosyaları oluşturulamadı.")
                return False
        except Exception as e:
            print(f"Ham veri dosyaları oluşturma hatası: {str(e)}")
            return False

    # Her şey başarılıysa True döndür
    return True

# get_weekday and get_date_from_weekday are not directly used in app.py's new logic
# Keeping them for reference but they can be removed if not used elsewhere
def get_weekday(date_str):
    """Convert date string to weekday name in Turkish"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        weekdays = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
        return weekdays[date_obj.weekday()]
    except:
        return None

def get_date_from_weekday(weekday):
    """Convert Turkish weekday name to a date string (Not relevant for interval files)"""
    # This function is not needed for the interval file approach
    weekdays = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
    try:
        # Bugünün tarihini al
        today = datetime.now()
        # Bugünün haftanın hangi günü olduğunu bul
        current_weekday = today.weekday()
        # İstenen günün indeksini bul
        target_weekday = weekdays.index(weekday)
        # Gün farkını hesapla
        days_diff = target_weekday - current_weekday
        # Tarihi ayarla
        target_date = today.replace(day=today.day + days_diff)
        return target_date.strftime('%Y-%m-%d')
    except:
        return None

# predict_traffic function is obsolete with interval-based data files
# def predict_traffic(detid, weekday, time_str):
#     """Belirli bir dedektör için belirli bir tarih ve saatte trafik tahmini yapar"""
#     # This function is likely obsolete with interval-based data files
#     # ... (rest of the obsolete function) ...
#     return None, None # This function is no longer used in the new approach

# predict_with_prophet function is replaced by simple_average_prediction in the new logic
# def predict_with_prophet(map_data1, map_data2, time_str):
#     """İki interval verisini kullanarak Prophet ile tahmin yapar"""
#     # This function is replaced by simple_average_prediction in the new logic
#     # ... (rest of the obsolete function) ...
#     print(f"Prophet tahmini sırasında hata: {str(e)}") # Keeping the error print from original
#     raise # Re-raise the exception as before

def prepare_road_data_from_map(map_data, edges_gdf):
    """Interval ham veri DataFrame'inden yol verilerini frontend formatına çevirir ve trafik verilerini atar."""
    roads_data = []
    
    print("Debug [prepare_road_data_from_map]: Fonksiyon başladı.")
    
    # Dedektör verilerini yükle
    detector_data = {}
    if isinstance(map_data, pd.DataFrame) and 'detid' in map_data.columns:
        for _, row in map_data.iterrows():
            detid = str(row['detid'])
            detector_data[detid] = {
                'flow': float(row['flow']) if pd.notna(row['flow']) else None,
                'occ': float(row['occ']) if pd.notna(row['occ']) else None
            }
        print(f"Debug: {len(detector_data)} dedektör verisi yüklendi.")
    else:
        print("Hata: Geçersiz map_data formatı.")
        return []

    global road_network_data
    if road_network_data is None:
        road_network_data = load_road_network_enhanced()
        if road_network_data is None:
            print("Hata: Yol ağı yüklenemedi.")
            return []

    # KDTree ve dedektör koordinatlarını al
    detector_tree = road_network_data.get('detector_tree')
    detector_coords = road_network_data.get('detector_coords')
    detector_data_df = road_network_data.get('detector_data')
    
    if detector_tree is None or detector_coords is None:
        print("Hata: Dedektör ağacı veya koordinatları eksik.")
        return []

    # Her yol için işlem yap
    for edge_index_key, edge in edges_gdf.iterrows():
        if not hasattr(edge.geometry, 'coords'):
            continue
            
        coords = list(edge.geometry.coords)
        coords_leaflet = [[lat, lon] for lon, lat in coords]
        
        # Yolun orta noktasını bul
        mid_idx = len(coords) // 2
        mid_point = coords[mid_idx]
        
        # En yakın 3 dedektörü bul
        _, nearest_indices = detector_tree.query([mid_point[0], mid_point[1]], k=3)
        nearest_detectors = [detector_data_df.iloc[idx] for idx in nearest_indices]
        
        # Trafik verilerini topla
        flows = []
        occs = []
        
        for det in nearest_detectors:
            detid = str(det['detid'])
            if detid in detector_data:
                traffic = detector_data[detid]
                if traffic['flow'] is not None:
                    flows.append(traffic['flow'])
                if traffic['occ'] is not None:
                    occs.append(traffic['occ'])
        
        # Ortalama değerleri hesapla
        avg_flow = sum(flows) / len(flows) if flows else None
        avg_occ = sum(occs) / len(occs) if occs else None

        roads_data.append({
            'id': str(edge_index_key),
            'coords': coords_leaflet,
            'flow': avg_flow,
            'occ': avg_occ,
            'detector_count': len(flows)
        })
    
    print(f"Debug: {len(roads_data)} yol verisi hazırlandı.")
    return convert_numpy_integers(roads_data)


# Modelleri kontrol et ve gerekirse oluştur - This now checks/creates interval files and road network file
check_and_create_models()

@app.route('/')
def index():
    return app.send_static_file('augsburg_traffic_map.html')

# get_available_intervals endpoint is redundant as get_date_time_options provides the same info
# Keeping it commented out for reference
# @app.route('/get_available_intervals')
# def get_available_intervals():
#     """Available weekday and interval combinations from map files."""
#     # ... (rest of the redundant function) ...
#     pass

@app.route('/get_date_time_options')
def get_date_time_options():
    """Provides available weekday and time interval options by scanning raw data files"""
    try:
        # Check if MODELS_DIR exists and is readable
        if not os.path.exists(MODELS_DIR):
            print(f"Hata: MODELS_DIR bulunamadı: {MODELS_DIR}")
            return jsonify({'error': f'Models directory not found: {MODELS_DIR}'}), 500
        
        # Get list of raw data files
        raw_data_files = [f for f in os.listdir(MODELS_DIR) 
                         if f.startswith("raw_data_interval_") and f.endswith(".pkl")]
        
        if not raw_data_files:
            print(f"Uyarı: {MODELS_DIR} dizininde ham veri dosyası bulunamadı.")
            return jsonify({'error': 'No raw data files found'}), 404

        # Regex to extract interval and weekday from filenames like raw_data_interval_300_Cuma.pkl
        pattern = re.compile(r'raw_data_interval_(\d+)_([^.]+)\.pkl')

        # Extract unique weekdays and intervals
        weekdays = set()
        intervals = set()
        time_display_map = {}  # interval -> "HH:MM" eşlemesi
        
        for filename in raw_data_files:
            match = pattern.match(filename)
            if match:
                try:
                    interval_str = match.group(1)
                    weekday = match.group(2)
                    interval = int(interval_str)
                    weekdays.add(weekday)
                    intervals.add(interval)
                    # Store the time display string for the interval
                    time_display_map[interval] = convert_seconds_to_time(interval)
                except ValueError:
                    print(f"Uyarı: Geçersiz interval formatı veya dosya adı bulundu: {filename}")
                    continue
            else:
                print(f"Uyarı: Beklenmeyen dosya adı formatı bulundu: {filename}. Atlanıyor.")

        # Create sorted lists
        sorted_weekdays = sorted(list(weekdays))
        sorted_intervals = sorted(list(intervals))
        
        # Create the ordered time_display list based on sorted intervals
        sorted_time_display = [time_display_map.get(i, "N/A") for i in sorted_intervals]

        response_data = {
            'weekdays': sorted_weekdays,
            'intervals': sorted_intervals,  # Return as list for JSON
            'time_display': sorted_time_display,  # Ordered list matching sorted_intervals
            'time_display_map': time_display_map  # Full map for convenience if needed
        }
        
        print("Debugging date_time_options response:", response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# get_traffic_data endpoint is obsolete as per user's approach
@app.route('/get_traffic_data')
def get_traffic_data():
    # This endpoint is obsolete. Keeping it as a placeholder.
    return jsonify({'error': 'This endpoint is obsolete. Use /get_predicted_traffic instead.'}), 400

# get_traffic_map endpoint seems to be intended for the new logic, renaming or using get_predicted_traffic
# The frontend calls /get_predicted_traffic, so let's update that one.
# @app.route('/get_traffic_map')
# def get_traffic_map():
#     # This endpoint will be replaced by the logic in get_predicted_traffic
#     pass

@app.route('/get_predicted_traffic')
def get_predicted_traffic():
    """Loads and returns traffic data for a specific weekday and interval from the pre-processed raw data files."""
    # Get the interval and weekday from the request
    weekday = request.args.get('weekday')
    time_str = request.args.get('time')  # "HH:MM" formatında

    if not weekday or not time_str or time_str.lower() == 'undefined' or time_str == '0':
        return jsonify({'error': 'weekday ve time parametreleri gereklidir ve geçerli olmalıdır.'}), 400

    try:
        # Convert "HH:MM" time string to seconds
        if ':' not in time_str:
            return jsonify({'error': 'Geçersiz zaman formatı. "HH:MM" formatında olmalıdır.'}), 400

        hour, minute = map(int, time_str.split(':'))
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return jsonify({'error': 'Geçersiz saat veya dakika değeri.'}), 400

        requested_seconds = hour * 3600 + minute * 60

        # Scan available interval files to find the closest match
        available_intervals = []
        if os.path.exists(MODELS_DIR):
            pattern = re.compile(r'raw_data_interval_(\d+)_([^.]+)\.pkl')
            for filename in os.listdir(MODELS_DIR):
                match = pattern.match(filename)
                if match:
                    try:
                        interval_seconds = int(match.group(1))
                        available_intervals.append(interval_seconds)
                    except ValueError:
                        continue
        
        available_intervals = sorted(list(set(available_intervals)))  # Ensure unique and sorted
        
        if not available_intervals:
            return jsonify({'error': 'Sunucuda hiç ham veri dosyası bulunamadı.'}), 500

        # Find the closest interval to the requested time
        closest_interval = min(available_intervals, key=lambda x: abs(x - requested_seconds))
        interval_to_load = closest_interval  # Use the closest interval found

        print(f"Debug: Requested time {time_str} ({requested_seconds}s), closest interval found: {closest_interval}s ({convert_seconds_to_time(closest_interval)})")

        # Construct the raw data file path based on the closest interval and weekday
        safe_interval_str = str(interval_to_load)
        safe_weekday = weekday.replace(' ', '_').replace('/', '_').replace('\\', '_')  # Ensure filename safety
        raw_data_file_path = os.path.join(MODELS_DIR, f'raw_data_interval_{safe_interval_str}_{safe_weekday}.pkl')

        print(f"Debug: Trying to load raw data file: {raw_data_file_path}")

        if not os.path.exists(raw_data_file_path):
            print(f"Hata: Ham veri dosyası bulunamadı: {raw_data_file_path}")
            return jsonify({'error': f'Raw data file not found for weekday {weekday} and time {time_str}.'}), 404

        # Load the raw data file
        with open(raw_data_file_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        print("Debug: Raw data loaded successfully.")
        print("Debug: raw_data structure:", raw_data.keys())

        # Ensure road_network_data is loaded to get edge geometries
        global road_network_data, edges
        if road_network_data is None or edges is None:
            print("Uyarı: Road network data uygulama başlatıldıktan sonra yüklenememiş, şimdi yükleniyor...")
            road_network_data = load_road_network_enhanced()  # Attempt to load again
            if road_network_data is None or edges is None:
                print("Hata: Yol ağı verisi yüklenemedi.")
                return jsonify({'error': 'Failed to load road network data.'}), 500

        # Use the globally loaded edges_gdf
        edges_gdf = edges
        if edges_gdf is None:  # Double check in case load_road_network_enhanced failed partially
            print("Hata: Yol ağı verisi edges geometries içermiyor (global değişken boş).")
            return jsonify({'error': 'Road network data is missing edge geometries.'}), 500

        # Prepare the response data for frontend
        roads_data_for_frontend = prepare_road_data_from_map(raw_data, edges_gdf)

        # Dedektör verilerini hazırla
        detector_points_data = []
        if 'detector_data' in road_network_data and 'detector_coords' in road_network_data:
            for idx, coords in enumerate(road_network_data['detector_coords']):
                detid = road_network_data['detector_data'].iloc[idx]['detid']
                detector_traffic = raw_data[raw_data['detid'] == detid]
                flow = detector_traffic['flow'].mean() if not detector_traffic.empty else None
                occ = detector_traffic['occ'].mean() if not detector_traffic.empty else None
                
                detector_points_data.append({
                    'coords': [coords[1], coords[0]],  # [lat, lon]
                    'flow': flow,
                    'occ': occ,
                    'detid': detid  # detid'yi eklediğinizden emin olun
                })

        response = {
            'weekday': weekday,
            'interval': interval_to_load,
            'time_display': convert_seconds_to_time(interval_to_load),
            'roads': roads_data_for_frontend,
            'detectors': detector_points_data
        }

        print("\n--- Debugging Response Data Sent to Frontend ---")
        print(f"Weekday: {response.get('weekday')}")
        print(f"Interval: {response.get('interval')}")
        print(f"Time Display: {response.get('time_display')}")
        print(f"Number of roads: {len(response.get('roads', []))}")
        print(f"Number of detectors: {len(response.get('detectors', []))}")
        print("Sample Road Data:")
        for i, road in enumerate(response.get('roads', [])[:5]):
            print(f"  Road {i+1}: ID={road.get('id')}, Flow={road.get('flow')}, Occ={road.get('occ')}, Coords Sample (first 2): {road.get('coords', [])[:2]}")
        print("Sample Detector Data:")
        for i, detector in enumerate(response.get('detectors', [])[:5]):
            print(f"  Detector {i+1}: ID={detector.get('detid')}, Flow={detector.get('flow')}, Occ={detector.get('occ')}, Coords: {detector.get('coords')}")
        print("---------------------------------------------")

        return jsonify(response)

    except FileNotFoundError:
        print(f"Hata (FileNotFoundError): Ham veri dosyası bulunamadı.")
        return jsonify({'error': f'Raw data file not found for weekday {weekday} and time {time_str}.'}), 404
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# simple_average_prediction function is now part of the create_traffic_models logic
# and the loading endpoint (/get_predicted_traffic) directly loads the result
# def simple_average_prediction(map_data1, map_data2):
#     """Prophet hatası durumunda basit bir ortalama tahmini yapar"""
#     # This logic is now part of the create_traffic_models script
#     pass

def load_interval_data(weekday, interval):
    """Belirtilen haftanın günü ve interval için veriyi yükler"""
    # This helper is now integrated directly into get_predicted_traffic
    pass

@app.route('/get_road_network')
def get_road_network():
    global road_network_data, edges
    if road_network_data is None or edges is None:
        road_network_data = load_road_network_enhanced()
        if road_network_data is None or edges is None:
            return jsonify({'error': 'Yol ağı verileri yüklenemedi.'}), 500

    # Dedektör verilerini hazırla
    detector_points_data = []
    if 'detector_coords' in road_network_data and 'detector_data' in road_network_data:
        for idx, coords in enumerate(road_network_data['detector_coords']):
            detid = road_network_data['detector_data'].iloc[idx]['detid']
            detector_points_data.append({
                'coords': [coords[1], coords[0]],  # [lat, lon]
                'detid': detid  # detid'yi eklediğinizden emin olun
            })

    return jsonify({
        'roads': [{
            'id': str(edge.name),
            'coords': [[lat, lon] for lon, lat in edge.geometry.coords]
        } for _, edge in edges.iterrows()],
        'detectors': detector_points_data  # Dedektörleri döndür
    })

if __name__ == '__main__':
    # Ensure models and road network are checked/created before running the app
    # This also loads the road network data into road_network_data global variable
    print("Uygulama başlatılıyor. Gerekli veri dosyaları kontrol ediliyor/oluşturuluyor...")
    if check_and_create_models():
        print("Gerekli veri dosyaları mevcut veya başarıyla oluşturuldu. Uygulama çalıştırılıyor.")
        # Set debug=True only during development for easier debugging
        app.run(debug=False) # Set debug=False for production
    else:
        print("Uygulama başlatılamadı: Gerekli modeller veya yol ağı oluşturulamadı/yüklenemedi.")


