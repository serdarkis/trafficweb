from flask import Flask, jsonify, request
import pandas as pd
import pickle
import traceback
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
import osmnx as ox
from datetime import datetime, timedelta
from prophet import Prophet
import json
import os
import subprocess
import re
from functools import lru_cache
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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

# Assuming the original raw data is in this file
RAW_DATA_FILE = os.path.join(MODELS_DIR, 'traffic_data_processed.csv')

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

def predict_with_prophet(detid, det_data, data_type):
    """Prophet ile tahmin yapar ve sonucu döndürür."""
    try:
        model = Prophet(
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        df = det_data[['ds', data_type]].rename(columns={data_type: 'y'}).dropna()
        if len(df) >= 2:
            model.fit(df)
            future = model.make_future_dataframe(periods=1, freq='h')
            forecast = model.predict(future)
            return (data_type, forecast['yhat'].iloc[-1])
        return (data_type, det_data[data_type].mean())
    except Exception as e:
        print(f"Prophet tahmin hatası (detid {detid}): {str(e)}")
        return (data_type, det_data[data_type].mean())

def process_detector(det, map_data):
    """Her dedektör için tahmin yapar ve sonuçları döndürür."""
    detid = str(det['detid'])
    det_data = map_data[map_data['detid'] == detid]
    
    if len(det_data) < 2:
        print(f"Uyarı: Dedektör {detid} için yetersiz veri ({len(det_data)} kayıt)")
        return None, None, None
    
    try:
        # Flow tahmini
        flow_result = predict_with_prophet(detid, det_data, 'flow')
        # Occupancy tahmini
        occ_result = predict_with_prophet(detid, det_data, 'occ')
        return flow_result[1], occ_result[1], detid
    except Exception as e:
        print(f"Prophet tahmin hatası (detid {detid}): {str(e)}")
        if not det_data.empty:
            return det_data['flow'].mean(), det_data['occ'].mean(), detid
        return None, None, None

def prepare_road_data_from_map(map_data, edges_gdf):
    """Ham veriyi işler ve sonuçları döndürür."""
    global road_network_data
    
    if road_network_data is None:
        road_network_data = load_road_network_enhanced()
        if road_network_data is None:
            print("Hata: Yol ağı verisi yüklenemedi!")
            return []

    detector_tree = road_network_data.get('detector_tree')
    detector_data_df = road_network_data.get('detector_data')
    
    if detector_tree is None or detector_data_df is None:
        print("Hata: Dedektör verileri eksik!")
        return []

    # Debug için sayaçlar
    total_edges = len(edges_gdf)
    processed_edges = 0
    
    # Timestamp sütununu hazırla
    map_data['ds'] = pd.to_datetime(map_data['day'])
    roads_data = []

    # Tüm dedektörler için tahmin yap (paralel)
    unique_detectors = detector_data_df['detid'].unique()
    detector_predictions = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_detector, {'detid': detid}, map_data): detid for detid in unique_detectors}
        for future in as_completed(futures):
            detid = futures[future]
            flow, occ, _ = future.result()
            if flow is not None and occ is not None:
                detector_predictions[detid] = {'flow': flow, 'occ': occ}

    # Tüm yolları işle
    for edge_index_key, edge in edges_gdf.iterrows():
        processed_edges += 1
        print(f"İşlenen yol: {processed_edges}/{total_edges} - {edge_index_key}")
        
        if not hasattr(edge.geometry, 'coords'):
            continue
            
        coords = list(edge.geometry.coords)
        coords_leaflet = [[lat, lon] for lon, lat in coords]
        mid_point = list(edge.geometry.coords)[len(coords)//2]
        
        # En yakın 3 dedektörü bul
        try:
            _, nearest_indices = detector_tree.query([mid_point[0], mid_point[1]], k=3)
            nearest_detectors = [detector_data_df.iloc[idx] for idx in nearest_indices]
        except Exception as e:
            print(f"Dedektör bulma hatası: {str(e)}")
            continue

        flow_predictions = []
        occ_predictions = []
        for det in nearest_detectors:
            detid = str(det['detid'])
            if detid in detector_predictions:
                flow_predictions.append(detector_predictions[detid]['flow'])
                occ_predictions.append(detector_predictions[detid]['occ'])

        # Ortalama değerleri hesapla
        avg_flow = np.mean(flow_predictions) if flow_predictions else None
        avg_occ = np.mean(occ_predictions) if occ_predictions else None
        
        # NaN değerlerini null ile değiştir
        if avg_flow is not None and np.isnan(avg_flow):
            avg_flow = None
        if avg_occ is not None and np.isnan(avg_occ):
            avg_occ = None

        roads_data.append({
            'id': str(edge_index_key),
            'coords': coords_leaflet,
            'flow': avg_flow,
            'occ': avg_occ,
            'detector_count': len(flow_predictions)
        })

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
        # This part processes road segments based on nearest detectors
        roads_data_for_frontend = prepare_road_data_from_map(raw_data, edges_gdf)

        # Dedektör verilerini hazırla - Use Prophet predictions for individual detectors
        detector_points_data = []
        if 'detector_data' in road_network_data and 'detector_coords' in road_network_data:
            detector_data_df = road_network_data.get('detector_data')
            detector_coords = road_network_data.get('detector_coords')

            if detector_data_df is not None and detector_coords is not None:
                # Get unique detector IDs present in the loaded raw data
                unique_detectors_in_raw_data = raw_data['detid'].unique()

                # Use ThreadPoolExecutor for processing detectors in parallel
                detector_predictions = {}
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {}
                    for detid in unique_detectors_in_raw_data:
                         # Filter raw data for the specific detector
                        det_data_subset = raw_data[raw_data['detid'] == detid].copy() # Use .copy() to avoid SettingWithCopyWarning
                        futures[executor.submit(process_detector, {'detid': detid}, det_data_subset)] = detid

                    for future in as_completed(futures):
                        detid = futures[future]
                        flow, occ, _ = future.result()
                        # Only store results if predictions were successful
                        if flow is not None and occ is not None:
                             # Convert numpy NaNs to None for JSON
                             flow_val = None if isinstance(flow, float) and np.isnan(flow) else flow
                             occ_val = None if isinstance(occ, float) and np.isnan(occ) else occ
                             detector_predictions[detid] = {'flow': flow_val, 'occ': occ_val}


                # Now populate detector_points_data with predictions
                # Match predictions with coordinates using detid
                for idx, row in detector_data_df.iterrows():
                    detid = row['detid']
                    # Ensure detid is handled consistently as string if needed
                    detid_str = str(detid)

                    if detid_str in detector_predictions:
                        coords = detector_coords[idx]
                        prediction = detector_predictions[detid_str]

                        detector_points_data.append({
                            'coords': [coords[1], coords[0]],  # [lat, lon]
                            'flow': prediction['flow'],
                            'occ': prediction['occ'],
                            'detid': detid_str
                        })
                    # Optional: include detectors without data/predictions, maybe with flow/occ as None
                    # else:
                    #      coords = detector_coords[idx]
                    #      detector_points_data.append({
                    #         'coords': [coords[1], coords[0]],  # [lat, lon]
                    #         'flow': None,
                    #         'occ': None,
                    #         'detid': str(detid)
                    #     })

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

# Function to calculate week number relative to the start date of the data
# This needs to be calculated dynamically based on the data loaded
# Or we can find the earliest date in the raw data once. Let's try loading and finding it.
@lru_cache(maxsize=1) # Cache the earliest date once loaded
def get_earliest_data_date():
    """Finds the earliest date in the raw traffic data."""
    try:
        if os.path.exists(RAW_DATA_FILE):
            # Read only the 'day' column and the first few rows to find the earliest date efficiently
            # Or read the whole column if necessary
            temp_df = pd.read_csv(RAW_DATA_FILE, sep=',', usecols=['day'], dtype={'day': str})
            if not temp_df.empty:
                # Convert 'day' column to datetime objects
                temp_df['day_dt'] = pd.to_datetime(temp_df['day'], errors='coerce')
                earliest_date = temp_df['day_dt'].min()
                if pd.notnull(earliest_date):
                    print(f"Earliest data date found: {earliest_date.strftime('%Y-%m-%d')}")
                    # Return the start of the week for the earliest date (Monday)
                    return earliest_date - timedelta(days=earliest_date.weekday())
            print("Could not determine earliest date from raw data.")
            return None
        else:
            print(f"Raw data file not found: {RAW_DATA_FILE}")
            return None
    except Exception as e:
        print(f"Error finding earliest data date: {str(e)}")
        return None

@app.route('/get_historical_detector_data')
def get_historical_detector_data():
    """Loads and returns historical raw traffic data for a specific detector from the relevant interval file, grouped by week."""
    detid = request.args.get('detid')
    # We also need the selected weekday from the frontend now to load the correct file
    weekday = request.args.get('weekday')
    time_str = request.args.get('time') # "HH:MM" formatında

    if not detid or not weekday or not time_str:
        return jsonify({'error': 'detid, weekday ve time parametreleri gereklidir.'}), 400

    try:
        # Convert "HH:MM" time string to seconds
        if ':' not in time_str:
            return jsonify({'error': 'Geçersiz zaman formatı. "HH:MM" formatında olmalıdır.'}), 400

        hour, minute = map(int, time_str.split(':'))
        requested_interval_seconds = hour * 3600 + minute * 60

        # Construct the raw data file path based on the requested interval and weekday
        safe_interval_str = str(requested_interval_seconds)
        safe_weekday = weekday.replace(' ', '_').replace('/', '_').replace('\\', '_')  # Ensure filename safety
        raw_data_file_path = os.path.join(MODELS_DIR, f'raw_data_interval_{safe_interval_str}_{safe_weekday}.pkl')

        print(f"Debug (Historical): Trying to load raw data file: {raw_data_file_path}")

        if not os.path.exists(raw_data_file_path):
            print(f"Hata: Ham veri dosyası bulunamadı: {raw_data_file_path}")
            # Suggest available options if file not found for exact interval/weekday
            available_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("raw_data_interval_") and f.endswith(".pkl")]
            if available_files:
                 # Try to find closest interval for the requested weekday if exact match not found
                 pattern = re.compile(r'raw_data_interval_(\d+)_([^.]+)\.pkl')
                 available_intervals_for_weekday = []
                 for filename in available_files:
                     match = pattern.match(filename)
                     if match and match.group(2) == safe_weekday:
                          try:
                              available_intervals_for_weekday.append(int(match.group(1)))
                          except ValueError:
                              continue
                 if available_intervals_for_weekday:
                     closest_interval = min(available_intervals_for_weekday, key=lambda x: abs(x - requested_interval_seconds))
                     closest_time_str = convert_seconds_to_time(closest_interval)
                     return jsonify({'error': f'{weekday} günü ve {time_str} saati için tam eşleşen veri bulunamadı. En yakın mevcut zaman aralığı {closest_time_str}.', 'closest_time': closest_time_str}), 404
                 else:
                      return jsonify({'error': f'{weekday} günü için herhangi bir zaman aralığında veri bulunamadı.'}), 404
            else:
                 return jsonify({'error': f'Models dizininde hiç ham veri dosyası bulunamadı.'}), 404


        # Load the raw data for the specific weekday and interval
        with open(raw_data_file_path, 'rb') as f:
            raw_data_interval = pickle.load(f)

        print(f"Raw data for {weekday} at {time_str} loaded. Shape: {raw_data_interval.shape}")

        # Filter data for the specific detector
        # Ensure detid is compared consistently (assuming string from URL and in dataframe)
        filtered_data = raw_data_interval[
            raw_data_interval['detid'].astype(str) == str(detid)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning


        print(f"Filtered data shape (detid={detid}): {filtered_data.shape}")

        if filtered_data.empty:
            return jsonify({'detid': detid, 'time': time_str, 'historical_data_by_week': [], 'message': f'Bu dedektör ({detid}) için {weekday}, {time_str} zaman aralığında ham veri bulunamadı.'})

        # Determine the earliest date in THIS filtered subset to calculate relative week numbers
        # This ensures week 1 is the first week present in the data for this specific detid/weekday/interval
        filtered_data['day_dt'] = pd.to_datetime(filtered_data['day'], errors='coerce')
        # Drop rows with invalid dates before finding min
        filtered_data.dropna(subset=['day_dt'], inplace=True)

        if filtered_data.empty:
             return jsonify({'detid': detid, 'time': time_str, 'historical_data_by_week': [], 'message': f'Bu dedektör ({detid}) için geçerli tarih verisi bulunamadı.'})

        earliest_date_in_subset = filtered_data['day_dt'].min()
        # Calculate week number relative to the start of the week of the earliest date in this subset
        earliest_date_subset_start_of_week = earliest_date_in_subset - timedelta(days=earliest_date_in_subset.weekday())

        # Calculate week number for each data point
        # Week number 1 starts from the earliest_date_subset_start_of_week
        filtered_data['week_number'] = ((filtered_data['day_dt'] - earliest_date_subset_start_of_week).dt.days // 7) + 1

        # Sort by week number and day
        filtered_data = filtered_data.sort_values(by=['week_number', 'day_dt'])

        # Group by week number and format the output
        historical_data_by_week = []
        # Iterate through unique week numbers in sorted order
        for week_num in sorted(filtered_data['week_number'].unique()):
             week_data = filtered_data[filtered_data['week_number'] == week_num]

             # Assuming there's only one entry per detector/interval/day in the raw data source
             # If there are multiple, we might need to average/sum, but for now, take the first
             if not week_data.empty:
                 # Get the data point for this week (should be only one for the specific detid/interval/day combination)
                 data_point = week_data.iloc[0]
                 historical_data_by_week.append({
                     'week_number': int(week_num), # Ensure it\'s int for JSON
                     'day': data_point['day'], # Include the specific date for context
                     'flow': convert_numpy_integers(data_point.get('flow')),
                     'occ': convert_numpy_integers(data_point.get('occ'))
                 })
             # else: This case should ideally not happen if week_num comes from unique values in filtered_data


        print(f"Debug (Historical): Prepared data for detid {detid} for {weekday} at {time_str} by week:", historical_data_by_week)

        return jsonify({'detid': detid, 'time': time_str, 'weekday': weekday, 'historical_data_by_week': historical_data_by_week})

    except FileNotFoundError:
        # This case is handled above now with a more specific error message and lookup
        # This catch might still be useful for unexpected file issues
        print(f"Hata (FileNotFoundError) yükleme sonrası?: {raw_data_file_path}")
        return jsonify({'error': f'Dosya yüklenirken beklenmedik hata: {raw_data_file_path}'}), 500
    except Exception as e:
        print(f"Genel Hata oluştu (Historical - from interval file): {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

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

