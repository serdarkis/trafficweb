import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point
import pickle
import os
import concurrent.futures # Add concurrent.futures for parallel pre-calculation

def create_road_network():
    """Yol ağını oluşturur ve kaydeder"""
    print("Yol ağı oluşturuluyor...")
    
    # Dedektör verilerini yükle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    detector_data = pd.read_csv(os.path.join(current_dir, 'dataset', 'filtered_augsburg_detectors.csv'), header=None, delimiter=',', encoding='MacRoman')
    detector_data.columns = ['detid', 'citycode', 'length', 'pos', 'fclass', 'road', 'limit', 'city', 'lanes', 'long_scaled', 'lat_scaled']
    detector_data = detector_data.dropna(subset=['long_scaled', 'lat_scaled'])
    detector_data['long'] = detector_data['long_scaled'].astype(float)
    detector_data['lat'] = detector_data['lat_scaled'].astype(float)
    print("Dedektör verileri yüklendi.")

    # Dedektörlerin merkez noktasını hesapla
    center_lat = detector_data['lat'].mean()
    center_lon = detector_data['long'].mean()
    print(f"\nMerkez nokta: {center_lat}, {center_lon}")

    # Merkez noktadan 10km yarıçapındaki yolları indir
    print("\nMerkez noktadan 10km yarıçapındaki yollar indiriliyor...")
    try:
        # Önce tüm Augsburg'u indir
        G = ox.graph_from_place('Augsburg, Germany', network_type='drive')
        G = ox.project_graph(G)
        
        # Merkez noktadan 10km yarıçapındaki alt grafiği al
        center_point = (center_lat, center_lon)
        combined_subgraph = ox.graph_from_point(center_point, dist=5000, network_type='drive')
        combined_subgraph = ox.project_graph(combined_subgraph)
        combined_subgraph.graph['crs'] = G.graph['crs']
        
        # Alt grafiği GeoDataFrame'e dönüştür
        nodes, edges = ox.graph_to_gdfs(combined_subgraph, nodes=True, edges=True)
        edges = edges.reset_index()
        
        # Koordinatları WGS84'e dönüştür
        edges = edges.to_crs(epsg=4326)
        nodes = nodes.to_crs(epsg=4326)
        print(f"Yol ağı indirildi ve WGS84'e dönüştürüldü. Toplam {len(edges)} yol ve {len(nodes)} node.")
    except Exception as e:
        print(f"Yol ağı indirme hatası: {e}")
        raise

    # Dedektör noktalarını GeoDataFrame'e çevir
    detector_points = gpd.GeoDataFrame(
        detector_data,
        geometry=[Point(xy) for xy in zip(detector_data['long'], detector_data['lat'])],
        crs="EPSG:4326"
    )

    # KDTree oluşturulduğundan emin olun
    detector_coords = np.array([[p.x, p.y] for p in detector_points.geometry])
    detector_tree = KDTree(detector_coords)  # Bu satır çalışıyor mu?
    print("KDTree oluşturuldu.")

    # --- PRE-CALCULATE NEAREST EDGE FOR EACH DETECTOR ---
    print("\nDedektörler için en yakın yol kenarları hesaplanıyor...")
    detector_to_edge_map = {}
    precalc_unmatched_count = 0

    # Dedektör verilerini Point objelerine çevir
    detector_points_list = [(row['detid'], Point(row['long'], row['lat'])) for index, row in detector_data.iterrows()]

    # Paralel olarak en yakın kenarları bul
    # max_workers'ı makul bir sayıda tutalım, örneğin 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(detector_points_list) or 1)) as executor:
        future_to_detid = {executor.submit(ox.nearest_edges, combined_subgraph, point.x, point.y, return_dist=False): detid for detid, point in detector_points_list}

        for future in concurrent.futures.as_completed(future_to_detid):
            detid = future_to_detid[future]
            try:
                nearest_edge = future.result()
                if nearest_edge is not None:
                    # ox.nearest_edges Projected graph üzerinde (u, v, key) döner
                    # Bu tuple'ı tamsayı tuple'ı olarak saklayalım
                    detector_to_edge_map[detid] = tuple(int(x) for x in nearest_edge)
                else:
                    # print(f"Uyarı: Dedektör {detid} için en yakın kenar bulunamadı.") # Çok fazla çıktı olabilir, uyarıyı kapat
                    precalc_unmatched_count += 1
            except Exception as exc:
                print(f"Dedektör {detid} için en yakın kenarı bulmada hata: {exc}. Atlanıyor.")
                precalc_unmatched_count += 1

    print(f"Hesaplandı: {len(detector_to_edge_map)} dedektör için en yakın kenar bulundu. {precalc_unmatched_count} eşleşmeyen.")
    # --- END OF PRE-CALCULATION ---

    # Yol ağını kaydet (traffic_models dizinine)
    MODELS_DIR = os.path.join(current_dir, "traffic_models")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Yol ağını genişletilmiş veri yapısıyla kaydet
    network_data = {
        'edges': edges,
        'nodes': nodes,
        'graph': combined_subgraph,
        'detector_to_edge_map': detector_to_edge_map,
        'detector_data': detector_data,
        'detector_points': detector_points,  # Dedektör noktalarını da kaydet
        'detector_coords': detector_coords,  # Dedektör koordinatları
        'detector_tree': detector_tree,  # KDTree
    }

    network_file_path = os.path.join(MODELS_DIR, 'road_network_enhanced.pkl')
    with open(network_file_path, 'wb') as f:
        pickle.dump(network_data, f)
    
    print(f"Yol ağı kaydedildi: {network_file_path}")
    return True

if __name__ == '__main__':
    create_road_network() 