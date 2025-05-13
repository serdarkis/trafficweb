import osmnx as ox
import networkx as nx
import pandas as pd
import folium
from shapely.geometry import Point, LineString
from geopandas import GeoSeries
from scipy.spatial import KDTree
import geopandas as gpd
import pickle

# 1. Yol ağını indir
G = ox.graph_from_place('Augsburg, Germany', network_type='drive')
G = ox.project_graph(G)
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
edges = edges.reset_index()

# 2. Dedektör verisini yükle
detector_data = pd.read_csv("filtered_augsburg_detectors.csv", header=None, delimiter=',', encoding='MacRoman')
detector_data.columns = ['detid', 'citycode', 'length', 'pos', 'fclass', 'road', 'limit', 'city', 'lanes', 'long_scaled', 'lat_scaled']
detector_data = detector_data.dropna(subset=['long_scaled', 'lat_scaled'])
detector_data['long'] = detector_data['long_scaled'].astype(float)
detector_data['lat'] = detector_data['lat_scaled'].astype(float)

# Trafik verilerini yükle
print("\nTrafik verileri yükleniyor...")
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

# Benzersiz tarihleri ve saatleri al
unique_dates = sorted(traffic_data['day'].unique())
unique_intervals = sorted(traffic_data['interval'].unique())

# İlk tarih ve saati al (default değerler için)
default_date = unique_dates[0]
default_interval = unique_intervals[0]

print(f"Benzersiz tarihler: {unique_dates}")
print(f"Benzersiz saatler: {unique_intervals}")

# Yol ağı ve node verilerini kaydet
print("\nYol ağı ve node verileri kaydediliyor...")

# Yol ağı verilerini kaydet
with open('road_network.pkl', 'wb') as f:
    pickle.dump({
        'edges': edges,
        'nodes': nodes,
        'G': G
    }, f)

print("Yol ağı ve node verileri kaydedildi.")

# HTML şablonunu oluştur
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Augsburg Trafik Haritası</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>
        #map {{
            height: 80vh;
        }}
        .controls {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        select {{
            padding: 8px;
            margin: 0 10px;
            border-radius: 4px;
            border: 1px solid #ced4da;
        }}
        button {{
            padding: 8px 16px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background: #0056b3;
        }}
    </style>
</head>
<body>
    <div class="controls">
        <select id="dateSelect">
            {date_options}
        </select>
        <select id="timeSelect">
            {time_options}
        </select>
        <button onclick="updateMap()">Haritayı Güncelle</button>
    </div>
    <div id="map"></div>

    <script>
        let map;
        let markers = [];
        let polylines = [];
        
        function initMap() {{
            map = L.map('map').setView([{center_lat}, {center_lon}], 13);
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);
            
            // İlk yükleme
            updateMap();
        }}
        
        function updateMap() {{
            // Önceki işaretçileri ve çizgileri temizle
            markers.forEach(marker => map.removeLayer(marker));
            polylines.forEach(line => map.removeLayer(line));
            markers = [];
            polylines = [];
            
            const date = document.getElementById('dateSelect').value;
            const time = document.getElementById('timeSelect').value;
            
            // AJAX isteği ile verileri al
            fetch(`/get_traffic_data?date=${{date}}&time=${{time}}`)
                .then(response => response.json())
                .then(data => {{
                    // Dedektörleri ekle
                    data.detectors.forEach(detector => {{
                        const marker = L.circleMarker([detector.lat, detector.lon], {{
                            radius: 5,
                            color: 'blue',
                            fill: true
                        }}).addTo(map);
                        
                        marker.bindPopup(`
                            Dedektör ID: ${{detector.id}}<br>
                            Flow: ${{detector.flow.toFixed(1)}}<br>
                            Occ: ${{detector.occ.toFixed(1)}}<br>
                            Speed: ${{detector.speed.toFixed(1)}}
                        `);
                        
                        markers.push(marker);
                    }});
                    
                    // Yolları çiz
                    data.roads.forEach(road => {{
                        const color = getRoadColor(road.flow, road.occ);
                        const polyline = L.polyline(road.coords, {{
                            color: color,
                            weight: 3,
                            opacity: 0.8
                        }}).addTo(map);
                        
                        polyline.bindPopup(`
                            Trafik Akışı: ${{road.flow.toFixed(1)}} araç/saat-şerit<br>
                            Doluluk Oranı: ${{road.occ.toFixed(1)}}%<br>
                            Ortalama Hız: ${{road.speed.toFixed(1)}} km/saat
                        `);
                        
                        polylines.push(polyline);
                    }});
                }});
        }}
        
        function getRoadColor(flow, occ) {{
            if (flow <= 0 || occ <= 0) return 'gray';
            if (flow < 200 && occ < 30) return 'green';
            if (flow < 500 && occ < 50) return 'yellow';
            if (flow < 1000 && occ < 70) return 'orange';
            return 'red';
        }}
        
        // Sayfa yüklendiğinde haritayı başlat
        window.onload = initMap;
    </script>
</body>
</html>
"""

# Tarih ve saat seçeneklerini oluştur
date_options = "\n".join([f'<option value="{date}">{date}</option>' for date in unique_dates])
time_options = "\n".join([f'<option value="{interval}">{interval}</option>' for interval in unique_intervals])

# HTML şablonunu doldur
html_content = html_template.format(
    date_options=date_options,
    time_options=time_options,
    center_lat=detector_data['lat'].mean(),
    center_lon=detector_data['long'].mean()
)

# HTML dosyasını kaydet
with open("augsburg_traffic_map.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("\nHarita başarıyla kaydedildi.")

# Trafik verilerini yükledikten hemen sonra kontrol edelim
print("\nTrafik verileri kontrol ediliyor...")
print("İlk 5 trafik verisi:")
print(traffic_data.head())
print("\nTrafik verilerinin sütun tipleri:")
print(traffic_data.dtypes)

# Her dedektör için ilk trafik değerini al
first_traffic_values = {}
for detid in detector_data['detid']:
    # Trafik verisinde dedektör ID'sini bul
    det_traffic = traffic_data[traffic_data['detid'] == detid]
    if not det_traffic.empty:
        # İlk kaydı al
        first_record = det_traffic.iloc[0]
        first_traffic_values[detid] = {
            'flow': float(first_record['flow']) if pd.notnull(first_record['flow']) else 0,
            'occ': float(first_record['occ']) if pd.notnull(first_record['occ']) else 0,
            'speed': float(first_record['speed']) if pd.notnull(first_record['speed']) else 0
        }
    else:
        first_traffic_values[detid] = {
            'flow': 0,
            'occ': 0,
            'speed': 0
        }

print(f"\n{len(first_traffic_values)} dedektör için trafik değeri bulundu.")

# first_traffic_values'ı kontrol edelim
print("\nTrafik değerleri kontrol ediliyor...")
for detid, values in list(first_traffic_values.items())[:5]:
    print(f"Dedektör {detid}: Flow={values['flow']}, Occ={values['occ']}, Speed={values['speed']}")

# Sıfır olmayan değerleri kontrol et
nonzero_values = {detid: values for detid, values in first_traffic_values.items() 
                 if values['flow'] > 0 or values['occ'] > 0 or values['speed'] > 0}
print(f"\nSıfır olmayan değerlere sahip {len(nonzero_values)} dedektör bulundu.")
if nonzero_values:
    print("\nSıfır olmayan değerler:")
    for detid, values in list(nonzero_values.items())[:5]:
        print(f"Dedektör {detid}: Flow={values['flow']}, Occ={values['occ']}, Speed={values['speed']}")

# Trafik verilerinin dağılımını kontrol et
print("\nTrafik verilerinin dağılımı:")
print("Flow değerlerinin dağılımı:")
print(traffic_data['flow'].describe())
print("\nOcc değerlerinin dağılımı:")
print(traffic_data['occ'].describe())
print("\nSpeed değerlerinin dağılımı:")
print(traffic_data['speed'].describe())

# Trafik verilerinin doğru yüklendiğinden emin olalım
print("\nTrafik verilerinin ilk 5 satırı:")
print(traffic_data.head())
print("\nTrafik verilerinin sütun tipleri:")
print(traffic_data.dtypes)

# 3. Merkez noktayı hesapla
print("\nMerkez nokta hesaplanıyor...")
center_lat = detector_data['lat'].mean()
center_lon = detector_data['long'].mean()
print(f"Merkez nokta: {center_lat}, {center_lon}")

# 4. Merkez noktadan 10km yarıçapındaki yolları indir
print("\nMerkez noktadan 10km yarıçapındaki yollar indiriliyor...")
try:
    # Merkez noktadan 10km yarıçapındaki yolları indir
    combined_subgraph = ox.graph_from_point((center_lat, center_lon), dist=5000, network_type='drive')
    # Projeksiyon yap
    combined_subgraph = ox.project_graph(combined_subgraph)
    # CRS bilgisini ekle
    combined_subgraph.graph['crs'] = G.graph['crs']
    print(f"  - {combined_subgraph.number_of_nodes()} node ve {combined_subgraph.number_of_edges()} yol indirildi")
except Exception as e:
    print(f"  - HATA: {e}")
    # Hata durumunda boş bir grafik oluştur
    combined_subgraph = G.copy()
    combined_subgraph.clear()
    combined_subgraph.graph['crs'] = G.graph['crs']

print(f"\nToplam {combined_subgraph.number_of_nodes()} node ve {combined_subgraph.number_of_edges()} yol indirildi.")

# 5. Yakın noktaları birleştir
print("\nYakın noktalar birleştiriliyor...")
nodes_combined, edges_combined = ox.graph_to_gdfs(combined_subgraph)

# Koordinatları WGS84'e dönüştür
print("\nKoordinatlar WGS84'e dönüştürülüyor...")
edges_combined = edges_combined.to_crs(epsg=4326)  # WGS84
nodes_combined = nodes_combined.to_crs(epsg=4326)  # WGS84

# Yol verilerini kontrol et
print("\nYol verilerini kontrol et:")
print(f"Toplam yol sayısı: {len(edges_combined)}")
print("\nİlk 5 yolun detaylı bilgileri:")
for idx, edge in edges_combined.head().iterrows():
    if edge.geometry is not None:
        coords = list(edge.geometry.coords)
        print(f"\nYol {idx}:")
        print(f"  - İlk koordinat: {coords[0]}")
        print(f"  - Son koordinat: {coords[-1]}")
        print(f"  - Yol adı: {edge.get('name', 'Bilinmiyor')}")
        print(f"  - Uzunluk: {edge.get('length', 0):.1f} metre")
        print(f"  - Koordinat sayısı: {len(coords)}")

# Dedektör noktalarını GeoDataFrame'e çevir ve WGS84'e dönüştür
detector_points = gpd.GeoDataFrame(
    detector_data,
    geometry=[Point(xy) for xy in zip(detector_data['long'], detector_data['lat'])],
    crs="EPSG:4326"  # WGS84
)

# Yakın noktaları bul ve birleştir (10 metre mesafe eşiği)
print("\nNoktalar birleştiriliyor...")
merged_points = []
used_indices = set()
merged_groups = []  # Birleştirilen noktaların gruplarını tutacak liste

for i, point1 in detector_points.iterrows():
    if i in used_indices:
        continue
    
    close_points = [i]
    for j, point2 in detector_points.iterrows():
        if i != j and j not in used_indices:
            # Mesafeyi metre cinsinden hesapla
            dist = point1.geometry.distance(point2.geometry) * 111000  # yaklaşık olarak 1 derece = 111km
            if dist < 10:  # 10 metre
                close_points.append(j)
                print(f"Nokta {i} (ID: {detector_data.iloc[i]['detid']}) ve {j} (ID: {detector_data.iloc[j]['detid']}) birleştirildi. Mesafe: {dist:.1f} metre")
    
    # Yakın noktaların ortalamasını al
    avg_lat = detector_data.iloc[close_points]['lat'].mean()
    avg_lon = detector_data.iloc[close_points]['long'].mean()
    merged_points.append((avg_lon, avg_lat))
    used_indices.update(close_points)
    merged_groups.append(close_points)  # Birleştirilen noktaların listesini kaydet

print(f"\nBirleştirme grupları:")
for idx, group in enumerate(merged_groups):
    det_ids = [detector_data.iloc[i]['detid'] for i in group]
    print(f"Grup {idx+1}: Dedektör ID'leri {det_ids}")
    for detid in det_ids:
        if detid in first_traffic_values:
            print(f"  - Dedektör {detid}: {first_traffic_values[detid]}")

print(f"\n{len(detector_data)} noktadan {len(merged_points)} birleştirilmiş nokta oluşturuldu.")

# 6. Birleştirilmiş noktaları en yakın yola sabitle
print("\nNoktalar en yakın yollara sabitleniyor...")
snapped_points = []
road_traffic = {}  # Yolların trafik değerlerini tutacak sözlük

for i, (lon, lat) in enumerate(merged_points):
    # En yakın yolu bul
    point = Point(lon, lat)
    nearest_edge = None
    min_dist = float('inf')
    
    for idx, edge in edges_combined.iterrows():
        if edge.geometry is not None:
            # Mesafeyi metre cinsinden hesapla
            dist = point.distance(edge.geometry) * 111000  # yaklaşık olarak 1 derece = 111km
            if dist < min_dist:
                min_dist = dist
                nearest_edge = edge
    
    if nearest_edge is not None:
        # Noktayı en yakın yola sabitle
        nearest_point = nearest_edge.geometry.interpolate(nearest_edge.geometry.project(point))
        snapped_points.append((nearest_point.x, nearest_point.y))
        
        # Yolun trafik değerini hesapla
        group = merged_groups[i]
        det_ids = [detector_data.iloc[j]['detid'] for j in group]
        traffic_values = [first_traffic_values.get(detid, {'flow': 0, 'occ': 0, 'speed': 0}) for detid in det_ids]
        
        # Ortalama trafik değerlerini hesapla
        avg_flow = sum(t['flow'] for t in traffic_values) / len(traffic_values) if traffic_values else 0
        avg_occ = sum(t['occ'] for t in traffic_values) / len(traffic_values) if traffic_values else 0
        avg_speed = sum(t['speed'] for t in traffic_values) / len(traffic_values) if traffic_values else 0
        
        # Yolun ID'sini al ve trafik değerini kaydet
        road_id = idx  # edges_combined'in indeksini kullan
        road_traffic[road_id] = {
            'flow': avg_flow,
            'occ': avg_occ,
            'speed': avg_speed
        }
        
        print(f"Nokta ({lon}, {lat}) yola sabitlendi. Mesafe: {min_dist:.1f} metre, Trafik: {avg_flow:.1f} araç/saat-şerit, {avg_occ:.1f}% doluluk, {avg_speed:.1f} km/saat")

# 1. Dedektör noktalarını yol ağına node olarak ekle ve yolları böl
print("\nDedektör noktaları yol ağına node olarak ekleniyor ve yollar bölünüyor...")

# Orijinal edges_combined GeoDataFrame'ini kopyala
edges_new = edges_combined.copy()
nodes_new = nodes_combined.copy()

# Her dedektör noktası için:
for i, (lon, lat) in enumerate(snapped_points):
    det_point = Point(lon, lat)
    min_dist = float('inf')
    nearest_edge_idx = None
    nearest_edge_geom = None
    nearest_edge_row = None
    
    # En yakın yolu bul
    for idx, edge in edges_new.iterrows():
        if edge.geometry is not None:
            dist = det_point.distance(edge.geometry) * 111000
            if dist < min_dist:
                min_dist = dist
                nearest_edge_idx = idx
                nearest_edge_geom = edge.geometry
                nearest_edge_row = edge
    
    if nearest_edge_idx is not None and min_dist < 50:
        # En yakın noktayı bul
        proj_dist = nearest_edge_geom.project(det_point)
        snapped_point = nearest_edge_geom.interpolate(proj_dist)
        
        # Yeni node olarak ekle
        new_node_id = f'det_{i}'
        new_node_data = {
            'x': snapped_point.x,
            'y': snapped_point.y,
            'geometry': snapped_point
        }
        nodes_new.loc[new_node_id] = new_node_data
        
        # Edge'i ikiye böl
        coords = list(nearest_edge_geom.coords)
        split_idx = min(range(len(coords)), key=lambda j: Point(coords[j]).distance(snapped_point))
        
        # İlk parça
        if split_idx > 0:  # En az 2 nokta olduğundan emin ol
            geom1 = LineString(coords[:split_idx+1] + [(snapped_point.x, snapped_point.y)])
        else:
            geom1 = LineString([coords[0], (snapped_point.x, snapped_point.y)])
        
        # İkinci parça
        if split_idx < len(coords) - 1:  # En az 2 nokta olduğundan emin ol
            geom2 = LineString([(snapped_point.x, snapped_point.y)] + coords[split_idx+1:])
        else:
            geom2 = LineString([(snapped_point.x, snapped_point.y), coords[-1]])
        
        # Koordinat listelerinin boş olmadığından emin ol
        if len(geom1.coords) < 2 or len(geom2.coords) < 2:
            print(f"Uyarı: Yol bölünemedi - yetersiz nokta sayısı")
            continue
        
        # Eski edge'i sil
        edges_new = edges_new.drop(nearest_edge_idx)
        
        # Yeni edge'leri ekle
        new_edge1 = nearest_edge_row.copy()
        new_edge2 = nearest_edge_row.copy()
        
        new_edge1['geometry'] = geom1
        new_edge2['geometry'] = geom2
        
        new_edge1['u'] = nearest_edge_row.name  # orijinal başlangıç noktası
        new_edge1['v'] = new_node_id  # yeni node
        
        new_edge2['u'] = new_node_id  # yeni node
        new_edge2['v'] = nearest_edge_row.name  # orijinal bitiş noktası
        
        edges_new = pd.concat([edges_new, pd.DataFrame([new_edge1])], ignore_index=True)
        edges_new = pd.concat([edges_new, pd.DataFrame([new_edge2])], ignore_index=True)

# 2. NetworkX grafiğini yeni node-edge yapısıyla oluştur
G_new = nx.Graph()

# Node'ları ekle
for idx, node in nodes_new.iterrows():
    G_new.add_node(idx, pos=(node['x'], node['y']))

# Edge'leri ekle
for idx, edge in edges_new.iterrows():
    G_new.add_edge(edge['u'], edge['v'], 
                  edge_id=idx,
                  geometry=edge['geometry'])

# 3. Trafik verilerini yeni edge'lere ata
print("\nTrafik verileri yollara atanıyor...")

# Her edge için trafik verilerini hesapla
for idx, edge in edges_new.iterrows():
    u, v = edge['u'], edge['v']
    
    # Node'ların trafik verilerini al
    u_data = None
    v_data = None
    
    # Eğer node bir dedektör ise
    if isinstance(u, str) and u.startswith('det_'):
        det_idx = int(u.split('_')[1])
        if det_idx < len(merged_groups):
            group = merged_groups[det_idx]
            det_ids = [detector_data.iloc[j]['detid'] for j in group]
            traffic_values = []
            for detid in det_ids:
                if detid in first_traffic_values:
                    traffic_values.append(first_traffic_values[detid])
            if traffic_values:
                u_data = {
                    'flow': sum(t['flow'] for t in traffic_values) / len(traffic_values),
                    'occ': sum(t['occ'] for t in traffic_values) / len(traffic_values),
                    'speed': sum(t['speed'] for t in traffic_values) / len(traffic_values)
                }
    
    if isinstance(v, str) and v.startswith('det_'):
        det_idx = int(v.split('_')[1])
        if det_idx < len(merged_groups):
            group = merged_groups[det_idx]
            det_ids = [detector_data.iloc[j]['detid'] for j in group]
            traffic_values = []
            for detid in det_ids:
                if detid in first_traffic_values:
                    traffic_values.append(first_traffic_values[detid])
            if traffic_values:
                v_data = {
                    'flow': sum(t['flow'] for t in traffic_values) / len(traffic_values),
                    'occ': sum(t['occ'] for t in traffic_values) / len(traffic_values),
                    'speed': sum(t['speed'] for t in traffic_values) / len(traffic_values)
                }
    
    # Trafik verilerini hesapla
    if u_data and v_data:
        flow = (u_data['flow'] + v_data['flow']) / 2
        occ = (u_data['occ'] + v_data['occ']) / 2
        speed = (u_data['speed'] + v_data['speed']) / 2
    elif u_data:
        flow, occ, speed = u_data['flow'], u_data['occ'], u_data['speed']
    elif v_data:
        flow, occ, speed = v_data['flow'], v_data['occ'], v_data['speed']
    else:
        flow, occ, speed = 0, 0, 0
    
    # Edge'e trafik verilerini ata
    G_new[u][v]['flow'] = flow
    G_new[u][v]['occ'] = occ
    G_new[u][v]['speed'] = speed
    
    # GeoDataFrame'e de ekle
    edges_new.loc[idx, 'flow'] = flow
    edges_new.loc[idx, 'occ'] = occ
    edges_new.loc[idx, 'speed'] = speed

# Trafik verilerini ağa yay
print("\nTrafik verileri ağa yayılıyor...")
for iteration in range(3):  # 3 iterasyon
    # Her edge için trafik değerlerini güncelle
    for u, v, data in G_new.edges(data=True):
        # Edge'in bağlı olduğu node'ların trafik değerlerini al
        u_flow = sum(d['flow'] for _, _, d in G_new.edges(u, data=True))
        v_flow = sum(d['flow'] for _, _, d in G_new.edges(v, data=True))
        
        # Edge'in bağlı olduğu diğer edge'lerin trafik değerlerini al
        u_edges = [d['flow'] for _, _, d in G_new.edges(u, data=True) if _ != v]
        v_edges = [d['flow'] for _, _, d in G_new.edges(v, data=True) if _ != u]
        
        # Yeni trafik değerini hesapla
        new_flow = (data['flow'] + sum(u_edges) + sum(v_edges)) / (len(u_edges) + len(v_edges) + 1)
        
        # Değerleri güncelle
        G_new[u][v]['flow'] = new_flow
        edges_new.loc[edges_new.index[(edges_new['u'] == u) & (edges_new['v'] == v)], 'flow'] = new_flow

# 4. Harita çizimini güncelle
print("\nHarita güncelleniyor...")

# Haritayı oluştur
mymap = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# Dedektör noktalarını ekle
for idx, row in detector_data.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=5,
        color='blue',
        fill=True,
        popup=f"Dedektör ID: {row['detid']}<br>Flow: {first_traffic_values.get(row['detid'], {}).get('flow', 0):.1f}<br>Occ: {first_traffic_values.get(row['detid'], {}).get('occ', 0):.1f}<br>Speed: {first_traffic_values.get(row['detid'], {}).get('speed', 0):.1f}"
    ).add_to(mymap)

# Yolları çiz
for idx, edge in edges_new.iterrows():
    if edge.geometry is not None:
        try:
            # Koordinatları Folium için doğru sırayla al
            coords = [(y, x) for x, y in edge.geometry.coords]
            
            # Trafik verilerine göre renk belirle
            flow = edge.get('flow', 0)
            occ = edge.get('occ', 0)
            
            if flow <= 0 or occ <= 0:
                color = 'gray'
            elif flow < 200 and occ < 30:
                color = 'green'
            elif flow < 500 and occ < 50:
                color = 'yellow'
            elif flow < 1000 and occ < 70:
                color = 'orange'
            else:
                color = 'red'
            
            # Popup bilgisi
            popup = f"""Trafik Akışı: {flow:.1f} araç/saat-şerit<br>
            Doluluk Oranı: {occ:.1f}%<br>
            Ortalama Hız: {edge.get('speed', 0):.1f} km/saat"""
            
            folium.PolyLine(
                coords,
                color=color,
                weight=3,
                opacity=0.8,
                popup=popup
            ).add_to(mymap)
            
        except Exception as e:
            print(f"Hata: {e}")
            continue

# Haritayı kaydet
mymap.save("augsburg_traffic_map.html")
print("\nHarita başarıyla kaydedildi.")