import pandas as pd
import pickle
import os
# Removed Prophet as per user request for interval-based average data
# from prophet import Prophet
from datetime import datetime
import asyncio
import concurrent.futures
# tqdm is not strictly needed
# from tqdm import tqdm
import time
# osmnx ve networkx kaldırıldı çünkü yol ağına atama yapılmıyor
# from shapely.geometry import Point # Point de kaldırıldı

def get_weekday(date_str):
    """Convert date string to weekday name in Turkish"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        weekdays = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
        return weekdays[date_obj.weekday()]
    except Exception as e: # Daha spesifik hata yakalama
        # print(f"Uyarı: Geçersiz tarih formatı: {date_str} - {e}") # Avoid excessive printing
        return None

def convert_seconds_to_time(seconds):
    """Saniyeyi HH:MM formatına çevirir (300 → '00:05')"""
    try:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        # secs = seconds % 60 # Not needed for HH:MM format
        return f"{hours:02d}:{minutes:02d}"
    except Exception as e: # Daha spesifik hata yakalama
        # print(f"Uyarı: Geçersiz saniye değeri: {seconds} - {e}") # Avoid excessive printing
        return "N/A"

# Define a function to process a single weekday and interval
def process_interval_data(args):
    """Processes and saves all RAW detector data for a single (weekday, interval) combination."""
    # Artık average_data_by_weekday_interval_detid yerine cleaned_traffic_data alıyoruz
    weekday, interval, cleaned_traffic_data, MODELS_DIR = args

    time_display = convert_seconds_to_time(interval)
    print(f"--- İşleniyor: {weekday}, Interval {interval} ({time_display}) ---")

    # Filter the CLEANED RAW data for the current (weekday, interval)
    # Bu DataFrame o zaman dilimine ait TÜM HAM veri satırlarını içerir
    current_slice_raw_data = cleaned_traffic_data[
        (cleaned_traffic_data['weekday'] == weekday) &
        (cleaned_traffic_data['interval'] == interval)
    ].copy()

    print(f"Debug: {weekday}, Interval {interval} için ham veri satırı sayısı: {len(current_slice_raw_data)}")

    if current_slice_raw_data.empty:
        print(f"Uyarı: {weekday}, Interval {interval} için ham veri bulunamadı, dosya oluşturulmuyor.")
        return False # Indicate failure to process this interval

    # --- KAYDETME MANTIĞI (Doğrudan filtrelenmiş RAW DataFrame'i kaydet) ---

    # Kaydedilecek veri doğrudan filtrelenmiş raw DataFrame'dir
    data_to_save = current_slice_raw_data # Tüm sütunları kaydedebiliriz veya sadece gerekli olanları seçebiliriz

    # Dosya adını güncelledik, sadece interval ve weekday içerecek (veri tipi belirtilebilir)
    safe_interval_str = str(interval)
    safe_weekday = weekday.replace(' ', '_').replace('/', '_').replace('\\', '_')
    map_file_path = os.path.join(MODELS_DIR, f'raw_data_interval_{safe_interval_str}_{safe_weekday}.pkl') # raw_data_interval olarak güncellendi

    try:
        print(f"Debug: {weekday}, Interval {interval} için dosya kaydediliyor: {map_file_path}")
        with open(map_file_path, 'wb') as f:
            pickle.dump(data_to_save, f) # Filtrelenmiş RAW DataFrame'i kaydet
        print(f"Debug: {map_file_path} dosyasına {len(data_to_save)} adet ham veri satırı kaydedildi.")
        return True # Indicate success
    except Exception as e:
        print(f"Hata: {map_file_path} dosyası kayıt sırasında hata: {str(e)}")
        return False # Indicate failure

# async keyword kaldırıldı
def generate_weekday_interval_data_files():
    """Tüm benzersiz (haftanın günü, interval) kombinasyonları için ilgili ham trafik verisi dosyalarını oluşturur (Paralel)."""
    start_time = time.time()
    print("\n=== Haftanın Günü ve Interval Bazlı İLGİLİ Ham Trafik Verisi Oluşturma İşlemi Başladı (Paralel) ===")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(current_dir, "traffic_models")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Load traffic data (unchanged)
    traffic_data_path = os.path.join(current_dir, 'dataset', 'filtered_augsburg_new.csv')
    if not os.path.exists(traffic_data_path):
        print(f"Hata: Trafik verisi dosyası bulunamadı: {traffic_data_path}")
        return False
    try:
        traffic_data_all = pd.read_csv(traffic_data_path, sep=',',
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
        print(f"Toplam ham veri satırı: {len(traffic_data_all)}") # Debug print

    except Exception as e:
        print(f"Hata: Trafik verileri yüklenirken hata oluştu: {str(e)}")
        return False

    # Add weekday column to traffic data (unchanged)
    traffic_data_all['weekday'] = traffic_data_all['day'].apply(get_weekday)
    initial_rows = len(traffic_data_all)
    # Geçersiz gün bilgisi olan satırları temizle (bu adım hala gerekli)
    cleaned_traffic_data = traffic_data_all.dropna(subset=['weekday']).copy()
    dropped_rows = initial_rows - len(cleaned_traffic_data)
    print(f"Haftanın günü bilgisi trafik verilerine eklendi ve temizlendi. Geçersiz gün bilgisi olan {dropped_rows} satır silindi.")
    print(f"Temizlenmiş ham veri satırı sayısı: {len(cleaned_traffic_data)}") # Debug print

    # Gruplama ve ortalama alma adımı KALDIRILDI
    # print("(Haftanın günü, interval, detid) bazında ortalama değerler hesaplanıyor...")
    # average_data_by_weekday_interval_detid = cleaned_traffic_data.groupby([
    #     'weekday', 'interval', 'detid'
    # ]).agg({
    #     'flow': 'mean',
    #     'occ': 'mean'
    # }).reset_index()
    # print(f"Ortalama değerler {len(average_data_by_weekday_interval_detid)} satır için hesaplandı.")


    # Get unique (weekday, interval) combinations from the cleaned data
    unique_weekday_intervals = sorted(cleaned_traffic_data[
        ['weekday', 'interval']
    ].drop_duplicates().values.tolist())
    print(f"Toplam {len(unique_weekday_intervals)} benzersiz (haftanın günü, interval) kombinasyonu için veri işlenecek.")

    # Yol ağı verileri veya dedektör eşleşme haritası burada gerekli değil ve yüklenmiyor.

    # Prepare arguments for parallel processing
    # process_interval_data fonksiyonuna cleaned_traffic_data DataFrame'ini geçiriyoruz
    task_args = [(weekday, interval, cleaned_traffic_data, MODELS_DIR)
                 for weekday, interval in unique_weekday_intervals]

    # Process intervals in parallel
    results = []
    # Paralel işlemci sayısını makul bir seviyede tutalım, örneğin 8 veya interval sayısı kadar
    # Bu işlem artık disk I/O ve DataFrame filtreleme ağırlıklı olacak
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(unique_weekday_intervals) or 1)) as executor:
        # tqdm is commented out, but could be added here around executor.map if desired
        # results = list(tqdm(executor.map(process_interval_data, task_args), total=len(task_args), desc="Processing Intervals"))
        results = list(executor.map(process_interval_data, task_args))

    # Summary (unchanged)
    success_count = sum(results)
    end_time = time.time()
    total_time = end_time - start_time

    print("\n=== Haftanın Günü ve Interval Bazlı İLGİLİ Ham Trafik Verisi Oluşturma İşlemi Tamamlandı (Paralel) ===")
    print(f"Toplam süre: {total_time/60:.2f} dakika")
    print(f"Başarılı dosya sayısı: {success_count}")
    print(f"Başarısız dosya sayısı: {len(unique_weekday_intervals) - success_count}")

    return success_count > 0

if __name__ == '__main__':
    # Yol ağı oluşturma adımı bu script için bir ön şart değildir.
    print("Interval bazlı İLGİLİ ham dedektör verisi dosyalarını oluşturmak için devam ediliyor...")

    # Scripti çalıştır
    generate_weekday_interval_data_files() 