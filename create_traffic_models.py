import pandas as pd
import pickle
import os
from datetime import datetime
import asyncio
import concurrent.futures
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
    """Tüm benzersiz (haftanın günü, interval) kombinasyonları için ilgili ham trafik verisi dosyalarını oluşturur (Paralel - Chunk Bazlı)."""
    start_time = time.time()
    print("\n=== Haftanın Günü ve Interval Bazlı İLGİLİ Ham Trafik Verisi Oluşturma İşlemi Başladı (Paralel - Chunk Bazlı) ===")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(current_dir, "traffic_models")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    traffic_data_path = os.path.join(current_dir, 'dataset', 'filtered_augsburg_new.csv')
    if not os.path.exists(traffic_data_path):
        print(f"Hata: Trafik verisi dosyası bulunamadı: {traffic_data_path}")
        return False

    CHUNK_SIZE = 500000 # Define chunk size
    accumulated_data = {} # Dictionary to accumulate data per (weekday, interval)
    total_rows_read = 0
    total_cleaned_rows = 0

    print(f"Trafik verileri '{traffic_data_path}' dosyasından {CHUNK_SIZE} satırlık parçalar halinde okunuyor...")

    try:
        # Read CSV in chunks
        for i, chunk in enumerate(pd.read_csv(traffic_data_path, sep=',',
                                              dtype={
                                                  'day': str,
                                                  'interval': int,
                                                  'detid': str,
                                                  'flow': float,
                                                  'occ': float,
                                                  'speed': float,
                                                  'city': str
                                              }, chunksize=CHUNK_SIZE)):
            print(f"Chunk {i+1} okunuyor, {len(chunk)} satır.")
            total_rows_read += len(chunk)

            # Process the chunk: add weekday and clean
            chunk['weekday'] = chunk['day'].apply(get_weekday)
            cleaned_chunk = chunk.dropna(subset=['weekday']).copy()
            total_cleaned_rows += len(cleaned_chunk)

            # Accumulate data from this chunk into the dictionary
            # Group by weekday and interval within the chunk
            grouped_chunk = cleaned_chunk.groupby(['weekday', 'interval'])

            for (weekday, interval), group_df in grouped_chunk:
                key = (weekday, interval)
                if key not in accumulated_data:
                    accumulated_data[key] = pd.DataFrame() # Initialize if first time seeing this key
                # Append the group data to the accumulated DataFrame for this key
                # Using pd.concat can be more efficient than df.append (which is deprecated)
                accumulated_data[key] = pd.concat([accumulated_data[key], group_df], ignore_index=True)
            print(f"Chunk {i+1} işlendi ve veriler akümüle edildi.")

        print(f"\nToplam {total_rows_read} ham veri satırı okundu.")
        print(f"Toplam {total_cleaned_rows} satır temizlendi ve işlenmeye hazır.")

    except Exception as e:
        print(f"Hata: Trafik verileri chunklar halinde yüklenirken veya işlenirken hata oluştu: {str(e)}")
        return False

    if not accumulated_data:
        print("Uyarı: İşlenecek temizlenmiş veri bulunamadı.")
        return False

    # Prepare arguments for parallel processing using the accumulated data
    task_args = [(weekday, interval, data_df, MODELS_DIR)
                 for (weekday, interval), data_df in accumulated_data.items()]

    print(f"\nToplam {len(task_args)} benzersiz (haftanın günü, interval) kombinasyonu için dosya kaydedilecek.")

    # Process saving in parallel
    results = []
    # Paralel işlemci sayısını makul bir seviyede tutalım
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(task_args) or 1)) as executor:
        results = list(executor.map(process_interval_data, task_args))

    # Summary (unchanged)
    success_count = sum(results)
    end_time = time.time()
    total_time = end_time - start_time

    print("\n=== Haftanın Günü ve Interval Bazlı İLGİLİ Ham Trafik Verisi Oluşturma İşlemi Tamamlandı (Paralel - Chunk Bazlı) ===")
    print(f"Toplam süre: {total_time/60:.2f} dakika")
    print(f"Başarılı dosya sayısı: {success_count}")
    print(f"Başarısız dosya sayısı: {len(task_args) - success_count}")
    print(f"İşlem tamamlandığında toplam {total_rows_read} ham satır okundu ve {total_cleaned_rows} temizlenmiş satır işlendi.")


    return success_count > 0

if __name__ == '__main__':
    # Yol ağı oluşturma adımı bu script için bir ön şart değildir.
    print("Interval bazlı İLGİLİ ham dedektör verisi dosyalarını chunklar halinde okuyup oluşturmak için devam ediliyor...")

    # Scripti çalıştır
    generate_weekday_interval_data_files() 