import numpy as np
import librosa
import soundfile as sf
import os
import glob
import time
from collections import defaultdict
from scipy.ndimage import maximum_filter
import warnings

warnings.filterwarnings('ignore')

# 1. константы и параметры алгоритма
TARGET_SR = 22050
N_FFT_FINAL = 4096
HOP_FINAL = 1024
NEIGHBORHOOD = 20
THRESHOLD_DB = -40

FAN_OUT = 10
T_MIN = 2
T_MAX = 80
MATCH_THRESHOLD = 8  # порог уверенности

# 2. ядро DSP & Hashing
def compute_constellation(y, sr, n_fft, hop_length, neighborhood, threshold_db):
    # Если она ничтожна (меньше 1e-4), выходим
    if np.max(np.abs(y)) < 1e-4:
        # Возвращаем пустые массивы для частот и времени, 
        # и пустую матрицу децибел, чтобы не ломать логику вызовов.
        return np.array([], dtype=int), np.array([], dtype=int), np.array([[]])
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    
    # Поиск локальных максимумов
    local_max = maximum_filter(D_db, size=neighborhood) == D_db
    background_threshold = D_db > threshold_db
    peaks = local_max & background_threshold
    
    frequencies, times = np.where(peaks)
    return frequencies, times, D_db

def generate_hashes(fi, ti, fan_out=FAN_OUT, t_min=T_MIN, t_max=T_MAX):
    if len(fi) == 0: return []
    order = np.argsort(ti)
    fi_s, ti_s = fi[order], ti[order]
    hashes = []
    n = len(fi_s)
    
    for i in range(n):
        f1, t1 = fi_s[i], ti_s[i]
        j_start = np.searchsorted(ti_s, t1 + t_min)
        j_end = np.searchsorted(ti_s, t1 + t_max, side='right')
        
        if j_start >= j_end: continue
        
        candidates = np.arange(j_start, j_end)
        if len(candidates) > fan_out:
            candidates = candidates[np.linspace(0, len(candidates)-1, fan_out, dtype=int)]
            
        for j in candidates:
            f2, t2 = fi_s[j], ti_s[j]
            dt = int(t2 - t1)
            if dt > 0:
                hashes.append(((int(f1), int(f2), dt), int(t1)))
    return hashes

# 3. генерируем искажения (Стресс-тесты)
def apply_degradation(y, noise_type, snr_db=None):
    if noise_type == '1': # Белый шум
        signal_power = np.mean(y**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        return y + noise
    elif noise_type == '2': # Клиппинг (Перегруз микрофона)
        threshold = 0.5 # 50% срез
        return np.clip(y, -threshold, threshold)
    return y # без искажений

# 4. движок бд Shazam Engine
class ShazamEngine:
    def __init__(self, db_folder="audio_database"):
        self.db_folder = db_folder
        self.DATABASE = defaultdict(list)
        self.TRACK_REGISTRY = {}
        self.is_loaded = False
        
    def build_database(self):
        print("\nСканирование базы данных...")
        wav_files = glob.glob(os.path.join(self.db_folder, "*.wav"))
        
        if not wav_files:
            print(f"ОШИБКА: Папка '{self.db_folder}' пуста или не существует!")
            return
            
        t_start = time.time()
        for track_id, file_path in enumerate(wav_files):
            name = os.path.basename(file_path)
            y, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
            
            if np.max(np.abs(y)) < 1e-4:
                self.TRACK_REGISTRY[track_id] = {"name": name, "path": file_path, "n_peaks": 0}
                continue
                
            y_norm = y / (np.max(np.abs(y)) + 1e-12)
            fi_db, ti_db, _ = compute_constellation(y_norm, TARGET_SR, N_FFT_FINAL, HOP_FINAL, NEIGHBORHOOD, THRESHOLD_DB)
            h_list = generate_hashes(fi_db, ti_db)
            
            for (key, t_anc) in h_list:
                self.DATABASE[key].append((track_id, t_anc))
                
            self.TRACK_REGISTRY[track_id] = {"name": name, "path": file_path, "n_peaks": len(fi_db)}
            
        self.is_loaded = True
        print(f"База загружена: {len(self.TRACK_REGISTRY)} треков, {len(self.DATABASE)} уникальных хешей ({time.time()-t_start:.2f} сек)")

    def recognize(self, query_path):
        if not os.path.exists(query_path):
            print(f"Файл {query_path} не найден!")
            return
            
        t_start = time.time()
        y_q, _ = librosa.load(query_path, sr=TARGET_SR)
        y_q = y_q / (np.max(np.abs(y_q)) + 1e-12)
        
        fi_q, ti_q, _ = compute_constellation(y_q, TARGET_SR, N_FFT_FINAL, HOP_FINAL, NEIGHBORHOOD, THRESHOLD_DB)
        q_hashes = generate_hashes(fi_q, ti_q)
        
        matches = defaultdict(list)
        # Fuzzy Search (плюс минус 1 бин)
        for (f1, f2, dt), t_q in q_hashes:
            for df1 in (-1, 0, 1):
                for df2 in (-1, 0, 1):
                    key = (f1+df1, f2+df2, dt)
                    if key in self.DATABASE:
                        for tid, t_db in self.DATABASE[key]:
                            matches[tid].append(t_db - t_q)
                            
        best_tid, best_score, best_offset = None, 0, 0
        for tid, deltas in matches.items():
            if not deltas: continue
            vals, counts = np.unique(deltas, return_counts=True)
            idx = np.argmax(counts)
            if counts[idx] > best_score:
                best_score, best_tid, best_offset = counts[idx], tid, vals[idx]
                
        search_time = (time.time() - t_start) * 1000
        
        total_q_hashes = len(q_hashes)

        # Считаем процент уверенности
        # Мы используем коэффициент (например, / 10), так как в шуме 
        # 10% совпавших хешей — это уже почти 100% гарантия успеха.
        confidence = (best_score / total_q_hashes) * 1000 # Коэффициент подбирается экспериментально
        confidence = min(100.0, confidence) # Не выше 100%
        print("\n" + "="*50)
        if best_tid is not None and best_score >= MATCH_THRESHOLD:
            name = self.TRACK_REGISTRY[best_tid]['name']
            offset_sec = best_offset * HOP_FINAL / TARGET_SR
            print(f"  СОВПАДЕНИЕ НАЙДЕНО!")
            print(f"  Трек: {name}")
            print(f"  Уверенность: {confidence:.1f}% ({best_score} хешей)")
            print(f"  Время в оригинале: {offset_sec:.2f} сек")
            print(f"  Согласованных хешей: {best_score}")
            print(f"  Скорость поиска: {search_time:.1f} мс")
        else:
            print("  ТРЕК НЕ ОПОЗНАН")
            print(f"  (Лучший результат: {best_score} хешей, Порог: {MATCH_THRESHOLD})")
        print("="*50)


# 5. интерфейс CLI
def main():
    print(r"""
    Аудио-поисковый движок
    """)
    
    engine = ShazamEngine()
    engine.build_database()
    
    if not engine.is_loaded:
        return
        
    while True:
        print("\n--- ГЛАВНОЕ МЕНЮ ---")
        print("1. Сгенерировать зашумленный фрагмент")
        print("2. Распознать аудиофайл")
        print("3. Выход")
        choice = input("Выберите действие (1-3): ")
        
        if choice == '3':
            print("Завершение работы...")
            break
            
        elif choice == '1':
            print("\nДоступные треки в базе:")
            for tid, info in engine.TRACK_REGISTRY.items():
                print(f"  [{tid + 1}] {info['name']}")
                
            try:
                user_input = int(input("Введите номер трека (1-10): "))
                t_idx = user_input - 1
                path = engine.TRACK_REGISTRY[t_idx]['path']
                
                start_sec = float(input("С какой секунды вырезать фрагмент? (например, 15): "))
                duration = float(input("Длительность фрагмента в секундах? (например, 10): "))
                
                print("\nТипы искажений:")
                print("1. Белый шум (White Noise)")
                print("2. Перегруз микрофона (Clipping)")
                print("3. Без искажений (Чистый срез)")
                noise_type = input("Выберите тип (1-3): ")
                
                snr = 0
                if noise_type == '1':
                    snr = float(input("Введите уровень шума SNR в дБ (например, 5): "))
                
                print("\nОбработка...")
                y_full, _ = librosa.load(path, sr=TARGET_SR, mono=True)
                y_full = y_full / (np.max(np.abs(y_full)) + 1e-12)
                
                # Вырезаем по границам фреймов
                start_frame = int(start_sec * TARGET_SR / HOP_FINAL)
                s1 = start_frame * HOP_FINAL
                s2 = s1 + int(duration * TARGET_SR)
                y_cut = y_full[s1:min(s2, len(y_full))]
                
                y_noisy = apply_degradation(y_cut, noise_type, snr)
                
                out_name = "test_noisy.wav"
                sf.write(out_name, y_noisy, TARGET_SR)
                print(f"Фрагмент успешно сохранен как '{out_name}' в папке проекта")
                
            except Exception as e:
                print(f"Ошибка ввода: {e}")
                
        elif choice == '2':
            query_file = input("\nВведите имя файла для распознавания (или нажмите Enter для 'test_query.wav'): ")
            if not query_file:
                query_file = "test_noisy.wav"
                
            engine.recognize(query_file)

if __name__ == "__main__":
    main()