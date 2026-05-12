import numpy as np
import librosa
import soundfile as sf
import os
import glob
import time
from collections import defaultdict
from scipy.ndimage import maximum_filter
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Скрываем предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Пытаемся импортировать тяжелые ИИ-библиотеки
try:
    import openl3
    print("Нейросеть OpenL3 успешно подключена")
except ImportError:
    print("ОШИБКА: Библиотека openl3 не установлена")
    exit()

# 1. константы и параметры алгоритма
TARGET_SR = 22050
N_FFT_FINAL = 4096
HOP_FINAL = 1024
NEIGHBORHOOD = 20
THRESHOLD_DB = -40

FAN_OUT = 10
T_MIN = 2
T_MAX = 80
MATCH_THRESHOLD = 8  # Порог для быстрого поиска

# 2. ядро DSP
def compute_constellation(y, sr, n_fft, hop_length, neighborhood, threshold_db):
    # Если она ничтожна (меньше 1e-4), выходим немедленно.
    if np.max(np.abs(y)) < 1e-4:
        # Возвращаем пустые массивы для частот и времени, 
        # и пустую матрицу децибел, чтобы не ломать логику вызовов.
        return np.array([], dtype=int), np.array([], dtype=int), np.array([[]])
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    local_max = maximum_filter(D_db, size=neighborhood) == D_db
    background_threshold = D_db > threshold_db
    peaks = local_max & background_threshold
    return np.where(peaks)[0], np.where(peaks)[1], D_db


def generate_hashes(fi, ti, fan_out=FAN_OUT, t_min=T_MIN, t_max=T_MAX):
    if len(fi) == 0: return []
    order = np.argsort(ti)
    fi_s, ti_s = fi[order], ti[order]
    hashes = []
    for i in range(len(fi_s)):
        f1, t1 = fi_s[i], ti_s[i]
        j_start = np.searchsorted(ti_s, t1 + t_min)
        j_end = np.searchsorted(ti_s, t1 + t_max, side='right')
        if j_start >= j_end: continue
        candidates = np.arange(j_start, j_end)
        if len(candidates) > fan_out:
            candidates = candidates[np.linspace(0, len(candidates)-1, fan_out, dtype=int)]
        for j in candidates:
            dt = int(ti_s[j] - t1)
            if dt > 0: hashes.append(((int(f1), int(fi_s[j]), dt), int(t1)))
    return hashes

def apply_degradation(y, noise_type, snr_db=None):
    if noise_type == '1':
        noise = np.random.normal(0, np.sqrt(np.mean(y**2) / (10 ** (snr_db / 10))), len(y))
        return y + noise
    elif noise_type == '2': return np.clip(y, -0.5, 0.5)
    return y

# 3. гибридный движок (Classic + AI)
class ShazamHybridEngine:
    def __init__(self, db_folder="audio_database"):
        self.db_folder = db_folder
        self.DATABASE = defaultdict(list)
        self.TRACK_REGISTRY = {}
        self.is_loaded = False
        self.ai_model = None
        
    def build_database(self):
        print("\nСканирование базы данных (Classic DSP) ")
        wav_files = glob.glob(os.path.join(self.db_folder, "*.wav"))
        t_start = time.time()
        for track_id, file_path in enumerate(wav_files):
            name = os.path.basename(file_path)
            y, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
            if np.max(np.abs(y)) < 1e-4:
                self.TRACK_REGISTRY[track_id] = {"name": name, "path": file_path, "duration": 0}
                continue
            y_norm = y / (np.max(np.abs(y)) + 1e-12)
            fi_db, ti_db, _ = compute_constellation(y_norm, TARGET_SR, N_FFT_FINAL, HOP_FINAL, NEIGHBORHOOD, THRESHOLD_DB)
            for (key, t_anc) in generate_hashes(fi_db, ti_db):
                self.DATABASE[key].append((track_id, t_anc))
            self.TRACK_REGISTRY[track_id] = {"name": name, "path": file_path, "duration": len(y)/TARGET_SR}
        self.is_loaded = True
        print(f"База загружена за {time.time()-t_start:.2f} сек.")

    def load_ai_model(self):
        if self.ai_model is None:
            print("\nЗагрузка модели OpenL3 в память ")
            self.ai_model = openl3.models.load_audio_embedding_model(
                input_repr="mel256", content_type="music", embedding_size=512)
            print("Нейросеть готова к работе.")

    def deep_recognize(self, query_path):
        """Двухуровневое распознавание: Классика -> ИИ-проверка"""
        t_start = time.time()
        
        # первый уровень - быстрый классический поиск
        print("\nКлассический поиск по хешам ")
        y_q, _ = librosa.load(query_path, sr=TARGET_SR)
        y_q = y_q / (np.max(np.abs(y_q)) + 1e-12)
        
        fi_q, ti_q, _ = compute_constellation(y_q, TARGET_SR, N_FFT_FINAL, HOP_FINAL, NEIGHBORHOOD, THRESHOLD_DB)
        q_hashes = generate_hashes(fi_q, ti_q)
        
        matches = defaultdict(list)
        for (f1, f2, dt), t_q in q_hashes:
            for df1 in (-1, 0, 1):
                for df2 in (-1, 0, 1):
                    key = (f1+df1, f2+df2, dt)
                    if key in self.DATABASE:
                        for tid, t_db in self.DATABASE[key]: matches[tid].append(t_db - t_q)
                            
        best_tid, best_score, best_offset = None, 0, 0
        for tid, deltas in matches.items():
            if not deltas: continue
            vals, counts = np.unique(deltas, return_counts=True)
            idx = np.argmax(counts)
            if counts[idx] > best_score:
                best_score, best_tid, best_offset = counts[idx], tid, vals[idx]
                
        if best_tid is None or best_score < MATCH_THRESHOLD:
            print("УРОВЕНЬ 1: Трек не найден в базе. Поиск прерван ")
            return

        name = self.TRACK_REGISTRY[best_tid]['name']
        offset_sec = best_offset * HOP_FINAL / TARGET_SR
        confidence_classic = min(100.0, (best_score / len(q_hashes)) * 1000)
        
        print(f"Найдено совпадение: {name} (Смещение: {offset_sec:.1f}с)")
        print(f"Уверенность алгоритма Ванга: {confidence_classic:.1f}%")

        # уровень 2 - ИИ-Верификация (OpenL3)
        print("\nЗапуск семантической ИИ-верификации ")
        self.load_ai_model()
        
        # Получаем глобальный вектор запроса
        emb_query, _ = openl3.get_audio_embedding(y_q, TARGET_SR, model=self.ai_model, hop_size=0.1, verbose=0)
        query_vector = np.mean(emb_query, axis=0).reshape(1, -1)
        
        # Вырезаем кусок из оригинала, куда указал первый алгоритм
        orig_path = self.TRACK_REGISTRY[best_tid]['path']
        y_orig, _ = librosa.load(orig_path, sr=TARGET_SR, offset=offset_sec, duration=len(y_q)/TARGET_SR)
        
        # Получаем вектор оригинального куска
        emb_orig, _ = openl3.get_audio_embedding(y_orig, TARGET_SR, model=self.ai_model, hop_size=0.1, verbose=0)
        orig_vector = np.mean(emb_orig, axis=0).reshape(1, -1)
        
        # Считаем косинусное расстояние
        similarity = cosine_similarity(query_vector, orig_vector)[0][0] * 100
        
        search_time = (time.time() - t_start)
        
        # финал
        print("\n" + "="*50)
        print("ФИНАЛЬНЫЙ ОТЧЕТ СИСТЕМЫ (HYBRID ENGINE)")
        print("="*50)
        print(f"Трек: {name}")
        print(f"Точное время: {offset_sec:.2f} сек")
        print(f"Время анализа: {search_time:.2f} сек")
        print("-" * 50)
        print(f"Метрика 1 (DSP Hashes): {confidence_classic:.1f}%")
        print(f"Метрика 2 (AI Cosine) : {similarity:.1f}%")
        print("-" * 50)
        
        if similarity > 75.0:
            print("НЕЙРОСЕТЬ ПОДТВЕРЖДАЕТ: Звуковой профиль полностью совпадает!")
        elif similarity > 50.0:
            print("НЕЙРОСЕТЬ СОМНЕВАЕТСЯ: Возможны сильные искажения, но ритм похож")
        else:
            print("НЕЙРОСЕТЬ ОТВЕРГАЕТ: Это ложное математическое совпадение (False Positive)")
        print("="*50)

# 4. интерфейс
def main():
    print(r"""
 
    Heavy AI Analysis Engine
    """)
    
    engine = ShazamHybridEngine()
    engine.build_database()
    
    if not engine.is_loaded: return
        
    while True:
        print("\n--- ГЛАВНОЕ МЕНЮ ---")
        print("1. Сгенерировать зашумленный фрагмент")
        print("2. Глубокое распознавание (DSP + AI Verification)")
        print("3. Выход")
        choice = input("Выберите действие (1-3): ")
        
        if choice == '3': break
            
        elif choice == '1':
            # Тот же блок генерации, что и в app.py
            print("\nДоступные треки:")
            for tid, info in engine.TRACK_REGISTRY.items():
                print(f"  [{tid+1}] {info['name']}")
            try:
                t_idx = int(input("Номер трека (1-10): ")) - 1
                path = engine.TRACK_REGISTRY[t_idx]['path']
                start_sec = float(input("С какой секунды: "))
                duration = float(input("Длительность: "))
                print("\n1. Белый шум")
                print("\n2. Клиппинг")
                print("\n3. Чистый")
                n_type = input("Тип искажения: ")
                snr = float(input("SNR (дБ): ")) if n_type == '1' else 0
                
                y_f, _ = librosa.load(path, sr=TARGET_SR)
                s1 = int(start_sec * TARGET_SR / HOP_FINAL) * HOP_FINAL
                y_c = y_f[s1:s1 + int(duration * TARGET_SR)]
                sf.write("noisy_ai.wav", apply_degradation(y_c, n_type, snr), TARGET_SR)
                print("Фрагмент сохранен как 'noisy_ai.wav'")
            except Exception as e: print(f"Ошибка: {e}")
                
        elif choice == '2':
            q_file = input("Имя файла: ") or "noisy_ai.wav"
            if os.path.exists(q_file):
                engine.deep_recognize(q_file)
            else:
                print("Файл не найден.")

if __name__ == "__main__":
    main()