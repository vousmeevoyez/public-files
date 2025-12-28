# EVALUASI METODOLOGI PENELITIAN: EDM-Fuzzy Entropy untuk Deteksi Fault Multi-Sensor dengan ANN

## RINGKASAN EKSEKUTIF

Notebook ini mengimplementasikan pipeline deteksi fault multi-sensor menggunakan EDM-Fuzzy Entropy dan ANN. Secara keseluruhan, **alur perhitungan sudah masuk akal secara ilmiah**, namun terdapat beberapa **inkonsistensi dan potensi kesalahan** yang perlu diperbaiki untuk memastikan validitas hasil eksperimen.

---

## A. PERHITUNGAN CV (COEFFICIENT OF VARIATION)

### ✅ **BAGIAN YANG SUDAH BENAR:**

1. **Skala sebagai variabel** (POIN A.1):
   - ✅ Skala didefinisikan sebagai variabel `SCALES = list(range(1, 11))` di Cell 16
   - ✅ Dapat diubah untuk uji coba nilai CV yang berbeda
   - ✅ Implementasi sudah benar

2. **Dataset mix 2 fault** (POIN A.2):
   - ✅ Fungsi `test_cv_with_mixed_faults()` menggunakan `fault_configs_mix2`
   - ✅ Konfigurasi mix 2 fault dibuat dengan `create_mix2_fault_configs()`
   - ✅ Dataset mencakup fault tunggal dan mix 2 fault

3. **Output entropy dan CV** (POIN A.3):
   - ✅ Fungsi `calculate_cv_for_entropy_scales()` menghitung CV per skala
   - ✅ Fungsi `visualize_cv_and_entropy_stability()` menampilkan nilai entropy dan CV
   - ✅ Output mencakup: `scale`, `mean_entropy`, `std_entropy`, `cv`, `stability`

### ⚠️ **MASALAH YANG DITEMUKAN:**

1. **CV Calculation Error di Output:**
   ```
   Warning: No valid entropy data found for this scenario.
   ```
   - **Masalah**: Output menunjukkan "No valid entropy data found" untuk semua skenario mix 2 fault
   - **Penyebab Potensial**: 
     - Window size terlalu besar untuk data yang tersedia setelah coarse-graining
     - Entropy function gagal menghitung untuk window tertentu
     - Data setelah fault injection tidak valid
   - **Dampak**: CV tidak dapat dihitung, sehingga analisis kestabilan entropy tidak valid
   - **Perbaikan**: Perlu debugging untuk memastikan entropy calculation berhasil

2. **Konsistensi Data untuk CV:**
   - CV dihitung dari entropy values per skala, namun tidak jelas apakah entropy dihitung pada data yang sama (baseline vs fault)
   - **Rekomendasi**: Pastikan CV dihitung dengan membandingkan entropy pada kondisi normal vs fault untuk setiap skala

---

## B. PROSES HITUNG EDM-FUZZY ENTROPY

### ✅ **BAGIAN YANG SUDAH BENAR:**

1. **Entropy per sensor** (POIN B.1):
   - ✅ Fungsi `calculate_edmfuzzy_entropy_per_sensor_vectorized()` menghitung entropy per sensor
   - ✅ Fault injection dilakukan per sensor melalui `create_fault_scenario_per_sensor()`
   - ✅ Setiap sensor memiliki label terpisah: `label_fault_{sensor}`

2. **Output vektor entropy** (POIN B.2):
   - ✅ Output berbentuk dictionary: `{sensor: {scale: np.array([...])}}`
   - ✅ Setiap sensor memiliki vektor entropy untuk setiap skala: E1(1..S), E2(1..S), E3(1..S), E4(1..S)
   - ✅ Struktur data sesuai dengan metodologi

### ⚠️ **MASALAH YANG DITEMUKAN:**

1. **Mapping Index Entropy ke Original Data:**
   ```python
   # Di calculate_edmfuzzy_entropy_per_sensor_vectorized()
   if scale > 1:
       start_indices = orig_indices * scale
       end_indices = np.minimum(start_indices + window * scale, len(df))
   ```
   - **Masalah**: Mapping dari window indices ke original indices mungkin tidak akurat
   - **Penyebab**: 
     - Coarse-graining mengurangi panjang data (len(scaled) = len(original) // scale)
     - Window indices dari scaled data perlu dikonversi dengan benar ke original indices
   - **Dampak**: Entropy values mungkin tidak sesuai dengan timestamp yang benar
   - **Perbaikan**: Verifikasi mapping dengan contoh data kecil

2. **Forward Fill untuk NaN:**
   ```python
   entropy_vector = pd.Series(entropy_vector, dtype=float).ffill().bfill().fillna(0).to_numpy()
   ```
   - **Masalah**: Forward fill dan backward fill dapat mengisi nilai entropy dengan nilai yang tidak valid
   - **Dampak**: Entropy di awal/akhir data mungkin tidak representatif
   - **Rekomendasi**: 
     - Jangan fill dengan 0, lebih baik tetap NaN atau gunakan nilai entropy dari window terdekat
     - Atau, pastikan semua window menghasilkan nilai entropy yang valid

3. **Konsistensi Window Size:**
   - Window size default = 256, step = 16
   - Untuk skala besar (misalnya scale=10), data setelah coarse-graining menjadi sangat pendek
   - **Rekomendasi**: Window size sebaiknya disesuaikan dengan panjang data setelah coarse-graining

---

## C. PROSES FAULT DETECTION MENGGUNAKAN ANN

### ✅ **BAGIAN YANG SUDAH BENAR:**

1. **Input gabungan vektor** (POIN C.1):
   - ✅ Fungsi `prepare_combined_entropy_features()` menggabungkan E1, E2, E3, E4
   - ✅ Input berbentuk: `[E1(1), E1(2), ..., E1(S), E2(1), ..., E4(S)]`
   - ✅ Implementasi sudah benar

2. **Input node = 4 × jumlah skala** (POIN C.2):
   - ✅ `input_size = X_combined.shape[1]` = 4 × len(scales)
   - ✅ Untuk 10 skala: input_size = 40 nodes
   - ✅ Sesuai dengan metodologi

3. **Hidden layer best practice** (POIN C.3):
   - ✅ Fungsi `calculate_hidden_layer_size()` mengimplementasikan beberapa metode:
     - `heaton`: H1 antara input dan output, H2 = H1/2
     - `two_thirds`: H1 = 2/3 × input + output, H2 = H1/2
     - `twice`: H1 < 2 × input, H2 = H1/2
     - `half`: H1 = I, H2 = I/2
   - ✅ Mengikuti best practice dari Heaton Research

4. **Grid search setelah arsitektur** (POIN C.4):
   - ✅ Grid search dilakukan setelah penentuan arsitektur
   - ✅ Klasifikasi ANN dilakukan setelah grid search
   - ✅ Urutan sudah benar

### ⚠️ **MASALAH YANG DITEMUKAN:**

1. **Label Selection untuk Multi-Sensor:**
   ```python
   # Di run_fault_detection_with_combined_entropy()
   for sensor in sensors:
       label_sensor = f'label_fault_{sensor}'
       if label_sensor in df.columns:
           y = df[label_sensor].fillna(0).to_numpy().ravel()
           break
   ```
   - **Masalah**: Hanya menggunakan label dari sensor pertama yang ditemukan
   - **Dampak**: 
     - Jika input adalah gabungan E1-E4, tetapi label hanya dari satu sensor, model tidak belajar deteksi multi-sensor
     - Tidak jelas apakah ini untuk deteksi per sensor atau multi-sensor
   - **Rekomendasi**: 
     - Jika deteksi multi-sensor: buat label gabungan (OR dari semua sensor labels)
     - Jika deteksi per sensor: jalankan terpisah untuk setiap sensor

2. **Data Leakage Potensial:**
   - Entropy dihitung dari data yang sudah mengandung fault (karena `df[sensor] = y_fault` di `create_fault_scenario_per_sensor()`)
   - **Masalah**: Entropy seharusnya dihitung dari data original, kemudian fault di-inject untuk label
   - **Dampak**: Entropy sudah terpengaruh oleh fault, sehingga tidak representatif untuk baseline
   - **Perbaikan**: 
     - Hitung entropy dari data original (tanpa fault)
     - Inject fault hanya untuk label, bukan untuk perhitungan entropy

3. **Time-Ordered Split:**
   ```python
   X_tr = X_combined[:n_tr]
   X_va = X_combined[n_tr:n_tr+n_va]
   X_te = X_combined[n_tr+n_va:]
   ```
   - ✅ Split time-ordered sudah benar
   - ⚠️ Namun, perlu memastikan entropy vectors juga time-ordered (tidak ada shuffle)

4. **Threshold Selection:**
   ```python
   precs, recs, thrs = precision_recall_curve(y_va, p_va)
   f1s = 2 * precs * recs / (precs + recs + 1e-12)
   best_idx = np.argmax(f1s)
   t_star = float(thrs[best_idx])
   ```
   - ✅ Threshold dipilih dari validation set (tidak ada data leakage)
   - ✅ Implementasi sudah benar

5. **Grid Search Limited:**
   - Grid search hanya menguji 4 konfigurasi (heaton, two_thirds, twice, half)
   - **Rekomendasi**: Pertimbangkan menambahkan lebih banyak variasi (misalnya single layer, 3 layers, dll.)

---

## D. KESIMPULAN DAN REKOMENDASI PERBAIKAN

### ✅ **YANG SUDAH BENAR:**
1. Struktur umum metodologi sudah sesuai
2. Implementasi CV calculation sudah benar secara konsep
3. Output entropy per sensor sudah berbentuk vektor sesuai metodologi
4. Input ANN sudah menggabungkan E1-E4 dengan benar
5. Hidden layer mengikuti best practice
6. Grid search dilakukan setelah penentuan arsitektur

### ⚠️ **YANG PERLU DIPERBAIKI (PRIORITAS TINGGI):**

1. **Fix CV Calculation Error:**
   - Debug mengapa entropy calculation gagal untuk mix 2 fault
   - Pastikan window size disesuaikan dengan panjang data setelah coarse-graining
   - Verifikasi bahwa entropy function dapat menghandle data dengan fault

2. **Fix Data Leakage:**
   - Hitung entropy dari data original (tanpa fault)
   - Inject fault hanya untuk label, bukan untuk perhitungan entropy
   - Atau, dokumentasikan dengan jelas bahwa entropy dihitung dari data dengan fault (jika ini adalah desain penelitian)

3. **Fix Label Selection:**
   - Jika deteksi multi-sensor: buat label gabungan dari semua sensor
   - Jika deteksi per sensor: jalankan terpisah untuk setiap sensor dengan input yang sesuai

4. **Improve Index Mapping:**
   - Verifikasi mapping dari window indices ke original indices
   - Pastikan entropy values sesuai dengan timestamp yang benar

### 📝 **YANG PERLU DIPERBAIKI (PRIORITAS MENENGAH):**

5. **Improve Forward Fill:**
   - Jangan fill dengan 0, lebih baik tetap NaN atau gunakan nilai dari window terdekat
   - Atau, pastikan semua window menghasilkan nilai entropy yang valid

6. **Expand Grid Search:**
   - Tambahkan lebih banyak variasi arsitektur (single layer, 3 layers, dll.)
   - Pertimbangkan hyperparameter tuning (learning rate, activation function, dll.)

7. **Add Validation:**
   - Tambahkan validasi untuk memastikan entropy vectors memiliki panjang yang sama
   - Validasi bahwa semua sensor memiliki entropy untuk semua skala

---

## E. PERBAIKAN MINIMUM YANG DIPERLUKAN

### **1. Fix CV Calculation (KRITIS):**
```python
# Di test_cv_with_mixed_faults(), pastikan:
# - Window size disesuaikan dengan panjang data setelah coarse-graining
# - Entropy function dapat menghandle data dengan fault
# - Validasi bahwa entropy values tidak semua NaN
```

### **2. Fix Data Leakage (KRITIS):**
```python
# Di calculate_edmfuzzy_entropy_per_sensor_vectorized():
# - Hitung entropy dari data original (df_base), bukan dari df yang sudah di-inject fault
# - Atau, dokumentasikan dengan jelas bahwa ini adalah desain penelitian

# Di create_fault_scenario_per_sensor():
# - Jangan update df[sensor] = y_fault sebelum entropy calculation
# - Atau, simpan data original terlebih dahulu
```

### **3. Fix Label Selection (PENTING):**
```python
# Di run_fault_detection_with_combined_entropy():
# - Jika multi-sensor: y = (df['label_fault_sensor1'] | df['label_fault_sensor2'] | ...).astype(int)
# - Jika per sensor: jalankan terpisah untuk setiap sensor
```

### **4. Improve Index Mapping (PENTING):**
```python
# Verifikasi mapping dengan contoh data kecil
# Pastikan entropy values sesuai dengan timestamp yang benar
```

---

## F. VALIDASI HASIL EKSPERIMEN

Setelah perbaikan, validasi bahwa:
1. ✅ CV dapat dihitung untuk semua skenario (normal, fault tunggal, mix 2 fault)
2. ✅ Entropy vectors memiliki panjang yang sama untuk semua sensor dan skala
3. ✅ Entropy dihitung dari data yang konsisten (original atau dengan fault, sesuai desain)
4. ✅ Label sesuai dengan tujuan deteksi (multi-sensor atau per sensor)
5. ✅ ANN dapat dilatih dan dievaluasi dengan hasil yang masuk akal

---

**Dokumen ini dibuat berdasarkan analisis notebook `final_databaru_optimized_revisi_2_pisah.ipynb`**

