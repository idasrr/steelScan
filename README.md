# SteelScan: Klasifikasi Cacat Permukaan Baja

Klasifikasi 6 jenis cacat permukaan baja canai panas (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches) dari gambar grayscale. Fokus proyek ini adalah **evaluasi yang jujur**: baseline, validasi silang, ablasi, uji ketahanan, Grad-CAM, dan analisis kesalahan, bukan sekadar satu angka akurasi.

**Demo:** [isi link Hugging Face Spaces]

**Data:** NEU surface defect database (Song dan Yan, Northeastern University), 1.800 gambar grayscale 200×200, 300 per kelas, diambil dari Kaggle (`kaustubhdikshit/neu-surface-defect-database`). Lisensi tidak dinyatakan di halaman Kaggle; data asli milik pembuatnya dan dipakai untuk tujuan edukasi. [isi sitasi pembuat asli]

## Pendekatan
| Tahap | Isi |
|---|---|
| Preprocessing | OpenCV: CLAHE (kontras lokal) dan resize ke 224×224 |
| Baseline | Fitur HOG + SVM RBF (scikit-learn) |
| Model utama | MobileNetV2 (ImageNet) sebagai ekstraktor fitur beku, dengan head klasifikasi yang dilatih (TensorFlow/Keras); augmentasi flip, rotasi kecil, kontras |
| Evaluasi | 20% data (360 gambar) disisihkan sebagai test set; stratified 5-fold CV pada 1.440 gambar sisanya |
| Ablasi | Dengan dan tanpa fine-tuning 30 layer terakhir backbone |
| Ketahanan | Gangguan kecerahan, noise Gaussian, dan blur pada gambar test |
| Interpretasi | Grad-CAM dan analisis contoh salah prediksi |
| Aplikasi | Streamlit: upload gambar, prediksi, Grad-CAM, log prediksi ke SQLite, peringatan jika satu jenis cacat melonjak |

## Hasil

**Validasi silang 5-fold** (epoch tetap, tanpa early stopping; batas atas CI dipotong di 1,0):

| Model | Akurasi (95% CI) | Macro-F1 (95% CI) |
|---|---|---|
| HOG + SVM | 0,8611 [0,8332, 0,8891] | 0,8596 [0,8318, 0,8874] |
| MobileNetV2 (final) | 0,9958 [0,9911, 1,0000] | 0,9958 [0,9911, 1,0000] |

**Test set (360 gambar):** MobileNetV2 akurasi **0,9917** (3 salah), interval kepercayaan 95% (Wilson) sekitar 0,976 sampai 0,997; HOG + SVM akurasi 0,8889.

**Ablasi fine-tuning.** Membuka 30 layer terakhir backbone (tahap 2) tidak memperbaiki hasil bersih, tetapi membuat model lebih tahan terhadap blur dan noise:

| Konfigurasi | Akurasi CV | Akurasi test | Noise σ=20 | Blur kernel 7 |
|---|---|---|---|---|
| Tanpa fine-tuning (final) | 0,9958 | 0,9917 | 0,753 | 0,775 |
| Dengan fine-tuning | 0,9653 | 0,9833 | 0,808 | 0,936 |

Model final dipilih tanpa fine-tuning karena val loss dan akurasi CV lebih baik, bukan berdasarkan test set. Ada trade-off: untuk kondisi gambar yang tidak bersih (blur), versi dengan fine-tuning lebih stabil.

**Uji ketahanan** (akurasi test; model SVM dilatih tanpa augmentasi, CNN dengan flip, rotasi, dan kontras):

| Gangguan | CNN (final) | HOG + SVM |
|---|---|---|
| Tanpa gangguan | 0,992 | 0,889 |
| Kecerahan -60 / +60 | 0,997 / 0,978 | 0,850 / 0,700 |
| Noise σ = 5 / 10 / 20 | 0,978 / 0,964 / 0,753 | 0,656 / 0,478 / 0,267 |
| Blur kernel 3 / 5 / 7 | 0,972 / 0,925 / 0,775 | 0,689 / 0,339 / 0,214 |

CNN jauh lebih tahan daripada baseline klasik terhadap kecerahan, noise ringan, dan blur ringan, tetapi turun tajam pada noise berat dan blur berat. Model tidak dilatih dengan augmentasi noise atau blur.

**Per kelas, kesalahan, dan Grad-CAM:** [isi dari classification report, contoh 3 gambar yang salah, dan temuan Grad-CAM]

**Inferensi:** rata-rata sekitar 172 ms per gambar di runtime Colab (30 panggilan, termasuk panggilan pertama yang lebih lambat, jadi angka kasar). Belum diukur di CPU Space.

## Keterbatasan
- Data kecil dan terkurasi (kondisi laboratorium); belum diuji di lini produksi. Akurasi mendekati batas atas, jadi selisih kecil antar konfigurasi tidak boleh ditafsirkan berlebihan; satu gambar setara 0,28 poin persen akurasi test, dan hasil antar-run bervariasi sekitar satu gambar.
- CI dihitung dari 5 fold yang tidak sepenuhnya independen: perkiraan kasar.
- Fine-tuning pada run pertama menurunkan akurasi latih di awal tahap 2; penyebabnya belum diselidiki.
- Gangguan pada uji ketahanan diterapkan setelah preprocessing (penyederhanaan).
- Grad-CAM adalah alat bantu interpretasi, bukan bukti kausal.
- Gambar di luar domain NEU menghasilkan prediksi yang tidak bermakna.

## Menjalankan
1. Buka `train_steelscan.ipynb` di Google Colab (runtime GPU), jalankan semua sel. Hasil: `steel_cnn.keras`, `results.json`, `cv_results.csv`, `robustness.csv`.
2. Taruh `steel_cnn.keras` di folder yang sama dengan `app.py` dan `steel_utils.py`.
3. `pip install -r requirements.txt`, lalu `streamlit run app.py`.

## Struktur
`train_steelscan.ipynb` (training dan evaluasi) · `steel_utils.py` (preprocessing, Grad-CAM) · `app.py` (Streamlit) · `requirements.txt`
