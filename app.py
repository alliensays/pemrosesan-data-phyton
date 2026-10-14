from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import shap 

app = Flask(__name__)
CORS(app)

# 1. Load Model dan Scaler
model = joblib.load('model_rf_bumil.pkl')
scaler = joblib.load('scaler_bumil.pkl')

# Siapkan Explainer SHAP di luar route biar enteng
explainer = shap.TreeExplainer(model)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 1. BUNGKUS JADI DATAFRAME
        df_input = pd.DataFrame([{
            'Age': float(data.get('usia', 0)),
            'SystolicBP': float(data.get('sistolik', 0)),
            'DiastolicBP': float(data.get('diastolik', 0)),
            'BS': float(data.get('gula_darah', 0)),
            'BodyTemp': float(data.get('suhu_tubuh', 0)),
            'HeartRate': float(data.get('detak_jantung', 0))
        }])

        # 2. NORMALISASI & KEMBALIKAN KE DATAFRAME
        input_scaled = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)

        # 3. PREDIKSI
        prediction = model.predict(input_scaled)
        hasil_prediksi = str(prediction[0]) 

        # 4. SHAP LOGIC
        nama_fitur = np.array(['Usia', 'Sistolik', 'Diastolik', 'Gula Darah', 'Suhu Tubuh', 'Detak Jantung'])
        
        shap_values = explainer.shap_values(input_scaled)
        class_idx = list(model.classes_).index(hasil_prediksi)
        
        if isinstance(shap_values, list):
            kontribusi_lokal = shap_values[class_idx][0]
        else:
            if len(shap_values.shape) == 3:
                kontribusi_lokal = shap_values[0, :, class_idx]
            else:
                kontribusi_lokal = shap_values[0]

       # 5. CARI SEMUA FAKTOR DOMINAN YANG MENYIMPANG (> 0)
        index_terurut = np.argsort(kontribusi_lokal)[::-1]
        
        faktor_dominan = []
        if hasil_prediksi != 'low risk': 
            for idx in index_terurut:
                # Syarat: Nilai SHAP harus > 0 (meningkatkan risiko)
                if kontribusi_lokal[idx] > 0: 
                    faktor_dominan.append(nama_fitur[idx])

        if 'Sistolik' in faktor_dominan or 'Diastolik' in faktor_dominan:
            # 1. Buat list baru untuk nampung hasil konversi istilah
            faktor_terformat = []
            status_tensi_dimasukkan = False
            
            for item in faktor_dominan:
                if item in ['Sistolik', 'Diastolik']:
                    if not status_tensi_dimasukkan:
                        faktor_terformat.append('Tekanan Darah')
                        status_tensi_dimasukkan = True
                else:
                    faktor_terformat.append(item)
            
            # Kembalikan hasil konversi ke variabel utama
            faktor_dominan = faktor_terformat

        # 6. KONDISIKAN PANJANG TEKS (Sekarang teksnya sudah lebih rapi)
        if faktor_dominan:
            if len(faktor_dominan) > 3:
                teks_faktor_dominan = ", ".join(faktor_dominan[:3]) + "..."
            else:
                teks_faktor_dominan = ", ".join(faktor_dominan)
        else:
            teks_faktor_dominan = "Tidak ada indikator spesifik"
   
        print(f"\nData masuk: {df_input.values[0]}")
        print(f"Hasil Prediksi Asli Model: {hasil_prediksi}")
        print(f"Faktor Dominan (SHAP): {teks_faktor_dominan}")

        return jsonify({
            'status': 'success',
            'hasil': hasil_prediksi,
            'faktor_dominan': teks_faktor_dominan
        })

    except Exception as e:
        import traceback
        print(f"Error API: {str(e)}") 
        print(traceback.format_exc()) 
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)