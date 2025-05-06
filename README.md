# Chatbot Layanan Akta Kelahiran

Ini adalah aplikasi web Flask yang menyediakan informasi tentang layanan akta kelahiran dengan chatbot berbasis NLTK terintegrasi.

## Fitur

- Halaman landing yang responsif tentang layanan akta kelahiran
- Chatbot interaktif yang menjawab pertanyaan tentang akta kelahiran
- Pemrosesan bahasa alami berbasis NLTK 
- Sistem intents terstruktur untuk alur percakapan yang lebih baik

## Instalasi

1. Clone repositori ini

```bash
git clone https://github.com/nurMangg/web-dukcapil-chatbot.git
```

2. Instal dependensi yang diperlukan:

```bash
pip install -r requirements.txt
```

3. Latih model chatbot:

```bash
python train_chatbot.py
```

4. Jalankan aplikasi Flask:

```bash
python app.py
```

5. Akses aplikasi di http://127.0.0.1:5000/

## Struktur Proyek

- `app.py`: Aplikasi Flask utama
- `train_chatbot.py`: Skrip untuk melatih model chatbot
- `intents.json`: Berisi data pelatihan dengan intents, pola, dan respons
- `classes.pkl`: File pickle yang berisi kelas intent
- `chatbot_model.h5`: Model TensorFlow yang telah dilatih
- `templates/`: Berisi template HTML
- `static/`: Berisi CSS, JavaScript, dan gambar

## Cara Kerja

Chatbot menggunakan jaringan saraf yang dilatih pada intents yang telah ditentukan terkait layanan akta kelahiran. Ketika pengguna mengirimkan pesan, aplikasi:

1. Melakukan tokenisasi dan lemmatisasi input
2. Membuat bag of words
3. Memprediksi intent menggunakan model yang telah dilatih
4. Mengembalikan respons berdasarkan intent yang diprediksi

Jika file model tidak ditemukan, aplikasi akan kembali ke sistem pencocokan kata kunci sederhana.

## Kustomisasi

Untuk menyesuaikan respons chatbot, edit file `intents.json` dan latih ulang model menggunakan `train_chatbot.py`.

