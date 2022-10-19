# Laporan Proyek Machine Learning - Winata Pranata

## Domain Proyek
*Cryptocurrency* sudah ada sekitar sepuluh tahun lalu dan kini telah menjadi cukup populer, tersebar luas , serta dilingkupi juga atas banyak kontroversi dari perkembangan yang inovatif, *Cryptocurrency* adalah mata uang digital dimana transaksi dapat dilakukan dengan transaksi online, tidak seperti mata uang umum *Cryptocurrency* dirancang beradasarkan Kriptografi, *Bitcoin* adalah salah satu jenisnya. Karakteristik unik *Bitcoin* adalah fluktuasi harga harian dan selalu berubah setiap hari. Oleh karena itu dilakukan sebuah penelitian harga *bitcoin* dengan menggunakan *machine learning*, yang mana dengan adanya *machine learning* ini diharapkan dapat memprediksi harga *bitcoin* dimasa mendatang, Sehingga dapat mengurangi resiko kerugian.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang diatas, permasalahan yang akan diselesaikan yaitu:
- Bagaimana cara memproses data harga mata uang *Bitcoin* sehingga dapat di latih dengan baik oleh model?
- Bagaimana menentukan model *machine learning* yang dapat memprediksi harga *bitcoin* dengan baik di masa yang akan datang?

### Goals

Tujuan dibuatnya proyek ini adalah sebagai berikut:
- Menghasilkan data harga *bitcoin* yang dapat digunakan dengan baik untuk model *machine learning*
- Menentukan model *machine learning* yang paling sesuai yang dapat digunakan dengan baik di masa mendatang

## Data Understanding
Dataset yang digunakan pada proyek ini merupakan dataset riwayat harga mata uang *bitcoin* yang diperoleh dari website kaggle: [Cryptocurrency Bitcoin Historical Prices](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory?select=coin_Bitcoin.csv).

Informasi dari data sebagai berikut:

* Format dataset yaitu CSV (Comma-Seperated Values)
* Jumlah kolom data yang terdapat didalam dataset berjumlah 10 kolom, antara lain: _SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap_.
* Terdapat 2991 jumlah sample yang terdapat didalam dataset.
* Terdapat 6 kolom data yang memiliki tipe data Float yaitu (_High, Low, Open, Close, Volume, Marketcap_), 
* Terdapat 1 kolom data yang memiliki tipe data Integer yaitu (_SNo_)
* Terdapat 2 kolom data yang memiliki tipe data Object atau String yaitu (_Name, Symbol_)
* Tidak terdapat _missing value_ pada dataset
### **Variabel-variabel pada dataset adalah sebagai berikut:**

* Name: Nama mata uang kripto
* Symbol: Simbol mata uang kripto
* Date: Tanggal pencatatan data
* High : Harga tertinggi pada hari tertentu
* Low : Harga terendah pada hari tertentu
* Open : Harga pembukaan pada hari tertentu
* Close : Harga penutupan pada hari tertentu
* Volume : Volume transaksi pada hari tertentu
* Mastercap : Kapitalisasi pasar dalam USD

## Data Preparation
Teknik data preparation yang dilakukan:
* Menghapus data yang tidak diperlukan dan merubah nama column

  Menangani Outlier
  
  Jika kita lihat visualisai outlier dibawah beberapa data numeric memiliki data outlier, untuk mengatasi outlier tersebut disini menggunakan teknik IQR Method yaitu dengan menghapus data yang berada diluar interquartile range. Interquartile merupakan range diantara kuartil pertama(25%) dan kuartil ketiga(75%).
  
  ![image](https://user-images.githubusercontent.com/62703894/196564317-27c0fb5e-c250-44c2-9000-37a0cb1312dd.png)
  
* Melakukan pembagian dataset
  Rasio perbandingan dataset pada proyek ini yaitu dengan membaginya menjadi 80% data training dan 20% data testing.

  ![image](https://user-images.githubusercontent.com/62703894/196565164-60390902-8c69-409c-9324-c438ea061457.png)
  
  Pada matriks korelasi di atas dapat disimpulkan bahwa kebanyakan variabel memiliki keterikatan dan korelasi yang kuat antar variabel lainnya, dimana nilai korelasi antar variabel bernilai lebih dari 0.8 atau mendekati 1.
  ![image](https://user-images.githubusercontent.com/62703894/196596796-4ec460e8-0aa6-4191-bff7-f14588fc94e5.png)
  
  Fitur Close Price pada harga Bitcoin menjadi target prediksi kali ini, maka dapat disimpulkan bahwa peningkatan harga bitcoin sebanding dengan penurunan jumlah sampel
  
  ![image](https://user-images.githubusercontent.com/62703894/196596845-426e2f25-9e6d-448d-b071-5157f6b95ff9.png)
  
  Korelasi yang terdapat dalam fitur Close pada sumbu y dengan fitur High, Low, Open, dan Marketcap termasuk korelasi yang tinggi. Sedangkan fitur Volume korelasi nya cukup lemah, sebaran datanya tidak membentuk pola.


  
* Normalisasi data
  Normalisasi data menggunakan library MinMaxScaler. MinMaxScaler mentransformasikan fitur dengan menskalakan setiap fitur ke rentang tertentu. Library ini menskalakan dan mentransformasikan setiap fitur secara individual sehingga berada dalam rentang yang diberikan pada set pelatihan, pada library ini memiliki range default antara 0 dan 1.

## Modeling

# Support Vector Regression
Support Vector Regression adalah algoritma pembelajaran supervised yang digunakan untuk memprediksi nilai diskrit. Support Vector Regression menggunakan prinsip yang sama dengan SVM. Ide dasar di balik SVR adalah menemukan garis yang paling sesuai. Dalam SVR, garis yang paling cocok adalah hyperplane yang memiliki jumlah poin maksimum. Adapun parameter yang digunakan sebagai berikut:
* kernel = rbf. Parameter ini merupakan metode yang digunakan untuk mengambil data sebagai input dan mengubahnya menjadi bentuk pemrosesan data yang diperlukan.
* gamma = 0.003. Secara intuitif, parameter gamma menentukan seberapa jauh pengaruh satu contoh pelatihan mencapai, dengan nilai rendah berarti 'jauh' dan nilai tinggi berarti 'dekat'. Parameter gamma dapat dilihat sebagai kebalikan dari radius pengaruh sampel yang dipilih oleh model sebagai vektor pendukung.
* C (parameter Regularisasi) = 100000. Parameter C menukar klasifikasi yang benar dari contoh pelatihan terhadap maksimalisasi margin fungsi keputusan. Untuk nilai C yang lebih besar, margin yang lebih kecil akan diterima jika fungsi keputusan lebih baik dalam mengklasifikasikan semua titik pelatihan dengan benar. C yang lebih rendah akan mendorong margin yang lebih besar, oleh karena itu fungsi keputusan yang lebih sederhana, dengan mengorbankan akurasi pelatihan. Dengan kata lain C berperilaku sebagai parameter regularisasi dalam SVR.

Model ini melakukan pelatihan untuk mendapatkan error seminimal mungkin.

# K-Nearest Neighbours
K-nearest neighbor (kNN) adalah algoritma pembelajaran mesin supervised yang dapat digunakan untuk menyelesaikan tugas klasifikasi dan regresi. Parameter yang digunakan pada model ini hanya akan menggunakan 1 parameter yaitu n_neighbours. Jumlah neighbours yang di gunakan yaitu sejumlah 5 neighbours. Kemudian, untuk menentukan titik mana dalam data yang paling mirip dengan input baru, KNN menggunakan perhitungan ukuran jarak. Metrik ukuran jarak yang digunakan secara default pada library sklearn adalah Minkowski distance.

# Random Forest
Random Forest adalah Algoritma Pembelajaran Mesin Supervised yang digunakan secara luas dalam masalah Klasifikasi dan Regresi. Itu membangun pohon keputusan pada sampel yang berbeda dan mengambil suara mayoritas mereka untuk klasifikasi dan rata-rata dalam kasus regresi. Parameter yang digunakan diantaranya:
* n_estimator: jumlah trees (pohon) di forest. Di sini kita set n_estimator=6.
* max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. Di sini kita set max_depth=16.


## Evaluation
Metrik evaluasi yang digunakan yaitu mean squared error (MSE) yang mana metrik ini merupakan ukuran seberapa dekat garis pas dengan titik data. Untuk setiap titik data, model mengambil jarak secara vertikal dari titik ke nilai y yang sesuai pada kecocokan kurva (kesalahan), dan kuadratkan nilainya.

![image](https://user-images.githubusercontent.com/62703894/196490785-5fb24087-efb7-4668-8acb-d31e52cd983e.png)

Keterangan:

At = Nilai Aktual permintaan

Ft = Nilai hasil prediksi

n = banyaknya data

Visualisasi metrik mean squared error

![image](https://user-images.githubusercontent.com/62703894/196492062-9afbf587-29d5-4ffb-8055-762e84ce6e2b.png)

Dapat kita lihat dari gambar diatas bahwa MSE pada model Random Forest merupakan MSE yang paling rendah dari model yang lain

![image](https://user-images.githubusercontent.com/62703894/196493241-314442c2-233a-475c-a4c5-cacc8f11f86c.png)

Dari ketiga model yang telah digunakan dapat kita lihat bahwa prediksi mendekati nilai yang sebenarnya dan model random forest yang paling mendekati nilai sebenarnya.
