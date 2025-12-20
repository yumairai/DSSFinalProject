import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from rapidfuzz import fuzz, process
from fuzzywuzzy import process as fuzzy_process

# ======================================================================
# LOAD MODEL
# ======================================================================
def load_model_features(model_file: str, feature_names_file: str) -> Tuple[Optional[object], Optional[list]]:
    try:
        model = joblib.load(model_file)
        feature_names = joblib.load(feature_names_file)
        return model, feature_names
    except Exception as e:
        st.error(f"❌ Error memuat model: {str(e)}")
        return None, None

def get_feature_metadata() -> Dict[str, Dict]:
    """
    Metadata untuk setiap feature: kategori dan tipe (benefit/cost)
    """
    return {
        # Demografi
        'Age': {'category': 'Demographics', 'type': 'Benefit', 'desc': 'Usia pelanggan'},
        'Gender': {'category': 'Demographics', 'type': 'Benefit', 'desc': 'Jenis kelamin'},
        'Income': {'category': 'Demographics', 'type': 'Benefit', 'desc': 'Pendapatan'},
        'AgeGroup': {'category': 'Demographics', 'type': 'Benefit', 'desc': 'Kelompok usia'},
        'YoungCustomer': {'category': 'Demographics', 'type': 'Benefit', 'desc': 'Pelanggan muda'},
        'SeniorCustomer': {'category': 'Demographics', 'type': 'Benefit', 'desc': 'Pelanggan senior'},
        'HighIncome': {'category': 'Demographics', 'type': 'Benefit', 'desc': 'Pendapatan tinggi'},
        
        # Perilaku Kunjungan
        'VisitFrequency': {'category': 'Visit Behavior', 'type': 'Benefit', 'desc': 'Frekuensi kunjungan'},
        'AverageSpend': {'category': 'Visit Behavior', 'type': 'Cost', 'desc': 'Rata-rata pengeluaran'},
        'PreferredCuisine': {'category': 'Visit Behavior', 'type': 'Benefit', 'desc': 'Masakan favorit'},
        'TimeOfVisit': {'category': 'Visit Behavior', 'type': 'Benefit', 'desc': 'Waktu kunjungan'},
        'GroupSize': {'category': 'Visit Behavior', 'type': 'Benefit', 'desc': 'Ukuran grup'},
        'DiningOccasion': {'category': 'Visit Behavior', 'type': 'Benefit', 'desc': 'Jenis acara makan'},
        'MealType': {'category': 'Visit Behavior', 'type': 'Benefit', 'desc': 'Tipe makanan'},
        'FrequentVisitor': {'category': 'Visit Behavior', 'type': 'Benefit', 'desc': 'Pengunjung sering'},
        
        # Status/Interaksi
        'OnlineReservation': {'category': 'Customer Status', 'type': 'Benefit', 'desc': 'Reservasi online'},
        'DeliveryOrder': {'category': 'Customer Status', 'type': 'Benefit', 'desc': 'Order delivery'},
        'LoyaltyProgramMember': {'category': 'Customer Status', 'type': 'Benefit', 'desc': 'Member loyalty'},
        'OnlineUser': {'category': 'Customer Status', 'type': 'Benefit', 'desc': 'Pengguna online'},
        'LoyalCustomer': {'category': 'Customer Status', 'type': 'Benefit', 'desc': 'Pelanggan setia'},
        
        # Rating/Feedback
        'WaitTime': {'category': 'Service Quality', 'type': 'Cost', 'desc': 'Waktu tunggu'},
        'ServiceRating': {'category': 'Service Quality', 'type': 'Benefit', 'desc': 'Rating layanan'},
        'FoodRating': {'category': 'Service Quality', 'type': 'Benefit', 'desc': 'Rating makanan'},
        'AmbianceRating': {'category': 'Service Quality', 'type': 'Benefit', 'desc': 'Rating suasana'},
        'TotalRating': {'category': 'Service Quality', 'type': 'Benefit', 'desc': 'Total rating'},
        'AvgRating': {'category': 'Service Quality', 'type': 'Benefit', 'desc': 'Rata-rata rating'},
        'RatingStd': {'category': 'Service Quality', 'type': 'Cost', 'desc': 'Standar deviasi rating'},
        'MaxRating': {'category': 'Service Quality', 'type': 'Benefit', 'desc': 'Rating maksimum'},
        'MinRating': {'category': 'Service Quality', 'type': 'Cost', 'desc': 'Rating minimum'},
        'RatingRange': {'category': 'Service Quality', 'type': 'Cost', 'desc': 'Rentang rating'},
        
        # Engineered Features
        'SpendPerPerson': {'category': 'Financial', 'type': 'Cost', 'desc': 'Pengeluaran per orang'},
        'SpendToIncomeRatio': {'category': 'Financial', 'type': 'Cost', 'desc': 'Rasio spend/income'},
        'HighSpender': {'category': 'Financial', 'type': 'Benefit', 'desc': 'Pengeluaran tinggi'},
        'LongWait': {'category': 'Service Quality', 'type': 'Cost', 'desc': 'Tunggu lama'},
        'WaitToService': {'category': 'Service Quality', 'type': 'Cost', 'desc': 'Rasio wait/service'},
        'Rating_x_Loyalty': {'category': 'Interaction', 'type': 'Benefit', 'desc': 'Rating × Loyalty'},
        'Rating_x_Frequency': {'category': 'Interaction', 'type': 'Benefit', 'desc': 'Rating × Frequency'},
        'Wait_x_Service': {'category': 'Interaction', 'type': 'Cost', 'desc': 'Wait × Service'},
        'Spend_x_Rating': {'category': 'Interaction', 'type': 'Benefit', 'desc': 'Spend × Rating'},
        'LargeGroup': {'category': 'Visit Behavior', 'type': 'Benefit', 'desc': 'Grup besar'},
        'Solo': {'category': 'Visit Behavior', 'type': 'Benefit', 'desc': 'Makan sendiri'},
        'ConsistentQuality': {'category': 'Service Quality', 'type': 'Benefit', 'desc': 'Kualitas konsisten'},
    }

def get_strategy_feature_mapping() -> Dict[str, Dict]:
    """
    Mapping strategi ke features yang relevan dengan bobot strategis
    Setiap strategi memiliki deskripsi lengkap dan contoh implementasi
    """
    return { 
    # ========== A. STRATEGI BERBASIS DEMOGRAFI ==========
    
    "A1: Program Diskon Pelajar/Mahasiswa": {
        'features': {
            'Age': 0.25,
            'YoungCustomer': 0.20,
            'AgeGroup': 0.15,
            'Income': 0.12,
            'AverageSpend': 0.10,
            'SpendToIncomeRatio': 0.08,
            'VisitFrequency': 0.05,
            'GroupSize': 0.03,
            'TimeOfVisit': 0.02,
            'OnlineUser': 0.00
        },
        'description': '''Strategi ini menargetkan segmen pelajar dan mahasiswa (usia 17–25 tahun) yang memiliki 
        daya beli relatif terbatas, namun berpotensi menjadi pelanggan loyal dalam jangka panjang. Program ini 
        menawarkan diskon khusus dengan menunjukkan kartu pelajar atau mahasiswa, paket hemat untuk kelompok 
        belajar, serta promo di jam tertentu seperti setelah jam sekolah (15:00–18:00). Dengan membangun loyalitas 
        sejak dini, restoran dapat menciptakan basis pelanggan yang kuat untuk masa depan.''',
        'implementation': [
            'Diskon 15–20% dengan menunjukkan kartu pelajar/mahasiswa yang masih berlaku',
            'Paket "Study Group" untuk 4–6 orang dengan harga khusus',
            'Fasilitas Wi-Fi gratis tanpa batas dan colokan listrik di setiap meja',
            'Menu ekonomis dengan porsi mencukupi (Rp 25.000–40.000)',
            'Loyalty card khusus pelajar: beli 5 kali gratis 1 kali',
            'Promo "After School Special" pukul 15:00–18:00',
            'Event rutin: Student Night setiap Jumat malam'
        ]
    },
    
    "A2: Paket Keluarga Premium": {
        'features': {
            'Age': 0.18,
            'AgeGroup': 0.15,
            'GroupSize': 0.20,
            'LargeGroup': 0.15,
            'AverageSpend': 0.12,
            'Income': 0.08,
            'DiningOccasion': 0.06,
            'TimeOfVisit': 0.03,
            'MealType': 0.03,
            'Gender': 0.00
        },
        'description': '''Program yang dirancang khusus untuk keluarga dengan anak-anak, dengan menawarkan paket 
        lengkap yang mencakup menu orang dewasa dan menu anak. Strategi ini mempertimbangkan kebutuhan keluarga 
        seperti kursi bayi, menu anak, area bermain, serta porsi makanan yang dapat dibagi. Target utama adalah 
        keluarga dengan pendapatan menengah ke atas (usia 30–50 tahun) yang mencari tempat makan nyaman untuk 
        quality time bersama keluarga, terutama saat akhir pekan dan momen spesial.''',
        'implementation': [
            'Paket "Happy Family" untuk 2 orang dewasa + 2 anak dengan harga bundling',
            'Kids Combo: hidangan utama + minuman + dessert + mainan edukatif',
            'Fasilitas kids corner dengan permainan yang aman dan mendidik',
            'Ketersediaan high chair dan peralatan makan anak di setiap area',
            'Menu khusus anak: nugget, pasta, dan pizza mini dengan gizi seimbang',
            'Promo "Family Weekend": diskon 20% untuk keluarga setiap Sabtu–Minggu',
            'Birthday package: gratis kue ulang tahun untuk reservasi makan keluarga'
        ]
    },
    
    "A3: Layanan Ramah Lansia": {
        'features': {
            'Age': 0.25,
            'SeniorCustomer': 0.25,
            'AgeGroup': 0.15,
            'ServiceRating': 0.12,
            'WaitTime': 0.08,
            'AmbianceRating': 0.06,
            'Income': 0.05,
            'VisitFrequency': 0.02,
            'TimeOfVisit': 0.02,
            'GroupSize': 0.00
        },
        'description': '''Strategi yang ditujukan khusus bagi pelanggan lansia (usia 60 tahun ke atas) dengan fokus 
        pada kenyamanan, kemudahan akses, dan pelayanan yang lebih personal. Program ini memperhatikan kebutuhan 
        khusus pelanggan lansia seperti menu rendah garam dan gula, porsi yang lebih kecil, serta pelayanan yang 
        sabar dan responsif. Dengan menyediakan fasilitas ramah lansia, restoran dapat menarik segmen pelanggan 
        yang umumnya loyal dan menghargai kualitas pelayanan.''',
        'implementation': [
            'Diskon khusus lansia sebesar 15% untuk usia 60 tahun ke atas setiap hari',
            'Kursi prioritas yang dekat dengan pintu masuk dan toilet',
            'Menu sehat: rendah garam, rendah gula, dan tinggi serat',
            'Pilihan porsi fleksibel (kecil atau reguler) dengan harga yang proporsional',
            'Staf yang dilatih untuk melayani pelanggan lansia dengan lebih sabar',
            'Program "Senior Morning": promo sarapan pukul 07:00–10:00',
            'Penyediaan kaca pembesar menu dan pencahayaan ruangan yang memadai'
        ]
    },
        
            "A4: Menu Sehat Senior": {
        'features': {
            'Age': 0.20,
            'SeniorCustomer': 0.20,
            'AgeGroup': 0.15,
            'FoodRating': 0.15,
            'Income': 0.10,
            'AvgRating': 0.08,
            'VisitFrequency': 0.05,
            'LoyalCustomer': 0.04,
            'ServiceRating': 0.03,
            'HighIncome': 0.00
        },
        'description': '''Menu khusus yang diformulasikan untuk mendukung kesehatan pelanggan lansia dengan 
        kandungan nutrisi seimbang, rendah sodium, rendah gula, tinggi serat, serta mudah dicerna. Strategi 
        ini menargetkan pelanggan lansia yang memiliki kesadaran tinggi terhadap kesehatan, serta keluarga 
        yang peduli pada asupan gizi orang tua mereka. Setiap menu dilengkapi dengan informasi nutrisi yang 
        jelas serta rekomendasi dari ahli gizi.''',
        'implementation': [
            'Menu "Golden Age" dengan 15 pilihan hidangan bernutrisi terukur',
            'Kolaborasi dengan ahli gizi dalam perancangan menu',
            'Label nutrisi yang jelas meliputi kalori, sodium, gula, dan serat',
            'Pilihan metode masak kukus, panggang, dan rebus untuk menghindari makanan berminyak',
            'Porsi dengan dominasi sayuran, protein rendah lemak, dan karbohidrat kompleks',
            'Konsultasi gizi gratis bagi member lansia',
            'Program mingguan "Healthy Thursday" dengan menu sehat baru setiap minggu'
        ]
    },
    
    "A5: Promo Weekend Generasi Muda": {
        'features': {
            'Age': 0.20,
            'YoungCustomer': 0.20,
            'TimeOfVisit': 0.15,
            'DiningOccasion': 0.12,
            'GroupSize': 0.10,
            'OnlineUser': 0.08,
            'VisitFrequency': 0.07,
            'AverageSpend': 0.05,
            'OnlineReservation': 0.03,
            'Gender': 0.00
        },
        'description': '''Program promosi yang dirancang khusus untuk generasi muda (usia 18–35 tahun) pada 
        akhir pekan dengan menggabungkan pengalaman bersantap, hiburan, dan aktivitas sosial. Target utama 
        adalah mahasiswa dan profesional muda yang mencari tempat berkumpul dan bersantai di akhir pekan. 
        Strategi ini memanfaatkan jam kunjungan puncak akhir pekan untuk memaksimalkan tingkat okupansi 
        melalui penawaran harga khusus.''',
        'implementation': [
            'Program "Weekend Vibes": diskon 25% untuk pelanggan usia 18–35 tahun setiap Sabtu–Minggu',
            'Promo beli 2 gratis 1 untuk minuman atau dessert di akhir pekan',
            'Live music atau penampilan DJ setiap Sabtu malam',
            'Area foto Instagramable dengan dekorasi estetik',
            'Promo "Bring Your Squad": gratis appetizer untuk grup berjumlah 5 orang atau lebih',
            'Happy hour diperpanjang setiap Jumat pukul 17:00–21:00',
            'Kontes media sosial: unggah foto dan tag akun restoran untuk undian hadiah'
        ]
    },
    
    "A6: Event Komunitas Generasi": {
        'features': {
            'Age': 0.18,
            'AgeGroup': 0.18,
            'YoungCustomer': 0.12,
            'SeniorCustomer': 0.10,
            'DiningOccasion': 0.12,
            'TimeOfVisit': 0.10,
            'GroupSize': 0.08,
            'VisitFrequency': 0.06,
            'LoyaltyProgramMember': 0.04,
            'ServiceRating': 0.02
        },
        'description': '''Strategi pembangunan komunitas melalui penyelenggaraan acara rutin yang disesuaikan 
        dengan segmen usia tertentu. Youth Night ditujukan bagi generasi muda (18–30 tahun) dengan suasana 
        yang energik dan modern, sedangkan Senior Morning ditujukan bagi pelanggan lansia (55 tahun ke atas) 
        dengan suasana yang tenang dan nyaman. Program ini bertujuan menciptakan rasa kebersamaan serta 
        meningkatkan frekuensi kunjungan ulang.''',
        'implementation': [
            'Program "Youth Night" setiap Jumat dengan live music, permainan, dan networking (19:00–23:00)',
            'Program "Senior Morning" setiap Rabu dengan sarapan prasmanan dan musik ringan (08:00–11:00)',
            'Tema acara bulanan seperti karaoke night, trivia night, dan turnamen permainan',
            'Harga khusus bagi peserta acara',
            'Poin loyalitas ganda bagi pelanggan yang menghadiri acara',
            'Komunitas WhatsApp Group untuk informasi acara dan promo',
            'Kerja sama dengan komunitas lokal seperti klub olahraga dan klub buku'
        ]
    },
    
    "A7: Kids Combo Spesial": {
        'features': {
            'GroupSize': 0.20,
            'LargeGroup': 0.15,
            'DiningOccasion': 0.15,
            'AverageSpend': 0.12,
            'TimeOfVisit': 0.10,
            'MealType': 0.10,
            'Age': 0.08,
            'FoodRating': 0.05,
            'ServiceRating': 0.03,
            'Income': 0.02
        },
        'description': '''Menu combo khusus anak-anak yang dirancang menarik, bergizi, dan terjangkau. Strategi 
        ini menargetkan keluarga dengan anak usia 2–12 tahun yang membutuhkan pilihan menu sesuai selera 
        anak. Setiap paket disajikan dengan mainan edukatif serta tampilan yang menyenangkan untuk 
        meningkatkan nafsu makan anak, sekaligus membantu orang tua memastikan asupan gizi yang sehat.''',
        'implementation': [
            'Paket "Little Star" (usia 2–5 tahun): nugget atau pasta + jus + buah + mainan',
            'Paket "Young Explorer" (usia 6–12 tahun): mini burger atau pizza + smoothie + dessert + mainan',
            'Penyajian kreatif dengan bentuk wajah senyum dan piring berwarna',
            'Mainan edukatif yang diperbarui setiap bulan sebagai koleksi',
            'Sayuran tersembunyi dalam menu melalui olahan saus',
            'Harga paket tetap Rp 35.000–50.000 sudah termasuk semua item',
            'Program "Kids Eat Free": 1 kids combo gratis untuk setiap pembelian 2 menu utama dewasa (weekday lunch)'
        ]
    },
    

        # ========== B. STRATEGI BERBASIS PENDAPATAN ==========
        
            "B1: VIP Membership Elite": {
        'features': {
            'Income': 0.25,
            'HighIncome': 0.20,
            'AverageSpend': 0.15,
            'HighSpender': 0.12,
            'LoyaltyProgramMember': 0.10,
            'ServiceRating': 0.08,
            'VisitFrequency': 0.05,
            'AmbianceRating': 0.03,
            'OnlineUser': 0.02,
            'Spend_x_Rating': 0.00
        },
        'description': '''Program keanggotaan eksklusif yang ditujukan bagi pelanggan berpendapatan tinggi 
        dengan berbagai manfaat premium dan layanan yang dipersonalisasi. Target utama program ini adalah 
        profesional, pemilik bisnis, dan eksekutif yang menghargai eksklusivitas serta bersedia membayar 
        lebih untuk pengalaman bersantap yang unggul. Program ini berfokus pada penciptaan pengalaman VIP 
        yang berkesan melalui layanan prioritas, menu eksklusif, dan berbagai hak istimewa khusus.''',
        'implementation': [
            'Biaya keanggotaan Rp 2.000.000 per tahun dengan manfaat senilai 3–4 kali lipat',
            'Akses private dining room dengan layanan butler',
            'Reservasi prioritas dengan jaminan tempat duduk kapan saja',
            'Menu eksklusif dengan bahan premium seperti wagyu, lobster, dan truffle',
            'Layanan valet parking gratis dan welcome drink',
            'Diskon 20% untuk seluruh menu serta poin loyalitas ganda',
            'Privilege ulang tahun: gratis menu premium untuk member dan 3 tamu',
            'Event eksklusif triwulanan seperti wine tasting dan chef table experience',
            'Layanan personal concierge untuk catering dan acara khusus'
        ]
    },
    
    "B2: Bundling Hemat Ekonomis": {
        'features': {
            'Income': 0.22,
            'SpendToIncomeRatio': 0.20,
            'AverageSpend': 0.18,
            'SpendPerPerson': 0.15,
            'VisitFrequency': 0.10,
            'GroupSize': 0.07,
            'TimeOfVisit': 0.05,
            'Age': 0.03,
            'MealType': 0.00,
            'OnlineUser': 0.00
        },
        'description': '''Strategi bundling dengan konsep value-for-money bagi pelanggan dengan anggaran 
        terbatas yang tetap menginginkan pengalaman bersantap yang memuaskan. Program ini menawarkan 
        paket hemat dengan kombinasi menu utama, minuman, serta dessert atau appetizer dengan harga 
        yang lebih terjangkau. Target utamanya adalah keluarga berpendapatan menengah ke bawah, 
        pelajar, serta pekerja muda yang sensitif terhadap harga.''',
        'implementation': [
            'Paket "Hemat Harian" (Rp 35.000): nasi, lauk utama, sayur, dan es teh',
            'Paket "Duo Ekonomis" (Rp 60.000): 2 menu utama dan 2 minuman',
            'Paket "Family Bundle" (Rp 150.000): paket lengkap untuk 4 orang',
            'Program "Mix & Match": pilih 1 menu utama, 1 side, dan 1 minuman dengan harga bundling',
            'Promo weekday lunch: Rp 30.000 all-in pukul 11:00–14:00',
            'Gratis upgrade ukuran minuman untuk semua paket hemat',
            'Kartu stempel loyalitas: beli 8 paket hemat gratis 1 paket',
            'Promo pembayaran non-tunai: diskon tambahan 5% untuk e-wallet'
        ]
    },
    
    "B3: Cashback High Spender": {
        'features': {
            'AverageSpend': 0.25,
            'HighSpender': 0.22,
            'Income': 0.15,
            'Spend_x_Rating': 0.12,
            'LoyaltyProgramMember': 0.10,
            'VisitFrequency': 0.08,
            'HighIncome': 0.05,
            'ServiceRating': 0.03,
            'TotalRating': 0.00,
            'SpendPerPerson': 0.00
        },
        'description': '''Program penghargaan berbasis cashback bagi pelanggan dengan tingkat pengeluaran 
        yang tinggi dan konsisten. Semakin besar total transaksi, semakin besar persentase cashback 
        yang diperoleh. Program ini bertujuan mempertahankan pelanggan bernilai tinggi serta mendorong 
        peningkatan nilai transaksi. Cashback dapat digunakan pada kunjungan berikutnya atau ditukar 
        dengan reward eksklusif.''',
        'implementation': [
            'Skema cashback bertingkat: transaksi Rp 200.000 (5%), Rp 500.000 (8%), Rp 1.000.000 (12%)',
            'Akumulasi cashback tersimpan secara digital dalam akun membership',
            'Bonus cashback pada hari tertentu, seperti Selasa dengan cashback ganda',
            'Cashback dapat digunakan sebagai potongan transaksi pada kunjungan berikutnya',
            'Reward khusus: cashback 20% bagi pelanggan dengan transaksi di atas Rp 2 juta per bulan',
            'Bonus referral: cashback Rp 50.000 untuk setiap teman yang bergabung dan bertransaksi minimal Rp 200.000',
            'Promo bulan ulang tahun: cashback tiga kali lipat untuk semua transaksi',
            'Penukaran cashback dengan hadiah atau item eksklusif',
            'Laporan triwulanan berisi ringkasan cashback yang diperoleh dan digunakan'
        ]
    },
    
    "B4: Personal Pricing Dynamic": {
        'features': {
            'Income': 0.20,
            'SpendToIncomeRatio': 0.18,
            'AverageSpend': 0.15,
            'HighIncome': 0.12,
            'HighSpender': 0.10,
            'VisitFrequency': 0.10,
            'LoyaltyProgramMember': 0.08,
            'OnlineUser': 0.05,
            'Age': 0.02,
            'SpendPerPerson': 0.00
        },
        'description': '''Strategi penetapan harga dinamis yang dipersonalisasi berdasarkan profil pendapatan 
        dan pola pengeluaran pelanggan. Program ini memanfaatkan analitik data untuk memberikan promo 
        yang relevan sesuai dengan kemampuan dan preferensi pelanggan. Pelanggan berpendapatan tinggi 
        memperoleh akses harga khusus untuk menu premium, sementara pelanggan dengan anggaran terbatas 
        mendapatkan diskon pada menu bernilai tinggi.''',
        'implementation': [
            'Rekomendasi menu berbasis AI sesuai anggaran dan preferensi pelanggan',
            'Voucher dinamis yang otomatis diberikan sesuai profil pengeluaran',
            'Segmentasi pelanggan: Gold (pendapatan tinggi), Silver (menengah), dan Bronze (hemat)',
            'Promo yang dipersonalisasi melalui aplikasi dengan penawaran berbeda untuk setiap pelanggan',
            'Strategi upselling cerdas yang disesuaikan dengan kapasitas belanja pelanggan',
            'Fasilitas cicilan 0% untuk transaksi di atas Rp 500.000 bagi pelanggan tertentu',
            'Notifikasi pengingat menu yang sesuai dengan anggaran pelanggan',
            'Laporan bulanan pola pengeluaran dan rekomendasi penghematan',
            'Kenaikan tier loyalitas otomatis berdasarkan pencapaian transaksi'
        ]
    },
    
    "B5: Premium Pay Later": {
        'features': {
            'Income': 0.22,
            'HighIncome': 0.18,
            'AverageSpend': 0.15,
            'HighSpender': 0.12,
            'OnlineUser': 0.10,
            'LoyaltyProgramMember': 0.10,
            'ServiceRating': 0.06,
            'VisitFrequency': 0.05,
            'Age': 0.02,
            'Spend_x_Rating': 0.00
        },
        'description': '''Fasilitas pembayaran cicilan atau pay-later untuk pembelian menu premium dan paket 
        catering bernilai tinggi. Program ini memberikan kemudahan bagi pelanggan berpendapatan tinggi 
        untuk menikmati layanan premium tanpa mengganggu arus kas secara langsung. Program ini 
        dijalankan melalui kerja sama dengan perusahaan fintech yang menyediakan opsi cicilan fleksibel 
        dengan suku bunga yang kompetitif.''',
        'implementation': [
            'Fasilitas cicilan 0% untuk transaksi di atas Rp 1.000.000 dengan tenor 3 bulan',
            'Kerja sama dengan penyedia pay-later seperti Kredivo, Akulaku, dan Atome',
            'Menu dan paket tertentu yang memenuhi syarat untuk pembayaran cicilan',
            'Limit kredit pra-persetujuan hingga Rp 10.000.000 bagi pelanggan loyal',
            'Skema Buy Now Pay Later dengan pembayaran dalam 30 hari tanpa bunga',
            'Pilihan tenor cicilan fleksibel 3, 6, atau 12 bulan',
            'Akses awal menu musiman bagi pengguna fasilitas pay-later',
            'Cicilan khusus untuk paket catering dengan minimal transaksi Rp 5.000.000',
            'Integrasi digital pengajuan cicilan melalui aplikasi atau website'
        ]
    },
    
    "B6: Upgrade Menu Premium": {
        'features': {
            'Income': 0.20,
            'HighIncome': 0.18,
            'AverageSpend': 0.15,
            'HighSpender': 0.12,
            'FoodRating': 0.10,
            'ServiceRating': 0.10,
            'AvgRating': 0.08,
            'LoyaltyProgramMember': 0.05,
            'Spend_x_Rating': 0.02,
            'AmbianceRating': 0.00
        },
        'description': '''Program peningkatan nilai transaksi dengan mengajak pelanggan berpendapatan tinggi 
        untuk mencoba menu premium melalui harga khusus atau manfaat tambahan. Strategi ini bertujuan 
        memperkenalkan menu dengan margin tinggi kepada pelanggan yang memiliki daya beli kuat melalui 
        penawaran menarik seperti rekomendasi pairing, menu spesial chef, atau item edisi terbatas.''',
        'implementation': [
            'Program "Taste the Premium" dengan diskon 30% bagi pelanggan yang pertama kali mencoba menu premium',
            'Rekomendasi wine pairing untuk setiap menu utama premium',
            'Menu signature chef yang hanya tersedia secara eksklusif',
            'Menu premium musiman seperti truffle season, lobster month, dan wagyu week',
            'Gratis appetizer untuk setiap pemesanan menu utama premium',
            'Undangan VIP tasting event untuk mencoba menu premium terbaru',
            'Opsi upgrade menu dengan tambahan Rp 50.000',
            'Paket set premium berupa 5-course tasting menu lengkap dengan wine pairing',
            'Reservasi prioritas untuk area makan premium'
        ]
    },
    
    "B7: Paket Ekonomis Value": {
        'features': {
            'Income': 0.18,
            'SpendToIncomeRatio': 0.20,
            'AverageSpend': 0.18,
            'SpendPerPerson': 0.15,
            'VisitFrequency': 0.10,
            'Age': 0.08,
            'GroupSize': 0.06,
            'TimeOfVisit': 0.03,
            'MealType': 0.02,
            'YoungCustomer': 0.00
        },
        'description': '''Paket menu dengan proposisi nilai terbaik bagi pelanggan yang sensitif terhadap 
        harga. Program ini berfokus pada pemberian nilai persepsi yang tinggi melalui optimasi porsi, 
        efisiensi bahan baku, dan kemasan yang cerdas. Target utama adalah keluarga, pelajar, dan pekerja 
        dengan anggaran terbatas yang tetap menginginkan pengalaman bersantap yang memuaskan.''',
        'implementation': [
            'Paket "Super Value" (Rp 39.000): porsi besar menu utama, minuman jumbo, dan isi ulang gratis',
            'Paket "Family Value Pack" (Rp 120.000): hidangan lengkap untuk 4 orang',
            'Menu spesial harian dengan harga ekonomis (Rp 25.000)',
            'Nasi gratis tanpa batas untuk seluruh paket ekonomis',
            'Gratis sup atau salad sebagai hidangan pembuka',
            'Promo kombo hemat dengan skema beli 1 gratis 1 untuk menu tertentu',
            'Buffet makan siang hari kerja: Rp 45.000 all-you-can-eat (11:00–14:00)',
            'Paket pelajar Rp 30.000 dengan menunjukkan kartu pelajar',
            'Diskon takeaway sebesar 10% untuk seluruh paket ekonomis'
        ]
    },

        # ========== C. STRATEGI BERBASIS KEBIASAAN KUNJUNGAN ==========
        
        "C1: Loyalty Stamp Card": {
            'features': {
            'VisitFrequency': 0.25,
            'FrequentVisitor': 0.20,
            'LoyalCustomer': 0.15,
            'LoyaltyProgramMember': 0.12,
            'Rating_x_Frequency': 0.10,
            'AverageSpend': 0.08,
            'ServiceRating': 0.05,
            'AvgRating': 0.03,
            'TimeOfVisit': 0.02,
            'OnlineUser': 0.00
        },
        'description': '''Program loyalitas tradisional yang dikombinasikan dengan sistem digital melalui 
        mekanisme stamp card. Setiap kunjungan pelanggan akan mendapatkan satu stamp, dan setelah mencapai 
        jumlah tertentu pelanggan berhak memperoleh hadiah gratis. Program ini sederhana, mudah dipahami, 
        dan efektif dalam mendorong kunjungan berulang. Sistem stamp digital memudahkan pelacakan, mencegah 
        kecurangan, serta menyediakan data yang bernilai untuk keperluan analitik.''',
        'implementation': [
            'Stamp card digital melalui aplikasi mobile (tanpa kartu fisik)',
            'Program "Buy 10 Get 1 Free": setiap 10 kunjungan mendapatkan 1 menu utama gratis',
            'Pengganda stamp: kunjungan di hari kerja mendapatkan 2 kali stamp',
            'Bonus stamp untuk membawa teman (1 stamp tambahan per teman)',
            'Reward progresif: 5 stamp (minuman gratis), 10 stamp (makanan gratis), 20 stamp (menu premium gratis)',
            'Masa berlaku stamp selama 6 bulan sejak kunjungan terakhir',
            'Fitur transfer stamp ke teman (maksimal 3 stamp per bulan)',
            'Bonus ulang tahun berupa tambahan 5 stamp',
            'Gamifikasi berupa badge untuk pencapaian milestone tertentu'
        ]
    },

    "C2: Frequent Visitor Privileges": {
        'features': {
            'VisitFrequency': 0.25,
            'FrequentVisitor': 0.22,
            'LoyalCustomer': 0.15,
            'Rating_x_Frequency': 0.12,
            'LoyaltyProgramMember': 0.10,
            'AverageSpend': 0.08,
            'ServiceRating': 0.05,
            'TimeOfVisit': 0.03,
            'VisitFrequency': 0.00,
            'OnlineReservation': 0.00
        },
        'description': '''Program khusus yang memberikan hak istimewa dan bentuk apresiasi kepada pelanggan 
        dengan frekuensi kunjungan tinggi. Semakin sering pelanggan berkunjung, semakin besar manfaat yang 
        diperoleh. Program ini menciptakan pengalaman layaknya pelanggan VIP tanpa biaya keanggotaan, 
        sepenuhnya berbasis frekuensi kunjungan. Strategi ini efektif untuk mempertahankan pelanggan tetap 
        dan meningkatkan rasa dihargai.''',
        'implementation': [
            'Program "Frequent Visitor Day" setiap Selasa dengan diskon 30% bagi pelanggan dengan 8+ kunjungan per bulan',
            'Sistem tier: Bronze (4x/bulan), Silver (8x/bulan), Gold (12x/bulan)',
            'Manfaat bertingkat: Bronze (diskon 10%), Silver (15%), Gold (20%)',
            'Prioritas tempat duduk tanpa antrean bagi pelanggan Gold',
            'Penyambutan personal oleh staf dengan menyebut nama pelanggan',
            'Gratis appetizer untuk Silver dan gratis dessert untuk Gold',
            'Perayaan ulang tahun dengan kue gratis dan ucapan khusus',
            'Akses informasi lebih awal terkait menu baru',
            'Grup WhatsApp VIP untuk reservasi dan penyampaian umpan balik'
        ]
    },

    "C3: Level Membership Tiers": {
        'features': {
            'LoyaltyProgramMember': 0.22,
            'VisitFrequency': 0.20,
            'FrequentVisitor': 0.15,
            'AverageSpend': 0.12,
            'LoyalCustomer': 0.10,
            'Rating_x_Frequency': 0.08,
            'Spend_x_Rating': 0.06,
            'ServiceRating': 0.04,
            'Income': 0.03,
            'HighSpender': 0.00
        },
        'description': '''Sistem keanggotaan bertingkat yang terdiri dari Bronze, Silver, Gold, dan Platinum 
        dengan manfaat yang meningkat berdasarkan kombinasi frekuensi kunjungan dan total pengeluaran. 
        Program ini dirancang dengan pendekatan gamifikasi untuk mendorong pelanggan naik tingkat melalui 
        target yang jelas serta hadiah yang menarik, sehingga menciptakan keterlibatan dan loyalitas jangka 
        panjang.''',
        'implementation': [
            'Bronze: pendaftaran gratis, diskon 5%, dan hadiah ulang tahun',
            'Silver: 10 kunjungan atau belanja Rp 1 juta/3 bulan → diskon 10% dan prioritas tempat duduk',
            'Gold: 25 kunjungan atau belanja Rp 3 juta/3 bulan → diskon 15%, appetizer gratis, dan event eksklusif',
            'Platinum: 50 kunjungan atau belanja Rp 8 juta/3 bulan → diskon 20%, private dining, dan layanan concierge',
            'Pemeliharaan tier dengan minimal 1 kunjungan per bulan',
            'Fast track: perhitungan kunjungan dan belanja ganda pada bulan pertama',
            'Setiap tier membuka akses menu eksklusif tertentu',
            'Evaluasi tahunan dengan masa tenggang sebelum penurunan tier',
            'Notifikasi otomatis saat mendekati target kenaikan tier'
        ]
    },

    "C4: Win-back Reminder System": {
        'features': {
            'VisitFrequency': 0.20,
            'FrequentVisitor': 0.15,
            'LoyalCustomer': 0.15,
            'OnlineUser': 0.12,
            'LoyaltyProgramMember': 0.10,
            'TimeOfVisit': 0.10,
            'AverageSpend': 0.08,
            'Rating_x_Frequency': 0.06,
            'ServiceRating': 0.04,
            'Age': 0.00
        },
        'description': '''Strategi re-engagement yang ditujukan bagi pelanggan dengan penurunan frekuensi 
        kunjungan atau yang sudah lama tidak berkunjung. Program ini menggunakan pengingat otomatis melalui 
        email, WhatsApp, atau notifikasi aplikasi dengan penawaran yang dipersonalisasi untuk mendorong 
        pelanggan kembali. Strategi ini penting untuk mencegah churn dan mengaktifkan kembali pelanggan 
        yang tidak aktif.''',
        'implementation': [
            'Pemicu otomatis setelah 2 minggu tanpa kunjungan bagi pelanggan reguler',
            'Kampanye "We Miss You" dengan diskon khusus 25%',
            'Pesan personal yang mencantumkan menu favorit dan tanggal kunjungan terakhir',
            'Penawaran bertahap: minggu ke-2 (15%), minggu ke-4 (20%), minggu ke-6 (30%)',
            'Voucher dengan masa berlaku terbatas 7 hari untuk menciptakan urgensi',
            'Pendekatan multi-kanal: email, WhatsApp, notifikasi aplikasi, dan SMS',
            'Permintaan umpan balik terkait alasan berhenti berkunjung dengan insentif',
            'Menu comeback eksklusif khusus pelanggan yang kembali',
            'Pelacakan tingkat reaktivasi dan nilai pelanggan jangka panjang'
        ]
    },

    "C5: Exclusive Frequent Benefits": {
        'features': {
            'FrequentVisitor': 0.25,
            'VisitFrequency': 0.20,
            'LoyalCustomer': 0.15,
            'Rating_x_Frequency': 0.12,
            'LoyaltyProgramMember': 0.10,
            'ServiceRating': 0.08,
            'AverageSpend': 0.05,
            'AvgRating': 0.03,
            'Spend_x_Rating': 0.02,
            'Income': 0.00
        },
        'description': '''Program manfaat eksklusif yang hanya dapat diakses oleh pelanggan dengan frekuensi 
        kunjungan tinggi untuk menciptakan rasa eksklusivitas dan penghargaan atas loyalitas. Program ini 
        memberikan akses ke menu khusus, layanan prioritas, dan pengalaman unik yang tidak tersedia bagi 
        pelanggan umum, sehingga meningkatkan ikatan emosional pelanggan.''',
        'implementation': [
            'Secret menu khusus pelanggan dengan 8+ kunjungan per bulan',
            'Chef table experience bulanan bagi 10 pelanggan paling loyal',
            'Akses pemesanan lebih awal untuk menu baru',
            'Jam makan eksklusif pagi hari (06:00–08:00) dengan menu khusus',
            'Tur dapur dan sesi pertemuan dengan chef',
            'Gratis valet parking untuk pelanggan frequent tier Gold',
            'Prioritas undangan untuk event dan perayaan khusus',
            'Hak kustomisasi menu sesuai preferensi pelanggan',
            'Meja favorit selalu disiapkan untuk pelanggan paling loyal'
        ]
    },

    # ========== D. STRATEGI BERBASIS POLA PENGELUARAN ==========
    
    "D1: Strategic Upselling": { 
        'features': {
            'AverageSpend': 0.25,
            'SpendPerPerson': 0.18,
            'HighSpender': 0.15,
            'Spend_x_Rating': 0.12,
            'FoodRating': 0.10,
            'ServiceRating': 0.08,
            'GroupSize': 0.06,
            'Income': 0.04,
            'VisitFrequency': 0.02,
            'TimeOfVisit': 0.00
        },
        'description': '''Teknik upselling yang terstruktur dan berbasis pelatihan untuk meningkatkan 
        nilai rata-rata transaksi dengan menawarkan upgrade, add-on, dan opsi premium. Staf dilatih 
        untuk mengidentifikasi peluang upselling serta memberikan rekomendasi yang relevan tanpa 
        terkesan memaksa. Strategi ini berfokus pada peningkatan pengalaman pelanggan sekaligus 
        meningkatkan pendapatan per transaksi.''',
        'implementation': [
            'Pelatihan staf: teknik upselling, pengetahuan produk, dan penentuan waktu yang tepat',
            'Penjualan sugestif: "Apakah ingin menambahkan truffle seharga Rp 25k?"',
            'Kombinasi minuman: sommelier/barista merekomendasikan pasangan minuman yang sesuai',
            'Upgrade ukuran: "Upgrade ke porsi besar hanya tambah Rp 15k"',
            'Protein premium: "Ganti ayam ke wagyu beef dengan tambahan Rp 50k"',
            'Rekomendasi dessert: menampilkan menu dessert atau dessert cart setelah hidangan utama',
            'Promosi appetizer: harga spesial appetizer jika dipesan sebelum hidangan utama',
            'Paket combo: "Tambah sup + salad hanya Rp 20k (hemat Rp 15k)"',
            'Insentif komisi: bonus untuk staf yang mencapai target upselling'
        ]
    },

    "D2: Smart Cross-selling": {
        'features': {
            'AverageSpend': 0.22,
            'SpendPerPerson': 0.18,
            'GroupSize': 0.15,
            'Spend_x_Rating': 0.12,
            'FoodRating': 0.10,
            'HighSpender': 0.10,
            'ServiceRating': 0.07,
            'DiningOccasion': 0.04,
            'MealType': 0.02,
            'VisitFrequency': 0.00
        },
        'description': '''Strategi cross-selling untuk mendorong pelanggan membeli produk pelengkap 
        atau dari kategori berbeda. Menggunakan analisis data untuk mengidentifikasi kombinasi menu 
        terbaik dan menciptakan paket yang menarik. Program ini bertujuan meningkatkan jumlah item 
        dalam satu transaksi sekaligus memberikan pengalaman bersantap yang lebih lengkap.''',
        'implementation': [
            'Menu "perfect pairs": menampilkan rekomendasi pasangan untuk setiap hidangan utama',
            'Diskon combo: "Tambah garlic bread Rp 15k (harga normal Rp 25k)"',
            'Paket minuman: pesan 2 minuman dengan harga spesial',
            'Dessert sharing: rekomendasi dessert untuk dibagi pada meja dengan 2+ orang',
            'Sampler appetizer: "Coba 3 appetizer Rp 60k (hemat 30%)"',
            'Promosi side dish: side dish tanpa batas dengan paket upgrade',
            'Paket lengkap: hidangan utama + minuman + dessert dengan harga diskon',
            'Rekomendasi digital: aplikasi otomatis menyarankan item pelengkap',
            'Visual merchandising: menampilkan foto kombinasi menu yang menggugah selera'
        ]
    },

    "D3: Spend More Get More": {
        'features': {
            'AverageSpend': 0.25,
            'HighSpender': 0.20,
            'SpendPerPerson': 0.15,
            'Spend_x_Rating': 0.12,
            'GroupSize': 0.10,
            'Income': 0.08,
            'VisitFrequency': 0.05,
            'LoyaltyProgramMember': 0.03,
            'ServiceRating': 0.02,
            'HighIncome': 0.00
        },
        'description': '''Program insentif bertingkat yang memberikan reward kepada pelanggan berdasarkan 
        ambang batas pengeluaran. Semakin besar nilai transaksi, semakin tinggi reward yang diperoleh. 
        Program ini mendorong pelanggan untuk mencapai level pengeluaran berikutnya dengan informasi 
        progres dan manfaat yang jelas, sehingga efektif meningkatkan nilai transaksi rata-rata dan 
        kepuasan pelanggan.''',
        'implementation': [
            'Reward berdasarkan threshold: Rp 150k (minuman gratis), Rp 300k (appetizer gratis), Rp 500k (dessert gratis)',
            'Pelacakan real-time: struk menampilkan total belanja dan threshold reward berikutnya',
            'Notifikasi "unlock reward": "Tambah Rp 25k untuk mendapatkan dessert gratis"',
            'Diskon bertingkat: Rp 200k (5%), Rp 400k (10%), Rp 800k (15%)',
            'Mystery reward: belanja Rp 1 juta+ mendapatkan hadiah premium kejutan',
            'Penggabungan grup: total belanja satu meja digabung untuk mencapai threshold',
            'Double points day: hari tertentu threshold reward dikurangi 50%',
            'Tier premium: belanja Rp 2 juta+ mendapatkan voucher eksklusif kunjungan berikutnya',
            'Milestone kuartalan: akumulasi belanja per kuartal untuk reward lebih besar'
        ]
    },

    "D4: Minimum Purchase Rewards": {
        'features': {
            'AverageSpend': 0.25,
            'SpendPerPerson': 0.18,
            'HighSpender': 0.15,
            'GroupSize': 0.12,
            'Spend_x_Rating': 0.10,
            'Income': 0.08,
            'VisitFrequency': 0.06,
            'FoodRating': 0.04,
            'ServiceRating': 0.02,
            'DiningOccasion': 0.00
        },
        'description': '''Program reward yang diaktifkan ketika pelanggan mencapai jumlah pembelian minimum 
        tertentu dalam satu transaksi. Manfaat diberikan secara langsung untuk menciptakan kepuasan 
        instan. Program ini sederhana, jelas, dan efektif dalam mendorong pelanggan menambahkan item 
        agar memenuhi syarat mendapatkan reward.''',
        'implementation': [
            '"Belanja Rp 200k, Gratis Appetizer" (pilihan dari 5 menu appetizer)',
            '"Belanja Rp 350k, Gratis Dessert Premium" (nilai hingga Rp 45k)',
            'Upgrade otomatis: belanja Rp 150k+ langsung upgrade ukuran minuman',
            'Side dish gratis: belanja Rp 250k+ mendapatkan refill roti tanpa batas',
            'Promo akhir pekan: threshold lebih rendah di Sabtu–Minggu (Rp 180k)',
            'Bonus takeaway: belanja Rp 300k+ gratis ongkir',
            'Pilihan reward: pelanggan dapat memilih reward dari beberapa opsi menu',
            'Dapat digabung promo lain: kompatibel dengan diskon member',
            'Indikator progres: menu menampilkan "Tambah Rp 50k lagi untuk dessert gratis"'
        ]
    },

    "D5: Personalized Upsell Engine": {
        'features': {
            'AverageSpend': 0.20,
            'SpendPerPerson': 0.18,
            'OnlineUser': 0.15,
            'Spend_x_Rating': 0.12,
            'VisitFrequency': 0.10,
            'FoodRating': 0.10,
            'HighSpender': 0.08,
            'LoyaltyProgramMember': 0.05,
            'ServiceRating': 0.02,
            'Income': 0.00
        },
        'description': '''Mesin rekomendasi berbasis kecerdasan buatan yang menganalisis riwayat transaksi, 
        preferensi, dan pola pengeluaran pelanggan untuk memberikan saran upselling yang dipersonalisasi 
        melalui aplikasi mobile. Setiap pelanggan menerima rekomendasi yang sesuai dengan selera dan 
        anggaran mereka, sehingga meningkatkan tingkat konversi dan kepuasan pelanggan tanpa bersifat 
        mengganggu.''',
        'implementation': [
            'Algoritma machine learning: menganalisis pesanan, rating, dan pengeluaran sebelumnya',
            'Rekomendasi cerdas: "Berdasarkan riwayat Anda, Anda mungkin menyukai..."',
            'Penyesuaian anggaran: rekomendasi sesuai dengan pola pengeluaran pelanggan',
            'Optimasi waktu: rekomendasi appetizer sebelum pesanan, dessert setelah hidangan utama',
            'Penawaran kontekstual: rekomendasi berbeda untuk makan siang/malam dan weekday/weekend',
            'A/B testing: peningkatan akurasi rekomendasi secara berkelanjutan',
            'Tambah sekali klik: item rekomendasi dapat ditambahkan dengan satu sentuhan',
            'Pelengkap menu: rekomendasi item yang belum dipesan (minuman/dessert)',
            'Social proof: "80% pelanggan yang memesan ini juga menambahkan..."'
        ]
    },


    # ========== E. STRATEGI BERBASIS PENGALAMAN LAYANAN ==========
    
    "E1: Digital Queue Management": {
        'features': {
            'WaitTime': 0.30,
            'LongWait': 0.25,
            'WaitToService': 0.15,
            'ServiceRating': 0.12,
            'OnlineReservation': 0.08,
            'OnlineUser': 0.05,
            'VisitFrequency': 0.03,
            'TimeOfVisit': 0.02,
            'Wait_x_Service': 0.00,
            'GroupSize': 0.00
        },
        'description': '''Sistem antrean digital modern yang dirancang untuk meminimalkan frustrasi 
        akibat waktu tunggu dan mengoptimalkan pengelolaan meja. Pelanggan dapat bergabung dalam antrean 
        melalui aplikasi, memperoleh pembaruan posisi antrean secara real-time, serta estimasi waktu 
        tunggu yang akurat. Sistem ini menghilangkan kerumunan di area masuk dan memberikan fleksibilitas 
        bagi pelanggan untuk menunggu dengan nyaman, sehingga secara signifikan meningkatkan persepsi 
        kualitas layanan.''',
        'implementation': [
            'Aplikasi antrean mobile: pelanggan dapat bergabung antrean dari rumah melalui smartphone',
            'Check-in QR code: pemindaian QR di pintu masuk untuk masuk antrean',
            'Pembaruan real-time: notifikasi push untuk progres antrean',
            'Estimasi waktu tunggu akurat: prediksi berbasis AI (akurasi ±5 menit)',
            'Antrean virtual: pelanggan tidak perlu berdiri di pintu masuk dan dapat menunggu di area sekitar',
            'Notifikasi SMS/WhatsApp: "Meja Anda akan siap dalam 5 menit"',
            'Jalur prioritas: fast-track untuk reservasi dan member loyalitas',
            'Dashboard antrean: layar digital menampilkan nomor antrean dan status',
            'Analitik: pelacakan jam sibuk dan rata-rata waktu tunggu untuk optimasi'
        ]
    },

    "E2: Fast Lane Reservation": {
        'features': {
            'OnlineReservation': 0.25,
            'WaitTime': 0.20,
            'LongWait': 0.15,
            'ServiceRating': 0.12,
            'WaitToService': 0.10,
            'OnlineUser': 0.08,
            'LoyaltyProgramMember': 0.05,
            'VisitFrequency': 0.03,
            'TimeOfVisit': 0.02,
            'Wait_x_Service': 0.00
        },
        'description': '''Sistem prioritas bagi pelanggan yang melakukan reservasi online untuk menghilangkan 
        atau secara signifikan mengurangi waktu tunggu. Pelanggan dengan reservasi mendapatkan jaminan 
        tempat duduk dan dapat melewati antrean reguler. Program ini mendorong penggunaan reservasi 
        online, meningkatkan pengalaman pelanggan, serta membantu restoran memprediksi permintaan dan 
        mengoptimalkan penjadwalan staf.''',
        'implementation': [
            'Jaminan tempat duduk: pelanggan reservasi duduk maksimal 5 menit setelah kedatangan',
            'Pemesanan mudah: reservasi melalui aplikasi, website, atau WhatsApp dengan konfirmasi instan',
            'Waktu toleransi: grace period 15 menit untuk keterlambatan',
            'Pengingat reservasi: notifikasi H-3, H-1, dan 2 jam sebelum kedatangan',
            'Preferensi tempat duduk: opsi permintaan meja atau area tertentu',
            'Opsi pre-order: pemesanan menu sebelum datang untuk mempercepat layanan',
            'Kebijakan pembatalan: pembatalan gratis hingga 2 jam sebelum jadwal',
            'Keuntungan reservasi: minuman sambutan gratis bagi pelanggan reservasi',
            'Prioritas jam sibuk: sangat bermanfaat pada jam makan malam Jumat–Minggu'
        ]
    },

    "E3: Real-time Wait Info": {
        'features': {
            'WaitTime': 0.28,
            'LongWait': 0.22,
            'OnlineUser': 0.15,
            'WaitToService': 0.12,
            'ServiceRating': 0.10,
            'OnlineReservation': 0.06,
            'TimeOfVisit': 0.04,
            'VisitFrequency': 0.02,
            'Wait_x_Service': 0.01,
            'Age': 0.00
        },
        'description': '''Penyediaan informasi waktu tunggu secara real-time yang ditampilkan melalui 
        berbagai kanal seperti aplikasi, website, Google Maps, dan layar di dalam restoran. Pelanggan 
        dapat membuat keputusan yang lebih tepat mengenai waktu kunjungan atau apakah akan bergabung 
        dalam antrean. Transparansi ini mengurangi frustrasi dan meningkatkan kepercayaan pelanggan, 
        serta terintegrasi dengan manajemen kapasitas untuk prediksi yang lebih akurat.''',
        'implementation': [
            'Tampilan live: waktu tunggu terkini di homepage website dan aplikasi mobile',
            'Integrasi Google Maps: informasi waktu tunggu muncul di profil bisnis Google',
            'Layar di restoran: papan digital menampilkan waktu tunggu dan panjang antrean',
            'Data historis: menampilkan estimasi waktu tunggu berdasarkan hari dan jam',
            'Kode warna: hijau (0–15 menit), kuning (15–30 menit), merah (30 menit+)',
            'Notifikasi push: pemberitahuan saat waktu tunggu berkurang signifikan',
            'Rekomendasi waktu terbaik: saran jam kunjungan dengan antrean minimal',
            'Indikator kapasitas: menampilkan persentase okupansi saat ini',
            'Integrasi API: aplikasi pihak ketiga dapat mengakses data waktu tunggu'
        ]
    },

    "E4: Dynamic Staffing": {
        'features': {
            'WaitTime': 0.25,
            'ServiceRating': 0.20,
            'LongWait': 0.15,
            'WaitToService': 0.12,
            'TimeOfVisit': 0.10,
            'Wait_x_Service': 0.08,
            'VisitFrequency': 0.05,
            'FoodRating': 0.03,
            'AvgRating': 0.02,
            'GroupSize': 0.00
        },
        'description': '''Optimasi penjadwalan staf berdasarkan prediksi permintaan untuk memastikan 
        kecukupan layanan pada jam sibuk dan menghindari kelebihan staf pada periode sepi. Sistem ini 
        memanfaatkan data historis, cuaca, acara khusus, dan faktor lainnya untuk memprediksi tingkat 
        kunjungan dan menyesuaikan jadwal kerja. Hasilnya adalah peningkatan kualitas layanan, penurunan 
        waktu tunggu, serta efisiensi biaya tenaga kerja.''',
        'implementation': [
            'Penjadwalan prediktif: prediksi permintaan berbasis AI hingga 2 minggu ke depan',
            'Shift fleksibel: staf dapat memperpanjang atau mempersingkat shift sesuai permintaan real-time',
            'Sistem on-call: staf cadangan siap dipanggil saat lonjakan tak terduga',
            'Cross-training: staf dilatih untuk menangani beberapa peran',
            'Penguatan jam sibuk: penambahan staf pada jam makan siang dan makan malam akhir pekan',
            'Penyesuaian real-time: manajer dapat memanggil staf tambahan saat antrean meningkat',
            'Alokasi staf: penempatan staf lebih banyak di area dengan lalu lintas tinggi',
            'Pemantauan kinerja: evaluasi efisiensi staf dan kepuasan pelanggan',
            'Program insentif: bonus bagi staf yang bekerja pada jam sibuk dan menantang'
        ]
    },

    "E5: Pre-order System": {
        'features': {
            'WaitTime': 0.25,
            'OnlineUser': 0.20,
            'OnlineReservation': 0.15,
            'WaitToService': 0.12,
            'ServiceRating': 0.10,
            'LongWait': 0.08,
            'TimeOfVisit': 0.05,
            'VisitFrequency': 0.03,
            'AverageSpend': 0.02,
            'Wait_x_Service': 0.00
        },
        'description': '''Sistem pre-order yang memungkinkan pelanggan memesan menu sebelum tiba, baik 
        untuk makan di tempat maupun takeaway. Proses persiapan makanan dimulai sebelum pelanggan 
        datang, sehingga secara signifikan mengurangi waktu tunggu setelah duduk. Sistem ini sangat 
        efektif untuk jam makan siang dan pelanggan bisnis yang memiliki keterbatasan waktu, serta 
        meningkatkan perputaran meja dan kepuasan pelanggan.''',
        'implementation': [
            'Pre-order berbasis aplikasi: pemesanan minimal 15 menit sebelum kedatangan',
            'Persiapan terjadwal: dapur dapat merencanakan dan mengoptimalkan urutan memasak',
            'Notifikasi kedatangan: pelanggan memberi tanda saat dalam perjalanan',
            'Jaminan waktu siap: makanan siap maksimal 5 menit setelah kedatangan terkonfirmasi',
            'Loket ekspres: konter khusus untuk pengambilan pre-order',
            'Opsi modifikasi: perubahan pesanan hingga 10 menit sebelum waktu terjadwal',
            'Integrasi pembayaran: pembayaran di muka melalui aplikasi',
            'Takeaway ekspres: pesanan siap tepat sesuai jadwal',
            'Pre-order grup: beberapa pelanggan dapat menambahkan item dalam satu pesanan grup'
        ]
    },

    "E6: Self-order Kiosk": {
        'features': {
            'WaitTime': 0.22,
            'ServiceRating': 0.18,
            'WaitToService': 0.15,
            'OnlineUser': 0.12,
            'LongWait': 0.10,
            'TimeOfVisit': 0.10,
            'AverageSpend': 0.06,
            'VisitFrequency': 0.04,
            'Age': 0.03,
            'YoungCustomer': 0.00
        },
        'description': '''Kiosk layanan mandiri untuk pemesanan dan pembayaran yang mengurangi ketergantungan 
        pada staf serta meminimalkan waktu tunggu pemesanan. Pelanggan dapat menelusuri menu secara visual, 
        menyesuaikan pesanan, dan melakukan pembayaran langsung. Sistem ini sangat menarik bagi pelanggan 
        yang melek teknologi dan efektif digunakan pada jam sibuk, sekaligus meningkatkan akurasi pesanan 
        dan peluang upselling melalui prompt strategis.''',
        'implementation': [
            'Kiosk layar sentuh: antarmuka ramah pengguna dengan gambar produk berukuran besar',
            'Metode pembayaran beragam: tunai, kartu, e-wallet, dan QRIS',
            'Kustomisasi pesanan: tambah/kurangi bahan dan atur tingkat kepedasan',
            'Menu visual: foto berkualitas tinggi dengan deskripsi menu',
            'Prompt upselling: "Apakah ingin menambahkan kentang goreng seharga Rp 15k?"',
            'Sistem nomor pesanan: pelanggan menerima nomor untuk pengambilan makanan',
            'Multi-bahasa: pilihan Bahasa Indonesia dan Inggris',
            'Mode cepat: pemesanan singkat untuk menu populer (tombol "Pesanan Biasa")',
            'Struk digital: opsi pengiriman email atau cetak struk',
            'Pemesanan paralel: beberapa kiosk untuk pemesanan simultan'
        ]
    },


    # ========== F. STRATEGI BERBASIS RATING & KUALITAS ==========
    
    "F7: Staff Performance Incentive Program": {
        'features': {
            'ServiceRating': 0.22,
            'AvgRating': 0.18,
            'ConsistentQuality': 0.15,
            'Rating_x_Loyalty': 0.12,
            'TotalRating': 0.10,
            'RatingStd': 0.08,
            'FoodRating': 0.06,
            'AmbianceRating': 0.05,
            'WaitTime': 0.04,
            'RatingRange': 0.00
        },
        'description': '''Program insentif berbasis kinerja untuk memotivasi staff dalam memberikan 
        service berkualitas tinggi secara konsisten. Reward dikaitkan langsung dengan customer ratings, 
        feedback positif, dan service KPIs. Program ini align staff motivation dengan business goals 
        dan mendorong ownership terhadap customer experience.''',
        'implementation': [
            'Monthly bonus: berdasarkan average service rating per staff',
            'Team-based incentive: bonus tim jika overall rating mencapai target',
            'Instant reward: voucher/points untuk feedback positif yang mention nama staff',
            'Leaderboard: display top-performing staff setiap bulan',
            'Non-monetary rewards: extra day off, preferred shifts',
            'Clear KPIs: transparansi metrics yang dinilai',
            'Fair weighting: adjust rating by shift difficulty (peak vs off-peak)',
            'Recognition ceremony: monthly appreciation meeting',
            'Performance review: feedback session berbasis data',
            'Continuous improvement plan: coaching untuk low performers'
        ]
    },

    "F8: Experience Personalization": {
        'features': {
            'AvgRating': 0.22,
            'Rating_x_Loyalty': 0.20,
            'TotalRating': 0.15,
            'ServiceRating': 0.13,
            'FoodRating': 0.10,
            'ConsistentQuality': 0.08,
            'VisitFrequency': 0.07,
            'AmbianceRating': 0.03,
            'Spend_x_Rating': 0.02,
            'RatingStd': 0.00
        },
        'description': '''Personalisasi experience berdasarkan customer history dan preferences. 
        Returning customers receive tailored service seperti preferred seating, favorite menu 
        recommendations, dan customized greetings. Personal touch meningkatkan emotional connection 
        dan customer loyalty.''',
        'implementation': [
            'Customer profile: store preferences (seat, spice level, allergies)',
            'Personal greeting: staff greet repeat customers by name',
            'Menu recommendation: suggest favorite atau frequently ordered items',
            'Special occasion tagging: birthday, anniversary recognition',
            'Custom pacing: adjust service speed based on customer type',
            'Loyalty integration: personalize perks berdasarkan tier',
            'CRM dashboard: quick access untuk staff',
            'Privacy compliance: opt-in personalization only',
            'Staff briefing: daily briefing tentang VIP/returning guests',
            'Feedback loop: refine personalization from customer responses'
        ]
    },

    "F9: End-to-End Experience Audit": {
        'features': {
            'AvgRating': 0.20,
            'ConsistentQuality': 0.18,
            'RatingStd': 0.15,
            'RatingRange': 0.12,
            'TotalRating': 0.10,
            'ServiceRating': 0.10,
            'FoodRating': 0.08,
            'AmbianceRating': 0.05,
            'WaitTime': 0.02,
            'MinRating': 0.00
        },
        'description': '''Audit menyeluruh terhadap entire customer journey dari arrival hingga payment. 
        Mengidentifikasi pain points, bottlenecks, dan inconsistencies di setiap touchpoint. 
        Pendekatan sistematis ini memastikan experience quality terjaga secara holistik, bukan parsial.''',
        'implementation': [
            'Journey mapping: breakdown every step of customer experience',
            'Touchpoint scoring: rate each interaction independently',
            'Cross-functional audit: FOH, BOH, cashier, cleanliness',
            'Weekly mini-audit: quick checks untuk critical touchpoints',
            'Monthly deep audit: comprehensive evaluation',
            'Issue prioritization: fix high-impact pain points first',
            'Owner assignment: setiap issue punya PIC jelas',
            'Timeline tracking: SLA untuk issue resolution',
            'Before-after comparison: measure improvement impact',
            'Audit report: management dashboard dengan trend analysis'
        ]
    },

    "F10: Continuous Improvement Loop": {
        'features': {
            'ConsistentQuality': 0.25,
            'AvgRating': 0.18,
            'TotalRating': 0.15,
            'RatingStd': 0.12,
            'Rating_x_Loyalty': 0.10,
            'ServiceRating': 0.08,
            'FoodRating': 0.07,
            'AmbianceRating': 0.03,
            'VisitFrequency': 0.02,
            'RatingRange': 0.00
        },
        'description': '''Sistem continuous improvement berbasis data yang mengintegrasikan feedback, 
        ratings, operational metrics, dan staff insights ke dalam improvement cycle yang berkelanjutan. 
        Fokus pada incremental improvements yang konsisten daripada perubahan besar yang sporadis.''',
        'implementation': [
            'Weekly review meeting: discuss ratings, complaints, anomalies',
            'Root cause analysis: 5-Why untuk recurring issues',
            'Improvement backlog: prioritized list of improvement actions',
            'Small experiments: test improvements di limited scope',
            'Data-driven decisions: validate changes with rating trends',
            'Staff suggestion system: encourage bottom-up improvements',
            'Documentation: log every change dan outcome',
            'Standard update: update SOP setelah improvement validated',
            'Knowledge sharing: best practices across shifts/teams',
            'Quarterly review: strategic evaluation of improvement impact'
        ]
    },
    
    # ========== G. STRATEGI BERBASIS RESERVASI & ONLINE BEHAVIOR ==========
    
    "G1: App-exclusive Deals": {
        'features': {
            'OnlineUser': 0.25,
            'OnlineReservation': 0.20,
            'LoyaltyProgramMember': 0.15,
            'VisitFrequency': 0.12,
            'AverageSpend': 0.10,
            'DeliveryOrder': 0.08,
            'ServiceRating': 0.05,
            'TimeOfVisit': 0.03,
            'Age': 0.02,
            'YoungCustomer': 0.00
        },
        'description': '''Exclusive promotions, discounts, dan benefits yang hanya available untuk 
        customers yang download dan use mobile app. Ini incentivize app adoption yang memberikan 
        restoran valuable data, direct communication channel, dan reduced dependency on third-party 
        platforms. App users typically have higher lifetime value dan engagement. Offers designed 
        untuk be compelling enough untuk drive app downloads.''',
        'implementation': [
            'Welcome offer: 25% off untuk first order via app',
            'Weekly app-only deals: different special setiap minggu',
            'Flash sales: limited-time offers pushed via app notification',
            'Early access: app users get first dibs pada seasonal menu',
            'Extra loyalty points: earn 2x points untuk app orders',
            'Free delivery: waived delivery fee untuk app orders',
            'Birthday special: exclusive birthday discount via app',
            'Gamification: spin the wheel, scratch card untuk random rewards',
            'Referral rewards: share app dengan teman dapat bonus',
            'Push notifications: personalized offers based pada user behavior'
        ]
    },
    
    "G2: Delivery Loyalty Program": {
        'features': {
            'DeliveryOrder': 0.30,
            'OnlineUser': 0.20,
            'VisitFrequency': 0.15,
            'LoyaltyProgramMember': 0.12,
            'FrequentVisitor': 0.10,
            'AverageSpend': 0.06,
            'OnlineReservation': 0.04,
            'ServiceRating': 0.02,
            'Rating_x_Frequency': 0.01,
            'Spend_x_Rating': 0.00
        },
        'description': '''Dedicated loyalty program specifically untuk delivery customers dengan benefits 
        seperti free delivery, priority fulfillment, dan exclusive delivery-only menu items. Delivery 
        customers have different behaviors dan needs dari dine-in customers, sehingga deserve 
        specialized program. Encourage repeat delivery orders dan increase basket size.''',
        'implementation': [
            'Delivery tiers: Bronze (free delivery 1x/week), Silver (3x/week), Gold (unlimited)',
            'Tier qualification: Bronze (4 orders/month), Silver (8), Gold (12)',
            'Priority delivery: Gold members get faster delivery slots',
            'Exclusive items: special delivery-only menu untuk members',
            'Packaging upgrade: premium packaging untuk loyalty members',
            'Order tracking: real-time tracking dengan ETA updates',
            'Delivery insurance: guarantee fresh arrival atau replacement',
            'Contactless delivery: prioritized untuk member safety',
            'Scheduled delivery: book delivery up to 3 days advance',
            'Group order: combine orders dari multiple addresses untuk free delivery'
        ]
    },
    
    "G3: Pre-reserve dengan Deposit": {
        'features': {
            'OnlineReservation': 0.28,
            'OnlineUser': 0.20,
            'AverageSpend': 0.15,
            'Income': 0.12,
            'DiningOccasion': 0.10,
            'TimeOfVisit': 0.07,
            'GroupSize': 0.05,
            'LargeGroup': 0.02,
            'ServiceRating': 0.01,
            'LoyaltyProgramMember': 0.00
        },
        'description': '''System untuk reserve tables dengan deposit yang convertible ke discount atau 
        menu credit. Ini reduce no-shows yang costly untuk restoran, especially untuk peak times 
        dan large groups. Deposit amount reasonable dan fully redeemable, jadi tidak punitive. 
        Customer benefit dari guaranteed seating dan possible extra perks. Win-win solution.''',
        'implementation': [
            'Deposit amount: Rp 50k/pax untuk regular hours, Rp 100k untuk peak hours',
            'Full redeemable: deposit converts to bill credit atau discount',
            'Bonus incentive: deposit + Rp 20k bonus credit (essentially 40% bonus)',
            'Easy payment: pay deposit via app, e-wallet, transfer',
            'Instant confirmation: immediate booking confirmation setelah deposit',
            'Flexible cancellation: full refund jika cancel 24h before',
            'Guaranteed seating: table reserved regardless of walk-in crowd',
            'Preferred seating: option untuk request specific table area',
            'Pre-order option: order menu in advance dengan deposit',
            'Group friendly: simplified deposit collection untuk large groups'
        ]
    },
    
    "G4: AI Menu Recommendation": {
        'features': {
            'OnlineUser': 0.25,
            'FoodRating': 0.20,
            'Rating_x_Frequency': 0.15,
            'VisitFrequency': 0.12,
            'AverageSpend': 0.10,
            'Spend_x_Rating': 0.08,
            'PreferredCuisine': 0.05,
            'OnlineReservation': 0.03,
            'DeliveryOrder': 0.02,
            'LoyaltyProgramMember': 0.00
        },
        'description': '''Machine learning powered recommendation engine yang analyze customer order 
        history, ratings, preferences, dan behavioral patterns untuk suggest menu items they'll 
        likely love. Recommendations become more accurate over time dengan more data. Helps customers 
        discover new items they might not try otherwise. Increase satisfaction dan cross-selling 
        opportunities.''',
        'implementation': [
            'Personalized homepage: "Recommended for you" section di app',
            'Smart search: prioritize items customer likely to enjoy',
            '"Try something new": suggest items outside usual preference untuk discovery',
            'Collaborative filtering: "Customers like you also enjoyed..."',
            'Dietary preferences: remember dan filter by dietary restrictions',
            'Occasion-based: different suggestions untuk lunch vs dinner, weekday vs weekend',
            'Weather-aware: suggest comfort food in rainy days, refreshing items when hot',
            'Trending items: highlight popular items dengan similar customers',
            'Rating prediction: show predicted rating customer will give to item',
            'One-click reorder: easy to reorder past favorites'
        ]
    },
    
    "G5: Gamification Rewards": {
        'features': {
            'OnlineUser': 0.25,
            'VisitFrequency': 0.20,
            'LoyaltyProgramMember': 0.18,
            'FrequentVisitor': 0.12,
            'OnlineReservation': 0.10,
            'Rating_x_Frequency': 0.08,
            'DeliveryOrder': 0.04,
            'ServiceRating': 0.02,
            'Age': 0.01,
            'YoungCustomer': 0.00
        },
        'description': '''Gamified loyalty program dengan elements seperti badges, achievements, 
        leaderboards, challenges, dan quests. Make earning rewards fun dan engaging beyond 
        transactional relationship. Particularly effective dengan millennials dan Gen Z. Create 
        emotional connection dan sense of accomplishment. Drive specific behaviors through 
        targeted challenges.''',
        'implementation': [
            'Achievement badges: "Early Bird" (5 breakfast orders), "Night Owl" (5 dinner after 20:00)',
            'Daily quests: complete challenge untuk bonus points (e.g. "Try new item today")',
            'Streak rewards: consecutive days/weeks visiting untuk multiplier bonus',
            'Leaderboard: monthly top spenders/visitors dengan exclusive prizes',
            'Limited edition badges: seasonal atau event-specific achievements',
            'Point multiplier: special days dengan 2x, 3x, 5x points',
            'Surprise rewards: random spin-the-wheel opportunities',
            'Social sharing: share achievements di social media untuk extra points',
            'Progress bars: visual progress toward next reward unlock',
            'Tier progression: level up system dengan increasing benefits'
        ]
    },
    
    "G6: Predictive Menu Alerts": {
        'features': {
            'OnlineUser': 0.25,
            'VisitFrequency': 0.20,
            'Rating_x_Frequency': 0.15,
            'PreferredCuisine': 0.12,
            'OnlineReservation': 0.10,
            'LoyaltyProgramMember': 0.08,
            'DeliveryOrder': 0.05,
            'FoodRating': 0.03,
            'TimeOfVisit': 0.02,
            'AverageSpend': 0.00
        },
        'description': '''Proactive notification system yang remind customers tentang favorite menu 
        items, alert ketika favorite menu available lagi (untuk seasonal items), dan suggest 
        optimal time untuk visit based on their patterns. Personalized dan timely communication 
        yang adds value tanpa being spammy. Uses AI untuk predict when customer likely mau order.''',
        'implementation': [
            'Favorite alert: "Your favorite Truffle Pasta is back!" untuk seasonal returns',
            'Craving predictor: "It\'s been 2 weeks since your last burger, craving one?"',
            'Time-based reminder: "Usually you visit on Friday lunch, special offer today!"',
            'New item match: "New item that matches your taste profile available"',
            'Weather-triggered: "Perfect rainy day untuk hot soup and comfort food"',
            'Optimal visit time: "Usually less crowded at 14:00-16:00 today"',
            'Promo match: "25% off your favorite category this week"',
            'Re-order suggestion: "Quick reorder your usual?" dengan one-tap order',
            'Limited availability: "Only 5 servings left today of..."',
            'Smart frequency: adjust notification frequency based pada engagement'
        ]
    },

    # ========== H. STRATEGI BERBASIS GROUP BEHAVIOR ==========
    
    "H1: Group Dining Packages": {
        'features': {
            'GroupSize': 0.30,
            'LargeGroup': 0.25,
            'AverageSpend': 0.15,
            'SpendPerPerson': 0.10,
            'DiningOccasion': 0.08,
            'Income': 0.05,
            'TimeOfVisit': 0.04,
            'ServiceRating': 0.02,
            'MealType': 0.01,
            'VisitFrequency': 0.00
        },
        'description': '''Specialized packages untuk groups of 4 or more dengan bundled pricing, 
        sharing menu options, dan dedicated service. Group dining profitable due to higher total 
        spend dan efficiency. Packages designed untuk different group types: family gatherings, 
        office celebrations, friend reunions. Include benefits seperti private/semi-private seating, 
        customizable menu, dan streamlined ordering.''',
        'implementation': [
            '"Family Feast" (4-6 pax): Rp 300k complete meal dengan variety',
            '"Party Pack" (8-10 pax): Rp 600k buffet-style dengan unlimited drinks',
            '"Corporate Package" (10-20 pax): Rp 1.2jt set menu + meeting room',
            'Sharing platters: large portions designed untuk sharing (ribs, pizza, pasta)',
            'Set menu: pre-selected courses untuk easy ordering dan service',
            'Customizable: allow substitutions within package untuk dietary needs',
            'Group discount: 15-20% off untuk groups 8+, 25% untuk 15+',
            'Private seating: sectioned area atau private room untuk larger groups',
            'Dedicated server: single server untuk entire group untuk consistency',
            'Easy billing: single bill atau flexible split billing options'
        ]
    },
    
    "H2: Solo Diner Comfort": {
        'features': {
            'Solo': 0.30,
            'GroupSize': 0.20,
            'ServiceRating': 0.15,
            'AverageSpend': 0.10,
            'AmbianceRating': 0.10,
            'TimeOfVisit': 0.07,
            'SpendPerPerson': 0.05,
            'VisitFrequency': 0.02,
            'Age': 0.01,
            'Income': 0.00
        },
        'description': '''Program untuk make solo diners feel comfortable dan welcome rather than awkward. 
        Many people hesitant untuk dine alone karena perceived stigma. Creating solo-friendly 
        environment opens new customer segment. Includes appropriate seating, reading materials, 
        attentive but not intrusive service, dan special solo menu portions. Solo diners often 
        become regulars jika treated well.''',
        'implementation': [
            'Solo-friendly seating: bar seats, window counters, small 2-tops',
            'Reading materials: newspapers, magazines, books available',
            'Free Wi-Fi: quality internet untuk working diners',
            'Solo portions: smaller portions at reduced price (60-70% of regular)',
            'Quick service: recognize solo diners may prefer faster service',
            'No judgment: staff trained untuk make solo diners feel welcome',
            '"Solo Special": special menu atau discount untuk solo diners',
            'Privacy: seating options that provide some privacy/comfort',
            'Charging ports: power outlets at solo-friendly seats',
            'Background music: ambient music so silence tidak awkward'
        ]
    },
    
    "H3: Large Group Reservations": {
        'features': {
            'LargeGroup': 0.30,
            'GroupSize': 0.25,
            'OnlineReservation': 0.15,
            'DiningOccasion': 0.12,
            'AverageSpend': 0.08,
            'TimeOfVisit': 0.05,
            'ServiceRating': 0.03,
            'Income': 0.02,
            'MealType': 0.00,
            'SpendPerPerson': 0.00
        },
        'description': '''Dedicated booking system dan special accommodations untuk groups of 10+. 
        Large groups require different planning, preparation, dan service approach. System allows 
        advance booking dengan menu pre-selection, table configuration planning, dan special 
        requests. High-value opportunity karena large groups generate significant revenue in 
        single seating, but need proper handling.''',
        'implementation': [
            'Advance booking: accept reservations untuk large groups up to 1 month ahead',
            'Deposit requirement: Rp 50k/pax deposit untuk guarantee booking',
            'Menu pre-selection: choose menu 48h before untuk kitchen preparation',
            'Table configuration: custom table arrangement untuk group size/preference',
            'Private space: dedicated room atau sectioned area untuk privacy',
            'Dedicated staff: assign specific servers untuk group',
            'Coordinated service: synchronized food service untuk large groups',
            'Event coordination: assist dengan decorations, cake, presentations',
            'Flexible payment: handle split bills atau group billing easily',
            'Special occasions: accommodate birthdays, celebrations dengan special touches'
        ]
    },
    
    "H4: Private Room Service": {
        'features': {
            'LargeGroup': 0.25,
            'GroupSize': 0.20,
            'Income': 0.15,
            'HighIncome': 0.12,
            'DiningOccasion': 0.10,
            'AverageSpend': 0.08,
            'ServiceRating': 0.05,
            'AmbianceRating': 0.03,
            'TimeOfVisit': 0.02,
            'MealType': 0.00
        },
        'description': '''Premium private dining rooms untuk groups yang desire privacy dan exclusivity. 
        Perfect untuk business meetings, intimate celebrations, atau VIP gatherings. Rooms equipped 
        dengan AV equipment, comfortable seating, dedicated service, dan customizable ambiance. 
        Command premium pricing due to exclusivity dan enhanced service. Requires minimum spend 
        atau room charge.''',
        'implementation': [
            'Multiple room sizes: Small (4-6 pax), Medium (8-12 pax), Large (15-20 pax)',
            'Room booking: advance reservation dengan minimum spend requirement',
            'Minimum spend: Rp 1.5jt (small), Rp 3jt (medium), Rp 5jt (large)',
            'Equipment: TV/projector, sound system, microphone untuk presentations',
            'Customizable: adjust lighting, temperature, music untuk preference',
            'Butler service: dedicated server exclusively untuk the room',
            'Custom menu: work dengan chef untuk special menu requests',
            'Privacy: soundproof, separate entrance untuk discretion',
            'Amenities: whiteboard, WiFi, charging stations, coat rack',
            'Extended hours: flexible dengan timing, can book beyond regular hours'
        ]
    },

    # ========== I. STRATEGI BERBASIS OCCASION & MEAL TYPE ==========
    
    "I1: Breakfast Champions": {
        'features': {
            'MealType': 0.25,
            'TimeOfVisit': 0.25,
            'AverageSpend': 0.15,
            'DiningOccasion': 0.12,
            'VisitFrequency': 0.10,
            'SpendPerPerson': 0.06,
            'GroupSize': 0.04,
            'ServiceRating': 0.02,
            'Age': 0.01,
            'Income': 0.00
        },
        'description': '''Specialized breakfast program untuk capture morning crowd dengan attractive 
        breakfast combos, quick service, dan morning-appropriate menu. Breakfast crowd typically 
        time-sensitive dan price-sensitive, so focus on value dan efficiency. Build morning routine 
        habit dengan consistent quality. Breakfast high-margin due to lower ingredient costs.''',
        'implementation': [
            'Breakfast combo: Rp 25k-45k complete breakfast (main + drink + bread)',
            'Early bird special: 20% off untuk order sebelum 08:00',
            'Quick service: guarantee 10-minute service untuk breakfast orders',
            'Grab & Go: pre-packed breakfast untuk busy commuters',
            'Coffee deals: unlimited refills untuk breakfast customers',
            'Healthy options: oatmeal, smoothie bowls, whole grain options',
            'Business breakfast: meeting-friendly seating dengan power outlets',
            'Breakfast subscription: monthly breakfast pass untuk regulars',
            'Weekend brunch: special expanded menu Sat-Sun 08:00-14:00',
            'Loyalty: breakfast stamp card - buy 10 get 1 free'
        ]
    },
    
    "I2: Romantic Dinner Experience": {
        'features': {
            'DiningOccasion': 0.30,
            'TimeOfVisit': 0.20,
            'AverageSpend': 0.15,
            'Income': 0.12,
            'AmbianceRating': 0.10,
            'GroupSize': 0.06,
            'ServiceRating': 0.04,
            'MealType': 0.02,
            'FoodRating': 0.01,
            'Solo': 0.00
        },
        'description': '''Premium romantic dining package untuk couples celebrating special occasions 
        atau seeking intimate dining experience. Focus on ambiance, privacy, dan special touches 
        that create memorable moments. Higher price point justified by enhanced experience. 
        Popular untuk anniversaries, proposals, Valentine, atau date nights. Requires reservation 
        dan allows customization.''',
        'implementation': [
            'Romantic package: Rp 500k-800k untuk couple dengan multi-course meal',
            'Private booth: intimate 2-person seating dengan curtain/divider',
            'Mood lighting: dimmed lighting, candles pada table',
            'Special setup: rose petals, personalized menu cards',
            'Wine pairing: sommelier recommendation untuk each course',
            'Live music: soft acoustic atau pianist pada certain nights',
            'Photo service: complimentary couple photo dengan Polaroid',
            'Surprise coordination: help plan proposals atau special surprises',
            'Dessert special: complimentary champagne dengan dessert',
            'Extended seating: no rush, enjoy evening at your pace'
        ]
    },
    
    "I3: Time-based Specials": {
        'features': {
            'TimeOfVisit': 0.30,
            'MealType': 0.20,
            'AverageSpend': 0.15,
            'DiningOccasion': 0.12,
            'VisitFrequency': 0.10,
            'SpendPerPerson': 0.06,
            'OnlineReservation': 0.04,
            'Age': 0.02,
            'GroupSize': 0.01,
            'ServiceRating': 0.00
        },
        'description': '''Different promotions dan menu highlights based on time of day untuk optimize 
        traffic throughout operational hours. Encourage visits during off-peak hours dengan attractive 
        offers. Match offerings dengan customer needs at different times - quick lunch, afternoon 
        tea, happy hour, late night snacks. Maximize utilization dan revenue per hour.''',
        'implementation': [
            'Lunch rush (11:00-14:00): express lunch sets Rp 35k dengan 15-min guarantee',
            'Afternoon delight (14:00-17:00): 30% off all desserts dan coffees',
            'Happy hour (17:00-19:00): buy 1 get 1 selected drinks dan appetizers',
            'Dinner prime (19:00-21:00): premium menu availability, no discount',
            'Late night (21:00-23:00): light bites menu, bar atmosphere, 20% off food',
            'Weekday lunch: business lunch set dengan quick service',
            'Weekend brunch: special expanded menu dengan bottomless drinks option',
            'Early bird dinner: 25% off untuk seated before 18:00',
            'Time-limited offers: flash deals pushed via app untuk fill slow hours',
            'Meal period packages: different set menus untuk breakfast/lunch/dinner'
        ]
    },
    
    "I4: Seasonal Event Menus": {
        'features': {
            'DiningOccasion': 0.25,
            'MealType': 0.18,
            'TimeOfVisit': 0.15,
            'FoodRating': 0.12,
            'AverageSpend': 0.10,
            'Income': 0.08,
            'VisitFrequency': 0.06,
            'GroupSize': 0.04,
            'ServiceRating': 0.02,
            'Age': 0.00
        },
        'description': '''Limited-time special menus untuk seasonal events dan holidays (Ramadan, Christmas, 
        Chinese New Year, Valentine, etc). Create excitement dan urgency dengan exclusive offerings 
        that celebrate the occasion. Strong cultural relevance drives traffic dan creates memorable 
        associations. Premium pricing acceptable untuk limited-time exclusive items.''',
        'implementation': [
            'Ramadan: Iftar buffet packages, Sahur specials, pre-order for family gatherings',
            'Christmas: festive set menus, turkey/ham options, Christmas decorations',
            'Chinese New Year: prosperity menu, lo hei, auspicious dishes',
            'Valentine: romantic dinners, couple packages, heart-themed presentations',
            'Halloween: themed menu items, spooky presentations, costume contest',
            'Independence Day: Indonesian traditional feast, national theme decorations',
            'Mother\'s/Father\'s Day: family packages dengan special honors',
            'New Year: countdown party packages, special celebration menu',
            'Limited availability: create urgency "Only available this month!"',
            'Pre-booking: accept advance reservations dengan deposit untuk peak days'
        ]
    },
    
    "I5: Happy Hour Extended": {
        'features': {
            'TimeOfVisit': 0.30,
            'AverageSpend': 0.18,
            'DiningOccasion': 0.15,
            'Age': 0.12,
            'YoungCustomer': 0.10,
            'GroupSize': 0.07,
            'MealType': 0.05,
            'VisitFrequency': 0.02,
            'OnlineUser': 0.01,
            'ServiceRating': 0.00
        },
        'description': '''Extended happy hour program particularly pada Fridays untuk capture after-work 
        crowd dan kick-start weekend vibe. Mix of food dan beverage promotions dengan lively atmosphere. 
        Target young professionals dan friends groups looking untuk unwind. Balance discounts dengan 
        volume untuk maintain profitability. Creates regular habit dan weekly traffic spike.''',
        'implementation': [
            'Friday extended: 15:00-21:00 happy hour (normal days 17:00-19:00)',
            'Buy 1 Get 1: all drinks dan selected appetizers',
            'Draft beer tower: special pricing untuk beer towers untuk sharing',
            'Snack platters: mixed appetizer platters at 40% off',
            'Live DJ: music dari 18:00 untuk party atmosphere',
            'Reserved sections: group-friendly seating arrangement',
            'Game zones: darts, board games untuk entertainment',
            'Social hours: networking-friendly environment untuk young professionals',
            'Punch cards: collect stamps during happy hour untuk rewards',
            'Extended seating: no time pressure, enjoy as long as wanted'
        ]
    },

    # ========== J. STRATEGI BERBASIS FITUR INTERAKSI ==========
    
    "J1: VIP Loyal Recognition": {
        'features': {
            'Rating_x_Loyalty': 0.30,
            'LoyalCustomer': 0.20,
            'LoyaltyProgramMember': 0.15,
            'Rating_x_Frequency': 0.12,
            'VisitFrequency': 0.10,
            'ServiceRating': 0.06,
            'AvgRating': 0.04,
            'AverageSpend': 0.02,
            'Spend_x_Rating': 0.01,
            'FrequentVisitor': 0.00
        },
        'description': '''Premium recognition program untuk most loyal customers dengan highest satisfaction 
        ratings. These are brand ambassadors yang deserve special treatment. Personalized service, 
        exclusive perks, dan public recognition. Hand-picked members based on loyalty score combining 
        frequency, ratings, spending, dan tenure. Ultra-VIP treatment untuk retain dan delight these 
        valuable customers.''',
        'implementation': [
            'VIP identification: top 50 customers identified quarterly based on loyalty metrics',
            'Personal greeting: staff know them by name, preferences memorized',
            'Surprise delights: random complimentary upgrades, desserts, drinks',
            'Birthday celebration: elaborate celebration dengan free premium meal',
            'Anniversary recognition: celebrate their loyalty anniversary',
            'Priority everything: reservations, seating, service, special requests',
            'Exclusive events: quarterly VIP-only events dengan chef, owner',
            'Behind scenes: kitchen tours, recipe sharing, cooking classes',
            'First taste: preview new menu items sebelum public launch',
            'Personal concierge: direct line untuk special requests, catering'
        ]
    },
    
    "J2: Service Recovery Program": {
        'features': {
            'Spend_x_Rating': 0.25,
            'RatingStd': 0.20,
            'ServiceRating': 0.15,
            'Rating_x_Loyalty': 0.12,
            'LoyalCustomer': 0.10,
            'FoodRating': 0.08,
            'AvgRating': 0.05,
            'VisitFrequency': 0.03,
            'AmbianceRating': 0.02,
            'RatingRange': 0.00
        },
        'description': '''Proactive system untuk identify dan recover customers who had negative experience 
        (indicated by drop in ratings). Immediate intervention untuk turn bad experience into positive 
        one. Statistics show customers whose complaints are resolved well become more loyal than those 
        who never complained. Critical untuk prevent churn dan rebuild trust.''',
        'implementation': [
            'Automated alerts: management notified instantly untuk ratings < 3 stars',
            'Immediate response: personal call/message within 24 hours',
            'Sincere apology: acknowledge issue without making excuses dan tunjukkan empati.',
            'Root cause investigation: understand exactly what went wrong (e.g., staff training issue, supply chain).',
            '**Kompensasi Proporsional:** Menawarkan kompensasi yang sepadan dengan kerugian (misalnya, free dessert untuk minor issue, full complimentary meal untuk major issue).',
            '**Follow-up:** Mengirimkan survey *follow-up* singkat 7 hari setelah *recovery* untuk memastikan kepuasan.',
            '**Preventive Action:** Menggunakan data dari keluhan yang diselesaikan untuk memperbarui SOP (Standard Operating Procedure) internal.'
        ]
    },
    "J3: High-Value Retention Perks": {
        'features': {
            'AverageSpend': 0.35,  # Fokus utama
            'Rating_x_Spend': 0.25, # Interaksi spend vs rating yang rendah
            'RatingStd': 0.15,     # Konsistensi yang buruk
            'LoyalCustomer': 0.10,
            'VisitFrequency': 0.05,
            'ServiceRating': 0.04,
            'FoodRating': 0.03,
            'Rating_x_Frequency': 0.02,
            'AmbianceRating': 0.01,
            'RatingRange': 0.00
        },
        'description': '''Strategi retensi yang sangat terfokus pada pelanggan yang memiliki riwayat 
        pembelanjaan tinggi (High-Spender) namun menunjukkan rating kepuasan yang rendah atau tidak konsisten. 
        Tujuan utamanya adalah menenangkan kekecewaan secara cepat, mencegah churn dari segmen pendapatan 
        kritis ini, dan membangun kembali hubungan melalui kompensasi premium yang proporsional dengan 
        nilai mereka. Ini adalah asuransi untuk pendapatan masa depan.''',
        'implementation': [
            'High-Spender Identification: Segmentasi pelanggan yang masuk kuartil atas AverageSpend (>Q3) tetapi memiliki AvgRating < 3.5.',
            'Targeted Offer: Mengirimkan kompensasi yang bernilai tinggi (misalnya, free premium meal atau diskon 50% pada kunjungan berikutnya).',
            'Feedback Loop: Manajer secara pribadi menghubungi untuk mendapatkan *feedback* mendalam mengenai pengalaman negatif mereka.',
            'Dedicated Service Channel: Memberikan jalur komunikasi prioritas untuk keluhan berikutnya.',
            'Proactive Check-in: Manajer melakukan *follow-up* personal pada kunjungan mereka berikutnya.'
        ]
    },
    "J4: Fast-Track Positif Loyalty": {
        'features': {
            'VisitFrequency': 0.30, # Sering datang
            'ServiceRating': 0.25,  # Sering memberi rating bagus (terutama service)
            'LoyaltyProgramMember': 0.15,
            'Rating_x_Frequency': 0.12, # Interaksi positif yang sering
            'FoodRating': 0.08,
            'AmbianceRating': 0.05,
            'AvgRating': 0.03,
            'LoyalCustomer': 0.01,
            'AverageSpend': 0.01,
            'Spend_x_Rating': 0.00
        },
        'description': '''Menciptakan jalur cepat (Fast-Track) ke level loyalitas yang lebih tinggi bagi pelanggan 
        yang menunjukkan frekuensi kunjungan tinggi dan secara konsisten memberikan rating positif (4 atau 5 bintang). 
        Tujuan strategi ini adalah mempercepat konversi mereka menjadi advokat merek dan memastikan mereka 
        merasa dihargai atas loyalitas dan *feedback* positif yang mereka berikan. Ini mendorong lingkaran 
        positif antara *feedback* dan *reward*.''',
        'implementation': [
            'Fast-Track Criteria: Pelanggan yang mencapai X kunjungan dalam Y bulan DAN memiliki AvgRating > 4.5 akan secara otomatis dinaikkan satu tingkat loyalitas (Tier Upgrade).',
            'Public Thank You: Memberikan shout-out atau ucapan terima kasih publik di platform media sosial mereka (dengan izin).',
            'Badge of Honor: Memberikan lencana digital atau *physical badge* yang menandakan status "Positive Advocate".',
            'Early Access: Memberi akses awal ke sistem reservasi atau promo musiman.',
            'Survey Reward: Hadiah kecil yang terikat dengan survey pasca-kunjungan untuk mendorong *feedback* positif yang berkelanjutan.'
            ]
        }
    }

def map_dataset_to_features(df: pd.DataFrame, 
                           model_features: List[str], 
                           min_features: int = 5
                          ) -> Tuple[bool, str, List[str], int, dict, pd.DataFrame]:
    """
    Smart Mapping dengan exact match, synonym, dan fuzzy match
    """
    df = df.copy()
    
    # Normalisasi nama kolom
    def normalize(col):
        return col.lower().replace("_", "").replace("-", "").replace(" ", "")
    
    df_norm = {normalize(c): c for c in df.columns}
    
    # Sinonim lengkap
    synonyms = {
        "waittime": ["waitingtime", "waiting_time", "queuetime", "wait", "waitduration"],
        "income": ["salary", "earning", "monthlyincome", "pendapatan", "gaji"],
        "averagespend": ["avgspend", "spending", "amountspent", "moneyspent", "spend", "totalspend"],
        "visitfrequency": ["visitcount", "numvisit", "freqvisit", "frequency", "visits"],
        "groupsize": ["pax", "guestcount", "peoplecount", "partysize", "guests"],
        "totalrating": ["overallrating", "ratingtotal", "total_rating"],
        "foodrating": ["ratingfood", "food_rating", "foodscore"],
        "servicerating": ["ratingservice", "service_rating", "servicescore"],
        "ambiancerating": ["ratingambiance", "ambiance_rating", "ambiencescore", "atmosphere", "ambiencerating"],
        "age": ["customerage", "customer_age", "usia"],
        "gender": ["sex", "jenis_kelamin"],
        "loyaltyprogrammember": ["ismember", "loyalty", "member", "loyaltymember"],
        "onlinereservation": ["reservation", "booking", "online_booking"],
        "deliveryorder": ["delivery", "takeout", "order_delivery"],
    }
    
    mapping_detail = {}
    matched_features = []
    missing_features = []
    
    # SMART MATCHING
    for feat in model_features:
        f_norm = normalize(feat)
        
        # 1) Exact match
        if f_norm in df_norm:
            col = df_norm[f_norm]
            mapping_detail[feat] = col
            matched_features.append(col)
            continue
        
        # 2) Synonym match
        if f_norm in synonyms:
            for syn in synonyms[f_norm]:
                syn_norm = normalize(syn)
                if syn_norm in df_norm:
                    col = df_norm[syn_norm]
                    mapping_detail[feat] = col
                    matched_features.append(col)
                    break
        
        if feat in mapping_detail:
            continue
        
        # 3) Fuzzy match
        best_match, score, _ = process.extractOne(
            f_norm,
            df_norm.keys(),
            scorer=fuzz.token_sort_ratio
        )
 
        if score >= 75:
            col = df_norm[best_match]
            mapping_detail[feat] = col
            matched_features.append(col)
            continue
        
        # 4) Fitur tidak ketemu
        missing_features.append(feat)
    
    # FITUR TURUNAN (Derived Features)
    derived = {}
    
    # Cek kolom yang ada di dataframe
    available_cols = set(df.columns)
    
    # SpendPerPerson
    if "TotalSpend" in available_cols and "GroupSize" in available_cols:
        df["SpendPerPerson"] = df["TotalSpend"] / df["GroupSize"].replace(0, 1)
        derived["SpendPerPerson"] = ["TotalSpend", "GroupSize"]
        if "SpendPerPerson" in model_features and "SpendPerPerson" not in mapping_detail:
            mapping_detail["SpendPerPerson"] = "SpendPerPerson"
            matched_features.append("SpendPerPerson")
    elif "AverageSpend" in available_cols and "GroupSize" in available_cols:
        df["SpendPerPerson"] = df["AverageSpend"] / df["GroupSize"].replace(0, 1)
        derived["SpendPerPerson"] = ["AverageSpend", "GroupSize"]
        if "SpendPerPerson" in model_features and "SpendPerPerson" not in mapping_detail:
            mapping_detail["SpendPerPerson"] = "SpendPerPerson"
            matched_features.append("SpendPerPerson")
    
    # SpendToIncomeRatio
    if "AverageSpend" in available_cols and "Income" in available_cols:
        df["SpendToIncomeRatio"] = df["AverageSpend"] / df["Income"].replace(0, 1)
        derived["SpendToIncomeRatio"] = ["AverageSpend", "Income"]
        if "SpendToIncomeRatio" in model_features and "SpendToIncomeRatio" not in mapping_detail:
            mapping_detail["SpendToIncomeRatio"] = "SpendToIncomeRatio"
            matched_features.append("SpendToIncomeRatio")
    
    # Rating aggregates
    rating_cols = ["ServiceRating", "FoodRating", "AmbianceRating"]
    rating_available = [col for col in rating_cols if col in available_cols]
    
    if len(rating_available) >= 2:
        df["AvgRating"] = df[rating_available].mean(axis=1)
        df["TotalRating"] = df[rating_available].sum(axis=1)
        df["RatingStd"] = df[rating_available].std(axis=1)
        df["MaxRating"] = df[rating_available].max(axis=1)
        df["MinRating"] = df[rating_available].min(axis=1)
        df["RatingRange"] = df["MaxRating"] - df["MinRating"]
        
        for new_feat in ["AvgRating", "TotalRating", "RatingStd", "MaxRating", "MinRating", "RatingRange"]:
            if new_feat in model_features and new_feat not in mapping_detail:
                mapping_detail[new_feat] = new_feat
                matched_features.append(new_feat)
        
        derived["RatingDerived"] = rating_available
    
    # Hapus duplikat
    matched_features = list(dict.fromkeys(matched_features))
    num_matched = len(matched_features)
    
    # Buat dataframe final dengan kolom yang matched
    df_final = df[[col for col in matched_features if col in df.columns]].copy()
    
    # VALIDASI
    if num_matched < min_features:
        message = (
            f"❌ Dataset terlalu umum. "
            f"Hanya {num_matched}/{len(model_features)} fitur yang cocok. "
            f"Minimal {min_features} fitur diperlukan."
        )
        return False, message, matched_features, num_matched, mapping_detail, df_final
    
    message = (
        f"✅ Dataset valid! {num_matched}/{len(model_features)} fitur berhasil di-map."
    )
    return True, message, matched_features, num_matched, mapping_detail, df_final

def build_topsis_matrix(matched_features: List[str], 
                       feature_importances: Dict[str, float],
                       strategy_mapping: Dict[str, Dict]) -> Tuple[pd.DataFrame, List[float], List[str]]:
    """
    Membangun decision matrix untuk TOPSIS berdasarkan features yang matched
    """
    feature_metadata = get_feature_metadata()
    
    # Filter strategi yang memiliki minimal 1 feature yang matched
    valid_strategies = {}
    for strategy, info in strategy_mapping.items():
        strategy_features = info['features']
        matched_strategy_features = {f: w for f, w in strategy_features.items() if f in matched_features}
        
        if len(matched_strategy_features) >= 1:
            # Normalize weights untuk features yang matched saja
            total_weight = sum(matched_strategy_features.values())
            normalized_weights = {f: w/total_weight for f, w in matched_strategy_features.items()}
            valid_strategies[strategy] = {
                'features': normalized_weights,
                'description': info['description']
            }
    
    if len(valid_strategies) == 0:
        return None, None, None
    
    # Buat decision matrix
    matrix_data = {}
    
    for feature in matched_features:
        scores = []
        for strategy in valid_strategies.keys():
            # Score = bobot strategi × feature importance
            strategy_weight = valid_strategies[strategy]['features'].get(feature, 0.0)
            feature_imp = feature_importances.get(feature, 0.0)
            score = strategy_weight * feature_imp * 100  # Scale up untuk visibility
            scores.append(score)
        
        matrix_data[feature] = scores
    
    decision_matrix = pd.DataFrame(matrix_data, index=list(valid_strategies.keys()))
    
    # Buat weights untuk TOPSIS (dari feature importance)
    weights = [feature_importances.get(f, 1.0/len(matched_features)) for f in matched_features]
    
    # Buat criteria types
    criteria_types = [feature_metadata.get(f, {}).get('type', 'Benefit') for f in matched_features]
    
    return decision_matrix, weights, criteria_types

