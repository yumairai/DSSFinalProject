import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from rapidfuzz import fuzz, process
from fuzzywuzzy import process

# ==============================================================================
# FUNGSI TOPSIS
# ==============================================================================
def calculate_topsis(decision_matrix: pd.DataFrame, 
                     weights: List[float], 
                     criteria_type: List[str]) -> pd.DataFrame:
    """
    Melakukan perhitungan TOPSIS.
    """
    if not np.isclose(sum(weights), 1.0):
        weights = np.array(weights) / sum(weights)
    
    X = decision_matrix.values
    R = X / np.sqrt((X**2).sum(axis=0))
    V = R * np.array(weights)
    
    A_plus = np.zeros(V.shape[1])
    A_minus = np.zeros(V.shape[1])
    
    for j in range(V.shape[1]):
        if criteria_type[j].lower() == 'benefit':
            A_plus[j] = V[:, j].max()
            A_minus[j] = V[:, j].min()
        elif criteria_type[j].lower() == 'cost':
            A_plus[j] = V[:, j].min()
            A_minus[j] = V[:, j].max()
    
    S_plus = np.sqrt(((V - A_plus)**2).sum(axis=1))
    S_minus = np.sqrt(((V - A_minus)**2).sum(axis=1))
    Closeness = S_minus / (S_plus + S_minus)
    
    closeness_series = pd.Series(Closeness, index=decision_matrix.index)
    
    results_df = pd.DataFrame({
        'Strategy': decision_matrix.index,
        'Closeness_Score': Closeness,
        'Rank': closeness_series.rank(method='dense', ascending=False).astype(int)
    }).sort_values(by='Closeness_Score', ascending=False).set_index('Strategy')
    
    return results_df

# ==============================================================================
# FUNGSI MAPPING & VALIDASI
# ==============================================================================
def load_model_features(model_file: str, feature_names_file: str) -> Tuple[Optional[object], Optional[list]]:
    """Load model dan feature names"""
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
            'description': '''Strategi ini menargetkan segmen pelajar dan mahasiswa (usia 17-25 tahun) yang memiliki 
            daya beli terbatas namun potensi menjadi pelanggan loyal jangka panjang. Program ini menawarkan diskon 
            khusus dengan menunjukkan kartu pelajar/mahasiswa, paket hemat untuk kelompok belajar, dan promo khusus 
            di jam-jam tertentu seperti setelah jam sekolah (15:00-18:00). Dengan membangun loyalitas sejak dini, 
            restoran dapat menciptakan customer base yang kuat untuk masa depan.''',
            'implementation': [
                'Diskon 15-20% dengan kartu pelajar/mahasiswa yang valid',
                'Paket "Study Group" untuk 4-6 orang dengan harga spesial',
                'Free Wi-Fi unlimited dan colokan listrik di setiap meja',
                'Menu ekonomis dengan porsi cukup (Rp 25.000 - 40.000)',
                'Loyalty card khusus pelajar: beli 5 gratis 1',
                'Promo "After School Special" jam 15:00-18:00',
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
            'description': '''Program yang dirancang untuk keluarga dengan anak-anak, menawarkan paket lengkap 
            yang mencakup menu untuk orang tua dan anak-anak. Strategi ini mempertimbangkan kebutuhan khusus 
            keluarga seperti high chair, kids menu, play area, dan porsi yang bisa dishare. Target utama adalah 
            keluarga middle-up income (30-50 tahun) yang mencari tempat makan nyaman untuk quality time bersama 
            keluarga, terutama di weekend dan momen spesial.''',
            'implementation': [
                'Paket "Happy Family" untuk 2 dewasa + 2 anak dengan harga bundling',
                'Kids Combo: main course + drink + dessert + mainan edukatif',
                'Fasilitas kids corner dengan permainan aman dan edukatif',
                'High chair dan kids cutlery tersedia di setiap area',
                'Menu khusus anak: nugget, pasta, pizza mini (porsi kecil, gizi seimbang)',
                'Promo "Family Weekend": diskon 20% untuk keluarga di Sabtu-Minggu',
                'Birthday package: gratis kue ulang tahun untuk reservasi family dining'
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
            'description': '''Strategi khusus untuk pelanggan senior (60+ tahun) dengan fokus pada kenyamanan, 
            kemudahan akses, dan pelayanan personal. Program ini menyadari bahwa pelanggan senior memiliki 
            kebutuhan khusus seperti menu rendah garam/gula, porsi lebih kecil, dan pelayanan yang lebih 
            sabar dan attentive. Dengan menyediakan fasilitas ramah lansia, restoran dapat menarik segmen 
            ini yang umumnya loyal dan appreciate good service.''',
            'implementation': [
                'Diskon senior 15% untuk usia 60+ (setiap hari)',
                'Kursi prioritas dekat pintu masuk dan toilet',
                'Menu sehat: low sodium, low sugar, high fiber options',
                'Porsi adjustable (small/regular) dengan harga proporsional',
                'Staf terlatih untuk membantu pelanggan senior (patience training)',
                '"Senior Morning": promo sarapan jam 07:00-10:00',
                'Kaca pembesar untuk membaca menu, pencahayaan yang baik'
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
            'description': '''Menu khusus yang diformulasikan untuk kesehatan senior dengan nutrisi seimbang, 
            rendah sodium, rendah gula, tinggi serat, dan mudah dicerna. Strategi ini menargetkan pelanggan 
            senior yang health-conscious dan keluarga yang peduli dengan kesehatan orang tua mereka. Setiap 
            menu dilengkapi dengan informasi nutrisi lengkap dan rekomendasi dari ahli gizi.''',
            'implementation': [
                'Menu "Golden Age": 15 pilihan menu dengan nutrisi terukur',
                'Kolaborasi dengan ahli gizi untuk desain menu',
                'Label jelas: kalori, sodium, gula, fiber di setiap menu',
                'Opsi steamed, grilled, dan boiled (menghindari deep fried)',
                'Porsi sayuran lebih banyak, protein lean, karbohidrat kompleks',
                'Free konsultasi gizi untuk member senior',
                'Weekly special: "Healthy Thursday" menu baru setiap minggu'
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
            'description': '''Program promo yang dirancang khusus untuk generasi muda (18-35 tahun) di weekend, 
            menggabungkan dining experience dengan entertainment dan social gathering. Target utama adalah 
            young professionals dan mahasiswa yang mencari tempat hangout di akhir pekan. Strategi ini 
            memanfaatkan peak hours weekend untuk maksimalkan occupancy dengan harga special.''',
            'implementation': [
                '"Weekend Vibes": diskon 25% untuk usia 18-35 (Sabtu-Minggu)',
                'Buy 2 Get 1 untuk minuman/dessert di weekend',
                'Live music atau DJ setiap Sabtu malam',
                'Instagram-worthy corner untuk foto (aesthetic decor)',
                'Promo "Bring Your Squad": gratis appetizer untuk grup 5+',
                'Happy hour extended: Jumat 17:00-21:00',
                'Social media contest: post & tag untuk lucky draw'
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
            'description': '''Strategi community building dengan mengadakan event reguler yang sesuai dengan 
            segmen usia tertentu. Youth Night untuk generasi muda (18-30 tahun) dengan suasana energik dan 
            modern, sementara Senior Morning untuk pelanggan senior (55+) dengan suasana tenang dan nyaman. 
            Event ini menciptakan sense of belonging dan meningkatkan repeat visits.''',
            'implementation': [
                '"Youth Night" setiap Jumat: live music, game, networking (19:00-23:00)',
                '"Senior Morning" setiap Rabu: breakfast buffet, light music (08:00-11:00)',
                'Tema bulanan: karaoke night, trivia night, game tournament',
                'Special pricing untuk event participants',
                'Loyalty points double untuk yang attend events',
                'Komunitas WhatsApp Group untuk info event dan promo',
                'Partnership dengan komunitas lokal (fitness club, book club, etc)'
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
            'description': '''Menu combo khusus anak-anak yang menarik, bergizi, dan harga terjangkau. Strategi 
            ini menargetkan keluarga dengan anak kecil (2-12 tahun) yang kesulitan mencari menu sesuai selera 
            anak. Setiap combo dilengkapi dengan mainan edukatif dan presentasi yang fun untuk meningkatkan 
            appetite anak. Program ini juga membantu orang tua yang ingin memastikan anak makan makanan sehat.''',
            'implementation': [
                'Paket "Little Star" (2-5 tahun): nugget/pasta + juice + fruit + toy',
                'Paket "Young Explorer" (6-12 tahun): mini burger/pizza + smoothie + dessert + toy',
                'Presentasi lucu: smiley face dari makanan, plate berwarna',
                'Mainan edukatif yang berubah setiap bulan (koleksi)',
                'Hidden vegetables dalam menu (pure sayuran dalam sauce)',
                'Harga flat Rp 35.000-50.000 all-in',
                'Kids eat free: 1 kids combo gratis per 2 adult main course (weekday lunch)'
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
            'description': '''Program membership eksklusif untuk pelanggan high-income dengan benefit premium 
            dan personalized service. Target adalah profesional, business owners, dan executives yang menghargai 
            exclusivity dan willing to pay premium untuk superior experience. Program ini fokus pada creating 
            VIP experience yang memorable dengan priority service, exclusive menu, dan special privileges.''',
            'implementation': [
                'Membership fee: Rp 2.000.000/tahun dengan benefit setara 3-4x lipat',
                'Private dining room dengan butler service',
                'Priority reservation: booking kapan saja, guaranteed seating',
                'Exclusive menu items: premium ingredients (wagyu, lobster, truffle)',
                'Complimentary valet parking dan welcome drink',
                '20% discount untuk semua menu + double loyalty points',
                'Birthday privilege: free premium course untuk member + 3 guests',
                'Quarterly exclusive event: wine tasting, chef table experience',
                'Personal concierge untuk catering dan special occasions'
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
            'description': '''Strategi bundling value-for-money untuk pelanggan dengan budget terbatas namun 
            tetap ingin menikmati dining experience yang memuaskan. Program ini menawarkan paket hemat dengan 
            kombinasi main course, drink, dan dessert/appetizer di harga yang lebih terjangkau. Target adalah 
            middle-low income families, students, dan young workers yang price-sensitive.''',
            'implementation': [
                'Paket "Hemat Harian" (Rp 35.000): nasi + main protein + sayur + es teh',
                'Paket "Duo Ekonomis" (Rp 60.000): 2 main course + 2 drinks',
                'Paket "Family Bundle" (Rp 150.000): 4 pax complete meal',
                '"Mix & Match": pilih 1 main + 1 side + 1 drink dengan harga bundling',
                'Weekday lunch special: Rp 30.000 all-in (11:00-14:00)',
                'Free upsize untuk drink di semua paket hemat',
                'Loyalty stamp: beli 8 paket hemat, dapat 1 gratis',
                'Cashless promo: tambahan diskon 5% untuk pembayaran e-wallet'
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
            'description': '''Program reward berbasis cashback untuk pelanggan yang konsisten spending tinggi 
            di restoran. Semakin besar spending, semakin besar cashback yang didapat. Program ini bertujuan 
            untuk retain high-value customers dan encourage mereka untuk increase spending. Cashback bisa 
            digunakan untuk visit berikutnya atau ditukar dengan exclusive rewards.''',
            'implementation': [
                'Tiered cashback: spend Rp 200k (5%), Rp 500k (8%), Rp 1M (12%)',
                'Cashback accumulated dalam membership account (digital)',
                'Bonus cashback untuk spending di hari tertentu (Tuesday: double cashback)',
                'Cashback dapat digunakan untuk discount di visit berikutnya',
                'Special reward: cashback 20% untuk spending > Rp 2jt/bulan',
                'Referral bonus: Rp 50k cashback untuk setiap teman yang join dan spend min Rp 200k',
                'Birthday month: triple cashback untuk semua transaksi',
                'Premium redemption: tukar cashback dengan exclusive items',
                'Quarterly report: statement cashback earned dan redeemed'
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
            'description': '''Strategi pricing yang dipersonalisasi berdasarkan profil income dan spending 
            pattern pelanggan. Menggunakan data analytics untuk memberikan promo yang relevan dengan 
            kemampuan dan preferensi spending customer. High-income customers mendapat akses ke premium 
            items dengan special pricing, sementara budget-conscious customers mendapat diskon untuk 
            value items.''',
            'implementation': [
                'AI-powered recommendation: suggest menu sesuai budget dan preferensi',
                'Dynamic voucher: otomatis dapat voucher sesuai profil spending',
                'Segmentasi: Gold (high-income), Silver (medium), Bronze (budget)',
                'Personalized promo via app: berbeda untuk setiap customer',
                'Smart upsell: suggest upgrade hanya jika sesuai spending capacity',
                'Flexible payment: cicilan 0% untuk bill > Rp 500k (high-income)',
                'Budget alert: notifikasi menu dalam budget customer',
                'Spending insights: monthly report spending pattern + recommendations',
                'Loyalty tier upgrade: automatic upgrade based on spending milestone'
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
            'description': '''Fasilitas pembayaran cicilan atau pay-later untuk pembelian menu premium dan 
            paket catering dengan nilai tinggi. Program ini memudahkan customer high-income untuk enjoy 
            premium dining tanpa beban cash flow langsung. Bekerjasama dengan fintech untuk provide 
            installment options yang flexible dan competitive rates.''',
            'implementation': [
                'Cicilan 0%: untuk transaksi > Rp 1jt (tenor 3 bulan)',
                'Partnership dengan Kredivo, Akulaku, Atome',
                'Special items eligible: premium tasting menu, catering package',
                'Pre-approved limit untuk loyal customers (up to Rp 10jt)',
                'Buy Now Pay Later: bayar dalam 30 hari tanpa bunga',
                'Flexible payment: pilih tenor 3/6/12 bulan',
                'Exclusive access: early bird untuk seasonal menu dengan pay-later',
                'Catering package: cicilan untuk event catering min Rp 5jt',
                'Digital integration: apply cicilan langsung via app/website'
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
            'description': '''Program upgrade untuk mengajak pelanggan high-income mencoba premium menu 
            dengan harga special atau benefit tambahan. Strategi ini memperkenalkan high-margin items 
            kepada customer yang memiliki purchasing power dengan offering yang menarik seperti pairing 
            recommendations, chef special, atau limited edition items.''',
            'implementation': [
                '"Taste the Premium": special price untuk first-time taster (diskon 30%)',
                'Wine pairing recommendation untuk setiap premium main course',
                'Chef signature dishes: exclusive items dari head chef',
                'Seasonal premium menu: truffle season, lobster month, wagyu week',
                'Complimentary appetizer untuk order premium main course',
                'VIP tasting event: undangan untuk try new premium items',
                'Upgrade option: tambah Rp 50k untuk upgrade ke premium version',
                'Premium set menu: 5-course tasting menu dengan wine pairing',
                'Exclusive reservation: priority booking untuk premium dining area'
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
            'description': '''Paket menu dengan value proposition terbaik untuk budget-conscious customers. 
            Fokus pada memberikan perceived value tinggi dengan harga terjangkau melalui portion optimization, 
            ingredient efficiency, dan smart packaging. Target adalah families, students, dan workers dengan 
            budget terbatas yang tetap ingin dining experience yang memuaskan.''',
            'implementation': [
                'Paket "Super Value" (Rp 39.000): large portion main + jumbo drink + free refill',
                '"Family Value Pack" (Rp 120.000): meal untuk 4 orang lengkap',
                'Daily special: menu ekonomis berbeda setiap hari (Rp 25.000)',
                'Unlimited rice untuk semua paket ekonomis',
                'Free soup atau salad sebagai starter',
                'Kombo hemat: 1+1 deals untuk selected items',
                'Weekday lunch buffet: Rp 45.000 all-you-can-eat (11:00-14:00)',
                'Student package: paket pelajar Rp 30.000 dengan kartu pelajar',
                'Takeaway discount: 10% off untuk paket ekonomis takeaway'
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
            'description': '''Program loyalty tradisional dengan digital twist menggunakan stamp card system. 
            Setiap kunjungan mendapat stamp, dan setelah mengumpulkan jumlah tertentu mendapat reward gratis. 
            Sistem ini simple, mudah dipahami, dan effective untuk encourage repeat visits. Digital stamp 
            memudahkan tracking dan prevent fraud sambil memberikan data valuable untuk analytics.''',
            'implementation': [
                'Digital stamp card via mobile app (paperless)',
                '"Buy 10 Get 1 Free": setiap 10 kunjungan dapat 1 main course gratis',
                'Stamp multiplier: kunjungan di weekday dapat 2x stamp',
                'Bonus stamp untuk bring friends (1 extra stamp per friend)',
                'Progressive rewards: 5 stamps (free drink), 10 stamps (free meal), 20 stamps (free premium)',
                'Stamp expiry: 6 bulan sejak kunjungan terakhir',
                'Transfer stamp: bisa share/gift stamp ke teman (max 3 stamps/month)',
                'Birthday bonus: 5 extra stamps di bulan ulang tahun',
                'Gamification: badges untuk milestone achievement'
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
            'description': '''Program khusus yang memberikan privilege dan recognition kepada frequent visitors. 
            Semakin sering berkunjung, semakin banyak benefit yang didapat. Program ini menciptakan sense of 
            VIP treatment tanpa membership fee, purely based on visit frequency. Ini effective untuk retain 
            regular customers dan make them feel valued and appreciated.''',
            'implementation': [
                '"Frequent Visitor Day" setiap Selasa: 30% off untuk customer dengan 8+ visits/bulan',
                'Tier system: Bronze (4x/month), Silver (8x/month), Gold (12x/month)',
                'Escalating benefits: Bronze (10% off), Silver (15% off), Gold (20% off)',
                'Priority seating: no queue untuk Gold members',
                'Personal greeting: staff recognize dan greet by name',
                'Complimentary: free appetizer untuk Silver+, free dessert untuk Gold',
                'Birthday celebration: free cake dan song untuk frequent visitors',
                'Exclusive preview: first to know about new menu',
                'VIP WhatsApp group: direct line untuk reservation dan feedback'
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
            'description': '''Sistem membership bertingkat (Bronze, Silver, Gold, Platinum) yang memberikan 
            escalating benefits berdasarkan kombinasi visit frequency dan total spending. Program ini 
            gamified untuk encourage customers untuk "level up" dengan clear milestones dan attractive 
            rewards di setiap tier. Ini creates engagement dan long-term loyalty.''',
            'implementation': [
                'Bronze: free signup, 5% discount, birthday treat',
                'Silver: 10 visits atau spend Rp 1jt/3bulan → 10% discount, priority seating',
                'Gold: 25 visits atau spend Rp 3jt/3bulan → 15% discount, free appetizer, exclusive events',
                'Platinum: 50 visits atau spend Rp 8jt/3bulan → 20% discount, private dining, concierge',
                'Tier maintenance: stay active dengan min 1 visit/month',
                'Fast track: double spending/visit count di first month',
                'Tier privileges: each tier unlock exclusive menu items',
                'Annual review: tier reset setiap tahun dengan grace perio',
                'Upgrade notification: alert ketika mendekati next tier milestone'
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
        'description': '''Strategi re-engagement untuk customer yang menurun frekuensi kunjungannya atau 
        sudah lama tidak visit. Menggunakan automated reminder via email/WhatsApp/push notification 
        dengan personalized offer untuk encourage them to come back. Program ini crucial untuk prevent 
        customer churn dan reactivate dormant customers.''',
        'implementation': [
            'Automated trigger: reminder setelah 2 minggu no-visit (untuk frequent visitors)',
            '"We Miss You" campaign: special 25% discount untuk comeback visit',
            'Personalized message: mention favorite menu dan last visit date',
            'Progressive offers: week 2 (15% off), week 4 (20% off), week 6 (30% off)',
            'Time-limited offer: voucher valid hanya 7 hari untuk create urgency',
            'Multi-channel: email, WhatsApp, push notification, SMS',
            'Feedback request: ask why they stopped coming (dengan incentive)',
            'Special comeback menu: exclusive item hanya untuk returning customers',
            'Reactivation tracking: measure comeback rate dan lifetime value'
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
        'description': '''Benefit eksklusif yang hanya bisa diakses oleh frequent visitors untuk create 
        sense of exclusivity dan reward loyalty. Program ini memberikan access ke special items, 
        priority service, dan unique experiences yang tidak available untuk regular customers. 
        Ini makes frequent visitors feel special dan appreciated.''',
        'implementation': [
            'Secret menu: akses ke off-menu items hanya untuk frequent visitors (8+ visits/month)',
            'Chef table experience: monthly exclusive untuk top 10 frequent visitors',
            'Early bird access: book new menu 1 minggu sebelum launch',
            'VIP hours: exclusive dining hours (06:00-08:00) dengan special menu',
            'Behind-the-scene tour: kitchen visit dan meet the chef',
            'Complimentary valet: free parking untuk Gold frequent visitors',
            'Priority for events: first invite untuk special events dan celebrations',
            'Customization privilege: request custom menu adjustments',
            '"Table Reserved" policy: favorite table always reserved untuk top frequenters'
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
        'description': '''Teknik upselling yang terstruktur dan terlatih untuk meningkatkan average 
        transaction value dengan suggest upgrades, add-ons, dan premium options. Staff dilatih untuk 
        identify opportunities dan make relevant suggestions tanpa being pushy. Fokus pada enhancing 
        customer experience sambil meningkatkan revenue per transaction.''',
        'implementation': [
            'Staff training: upselling techniques, product knowledge, timing',
            'Suggestive selling: "Would you like to add truffle untuk Rp 25k?"',
            'Drink pairing: sommelier/barista suggest perfect drink pairing',
            'Size upgrade: "Upgrade ke large portion hanya tambah Rp 15k"',
            'Premium protein: "Change chicken ke wagyu beef tambah Rp 50k"',
            'Dessert suggestion: show dessert menu atau dessert cart setelah main',
            'Appetizer promotion: special price appetizer jika order sebelum main',
            'Combo deals: "Add soup + salad hanya Rp 20k (hemat Rp 15k)"',
            'Commission incentive: bonus untuk staff yang achieve upsell target'
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
        'description': '''Cross-selling strategy untuk encourage customers membeli item complementary 
        atau dari kategori berbeda. Menggunakan data analytics untuk identify best pairings dan 
        create attractive combo deals. Program ini increase basket size dengan suggest items yang 
        enhance main order dan provide better overall experience.''',
        'implementation': [
            '"Perfect pairs" menu: show recommended pairings untuk setiap main dish',
            'Combo discounts: "Add garlic bread Rp 15k (harga normal Rp 25k)"',
            'Beverage bundle: order 2 drinks dapat harga special',
            'Dessert sharing: suggest sharing dessert untuk tables dengan 2+ pax',
            'Appetizer sampler: "Try 3 appetizers Rp 60k (hemat 30%)"',
            'Side dish promotion: unlimited side dish dengan upgrade package',
            'Complete meal offer: main + drink + dessert package dengan discount',
            'Digital menu suggestion: app automatically suggest pairing items',
            'Visual merchandising: display appetizing photos dari combo items'
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
        'description': '''Program incentive berlapis yang reward customers berdasarkan spending threshold. 
        Semakin besar spend, semakin valuable reward yang didapat. Program ini encourage customers 
        untuk reach next threshold dengan clear visibility pada progress dan rewards. Effective untuk 
        increase average ticket size dan customer satisfaction.''',
        'implementation': [
            'Threshold rewards: Rp 150k (free drink), Rp 300k (free appetizer), Rp 500k (free dessert)',
            'Real-time tracking: bill shows current spend dan next reward threshold',
            '"Unlock reward": notification "Tambah Rp 25k untuk unlock free dessert"',
            'Tiered discount: spend Rp 200k (5% off), Rp 400k (10% off), Rp 800k (15% off)',
            'Mystery reward: spend Rp 1jt+ untuk surprise premium gift',
            'Group pooling: combine spending untuk grup seating untuk reach threshold',
            'Double points day: certain days, threshold dikurangi 50%',
            'Premium tier: spend Rp 2jt+ dapat exclusive voucher untuk next visit',
            'Quarterly milestone: total spending per quarter untuk bigger rewards'
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
        'description': '''Program reward yang di-trigger ketika customer reach minimum purchase amount 
        tertentu dalam single transaction. Benefit langsung dan immediate untuk create instant 
        gratification. Program ini simple, clear, dan effective untuk push customers untuk add 
        more items untuk qualify reward.''',
        'implementation': [
            '"Spend Rp 200k, Get Free Appetizer" (pilihan dari 5 appetizer menu)',
            '"Spend Rp 350k, Get Free Premium Dessert" (valued up to Rp 45k)',
            'Automatic upgrade: spend Rp 150k+ otomatis upgrade drink size',
            'Complimentary sides: spend Rp 250k+ dapat unlimited bread/refill',
            'Weekend special: threshold lebih rendah di Sabtu-Minggu (Rp 180k untuk free item)',
            'Takeaway bonus: spend Rp 300k+ dapat free delivery (no delivery fee)',
            'Choice of reward: customer pilih reward dari menu options',
            'Stack with other promos: kombinable dengan member discount',
            'Progress indicator: menu shows "Add Rp 50k more to get free dessert"'
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
        'description': '''AI-powered recommendation engine yang analyze customer history, preferences, 
        dan spending pattern untuk provide personalized upsell suggestions via mobile app. Setiap 
        customer mendapat recommendations yang relevant dengan taste dan budget mereka, increasing 
        conversion rate dan customer satisfaction. Non-intrusive dan value-adding approach.''',
        'implementation': [
            'ML algorithm: analyze past orders, ratings, dan spending untuk predict preferences',
            'Smart suggestions: "Based on your history, you might like..."',
            'Budget-aware: suggest items within customer usual spending range',
            'Timing optimization: suggest appetizer before order, dessert after main',
            'Contextual offers: different suggestions untuk lunch vs dinner, weekday vs weekend',
            'A/B testing: continuously improve recommendation accuracy',
            'One-click add: easy to add suggested items dengan single tap',
            '"Complete your meal": suggest missing components (no drink? no dessert?)',
            'Social proof: "80% customers who ordered this also added..."'
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
        'description': '''Sistem antrian digital yang modern untuk minimize wait frustration dan optimize 
        table management. Customers dapat join queue via app, get real-time update posisi antrian, 
        dan estimated waiting time. Ini eliminate kerumunan di entrance dan give customers flexibility 
        untuk wait comfortably atau do other things sambil waiting. Improve perceived service quality 
        significantly.''',
        'implementation': [
            'Mobile queue app: join queue dari rumah via smartphone',
            'QR code check-in: scan QR di entrance untuk masuk antrian',
            'Real-time updates: push notification untuk queue progress',
            'Accurate wait time: AI prediction untuk estimated wait (±5 min accuracy)',
            'Virtual waiting: customers tidak perlu stand di entrance, bisa wait di sekitar area',
            'SMS/WhatsApp alert: "Your table is ready in 5 minutes"',
            'Priority lane: fast-track untuk reservations dan loyalty members',
            'Waiting dashboard: display screen showing queue number dan status',
            'Analytics: track peak hours, average wait time untuk optimization'
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
        'description': '''Priority system untuk customers yang book reservation online, eliminating atau 
        significantly reducing wait time. Reserved customers get guaranteed seating dan skip regular 
        queue. Program ini incentivize online booking behavior, improve customer experience, dan help 
        restoran better forecast demand dan optimize staffing.''',
        'implementation': [
            'Guaranteed seating: reservations get seated within 5 minutes dari arrival',
            'Easy booking: book via app, website, WhatsApp dengan instant confirmation',
            'Flexible timing: 15 min grace period untuk late arrival',
            'Reservation reminder: notification H-3, H-1, dan 2 hours before',
            'Preferred seating: option untuk request specific table/area',
            'Pre-order option: order menu sebelum datang untuk faster service',
            'Cancellation policy: free cancel up to 2 hours before',
            'Reservation perks: complimentary welcome drink untuk reserved customers',
            'Peak hours priority: especially valuable untuk Fri-Sun dinner hours'
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
        'description': '''Transparent real-time information tentang current wait time yang di-display di 
        berbagai channels (app, website, Google Maps, in-store display). Customers dapat make informed 
        decision kapan untuk visit atau whether to join queue. Transparency reduces frustration dan 
        improve trust. Integration dengan capacity management untuk accurate predictions.''',
        'implementation': [
            'Live display: current wait time di homepage website dan mobile app',
            'Google Maps integration: wait time muncul di Google business profile',
            'In-store display: digital board showing current wait dan queue length',
            'Historical data: show typical wait time untuk hari/jam tertentu',
            'Color coding: green (0-15min), yellow (15-30min), red (30min+)',
            'Push notification: alert ketika wait time berkurang significantly',
            '"Best time to visit": recommendation jam dengan minimal wait',
            'Capacity indicator: shows current occupancy percentage',
            'API integration: third-party apps bisa access wait time data'
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
        'description': '''Optimasi staffing berdasarkan predicted demand untuk ensure adequate service 
        during peak hours dan avoid over-staffing during slow periods. Menggunakan historical data, 
        weather, events, dan other factors untuk forecast traffic dan adjust staff schedule accordingly. 
        Result: better service, reduced wait time, optimal labor cost.''',
        'implementation': [
            'Predictive scheduling: AI forecast demand untuk 2 weeks ahead',
            'Flexible shifts: staff willing untuk extend/cut shift based on real-time demand',
            'On-call system: additional staff on standby untuk unexpected peaks',
            'Cross-training: staff dapat handle multiple roles untuk flexibility',
            'Peak hours reinforcement: extra staff untuk Fri-Sun dinner, lunch rush',
            'Real-time adjustment: manager dapat call additional staff jika queue meningkat',
            'Staff allocation: assign more staff ke high-traffic sections',
            'Performance tracking: monitor staff efficiency dan customer satisfaction',
            'Incentive program: bonus untuk staff yang work during challenging peak hours'
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
        'description': '''Sistem pre-order yang allow customers untuk order menu sebelum arrive, baik 
        untuk dine-in atau takeaway. Food preparation dimulai sebelum customer datang, drastically 
        reducing wait time setelah seated. Particularly effective untuk lunch crowds dan business 
        diners yang time-constrained. Improve table turnover dan customer satisfaction.''',
        'implementation': [
            'App-based pre-order: order minimum 15 menit sebelum arrival',
            'Scheduled preparation: kitchen dapat plan dan optimize cooking sequence',
            'Arrival notification: customer notify saat on the way untuk timing preparation',
            'Guaranteed ready time: food ready within 5 min dari confirmed arrival',
            'Express pickup: dedicated counter untuk pre-order pickup',
            'Modification option: dapat modify order up to 10 min before scheduled time',
            'Payment integration: pre-pay via app untuk seamless experience',
            'Takeaway express: pre-order takeaway ready di exact scheduled time',
            'Group pre-order: all members dapat add items ke single group order'
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
        'description': '''Self-service kiosk untuk order dan payment, reducing dependency pada staff dan 
        minimizing order wait time. Customers dapat browse menu dengan visual yang menarik, customize 
        order, dan pay directly. Particularly appealing untuk tech-savvy customers dan effective during 
        peak hours. Also increase order accuracy dan upsell through strategic prompts.''',
        'implementation': [
            'Touchscreen kiosk: user-friendly interface dengan large product images',
            'Multiple payment: accept cash, card, e-wallet, QRIS',
            'Customization: easy to add/remove ingredients, adjust spice level',
            'Visual menu: high-quality photos dengan descriptions',
            'Upsell prompts: "Would you like to add fries for Rp 15k?"',
            'Order number system: receive number untuk food pickup',
            'Multiple language: Bahasa, English options',
            'Express mode: quick order untuk popular items ("Usual" button)',
            'Digital receipt: email atau print receipt option',
            'Parallel ordering: multiple kiosks untuk simultaneous orders'
        ]
    },

    # ========== F. STRATEGI BERBASIS RATING & KUALITAS ==========
    
    "F1: Service Excellence Training": {
        'features': {
            'ServiceRating': 0.25,
            'ConsistentQuality': 0.20,
            'AvgRating': 0.15,
            'RatingStd': 0.12,
            'TotalRating': 0.10,
            'WaitTime': 0.08,
            'Rating_x_Loyalty': 0.05,
            'FoodRating': 0.03,
            'AmbianceRating': 0.02,
            'RatingRange': 0.00
        },
        'description': '''Program training komprehensif untuk staff mencakup customer service skills, 
        product knowledge, complaint handling, dan service recovery. Tujuan adalah create consistent 
        excellent service experience di setiap touch point. Includes role-playing, mystery shopper, 
        dan continuous feedback loops. Well-trained staff adalah key differentiator dan directly 
        impact customer satisfaction.''',
        'implementation': [
            'Onboarding training: 2-week comprehensive training untuk new hires',
            'Monthly refresher: 2-hour training session untuk existing staff',
            'Service standards: documented SOP untuk every customer interaction',
            'Product knowledge: weekly tasting session untuk new/seasonal items',
            'Complaint handling: specific protocols untuk different complaint scenarios',
            'Mystery shopper: monthly evaluation dengan detailed feedback',
            'Service awards: recognition untuk staff dengan highest service ratings',
            'Cross-training: FOH staff learn BOH operations untuk better understanding',
            'Customer feedback review: weekly team review dari customer comments',
            'Certification program: service excellence certification untuk career advancement'
        ]
    },
    
    "F2: Food Quality Assurance": {
        'features': {
            'FoodRating': 0.30,
            'ConsistentQuality': 0.20,
            'AvgRating': 0.15,
            'RatingStd': 0.12,
            'TotalRating': 0.10,
            'RatingRange': 0.06,
            'MaxRating': 0.04,
            'MinRating': 0.03,
            'ServiceRating': 0.00,
            'Spend_x_Rating': 0.00
        },
        'description': '''Sistem quality control yang strict untuk maintain food quality consistency 
        dan safety. Include supplier auditing, ingredient inspection, standardized recipes, cooking 
        procedures, dan plating guidelines. Weekly quality audits oleh chef dan periodic external 
        audits. Zero tolerance untuk quality compromises. Consistency adalah key untuk build 
        reliable brand reputation.''',
        'implementation': [
            'Supplier certification: approved vendor list dengan quality requirements',
            'Daily ingredient inspection: check freshness, temperature, appearance',
            'Standardized recipes: exact measurements, timing, temperature specifications',
            'Portion control: accurate portioning untuk consistency dan cost control',
            'Plating standards: photo guidelines untuk every dish presentation',
            'Taste testing: chef approval before every service shift',
            'Temperature monitoring: regular checks untuk storage dan cooking temps',
            'Weekly audit: comprehensive kitchen inspection dengan checklist',
            'Food waste tracking: analyze untuk identify quality issues',
            'Customer feedback integration: menu adjustment based on recurring feedback'
        ]
    },
    
    "F3: Ambience Enhancement": {
        'features': {
            'AmbianceRating': 0.30,
            'AvgRating': 0.18,
            'ConsistentQuality': 0.15,
            'TotalRating': 0.12,
            'RatingRange': 0.10,
            'ServiceRating': 0.08,
            'FoodRating': 0.04,
            'TimeOfVisit': 0.02,
            'DiningOccasion': 0.01,
            'RatingStd': 0.00
        },
        'description': '''Investment dalam interior design, lighting, music, cleanliness, dan overall 
        atmosphere untuk create inviting dan comfortable dining environment. Ambience significantly 
        impact dining experience dan willingness to return. Different areas dapat have different 
        vibes untuk cater different occasions (romantic, family, business). Regular refresh untuk 
        maintain contemporary appeal.''',
        'implementation': [
            'Professional interior design: cohesive theme dan aesthetic',
            'Lighting control: adjustable lighting untuk different time of day',
            'Curated playlist: music selection sesuai time dan crowd',
            'Comfortable seating: ergonomic chairs, appropriate table height',
            'Temperature control: AC yang adequate dan well-maintained',
            'Cleanliness: constant monitoring, immediate cleanup protocols',
            'Décor refresh: seasonal decoration updates untuk freshness',
            'Scent marketing: subtle pleasant aroma (fresh bread, coffee)',
            'Instagram spots: designed photo-worthy corners',
            'Noise management: acoustic panels untuk control noise level'
        ]
    },
    
    "F4: Real-time Feedback System": {
        'features': {
            'ServiceRating': 0.20,
            'FoodRating': 0.18,
            'AmbianceRating': 0.15,
            'AvgRating': 0.15,
            'ConsistentQuality': 0.12,
            'RatingStd': 0.08,
            'Rating_x_Loyalty': 0.06,
            'TotalRating': 0.04,
            'WaitTime': 0.02,
            'Spend_x_Rating': 0.00
        },
        'description': '''Digital feedback collection system yang allows customers memberikan rating dan 
        comment immediately setelah dining. Quick and easy feedback mechanism via QR code atau tablet 
        di meja. Real-time alerts untuk management jika ada negative feedback untuk immediate service 
        recovery. Data analyzed untuk continuous improvement dan staff coaching.''',
        'implementation': [
            'QR code feedback: scan QR di bill untuk instant feedback form',
            'Tablet on table: optional untuk premium sections',
            '5-star rating: separate ratings untuk food, service, ambiance',
            'Comment box: open text untuk specific feedback',
            'Real-time alerts: management notified instantly untuk rating < 3 stars',
            'Service recovery: immediate action untuk resolve issues before customer leaves',
            'Incentive: small reward (free dessert/drink) untuk complete feedback',
            'Dashboard: live dashboard untuk management monitoring',
            'Weekly report: aggregated feedback dengan insights untuk improvement',
            'Follow-up: personal response untuk customers who left detailed feedback'
        ]
    },
    
    "F5: Quality Guarantee Program": {
        'features': {
            'FoodRating': 0.25,
            'ServiceRating': 0.20,
            'AvgRating': 0.15,
            'ConsistentQuality': 0.15,
            'RatingStd': 0.10,
            'AmbianceRating': 0.08,
            'Rating_x_Loyalty': 0.05,
            'TotalRating': 0.02,
            'Spend_x_Rating': 0.00,
            'MinRating': 0.00
        },
        'description': '''Bold guarantee policy: jika customer tidak satisfied dengan food atau service, 
        kami replace gratis atau refund tanpa pertanyaan. Program ini demonstrate confidence dalam 
        quality dan prioritize customer satisfaction above short-term profit. Build tremendous trust 
        dan loyalty. Empowers staff untuk make decisions untuk resolve issues immediately.''',
        'implementation': [
            'Unconditional guarantee: tidak puas, ganti gratis atau refund 100%',
            'No questions asked: staff tidak argue atau question customer complaint',
            'Immediate resolution: replacement prepared immediately atau refund processed',
            'Staff empowerment: any staff dapat authorize guarantee tanpa manager approval',
            'Apology standards: sincere apology dan acknowledgment dari management',
            'Root cause analysis: investigate setiap guarantee claim untuk prevent recurrence',
            'Communication: guarantee policy prominently displayed di menu dan signage',
            'Recovery attempt: offer alternative solution sebelum default ke refund',
            'Follow-up: personal call atau message untuk ensure satisfaction',
            'Tracking: monitor guarantee rate untuk identify systemic quality issues'
        ]
    },
    
    "F6: Menu Standardization": {
        'features': {
            'FoodRating': 0.25,
            'ConsistentQuality': 0.25,
'RatingStd': 0.15,
            'RatingRange': 0.12,
            'AvgRating': 0.10,
            'TotalRating': 0.06,
            'ServiceRating': 0.04,
            'MaxRating': 0.02,
            'MinRating': 0.01,
            'AmbianceRating': 0.00
        },
        'description': '''Rigorous standardization dari every menu item untuk ensure exact same taste, 
        presentation, dan quality setiap kali. Detailed recipe cards, precise measurements, timing, 
        dan plating instructions. Regular audits dan retraining untuk maintain standards. Goal adalah 
        customer dapat expect same excellent quality whether they visit di pagi, siang, atau malam, 
        weekday atau weekend. Predictability builds trust.''',
        'implementation': [
            'Recipe database: digital recipe system dengan exact specifications',
            'Portion tools: scoops, ladles, scales untuk precise portioning',
            'Cooking timers: standardized cooking times dengan timer enforcement',
            'Plating photos: reference photos untuk every dish at every station',
            'Ingredient prep: centralized prep untuk consistency (sauces, marinades)',
            'Quality checklist: visual checklist sebelum dish goes out',
            'Cross-shift tasting: different shifts taste test untuk consistency',
            'New staff certification: must pass practical test sebelum solo cooking',
            'Blind taste test: periodic testing untuk identify deviations',
            'Supplier consistency: long-term contracts untuk stable ingredient quality'
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
        best_match, score = process.extractOne(
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

# ==============================================================================
# APLIKASI STREAMLIT
# ==============================================================================
def main():
    st.set_page_config(
        page_title="Sistem Rekomendasi Strategi - TOPSIS",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🍽️ Sistem Rekomendasi Strategi Restoran dengan TOPSIS")
    st.markdown("""
    Aplikasi ini menganalisis dataset pelanggan Anda dan memberikan rekomendasi strategi terbaik 
    menggunakan metode **TOPSIS** dengan bobot dari **Feature Importance Model ML**.
    """)
    
    # ==============================================================================
    # STEP 1: UPLOAD DATASET
    # ==============================================================================
    st.header("📤 Step 1: Upload Dataset Pelanggan")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload file CSV dataset pelanggan Anda",
            type=['csv'],
            help="Dataset harus memiliki kolom yang match dengan features model"
        )
    
    with col2:
        st.info("""
        **Format yang disupport:**
        - CSV file
        - Kolom harus match dengan features model
        """)
    
    if uploaded_file is None:
        st.info("👆 Upload dataset CSV untuk memulai analisis")
        
        # Tampilkan contoh format
        with st.expander("📋 Contoh Format Dataset", expanded=True):
            st.markdown("""
            Dataset Anda harus memiliki kolom yang match dengan features model. 
            Contoh kolom yang umum:
            
            **Service Quality:**
            - ServiceRating, FoodRating, AmbianceRating
            - WaitTime, AvgRating, TotalRating
            
            **Visit Behavior:**
            - VisitFrequency, AverageSpend, GroupSize
            - DiningOccasion, MealType
            
            **Customer Status:**
            - LoyaltyProgramMember, OnlineReservation
            - DeliveryOrder, LoyalCustomer
            """)
            
            example_df = pd.DataFrame({
                'CustomerID': [1, 2, 3],
                'ServiceRating': [4.5, 3.8, 4.2],
                'FoodRating': [4.0, 4.5, 3.9],
                'AmbianceRating': [4.2, 3.5, 4.0],
                'WaitTime': [15, 25, 10],
                'AverageSpend': [50000, 75000, 45000],
                'VisitFrequency': [5, 12, 3],
                'LoyaltyProgramMember': [1, 1, 0]
            })
            st.dataframe(example_df, use_container_width=True)
        
        st.stop()
    
    # Load dataset
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset berhasil dimuat: {df.shape[0]} baris × {df.shape[1]} kolom")
        
        with st.expander("👁️ Preview Dataset", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**Kolom ({len(df.columns)}):** {', '.join(df.columns.tolist())}")
    
    except Exception as e:
        st.error(f"❌ Error membaca file: {str(e)}")
        st.stop()
    
    # ==============================================================================
    # STEP 2: LOAD MODEL & MAPPING FEATURES
    # ==============================================================================
    st.header("🔗 Step 2: Mapping Kolom Dataset ke Features Model")
    
    model, feature_names = load_model_features("model_satisfied_v2.pkl", "feature_names.pkl")
    
    if model is None or feature_names is None:
        st.error("❌ Model atau feature names tidak dapat dimuat. Pastikan file tersedia.")
        st.stop()
    
    st.success(f"✅ Model dimuat: **{len(feature_names)} features** di model")
    
    # Mapping
    MIN_FEATURES = 5
    is_valid, message, matched_features, num_matched, mapping_detail, df_final = map_dataset_to_features(
        df, feature_names, MIN_FEATURES
    )
    
    # Tampilkan hasil mapping
    st.info(message)
    
    if not is_valid:
        st.error("Dataset tidak memenuhi kriteria minimum. Silakan upload dataset yang lebih lengkap.")
        st.stop()
    
    # Tampilkan mapping per kategori
    st.subheader("📋 Hasil Mapping Features")
    
    feature_metadata = get_feature_metadata()
    
    # Group by category
    categories = {}
    for feat in matched_features:
        # Cari feature name dari mapping_detail
        model_feat = None
        for mf, df_col in mapping_detail.items():
            if df_col == feat:
                model_feat = mf
                break
        
        if model_feat and model_feat in feature_metadata:
            cat = feature_metadata[model_feat]['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                'model_feature': model_feat,
                'dataset_column': feat,
                'type': feature_metadata[model_feat]['type'],
                'desc': feature_metadata[model_feat]['desc']
            })
    # Tampilkan per kategori
    for cat, features in categories.items():
        with st.expander(f"📁 {cat} ({len(features)} features)", expanded=True):
            cat_df = pd.DataFrame(features)
            st.dataframe(cat_df, use_container_width=True)

    # ==============================================================================
    # STEP 3: ANALISIS FEATURE IMPORTANCE
    # ==============================================================================
    st.header("📊 Step 3: Analisis Feature Importance")

    try:
        feature_importances = pd.Series(model.feature_importances_, index=feature_names)
        matched_importances = feature_importances[[mf for mf in mapping_detail.keys() if mf in feature_names]]
        
        # Normalize
        matched_importances = matched_importances / matched_importances.sum()
        
        # Tampilkan top features
        top_n = min(15, len(matched_importances))
        top_features = matched_importances.nlargest(top_n)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Top {top_n} Features (Normalized Importance)")
            
            import plotly.graph_objects as go
            
            fig = go.Figure(go.Bar(
                x=top_features.values,
                y=top_features.index,
                orientation='h',
                marker=dict(
                    color=top_features.values,
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Normalized Importance",
                yaxis_title="Features",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Feature Importance Table")
            importance_df = pd.DataFrame({
                'Feature': top_features.index,
                'Importance': top_features.values
            }).reset_index(drop=True)
            importance_df['Importance'] = importance_df['Importance'].apply(lambda x: f"{x:.4f}")
            st.dataframe(importance_df, use_container_width=True, height=500)
        
        # Simpan sebagai dict
        feature_importance_dict = matched_importances.to_dict()
        
    except Exception as e:
        st.error(f"❌ Error mengambil feature importance: {str(e)}")
        st.stop()

    # ==============================================================================
    # STEP 4: BUILD TOPSIS MATRIX & CALCULATE
    # ==============================================================================
    st.header("🎯 Step 4: Analisis TOPSIS & Rekomendasi Strategi")

    strategy_mapping = get_strategy_feature_mapping()

    # Build TOPSIS matrix
    decision_matrix, weights, criteria_types = build_topsis_matrix(
        list(matched_importances.index),
        feature_importance_dict,
        strategy_mapping
    )

    if decision_matrix is None:
        st.error("❌ Tidak ada strategi yang cocok dengan features yang terdeteksi.")
        st.stop()

    st.success(f"✅ {len(decision_matrix)} strategi yang relevan ditemukan!")

    # Tampilkan decision matrix
    with st.expander("📊 Decision Matrix", expanded=False):
        st.dataframe(decision_matrix.style.format("{:.4f}"), use_container_width=True)

    # Calculate TOPSIS
    try:
        topsis_results = calculate_topsis(decision_matrix, weights, criteria_types)
        
        st.subheader("🏆 Hasil Ranking Strategi")
        
        # Format hasil
        results_display = topsis_results.copy()
        results_display['Closeness_Score'] = results_display['Closeness_Score'].apply(lambda x: f"{x:.4f}")
        
        # Add description
        results_display['Description'] = results_display.index.map(
            lambda x: strategy_mapping[x]['description']
        )
        
        st.dataframe(
            results_display.style.highlight_max(subset=['Rank'], color='lightgreen'),
            use_container_width=True
        )
        
        # Visualisasi
        st.subheader("📈 Visualisasi Hasil")
        
        import plotly.graph_objects as go
        
        # Sort by rank
        topsis_sorted = topsis_results.sort_values('Rank')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=topsis_sorted['Closeness_Score'],
            y=topsis_sorted.index,
            orientation='h',
            marker=dict(
                color=topsis_sorted['Closeness_Score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            text=topsis_sorted['Rank'].apply(lambda x: f"Rank {x}"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="TOPSIS Closeness Score per Strategi",
            xaxis_title="Closeness Score",
            yaxis_title="Strategi",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 3 Recommendations
        st.subheader("💡 Top 3 Rekomendasi Strategi")
        
        top_3 = topsis_results.head(3)
        
        for idx, (strategy, row) in enumerate(top_3.iterrows(), 1):
            with st.expander(f"🥇 Rank {idx}: {strategy}", expanded=(idx==1)):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Closeness Score", f"{row['Closeness_Score']:.4f}")
                    st.metric("Rank", f"#{int(row['Rank'])}")
                
                with col2:
                    st.write("**Deskripsi:**")
                    st.info(strategy_mapping[strategy]['description'])

                    st.write("**Langkah Implementasi:**")
                    implementation_steps = strategy_mapping[strategy]['implementation']
                    formatted_steps = '\n* '.join(implementation_steps)
                    st.markdown('* ' + formatted_steps)
                    
                    st.write("**Features yang Relevan:**")
                    strategy_features = strategy_mapping[strategy]['features']
                    matched_strategy_features = {f: w for f, w in strategy_features.items() 
                                                if f in matched_importances.index}
                    
                    if matched_strategy_features:
                        feat_df = pd.DataFrame([
                            {
                                'Feature': f,
                                'Strategy Weight': f"{w:.3f}",
                                'Feature Importance': f"{feature_importance_dict[f]:.4f}",
                                'Combined Score': f"{w * feature_importance_dict[f]:.4f}"
                            }
                            for f, w in sorted(matched_strategy_features.items(), 
                                            key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(feat_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error menghitung TOPSIS: {str(e)}")
        st.stop()

    # ==============================================================================
    # STEP 5: EXPORT RESULTS
    # ==============================================================================
    st.header("💾 Step 5: Export Hasil")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export TOPSIS results
        csv_topsis = topsis_results.to_csv()
        st.download_button(
            label="📥 Download TOPSIS Results (CSV)",
            data=csv_topsis,
            file_name="topsis_results.csv",
            mime="text/csv"
        )

    with col2:
        # Export decision matrix
        csv_matrix = decision_matrix.to_csv()
        st.download_button(
            label="📥 Download Decision Matrix (CSV)",
            data=csv_matrix,
            file_name="decision_matrix.csv",
            mime="text/csv"
        )

    with col3:
        # Export feature mapping
        mapping_df = pd.DataFrame([
            {'Model Feature': k, 'Dataset Column': v}
            for k, v in mapping_detail.items()
        ])
        csv_mapping = mapping_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Mapping (CSV)",
            data=csv_mapping,
            file_name="feature_mapping.csv",
            mime="text/csv"
        )
    
if __name__ == '__main__':
    main()