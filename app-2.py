import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from rapidfuzz import fuzz, process

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
    """
    return {
        "A1: Tingkatkan Kualitas Layanan": {
            'features': {
                'ServiceRating': 0.30,
                'WaitTime': 0.25,
                'LongWait': 0.15,
                'WaitToService': 0.15,
                'Wait_x_Service': 0.10,
                'ConsistentQuality': 0.05
            },
            'description': 'Fokus pada peningkatan kualitas layanan dan pengurangan waktu tunggu'
        },
        "A2: Optimalkan Kualitas Makanan": {
            'features': {
                'FoodRating': 0.40,
                'AvgRating': 0.20,
                'TotalRating': 0.15,
                'RatingStd': 0.15,
                'ConsistentQuality': 0.10
            },
            'description': 'Fokus pada peningkatan dan konsistensi kualitas makanan'
        },
        "A3: Percepat Waktu Penyajian": {
            'features': {
                'WaitTime': 0.35,
                'LongWait': 0.25,
                'WaitToService': 0.20,
                'Wait_x_Service': 0.15,
                'ServiceRating': 0.05
            },
            'description': 'Fokus pada efisiensi operasional dan percepatan penyajian'
        },
        "A4: Program Loyalty & Retention": {
            'features': {
                'LoyaltyProgramMember': 0.25,
                'LoyalCustomer': 0.25,
                'VisitFrequency': 0.20,
                'FrequentVisitor': 0.15,
                'Rating_x_Loyalty': 0.10,
                'Rating_x_Frequency': 0.05
            },
            'description': 'Fokus pada peningkatan loyalitas dan retensi pelanggan'
        },
        "A5: Perbaiki Ambience & Kebersihan": {
            'features': {
                'AmbianceRating': 0.40,
                'AvgRating': 0.25,
                'TotalRating': 0.15,
                'RatingRange': 0.10,
                'ConsistentQuality': 0.10
            },
            'description': 'Fokus pada peningkatan suasana dan kenyamanan restoran'
        },
        "A6: Strategi Pricing & Value": {
            'features': {
                'AverageSpend': 0.25,
                'SpendPerPerson': 0.20,
                'SpendToIncomeRatio': 0.20,
                'HighSpender': 0.15,
                'Spend_x_Rating': 0.15,
                'Income': 0.05
            },
            'description': 'Fokus pada optimasi harga dan nilai yang diberikan'
        },
        "B1: Program Diskon Demografi": {
            'features': {
                'Age': 0.30,
                'AgeGroup': 0.25,
                'YoungCustomer': 0.20,
                'SeniorCustomer': 0.15,
                'Income': 0.10
            },
            'description': 'Program diskon berdasarkan segmen demografi'
        },
        "B2: Strategi Pendapatan": {
            'features': {
                'Income': 0.30,
                'HighIncome': 0.25,
                'SpendToIncomeRatio': 0.20,
                'HighSpender': 0.15,
                'AverageSpend': 0.10
            },
            'description': 'Strategi berbasis profil pendapatan pelanggan'
        },
        "C1: Loyalty Frequent Visitor": {
            'features': {
                'VisitFrequency': 0.35,
                'FrequentVisitor': 0.30,
                'LoyalCustomer': 0.20,
                'Rating_x_Frequency': 0.15
            },
            'description': 'Program khusus untuk pengunjung sering'
        },
        "D1: Upselling & Cross-selling": {
            'features': {
                'AverageSpend': 0.30,
                'SpendPerPerson': 0.25,
                'HighSpender': 0.20,
                'Spend_x_Rating': 0.15,
                'GroupSize': 0.10
            },
            'description': 'Strategi meningkatkan nilai transaksi per kunjungan'
        },
        "E1: Queue Management": {
            'features': {
                'WaitTime': 0.40,
                'LongWait': 0.30,
                'WaitToService': 0.20,
                'OnlineReservation': 0.10
            },
            'description': 'Optimasi sistem antrian dan reservasi'
        },
        "F1: Konsistensi Kualitas": {
            'features': {
                'RatingStd': 0.30,
                'RatingRange': 0.25,
                'ConsistentQuality': 0.20,
                'AvgRating': 0.15,
                'MaxRating': 0.05,
                'MinRating': 0.05
            },
            'description': 'Peningkatan konsistensi kualitas layanan dan produk'
        },
        "G1: Digital Experience": {
            'features': {
                'OnlineReservation': 0.30,
                'OnlineUser': 0.25,
                'DeliveryOrder': 0.25,
                'LoyaltyProgramMember': 0.15,
                'FrequentVisitor': 0.05
            },
            'description': 'Optimasi pengalaman digital dan online'
        },
        "H1: Group Dining Strategy": {
            'features': {
                'GroupSize': 0.40,
                'LargeGroup': 0.30,
                'Solo': 0.15,
                'AverageSpend': 0.10,
                'SpendPerPerson': 0.05
            },
            'description': 'Strategi khusus untuk kelompok makan'
        },
        "I1: Occasion-based Marketing": {
            'features': {
                'DiningOccasion': 0.35,
                'MealType': 0.30,
                'TimeOfVisit': 0.20,
                'GroupSize': 0.10,
                'AverageSpend': 0.05
            },
            'description': 'Marketing berbasis occasion dan waktu makan'
        },
        "J1: Customer Experience Recovery": {
            'features': {
                'Rating_x_Loyalty': 0.30,
                'Spend_x_Rating': 0.25,
                'RatingStd': 0.20,
                'ServiceRating': 0.15,
                'LoyalCustomer': 0.10
            },
            'description': 'Program pemulihan untuk pelanggan dengan pengalaman buruk'
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