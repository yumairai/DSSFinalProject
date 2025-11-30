import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st

# ==============================================================================
# FUNGSI TOPSIS
# ==============================================================================

def calculate_topsis(decision_matrix: pd.DataFrame, weights: List[float], criteria_type: List[str]) -> pd.DataFrame:
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

    # Buat pandas Series agar bisa pakai .rank()
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

def map_dataset_to_features(df: pd.DataFrame, model_features: List[str], min_features: int = 5) -> Tuple[bool, str, List[str], int]:
    """
    Map kolom dataset ke features model
    Returns: (is_valid, message, matched_features, num_matched)
    """
    matched_features = [f for f in model_features if f in df.columns]
    num_matched = len(matched_features)
    
    if num_matched < min_features:
        message = f"❌ Dataset terlalu umum. Hanya {num_matched} dari {len(model_features)} features yang teridentifikasi. Minimal {min_features} features diperlukan."
        return False, message, matched_features, num_matched
    else:
        message = f"✅ Dataset valid! {num_matched} features teridentifikasi dari {len(model_features)} features model."
        return True, message, matched_features, num_matched

def get_feature_weights(model, feature_names: List[str], matched_features: List[str]) -> Dict[str, float]:
    """
    Ambil feature importance dari model untuk features yang matched
    """
    feature_importances = pd.Series(model.feature_importances_, index=feature_names)
    
    # Filter hanya matched features
    matched_importances = feature_importances[matched_features]
    
    # Normalisasi
    if matched_importances.sum() == 0:
        normalized = np.ones(len(matched_importances)) / len(matched_importances)
    else:
        normalized = matched_importances / matched_importances.sum()
    
    return dict(zip(matched_features, normalized))

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
    Aplikasi ini menganalisis dataset pelanggan Anda dan memberikan rekomendasi strategi 
    terbaik menggunakan metode **TOPSIS** dengan bobot dari **Feature Importance Model ML**.
    """)

    # Definisi Strategi
    STRATEGIES = [
        "A1: Tingkatkan Kualitas Layanan",
        "A2: Optimalkan Kualitas Makanan",
        "A3: Percepat Waktu Penyajian",
        "A4: Program Loyalty & Retention",
        "A5: Perbaiki Ambience & Kebersihan",
        "A6: Strategi Pricing & Value"
    ]

    # ==============================================================================
    # STEP 1: UPLOAD DATASET
    # ==============================================================================
    st.header("📤 Step 1: Upload Dataset Pelanggan")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload file CSV dataset pelanggan Anda",
            type=['csv'],
            help="Dataset harus memiliki kolom yang match dengan features model ML"
        )
    
    with col2:
        st.info("""
        **Format yang disupport:**
        - CSV file
        - Kolom harus match dengan features model
        """)

    if uploaded_file is not None:
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
        is_valid, message, matched_features, num_matched = map_dataset_to_features(
            df, feature_names, MIN_FEATURES
        )
        
        # Tampilkan mapping per kategori
        st.subheader("📋 Hasil Mapping Features")
        
        feature_metadata = get_feature_metadata()
        
        # Group by category
        categories = {}
        for feat in feature_names:
            if feat in feature_metadata:
                cat = feature_metadata[feat]['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(feat)
        
        col1, col2 = st.columns(2)
        
        for idx, (category, features) in enumerate(categories.items()):
            with (col1 if idx % 2 == 0 else col2):
                with st.expander(f"📁 **{category}**", expanded=False):
                    matched = [f for f in features if f in matched_features]
                    missing = [f for f in features if f not in matched_features]
                    
                    if matched:
                        st.success(f"✅ Matched ({len(matched)}):")
                        for f in matched:
                            desc = feature_metadata.get(f, {}).get('desc', '')
                            st.write(f"  • {f} - {desc}")
                    
                    if missing:
                        st.warning(f"⚠️ Not Found ({len(missing)}):")
                        st.write(", ".join(missing))
        
        # Summary
        st.metric("Total Features Matched", f"{num_matched} / {len(feature_names)}", 
                  delta=f"{(num_matched/len(feature_names)*100):.1f}%")
        
        # ==============================================================================
        # STEP 3: VALIDASI DATASET
        # ==============================================================================
        st.header("✅ Step 3: Validasi Dataset")
        
        if not is_valid:
            st.error(message)
            st.warning("""
            **Saran:**
            - Pastikan dataset memiliki minimal 5 kolom yang match dengan features model
            - Features yang umum: ServiceRating, FoodRating, WaitTime, AverageSpend, VisitFrequency
            - Upload dataset dengan kolom yang lebih relevan dengan model
            """)
            st.stop()
        
        st.success(message)
        
        # Hitung bobot dari model
        weights_dict = get_feature_weights(model, feature_names, matched_features)
        
        st.info("📊 Bobot features dihitung dari Feature Importance model ML")
        
        # Tampilkan top features
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("🔝 Top 10 Features (by Weight)")
            top_features = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            top_df = pd.DataFrame(top_features, columns=['Feature', 'Weight'])
            top_df['Weight (%)'] = (top_df['Weight'] * 100).round(2)
            st.dataframe(top_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Feature Types")
            type_counts = {'Benefit': 0, 'Cost': 0}
            for feat in matched_features:
                feat_type = feature_metadata.get(feat, {}).get('type', 'Benefit')
                type_counts[feat_type] += 1
            
            st.write(f"🟢 **Benefit Features**: {type_counts['Benefit']} (higher is better)")
            st.write(f"🔴 **Cost Features**: {type_counts['Cost']} (lower is better)")
            
            st.write("\n**Examples:**")
            st.write("• Benefit: ServiceRating, FoodRating, VisitFrequency")
            st.write("• Cost: WaitTime, SpendPerPerson, RatingStd")
        
        # ==============================================================================
        # STEP 4: INPUT SKOR STRATEGI
        # ==============================================================================
        st.header("🎯 Step 4: Input Skor Dampak Strategi terhadap Features")
        
        st.info(f"📝 Berikan skor 1-5 untuk dampak setiap strategi terhadap **{num_matched} features** yang teridentifikasi")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Reset Skor", use_container_width=True):
                if 'matrix_data' in st.session_state:
                    del st.session_state.matrix_data
                st.rerun()
        
        # Default matrix dengan skor realistis
        if 'matrix_data' not in st.session_state:
            default_matrix = []
            
            for strategy in STRATEGIES:
                row = {}
                for feature in matched_features:
                    # Logic default scoring berdasarkan strategy dan feature
                    if "Layanan" in strategy or "Service" in strategy:
                        if "Service" in feature or "Wait" in feature:
                            row[feature] = 5
                        elif "Rating" in feature:
                            row[feature] = 4
                        else:
                            row[feature] = 3
                    
                    elif "Makanan" in strategy or "Food" in strategy:
                        if "Food" in feature or "Rating" in feature:
                            row[feature] = 5
                        elif "Consistent" in feature:
                            row[feature] = 4
                        else:
                            row[feature] = 3
                    
                    elif "Penyajian" in strategy or "Waktu" in strategy:
                        if "Wait" in feature:
                            row[feature] = 5
                        elif "Service" in feature:
                            row[feature] = 4
                        else:
                            row[feature] = 2
                    
                    elif "Loyalty" in strategy:
                        if "Loyalty" in feature or "Frequency" in feature:
                            row[feature] = 5
                        elif "Visit" in feature:
                            row[feature] = 4
                        else:
                            row[feature] = 3
                    
                    elif "Ambience" in strategy or "Kebersihan" in strategy:
                        if "Ambiance" in feature or "Ambience" in feature:
                            row[feature] = 5
                        elif "Rating" in feature:
                            row[feature] = 4
                        else:
                            row[feature] = 3
                    
                    elif "Pricing" in strategy or "Value" in strategy:
                        if "Spend" in feature or "Price" in feature:
                            row[feature] = 5
                        elif "Income" in feature:
                            row[feature] = 4
                        else:
                            row[feature] = 3
                    
                    else:
                        row[feature] = 3
                
                default_matrix.append(row)
            
            st.session_state.matrix_data = pd.DataFrame(
                default_matrix,
                index=STRATEGIES,
                columns=matched_features
            )
        
        matrix_input = st.session_state.matrix_data.copy()
        
        # Input UI - Gunakan tabs untuk setiap strategi
        tabs = st.tabs([f"**{s.split(':')[0]}**" for s in STRATEGIES])
        
        for idx, (tab, strategy) in enumerate(zip(tabs, STRATEGIES)):
            with tab:
                st.write(f"### {strategy}")
                
                # Tampilkan dalam grid
                num_cols = 4
                features_list = list(matched_features)
                
                for i in range(0, len(features_list), num_cols):
                    cols = st.columns(num_cols)
                    for j, col in enumerate(cols):
                        if i + j < len(features_list):
                            feature = features_list[i + j]
                            short_name = feature[:20] + "..." if len(feature) > 20 else feature
                            
                            matrix_input.loc[strategy, feature] = col.slider(
                                short_name,
                                min_value=1,
                                max_value=5,
                                value=int(st.session_state.matrix_data.loc[strategy, feature]),
                                key=f"slider_{idx}_{i+j}",
                                help=f"Dampak {strategy} terhadap {feature}"
                            )
        
        st.session_state.matrix_data = matrix_input
        
        # ==============================================================================
        # STEP 5: HITUNG TOPSIS
        # ==============================================================================
        st.divider()
        
        if st.button("🚀 Hitung Rekomendasi TOPSIS", type="primary", use_container_width=True):
            st.header("📊 Step 5: Hasil Analisis TOPSIS")
            
            # Tampilkan matriks keputusan (hanya top features untuk readability)
            st.subheader("Matriks Keputusan (Top 15 Features by Weight)")
            top_15_features = [f[0] for f in sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)[:15]]
            display_matrix = matrix_input[top_15_features]
            
            st.dataframe(
                display_matrix.style.background_gradient(cmap='RdYlGn', vmin=1, vmax=5),
                use_container_width=True
            )
            
            # Tentukan tipe kriteria dan weights - HARUS URUTAN YANG SAMA
            criteria_types = [feature_metadata.get(f, {}).get('type', 'Benefit') for f in matched_features]
            weights_list = [weights_dict[f] for f in matched_features]
            
            # VALIDASI: Pastikan dimensi match
            assert matrix_input.shape[1] == len(matched_features), f"Matrix columns ({matrix_input.shape[1]}) != matched features ({len(matched_features)})"
            assert len(weights_list) == len(matched_features), f"Weights ({len(weights_list)}) != matched features ({len(matched_features)})"
            assert len(criteria_types) == len(matched_features), f"Criteria types ({len(criteria_types)}) != matched features ({len(matched_features)})"
            
            # Hitung TOPSIS
            with st.spinner("⏳ Menghitung skor TOPSIS..."):
                results_df = calculate_topsis(
                    decision_matrix=matrix_input,
                    weights=weights_list,
                    criteria_type=criteria_types
                )
            
            st.success("✅ Perhitungan selesai!")
            
            # Hasil
            st.subheader("🏆 Hasil Perangkingan Strategi")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    results_df.style.format({'Closeness_Score': '{:.4f}'})
                                  .background_gradient(subset=['Closeness_Score'], cmap='Greens'),
                    use_container_width=True
                )
            
            with col2:
                best_strategy = results_df.index[0]
                best_score = results_df.loc[best_strategy, 'Closeness_Score']
                
                st.metric(
                    "🥇 Strategi Terbaik",
                    f"Rank {results_df.loc[best_strategy, 'Rank']}",
                    delta=f"Score: {best_score:.4f}"
                )
                
                st.success(f"**{best_strategy}**")
                
                st.markdown("**Top 3 Strategi:**")
                for idx, (strat, row) in enumerate(results_df.head(3).iterrows(), 1):
                    medal = ["🥇", "🥈", "🥉"][idx-1]
                    st.write(f"{medal} {strat} ({row['Closeness_Score']:.4f})")
            
            # Rekomendasi Aksi
            st.divider()
            st.subheader("💡 Rekomendasi Aksi")
            
            top3 = results_df.head(3)
            
            recommendations = {
                "A1": "• Pelatihan staf customer service\n• Implementasi sistem feedback real-time\n• Optimasi proses reservasi",
                "A2": "• Tingkatkan kualitas bahan baku\n• Standardisasi resep dan penyajian\n• Kontrol kualitas berkelanjutan",
                "A3": "• Optimasi workflow dapur\n• Upgrade sistem POS\n• Manajemen antrian yang lebih baik",
                "A4": "• Program loyalty rewards\n• Personalisasi komunikasi\n• Exclusive member benefits",
                "A5": "• Renovasi interior\n• Protokol kebersihan ketat\n• Ambience lighting dan musik",
                "A6": "• Review pricing strategy\n• Program promosi targeted\n• Value meal packages"
            }
            
            for idx, (strat, row) in enumerate(top3.iterrows(), 1):
                strategy_id = strat.split(':')[0]
                with st.expander(f"**#{idx} - {strat}** (Score: {row['Closeness_Score']:.4f})", expanded=(idx==1)):
                    st.markdown(recommendations.get(strategy_id, "Implementasikan strategi ini dengan fokus pada features teridentifikasi."))
            
            # Interpretasi
            with st.expander("📖 Cara Membaca Hasil", expanded=False):
                st.markdown(f"""
                ### Interpretasi Skor TOPSIS:
                - **Closeness Score**: 0-1, semakin mendekati 1 semakin optimal
                - **Rank**: Urutan strategi dari terbaik ke kurang optimal
                
                ### Metodologi:
                1. Dataset dianalisis dan di-mapping ke {num_matched} features model
                2. Bobot dihitung dari feature importance model ML
                3. TOPSIS mengevaluasi jarak setiap strategi ke kondisi ideal
                4. Strategi dengan closeness score tertinggi = paling optimal
                
                ### Feature Types:
                - **Benefit** ({type_counts['Benefit']} features): Nilai tinggi lebih baik
                - **Cost** ({type_counts['Cost']} features): Nilai rendah lebih baik
                
                ### Features Teridentifikasi:
                {', '.join(matched_features[:10])}{"..." if len(matched_features) > 10 else ""}
                """)
    
    else:
        st.info("👆 Upload dataset CSV untuk memulai analisis")
        
        # Tampilkan contoh format
        with st.expander("📋 Contoh Format Dataset", expanded=False):
            st.markdown("""
            Dataset Anda harus memiliki kolom yang match dengan features model. Contoh kolom yang umum:
            
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

if __name__ == '__main__':
    main()