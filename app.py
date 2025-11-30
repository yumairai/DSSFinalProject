import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
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
    
    # 1. Normalisasi Matriks Keputusan
    X = decision_matrix.values
    R = X / np.sqrt((X**2).sum(axis=0))

    # 2. Pembobotan Matriks Ternormalisasi
    V = R * np.array(weights)

    # 3. Penentuan Solusi Ideal Positif dan Negatif
    A_plus = np.zeros(V.shape[1])
    A_minus = np.zeros(V.shape[1])
    
    for j in range(V.shape[1]):
        if criteria_type[j].lower() == 'benefit':
            A_plus[j] = V[:, j].max()
            A_minus[j] = V[:, j].min()
        elif criteria_type[j].lower() == 'cost':
            A_plus[j] = V[:, j].min()
            A_minus[j] = V[:, j].max()

    # 4. Menghitung Jarak ke Solusi Ideal
    S_plus = np.sqrt(((V - A_plus)**2).sum(axis=1))
    S_minus = np.sqrt(((V - A_minus)**2).sum(axis=1))

    # 5. Menghitung Kedekatan Relatif
    Closeness = S_minus / (S_plus + S_minus)

    results_df = pd.DataFrame({
        'Strategy': decision_matrix.index,
        'Closeness_Score': Closeness,
        'Rank': Closeness.rank(method='dense', ascending=False).astype(int)
    }).sort_values(by='Closeness_Score', ascending=False).set_index('Strategy')
    
    return results_df

# ==============================================================================
# MAPPING FITUR KE KRITERIA BISNIS
# ==============================================================================

@st.cache_data
def get_topsis_weights(model_file: str, feature_names_file: str) -> Tuple[List[float], Dict[str, float]]:
    """
    Memuat model dan menghitung bobot kriteria TOPSIS.
    """
    try:
        model = joblib.load(model_file)
        feature_names = joblib.load(feature_names_file)
    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: {e.filename}")
        st.info("Pastikan file 'model_satisfied_v2.pkl' dan 'feature_names.pkl' ada di direktori yang sama.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error memuat file: {str(e)}")
        return None, None

    if len(model.feature_importances_) != len(feature_names):
        st.error("❌ Jumlah Feature Importance dan Feature Names tidak cocok.")
        return None, None

    feature_importances = pd.Series(model.feature_importances_, index=feature_names)

    # Definisi Mapping Kriteria
    criterion_mapping = {
        'C1_Rating_Consistency': ['AvgRating', 'FoodRating', 'AmbianceRating', 'ServiceRating', 'TotalRating', 'ConsistentQuality'],
        'C1_Rating_Cost': ['RatingStd', 'RatingRange', 'MaxRating', 'MinRating'], 
        'C2_Interaction_Loyalty': ['LoyaltyProgramMember', 'VisitFrequency', 'FrequentVisitor', 'LoyalCustomer', 'OnlineReservation', 'OnlineUser', 'Rating_x_Loyalty', 'Rating_x_Frequency'],
        'C3_Time_Efficiency_Cost': ['WaitTime', 'LongWait', 'WaitToService', 'Wait_x_Service'], 
        'C4_Financial': ['AverageSpend', 'SpendPerPerson', 'SpendToIncomeRatio', 'HighSpender', 'Spend_x_Rating'],
        'C5_Demography_Group': ['Age', 'Gender', 'Income', 'GroupSize', 'DiningOccasion', 'MealType', 'PreferredCuisine', 'DeliveryOrder', 'AgeGroup', 'YoungCustomer', 'SeniorCustomer', 'HighIncome', 'LargeGroup', 'Solo']
    }

    # Hitung skor untuk setiap kriteria dengan error handling
    try:
        c1_score = sum(feature_importances.get(f, 0) for f in criterion_mapping['C1_Rating_Consistency']) - \
                   sum(feature_importances.get(f, 0) for f in criterion_mapping['C1_Rating_Cost'])
        c2_score = sum(feature_importances.get(f, 0) for f in criterion_mapping['C2_Interaction_Loyalty'])
        c3_score = sum(feature_importances.get(f, 0) for f in criterion_mapping['C3_Time_Efficiency_Cost'])
        c4_score = sum(feature_importances.get(f, 0) for f in criterion_mapping['C4_Financial'])
        c5_score = sum(feature_importances.get(f, 0) for f in criterion_mapping['C5_Demography_Group'])
    except Exception as e:
        st.error(f"❌ Error menghitung skor kriteria: {str(e)}")
        return None, None
    
    # Normalisasi skor
    raw_scores = np.array([c1_score, c2_score, c3_score, c4_score, c5_score])
    min_score = raw_scores.min()
    positive_scores = raw_scores - min_score + 1e-6
    
    total_sum = positive_scores.sum()
    normalized_weights = positive_scores / total_sum
    
    weights_dict = {
        'C1: Rating & Konsistensi': normalized_weights[0],
        'C2: Interaksi & Loyalitas': normalized_weights[1],
        'C3: Efisiensi Waktu': normalized_weights[2],
        'C4: Profil Keuangan': normalized_weights[3],
        'C5: Demografi & Grup': normalized_weights[4]
    }
    
    return list(normalized_weights), weights_dict

# ==============================================================================
# APLIKASI STREAMLIT
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Sistem Rekomendasi Strategi Restoran",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🍽️ Sistem Rekomendasi Strategi Peningkatan Kepuasan Pelanggan")
    st.markdown("Sistem ini memadukan **Bobot Kriteria dari Model XGBoost** dengan **Skor Dampak Strategi** menggunakan metode TOPSIS.")

    # --------------------------------------------------------------------------
    # SIDEBAR: Bobot Kriteria
    # --------------------------------------------------------------------------
    st.sidebar.header("📊 Bobot Kriteria (dari Model)")
    
    topsis_weights, weights_dict = get_topsis_weights("model_satisfied_v2.pkl", "feature_names.pkl")
    
    if topsis_weights is None:
        st.stop()

    st.sidebar.success(f"✅ Total Bobot: {sum(topsis_weights):.4f}")
    st.sidebar.dataframe(
        pd.DataFrame(weights_dict.items(), columns=['Kriteria', 'Bobot']).set_index('Kriteria'),
        use_container_width=True
    )

    criteria_labels = list(weights_dict.keys())
    
    # Strategi Bisnis
    strategies = [
        "A1: Tingkatkan Kualitas Menu (Food Quality)",
        "A2: Pelatihan Staf Layanan (Service Quality)",
        "A3: Perbarui Suasana Restoran (Ambiance)",
        "A4: Investasi Sistem Pesanan (Wait Time)",
        "A5: Tingkatkan Program Loyalitas (Loyalty Program)"
    ]

    # --------------------------------------------------------------------------
    # INPUT SKOR DAMPAK STRATEGI
    # --------------------------------------------------------------------------
    st.header("1️⃣ Input Skor Dampak Strategi")
    st.info("📝 Berikan skor 1-5 untuk setiap strategi terhadap kriteria bisnis (5 = Sangat Efektif)")

    # Inisialisasi data
    if 'matrix_data' not in st.session_state:
        st.session_state.matrix_data = pd.DataFrame(
            3,  # Default value = 3
            index=strategies, 
            columns=criteria_labels
        )

    matrix_input = st.session_state.matrix_data.copy()
    
    # UI Input
    for i, strategy in enumerate(strategies):
        with st.expander(f"**{strategy}**", expanded=(i == 0)):
            cols = st.columns(len(criteria_labels))
            for j, criterion in enumerate(criteria_labels):
                matrix_input.loc[strategy, criterion] = cols[j].slider(
                    criterion.replace('C1: ', '').replace('C2: ', '').replace('C3: ', '').replace('C4: ', '').replace('C5: ', ''),
                    min_value=1, 
                    max_value=5, 
                    value=int(st.session_state.matrix_data.loc[strategy, criterion]),
                    key=f"slider_{i}_{j}"
                )
    
    st.session_state.matrix_data = matrix_input
    
    # --------------------------------------------------------------------------
    # TOMBOL HITUNG
    # --------------------------------------------------------------------------
    if st.button("🚀 Hitung Rekomendasi TOPSIS", type="primary", use_container_width=True):
        st.divider()
        
        st.header("2️⃣ Matriks Keputusan")
        st.dataframe(matrix_input, use_container_width=True)
        
        # Semua kriteria adalah 'Benefit'
        criteria_type = ['Benefit'] * len(criteria_labels)
        
        # Perhitungan TOPSIS
        with st.spinner("Menghitung..."):
            results_df = calculate_topsis(
                decision_matrix=matrix_input,
                weights=topsis_weights,
                criteria_type=criteria_type
            )
        
        st.header("3️⃣ Hasil Perangkingan")
        st.dataframe(
            results_df.style.format({'Closeness_Score': '{:.4f}'})
                          .background_gradient(subset=['Closeness_Score'], cmap='Greens'),
            use_container_width=True
        )
        
        best_strategy = results_df.index[0]
        best_score = results_df.loc[best_strategy, 'Closeness_Score']
        
        st.success(f"🏆 **Rekomendasi Strategi Utama:** {best_strategy}")
        st.metric("Skor TOPSIS", f"{best_score:.4f}", delta="Peringkat 1")
        
        with st.expander("ℹ️ Interpretasi Hasil"):
            st.markdown("""
            **Cara Membaca Hasil:**
            - **Rank 1** = Strategi dengan jarak terpendek dari **Solusi Ideal Positif (A+)** dan terjauh dari **Solusi Ideal Negatif (A-)**
            - **Closeness Score** berkisar 0-1, semakin mendekati 1 semakin baik
            - Peringkat menunjukkan strategi mana yang paling optimal setelah memperhitungkan bobot dari model prediksi kepuasan pelanggan
            """)

if __name__ == '__main__':
    main()