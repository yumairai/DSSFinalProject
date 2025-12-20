import streamlit as st
import pandas as pd

from topsis_utils import calculate_topsis
from data_mapping import (
    load_model_features,
    get_feature_metadata,
    get_strategy_feature_mapping,
    map_dataset_to_features,
    build_topsis_matrix
)
from ui_components import (
    set_page_style,
    show_progress_indicator,
    create_metric_card,
    show_user_guide,
    show_summary_stats,
    show_about
)

def main():
    st.set_page_config(
        page_title="SISTEM REKOMENDASI STRATEGI PENINGKATAN KEPUASAN PELANGGAN RESTORAN",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    set_page_style()
    
    # ==============================================================================
    # SIDEBAR - NAVIGATION & INFO
    # ==============================================================================
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: white; font-size: 48px;'>🍽️</h1>
            <h2 style='color: white;'>Restaurant Strategy</h2>
            <p style='color: rgba(255,255,255,0.8);'>Recommendation System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "📍 Navigation",
            ["🏠 Home", "📖 User Guide", "📊 Analysis Dashboard", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Stats (jika ada data yang di-upload)
        if 'df' in st.session_state:
            st.markdown("""
            <div style='background: rgba(255,255,255,0.1); padding: 15px; 
                        border-radius: 10px; color: white;'>
                <h4>📈 Quick Stats</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Data Rows", f"{len(st.session_state.df):,}")
            st.metric("Columns", f"{len(st.session_state.df.columns)}")
            
            if 'num_matched' in st.session_state:
                st.metric("Matched Features", f"{st.session_state.num_matched}")
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style='color: rgba(255,255,255,0.6); font-size: 12px; text-align: center;'>
            <p>Powered by TOPSIS & ML</p>
            <p>Version 2.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==============================================================================
    # MAIN CONTENT AREA
    # ==============================================================================
    
    # HOME PAGE
    if page == "🏠 Home":
        # Header
        st.markdown("""
        <div style='text-align: center; padding: 40px 0;'>
            <h1 style='font-size: 48px; margin-bottom: 10px;'>
                🍽️ SISTEM REKOMENDASI STRATEGI PENINGKATAN KEPUASAN PELANGGAN RESTORAN
            </h1>
            <p style='font-size: 20px; color: rgba(255,255,255,0.9);'>
                Analisis Data Pelanggan & Dapatkan Strategi Bisnis Terbaik
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show progress indicator
        current_step = 0
        if 'df' in st.session_state:
            current_step = 1
        if 'mapping_done' in st.session_state:
            current_step = 2
        if 'analysis_done' in st.session_state:
            current_step = 3
        if 'results_done' in st.session_state:
            current_step = 4
        
        show_progress_indicator(current_step)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ==============================================================================
        # STEP 1: UPLOAD DATASET
        # ==============================================================================
        with st.container():
            st.markdown("""
            <div style='background: white; padding: 30px; border-radius: 15px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 30px;'>
                <h2 style='color: #667eea;'>📤 Step 1: Upload Dataset Pelanggan</h2>
                <p style='color: #6b7280;'>
                    Upload file CSV yang berisi data pelanggan restoran Anda. Sistem akan otomatis 
                    mencocokkan kolom dengan features yang dibutuhkan.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Pilih file CSV Anda",
                    type=['csv'],
                    help="Upload file CSV dengan data pelanggan"
                )
            
            with col2:
                st.markdown("""
                <div style='background: #f0f9ff; padding: 20px; border-radius: 10px; 
                            border-left: 4px solid #667eea;'>
                    <strong style='color: #667eea;'>💡 Tips:</strong><br>
                    <ul style='color: #4b5563; font-size: 14px; margin-top: 10px;'>
                        <li>Format: CSV</li>
                        <li>Minimal 5 kolom match</li>
                        <li>Data bersih tanpa missing values banyak</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        if uploaded_file is None:
            # Show example dataset
            with st.expander("📋 Lihat Contoh Format Dataset", expanded=True):
                st.markdown("""
                <p style='color: #4b5563; margin-bottom: 15px;'>
                    Berikut contoh struktur data yang ideal untuk analisis:
                </p>
                """, unsafe_allow_html=True)
                
                example_df = pd.DataFrame({
                    'CustomerID': [1, 2, 3, 4, 5],
                    'ServiceRating': [4.5, 3.8, 4.2, 5.0, 3.5],
                    'FoodRating': [4.0, 4.5, 3.9, 4.8, 4.0],
                    'AmbianceRating': [4.2, 3.5, 4.0, 4.5, 3.8],
                    'WaitTime': [15, 25, 10, 12, 30],
                    'AverageSpend': [50000, 75000, 45000, 95000, 40000],
                    'VisitFrequency': [5, 12, 3, 8, 2],
                    'LoyaltyProgramMember': [1, 1, 0, 1, 0]
                })
                st.dataframe(example_df, use_container_width=True)
                
                # Download example
                csv_example = example_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Contoh Dataset",
                    csv_example,
                    "example_dataset.csv",
                    "text/csv"
                )
            
            st.stop()
        
        # Load dataset
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
            st.session_state.df = df

            st.success("✅ Dataset berhasil dimuat!")

        except Exception as e:
            st.error(f"❌ Error membaca file CSV: {e}")
            st.stop()



        with st.expander("👁️ Preview Dataset", expanded=True):
            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.markdown(
                f"""
                <div style="margin-top: 15px; color: #6b7280;">
                    <strong>Kolom yang terdeteksi ({len(df.columns)}):</strong><br>
                    {', '.join(map(str, df.columns))}
                </div>
                """,
                unsafe_allow_html=True
            )

        
        # ==============================================================================
        # STEP 2: MAPPING FEATURES
        # ==============================================================================
        with st.container():
            st.markdown("""
            <div style='background: white; padding: 30px; border-radius: 15px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 30px;'>
                <h2 style='color: #667eea;'>🔗 Step 2: Mapping Features</h2>
                <p style='color: #6b7280;'>
                    Sistem akan otomatis mencocokkan kolom dataset Anda dengan features model ML.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        model, feature_names = load_model_features("model_satisfied_v2.pkl", "feature_names.pkl")
        
        if model is None or feature_names is None:
            st.error("❌ Model tidak dapat dimuat. Pastikan file model tersedia.")
            st.stop()
        
        with st.spinner("🔄 Sedang melakukan mapping features..."):
            MIN_FEATURES = 5
            is_valid, message, matched_features, num_matched, mapping_detail, df_final = map_dataset_to_features(
                df, feature_names, MIN_FEATURES
            )
        
        st.session_state.num_matched = num_matched
        st.session_state.mapping_done = True
        
        # Show mapping results
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            create_metric_card(
                "Total Features Model",
                f"{len(feature_names)}",
                icon="🎯"
            )
        
        with col2:
            create_metric_card(
                "Features Matched",
                f"{num_matched}",
                icon="✅"
            )
        
        with col3:
            match_rate = (num_matched / len(feature_names) * 100)
            icon = "🎉" if match_rate >= 50 else "⚠️" if match_rate >= 30 else "❌"
            create_metric_card(
                "Match Rate",
                f"{match_rate:.1f}%",
                icon=icon
            )
        
        if not is_valid:
            st.error("""
            ❌ **Dataset tidak memenuhi kriteria minimum!**
            
            Dataset Anda hanya memiliki {} features yang cocok, minimal {} features diperlukan.
            
            **Saran:**
            - Pastikan nama kolom sesuai dengan standar (ServiceRating, WaitTime, dll)
            - Tambahkan kolom yang relevan dengan bisnis restoran
            - Lihat contoh format dataset di bagian User Guide
            """.format(num_matched, MIN_FEATURES))
            st.stop()
        
        st.success(message)
        
        # Detail mapping per kategori
        with st.expander("📊 Detail Mapping per Kategori", expanded=True):
            feature_metadata = get_feature_metadata()
            
            categories = {}
            for feat in matched_features:
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
                        'Model Feature': model_feat,
                        'Dataset Column': feat,
                        'Type': feature_metadata[model_feat]['type'],
                        'Description': feature_metadata[model_feat]['desc']
                    })
            
            # Tampilkan dengan tabs
            cat_tabs = st.tabs(list(categories.keys()))
            
            for tab, (cat_name, features) in zip(cat_tabs, categories.items()):
                with tab:
                    cat_df = pd.DataFrame(features)
                    st.dataframe(
                        cat_df.style.applymap(
                            lambda x: 'background-color: #d4edda' if x == 'Benefit' else 'background-color: #fff3cd',
                            subset=['Type']
                        ),
                        use_container_width=True
                    )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ==============================================================================
        # STEP 3: FEATURE IMPORTANCE ANALYSIS
        # ==============================================================================
        with st.container():
            st.markdown("""
            <div style='background: white; padding: 30px; border-radius: 15px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 30px;'>
                <h2 style='color: #667eea;'>📊 Step 3: Analisis Feature Importance</h2>
                <p style='color: #6b7280;'>
                    Melihat feature mana yang paling berpengaruh terhadap kepuasan pelanggan.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        try:
            feature_importances = pd.Series(model.feature_importances_, index=feature_names)
            matched_importances = feature_importances[[mf for mf in mapping_detail.keys() if mf in feature_names]]
            matched_importances = matched_importances / matched_importances.sum()
            
            st.session_state.analysis_done = True
            
            top_n = min(10, len(matched_importances))
            top_features = matched_importances.nlargest(top_n)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                import plotly.graph_objects as go
                
                fig = go.Figure(go.Bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    marker=dict(
                        color=top_features.values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    ),
                    text=[f'{v:.1%}' for v in top_features.values],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f"🏆 Top {top_n} Most Important Features",
                    xaxis_title="Normalized Importance",
                    yaxis_title="Features",
                    height=500,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div style='background: #f8f9ff; padding: 20px; border-radius: 10px;'>
                    <h4 style='color: #667eea;'>💡 Insight</h4>
                    <p style='color: #4b5563; font-size: 14px; line-height: 1.6;'>
                        Feature dengan importance tertinggi adalah faktor yang paling 
                        mempengaruhi kepuasan pelanggan di restoran Anda.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 3 features dengan penjelasan
                st.markdown("<br>", unsafe_allow_html=True)
                for idx, (feat, imp) in enumerate(top_features.head(3).items(), 1):
                    st.markdown(f"""
                    <div style='background: white; padding: 15px; border-radius: 8px; 
                                margin-bottom: 10px; border-left: 4px solid #667eea;'>
                        <strong style='color: #667eea;'>#{idx} {feat}</strong><br>
                        <span style='color: #10b981; font-size: 18px; font-weight: bold;'>
                            {imp:.1%}
                        </span>
                        <span style='color: #6b7280; font-size: 12px;'> importance</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            feature_importance_dict = matched_importances.to_dict()
            
        except Exception as e:
            st.error(f"❌ Error menganalisis feature importance: {str(e)}")
            st.stop()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ==============================================================================
        # STEP 4: TOPSIS ANALYSIS & RECOMMENDATIONS
        # ==============================================================================
        with st.container():
            st.markdown("""
            <div style='background: white; padding: 30px; border-radius: 15px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 30px;'>
                <h2 style='color: #667eea;'>🎯 Step 4: Analisis TOPSIS & Rekomendasi</h2>
                <p style='color: #6b7280;'>
                    Mendapatkan ranking strategi bisnis terbaik menggunakan metode TOPSIS.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        strategy_mapping = get_strategy_feature_mapping()
        
        with st.spinner("🔄 Menghitung ranking strategi..."):
            decision_matrix, weights, criteria_types = build_topsis_matrix(
                list(matched_importances.index),
                feature_importance_dict,
                strategy_mapping
            )
        
        if decision_matrix is None:
            st.error("❌ Tidak ada strategi yang cocok dengan features yang terdeteksi.")
            st.stop()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card(
                "Total Strategi",
                f"{len(decision_matrix)}",
                icon="🎯"
            )
        
        with col2:
            create_metric_card(
                "Kriteria Evaluasi",
                f"{len(weights)}",
                icon="📊"
            )
        
        with col3:
            create_metric_card(
                "Features Analyzed",
                f"{num_matched}",
                icon="✅"
            )
        
        # Calculate TOPSIS
        try:
            topsis_results = calculate_topsis(decision_matrix, weights, criteria_types)
            st.session_state.results_done = True
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Visualization
            st.markdown("""
            <h3 style='color: #667eea;'>📈 Ranking Strategi</h3>
            """, unsafe_allow_html=True)
            
            import plotly.graph_objects as go
            
            topsis_sorted = topsis_results.sort_values('Rank').head(10)
            
            fig = go.Figure()
            
            colors = ['#10b981' if i < 3 else '#667eea' for i in range(len(topsis_sorted))]
            
            fig.add_trace(go.Bar(
                x=topsis_sorted['Closeness_Score'],
                y=topsis_sorted.index,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                text=[f"Rank {r}" for r in topsis_sorted['Rank']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="🏆 Top 10 Strategi Rekomendasi",
                xaxis_title="Closeness Score (semakin tinggi semakin baik)",
                yaxis_title="",
                height=600,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # TOP 3 RECOMMENDATIONS dengan style yang lebih menarik
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <h3 style='color: #667eea;'>🏆 Top 3 Rekomendasi Strategi Terbaik</h3>
            <p style='color: #6b7280;'>Strategi-strategi ini paling cocok dengan profil pelanggan Anda</p>
            """, unsafe_allow_html=True)
            
            top_3 = topsis_results.head(3)
            
            medals = ["🥇", "🥈", "🥉"]
            colors_top = ["#ffd700", "#c0c0c0", "#cd7f32"]
            
            for idx, ((strategy, row), medal, color) in enumerate(zip(top_3.iterrows(), medals, colors_top)):
                with st.expander(f"{medal} Rank {idx+1}: {strategy}", expanded=(idx==0)):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                                    padding: 30px; border-radius: 15px; text-align: center; color: white;'>
                            <div style='font-size: 48px;'>{medal}</div>
                            <div style='font-size: 32px; font-weight: bold; margin: 15px 0;'>
                                {row['Closeness_Score']:.4f}
                            </div>
                            <div style='font-size: 14px;'>Closeness Score</div>
                            <div style='font-size: 48px; font-weight: bold; margin-top: 20px;'>
                                #{int(row['Rank'])}
                            </div>
                            <div style='font-size: 14px;'>Ranking</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### 📋 Deskripsi Strategi")
                        st.info(strategy_mapping[strategy]['description'])
                        
                        st.markdown("#### 🎯 Langkah Implementasi")
                        implementation_steps = strategy_mapping[strategy]['implementation']
                        
                        for step_idx, step in enumerate(implementation_steps, 1):
                            st.markdown(f"""
                            <div style='background: #f8f9ff; padding: 10px; border-radius: 5px; 
                                        margin-bottom: 8px; border-left: 3px solid #667eea;'>
                                <strong style='color: #667eea;'>{step_idx}.</strong> 
                                <span style='color: #4b5563;'>{step}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("#### 📊 Features yang Relevan")
                        strategy_features = strategy_mapping[strategy]['features']
                        matched_strategy_features = {f: w for f, w in strategy_features.items() 
                                                    if f in matched_importances.index}
                        
                        if matched_strategy_features:
                            feat_df = pd.DataFrame([
                                {
                                    'Feature': f,
                                    'Strategy Weight': f"{w:.3f}",
                                    'Model Importance': f"{feature_importance_dict[f]:.4f}",
                                    'Combined Score': f"{w * feature_importance_dict[f]:.4f}"
                                }
                                for f, w in sorted(matched_strategy_features.items(), 
                                                key=lambda x: x[1] * feature_importance_dict[x[0]], reverse=True)
                            ][:5])
                            
                            st.dataframe(
                                feat_df.style.background_gradient(subset=['Combined Score'], cmap='Greens'),
                                use_container_width=True
                            )
        
        except Exception as e:
            st.error(f"❌ Error menghitung TOPSIS: {str(e)}")
            st.stop()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ==============================================================================
        # STEP 5: EXPORT RESULTS
        # ==============================================================================
        with st.container():
            st.markdown("""
            <div style='background: white; padding: 30px; border-radius: 15px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 30px;'>
                <h2 style='color: #667eea;'>💾 Step 5: Export Hasil Analisis</h2>
                <p style='color: #6b7280;'>
                    Download hasil analisis untuk dokumentasi dan presentasi.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div style='background: #f0f9ff; padding: 20px; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 48px; margin-bottom: 10px;'>📊</div>
                    <h4 style='color: #667eea;'>TOPSIS Results</h4>
                    <p style='color: #6b7280; font-size: 14px;'>Ranking lengkap semua strategi</p>
                </div>
                """, unsafe_allow_html=True)
                
                csv_topsis = topsis_results.to_csv()
                st.download_button(
                    "⬇️ Download CSV",
                    csv_topsis,
                    "topsis_results.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("""
                <div style='background: #f0fdf4; padding: 20px; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 48px; margin-bottom: 10px;'>🎯</div>
                    <h4 style='color: #10b981;'>Decision Matrix</h4>
                    <p style='color: #6b7280; font-size: 14px;'>Matrix keputusan TOPSIS</p>
                </div>
                """, unsafe_allow_html=True)
                
                csv_matrix = decision_matrix.to_csv()
                st.download_button(
                    "⬇️ Download CSV",
                    csv_matrix,
                    "decision_matrix.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col3:
                st.markdown("""
                <div style='background: #fef3c7; padding: 20px; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 48px; margin-bottom: 10px;'>🔗</div>
                    <h4 style='color: #f59e0b;'>Feature Mapping</h4>
                    <p style='color: #6b7280; font-size: 14px;'>Pemetaan kolom dataset</p>
                </div>
                """, unsafe_allow_html=True)
                
                mapping_df = pd.DataFrame([
                    {'Model Feature': k, 'Dataset Column': v}
                    for k, v in mapping_detail.items()
                ])
                csv_mapping = mapping_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    csv_mapping,
                    "feature_mapping.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    # ==============================================================================
    # USER GUIDE PAGE
    # ==============================================================================
    elif page == "📖 User Guide":
        st.markdown("<br>", unsafe_allow_html=True)
        show_user_guide()
    
    # ==============================================================================
    # ANALYSIS DASHBOARD PAGE
    # ==============================================================================
    elif page == "📊 Analysis Dashboard":
        st.markdown("""
        <div style='text-align: center; padding: 40px 0;'>
            <h1 style='font-size: 48px;'>📊 Analysis Dashboard</h1>
            <p style='font-size: 20px; color: rgba(255,255,255,0.9);'>
                Summary lengkap dari hasil analisis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'df' not in st.session_state:
            st.warning("⚠️ Belum ada data yang di-upload. Silakan upload dataset terlebih dahulu di halaman Home.")
            st.stop()
        
        # Summary Statistics
        st.markdown("""
        <div style='background: white; padding: 30px; border-radius: 15px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 30px;'>
            <h2 style='color: #667eea;'>📈 Summary Statistics</h2>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.df
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Total Records", f"{len(df):,}", icon="📊")
        
        with col2:
            create_metric_card("Total Columns", f"{len(df.columns)}", icon="📋")
        
        with col3:
            if 'num_matched' in st.session_state:
                create_metric_card("Matched Features", f"{st.session_state.num_matched}", icon="✅")
            else:
                create_metric_card("Matched Features", "N/A", icon="❓")
        
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            create_metric_card("Missing Data", f"{missing_pct:.1f}%", icon="⚠️")
        
        # Data Quality Check
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: white; padding: 30px; border-radius: 15px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 30px;'>
            <h2 style='color: #667eea;'>🔍 Data Quality Check</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Missing Values per Column")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing': df.isnull().sum().values,
                'Percentage': (df.isnull().sum().values / len(df) * 100)
            }).sort_values('Missing', ascending=False)
            
            missing_df = missing_df[missing_df['Missing'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(
                    missing_df.style.background_gradient(subset=['Percentage'], cmap='Reds'),
                    use_container_width=True
                )
            else:
                st.success("✅ Tidak ada missing values!")
        
        with col2:
            st.markdown("#### Data Types")
            dtype_counts = df.dtypes.value_counts()
            
            import plotly.graph_objects as go
            
            fig = go.Figure(go.Pie(
                labels=dtype_counts.index.astype(str),
                values=dtype_counts.values,
                hole=0.4
            ))
            
            fig.update_layout(
                title="Distribution of Data Types",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Descriptive Statistics
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Descriptive Statistics", expanded=False):
            st.dataframe(df.describe(), use_container_width=True)
    
    # ==============================================================================
    # ABOUT PAGE
    # ==============================================================================
    elif page == "ℹ️ About":
        st.markdown("<br>", unsafe_allow_html=True)
        show_about()

if __name__ == '__main__':
    main()
