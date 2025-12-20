import streamlit as st
import pandas as pd

def set_page_style():
    """Custom CSS untuk styling yang lebih menarik"""
    st.markdown(
    """
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, rgb(102, 126, 234) 0%, rgb(118, 75, 162) 100%);
    }
    
    /* Card styling */
    .css-1r6slb0 {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #667eea;
    }
    
    /* Headers */
    h1 {
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2, h3 {
        color: #667eea;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, rgb(102, 126, 234) 0%, rgb(118, 75, 162) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    /* Progress indicator */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 20px 0;
        padding: 20px;
        background: white;
        border-radius: 10px;
    }
    
    .step {
        flex: 1;
        text-align: center;
        padding: 10px;
        position: relative;
    }
    
    .step.active {
        color: #667eea;
        font-weight: bold;
    }
    
    .step.completed {
        color: #10b981;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgb(102, 126, 234) 0%, rgb(118, 75, 162) 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9ff;
        border-radius: 5px;
    }
    </style>
    """
    , unsafe_allow_html=True)

def show_progress_indicator(current_step):
    """Tampilkan progress indicator untuk langkah-langkah"""
    steps = [
        ("📤", "Upload Data"),
        ("🔗", "Mapping"),
        ("📊", "Analysis"),
        ("🎯", "Results"),
        ("💾", "Export")
    ]
    
    cols = st.columns(len(steps))
    for idx, (col, (icon, label)) in enumerate(zip(cols, steps)):
        with col:
            if idx < current_step:
                st.markdown(f"""
                <div style='text-align: center; color: #10b981;'>
                    <div style='font-size: 32px;'>{icon}</div>
                    <div style='font-size: 12px; font-weight: bold;'>✓ {label}</div>
                </div>
                """, unsafe_allow_html=True)
            elif idx == current_step:
                st.markdown(f"""
                <div style='text-align: center; color: #667eea;'>
                    <div style='font-size: 32px;'>{icon}</div>
                    <div style='font-size: 12px; font-weight: bold;'>► {label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align: center; color: #9ca3af;'>
                    <div style='font-size: 32px;'>{icon}</div>
                    <div style='font-size: 12px;'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

def create_metric_card(label, value, delta=None, icon="📊"):
    """Buat metric card yang cantik"""
    delta_html = ""
    if delta:
        color = "#10b981" if delta > 0 else "#ef4444"
        delta_html = f"<div style='color: {color}; font-size: 14px;'>▲ {delta}</div>"
    
    st.markdown(f"""
    <div style='background: white; padding: 20px; border-radius: 10px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;'>
        <div style='font-size: 32px; margin-bottom: 10px;'>{icon}</div>
        <div style='color: #6b7280; font-size: 14px; margin-bottom: 5px;'>{label}</div>
        <div style='color: #667eea; font-size: 28px; font-weight: bold;'>{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def show_user_guide():
    st.markdown("""
    <div style="
        background:white;
        color:#374151;
        padding:32px;
        border-radius:16px;
        box-shadow:0 4px 12px rgba(0,0,0,0.1);
        line-height:1.8;
    ">

    <div style="margin-bottom:20px;">
        <span style="font-size:28px; font-weight:700; color:#667eea;">
            📖 Panduan Penggunaan
        </span>
    </div>

    <p>
        Sistem Rekomendasi Strategi Restoran merupakan aplikasi berbasis web
        yang digunakan untuk menganalisis data pelanggan dan menghasilkan
        rekomendasi strategi bisnis menggunakan kombinasi
        <strong>Machine Learning</strong> dan metode
        <strong>TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)</strong>.
    </p>

    <hr style="margin:24px 0;">

    <strong>Langkah-langkah Penggunaan:</strong>
    <ol>
        <li>Pengguna mengunggah dataset pelanggan restoran dalam format CSV</li>
        <li>Sistem melakukan pemetaan kolom dataset secara otomatis</li>
        <li>Data dianalisis menggunakan model dan metode TOPSIS</li>
        <li>Hasil rekomendasi strategi ditampilkan dalam bentuk peringkat</li>
        <li>Pengguna dapat mengunduh hasil analisis</li>
    </ol>

    <hr style="margin:24px 0;">

    <strong>Format Data yang Disarankan:</strong>
    <ul>
        <li>Rating layanan, makanan, dan suasana restoran</li>
        <li>Frekuensi kunjungan dan rata-rata pengeluaran pelanggan</li>
        <li>Data demografi seperti usia dan pendapatan</li>
        <li>Status loyalitas dan reservasi online</li>
    </ul>

    <div style="
        background:#f0f9ff;
        padding:16px;
        border-radius:8px;
        border-left:4px solid #667eea;
        margin-top:24px;
    ">
        <strong>Catatan:</strong><br>
        Sistem bersifat fleksibel dan dapat menyesuaikan dataset dengan
        struktur kolom yang berbeda melalui proses mapping.
    </div>

    </div>
    """, unsafe_allow_html=True)



def show_summary_stats(df, num_matched, total_features):
    """Tampilkan summary statistics dalam card"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Total Data", f"{len(df):,}", icon="📊")
    
    with col2:
        create_metric_card("Kolom Dataset", f"{len(df.columns)}", icon="📋")
    
    with col3:
        create_metric_card("Features Matched", f"{num_matched}/{total_features}", icon="✅")
    
    with col4:
        match_pct = (num_matched / total_features * 100)
        create_metric_card("Match Rate", f"{match_pct:.0f}%", icon="🎯")

def show_about():
    st.markdown("""
    <div style="
        background:white;
        color:#374151;
        padding:36px;
        border-radius:16px;
        box-shadow:0 4px 12px rgba(0,0,0,0.1);
        line-height:1.8;
    ">

    <div style="margin-bottom:20px;">
        <span style="font-size:28px; font-weight:700; color:#667eea;">
            ℹ️ Tentang Sistem
        </span>
    </div>

    <p>
        Sistem Rekomendasi Strategi Restoran dikembangkan sebagai sistem pendukung
        keputusan (Decision Support System) untuk membantu pemilik dan manajer
        restoran dalam menentukan strategi bisnis berbasis data pelanggan.
    </p>

    <hr style="margin:24px 0;">

    <strong>Tujuan Pengembangan:</strong>
    <ul>
        <li>Mendukung pengambilan keputusan yang objektif dan terukur</li>
        <li>Meningkatkan kepuasan dan loyalitas pelanggan</li>
        <li>Mengoptimalkan strategi pengembangan restoran</li>
    </ul>

    <hr style="margin:24px 0;">

    <strong>Metodologi:</strong>
    <ul>
        <li>Machine Learning untuk analisis pola kepuasan pelanggan</li>
        <li>TOPSIS sebagai metode pengambilan keputusan multikriteria</li>
        <li>Pemetaan kolom otomatis untuk fleksibilitas dataset</li>
    </ul>

    <hr style="margin:24px 0;">

    <strong>Teknologi yang Digunakan:</strong>
    <p>
        Python, Streamlit, Scikit-learn, Pandas, NumPy, dan Plotly
    </p>

    <div style="
        margin-top:32px;
        padding-top:16px;
        border-top:1px solid #e5e7eb;
        color:#9ca3af;
        font-size:14px;
        text-align:center;
    ">
        Restaurant Strategy Recommendation System<br>
        Version 2.0
    </div>

    </div>
    """, unsafe_allow_html=True)

