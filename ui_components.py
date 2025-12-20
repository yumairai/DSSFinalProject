import streamlit as st
import pandas as pd

def set_page_style():
    """Custom CSS tema kuning #ffc20f, hitam, dan putih"""
    st.markdown(
    """
    <style>
    /* ===== MAIN BACKGROUND ===== */
    .main {
        background-color: #ffffff;
    }

    /* ===== CARD ===== */
    .css-1r6slb0 {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.12);
        border-top: 5px solid #ffc20f;
    }

    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #000000;
    }

    [data-testid="stMetricLabel"] {
        color: #374151;
        font-weight: 600;
    }

    /* ===== HEADERS ===== */
    h1 {
        color: #000000;
        font-weight: 800;
        border-bottom: 4px solid #ffc20f;
        padding-bottom: 8px;
    }

    h2, h3 {
        color: #000000;
        font-weight: 700;
    }

    /* ===== BUTTONS ===== */
    .stButton>button {
        background-color: #ffc20f;
        color: #000000;
        border: none;
        border-radius: 25px;
        padding: 10px 28px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #e6ad00;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.25);
    }

    /* ===== ALERT / INFO ===== */
    .stAlert {
        border-radius: 10px;
        border-left: 6px solid #ffc20f;
        background-color: #fffbea;
        color: #000000;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] a {
        color: #ffc20f !important;
        font-weight: bold;
    }

    /* Metric di Sidebar */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffc20f !important;
        font-weight: bold;
    }

    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        color: #a7f3d0 !important;
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background-color: #fffbea;
        border-radius: 6px;
        border-left: 4px solid #ffc20f;
        font-weight: 600;
        color: #000000;
    }

    /* ===== TABLE ===== */
    thead tr th {
        background-color: #ffc20f;
        color: #000000;
        font-weight: bold;
    }

    tbody tr:nth-child(even) {
        background-color: #f9fafb;
    }

    </style>
    """,
    unsafe_allow_html=True
    )


def show_progress_indicator(current_step):
    steps = [
        ("Upload Data"),
        ("Mapping"),
        ("Analysis"),
        ("Results"),
        ("Export")
    ]

    cols = st.columns(len(steps))
    for i, ((label), col) in enumerate(zip(steps, cols)):
        with col:
            if i < current_step:
                color = "#000000"
                prefix = "✓"
            elif i == current_step:
                color = "#ffc20f"
                prefix = "►"
            else:
                color = "#9ca3af"
                prefix = ""

            st.markdown(f"""
            <div style="text-align:center; color:{color};">
                <div style="font-size:16px; font-weight:700;">{prefix} {label}</div>
            </div>
            """, unsafe_allow_html=True)

def create_metric_card(label, value, delta=None):
    """Buat metric card yang cantik"""
    delta_html = ""
    if delta is not None:
        color = "#16a34a" if delta > 0 else "#dc2626"
        delta_html = f"<div style='color:{color}; font-size:14px;'>▲ {delta}</div>"

    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            padding:18px 20px;
            border-radius:14px;
            text-align:center;
            border-top:5px solid #ffc20f;
            box-shadow:0 4px 10px rgba(0,0,0,0.12);
        ">
            <div style="font-size:13px; color:#000000; margin-bottom:6px;">
                {label}
            </div>

            <div style="font-size:30px; font-weight:800; color:#000000;">
                {value}
            </div>

            {delta_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def show_user_guide():
    st.markdown("""
    <div style="
        color:#000000;
        border-left:4px solid #ffc20f;
        background:#fff6d6;
        padding:32px;
        border-radius:16px;
        box-shadow:0 4px 12px rgba(0,0,0,0.1);
        line-height:1.8;
    ">

    <div style="margin-bottom:20px;">
        <span style="font-size:28px; font-weight:700; color:#667eea;">
            Panduan Penggunaan
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
        create_metric_card("Total Data", f"{len(df):,}")
    
    with col2:
        create_metric_card("Kolom Dataset", f"{len(df.columns)}")
    
    with col3:
        create_metric_card("Features Matched", f"{num_matched}/{total_features}")
    
    with col4:
        match_pct = (num_matched / total_features * 100)
        create_metric_card("Match Rate", f"{match_pct:.0f}%")

def show_about():
    st.markdown("""
    <div style="
        color:#000000;
        border-left:4px solid #ffc20f;
        background:#fff6d6;
        padding:36px;
        border-radius:16px;
        box-shadow:0 4px 12px rgba(0,0,0,0.1);
        line-height:1.8;
    ">

    <div style="margin-bottom:20px;">
        <span style="font-size:28px; font-weight:700; color:#667eea;">
            Tentang Sistem
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

