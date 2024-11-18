import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker

# Styling CSS
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
    background: linear-gradient(to bottom, #eef2f3, #8e9eab);
}
header {
    background-color: #6a11cb;
    color: white;
    text-align: center;
    padding: 10px 0;
    border-radius: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<header><h2>Aplikasi Analisis Media Sosial</h2></header>", unsafe_allow_html=True)

# Sidebar Menu with streamlit-option-menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigasi",
        options=["Home", "Input Data", "Preprocessing", "Analysis", "Visualizations", "About Us"],
        icons=["house", "upload", "gear", "bar-chart", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "icon": {"color": "#6a11cb", "font-size": "20px"},
            "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "#ddd"},
            "nav-link-selected": {"background-color": "#6a11cb", "color": "white"},
        },
    )

# Global variable for session data
if "data" not in st.session_state:
    st.session_state["data"] = None

# Define functions for each menu
def home():
    st.header("Selamat Datang")
    st.write("""
    Aplikasi ini membantu Anda menganalisis dampak media sosial terhadap emosi.  
    Anda dapat:
    - Mengunggah dataset atau menghasilkan data.
    - Membersihkan dan memproses data.
    - Melakukan analisis clustering atau sentiment analysis.
    - Memvisualisasikan hasil analisis.
    """)

def input_data():
    st.header("Input Data")
    fake = Faker()

    # Generate or Upload Data
    choice = st.radio("Pilih cara mendapatkan data:", ["Unggah Dataset", "Hasilkan Data"])
    if choice == "Unggah Dataset":
        uploaded_file = st.file_uploader("Unggah dataset CSV", type=["csv"])
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state['data'] = data  # Save to session state
                st.write(f"Dataset berisi {data.shape[0]} baris dan {data.shape[1]} kolom.")
                st.dataframe(data)
            except Exception as e:
                st.error("File yang diunggah tidak valid. Harap unggah file CSV yang benar.")
    elif choice == "Hasilkan Data":
        num_samples = st.slider("Jumlah data yang dihasilkan:", 10, 100, 50)
        data = pd.DataFrame({
            "User_ID": [fake.uuid4() for _ in range(num_samples)],
            "Daily_Usage_Time (minutes)": np.random.randint(30, 600, num_samples),
            "Posts_Per_Day": np.random.randint(0, 50, num_samples),
            "Dominant_Emotion": np.random.choice(["Happy", "Sad", "Angry", "Neutral"], num_samples),
            "Platform": np.random.choice(["Instagram", "Facebook", "Twitter"], num_samples),
            "Comment": [fake.sentence(nb_words=10) for _ in range(num_samples)]
        })
        st.session_state['data'] = data  # Save to session state
        st.write(f"Dataset simulasi berisi {data.shape[0]} baris dan {data.shape[1]} kolom.")
        st.dataframe(data)


def preprocessing():
    st.header("Preprocessing Data")
    if st.session_state['data'] is None:
        st.warning("Harap unggah atau hasilkan data terlebih dahulu.")
    else:
        # Retrieve data from session state
        data = st.session_state['data']

        st.subheader("Statistik Sebelum Preprocessing")
        st.write("Statistik awal dataset:")
        st.dataframe(data.describe(include="all").T)

        # Hapus data duplikat
        st.subheader("Menghapus Data Duplikat")
        data = data.drop_duplicates()
        st.write(f"Jumlah data setelah menghapus duplikat: {data.shape[0]} baris")

        # Mengatasi Nilai Kosong
        st.subheader("Mengatasi Nilai Kosong")
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                if data[col].dtype in ["float64", "int64"]:
                    data[col].fillna(data[col].mean(), inplace=True)
                else:
                    data[col].fillna(data[col].mode()[0], inplace=True)

        # Validasi kolom 'Age'
        if 'Age' in data.columns:
            st.subheader("Validasi Kolom 'Age'")
            data['Age'] = pd.to_numeric(data['Age'], errors='coerce')  # Konversi ke numerik
            data = data.dropna(subset=['Age'])  # Hapus baris dengan Age NaN

        # Validasi kolom 'Gender'
        if 'Gender' in data.columns:
            st.subheader("Validasi Kolom 'Gender'")
            valid_genders = ['Male', 'Female', 'Non-binary']
            data['Gender'] = data['Gender'].apply(lambda x: x if x in valid_genders else None)
            data.dropna(subset=['Gender'], inplace=True)

        st.subheader("Statistik Setelah Preprocessing")
        st.dataframe(data.describe(include="all").T)

        # Save processed data back to session state
        st.session_state['data'] = data
        st.success("Preprocessing selesai!")


def analysis():
    st.header("Analisis Data")
    if st.session_state['data'] is None:
        st.warning("Harap unggah atau hasilkan data terlebih dahulu.")
    else:
        data = st.session_state['data']

        st.subheader("1. Hubungan Media Sosial dengan Emosi")
        platform_emotion = data.groupby('Platform')['Dominant_Emotion'].value_counts(normalize=True).unstack()
        st.write("Berikut adalah proporsi emosi dominan pada tiap platform media sosial:")
        st.dataframe(platform_emotion)

        st.write("*Penjelasan:* Angka di tabel di atas mewakili proporsi emosi tertentu pada tiap platform media sosial. Misalnya, 0.40 berarti 40% pengguna pada platform tersebut menunjukkan emosi tersebut.")

        # Hubungan Gender dengan Jumlah Postingan
        st.subheader("2. Hubungan Gender dengan Jumlah Postingan")
        gender_post = data.groupby('Gender')['Posts_Per_Day'].mean().reset_index()
        st.write("Rata-rata jumlah postingan per hari berdasarkan gender:")
        st.dataframe(gender_post)

        st.write("*Penjelasan:* Tabel ini menunjukkan rata-rata jumlah postingan per hari untuk tiap gender. Gender dapat berupa kategori seperti 'Male' atau 'Female'.")

        # Analisis 3: Hubungan Umur dengan Jumlah Postingan
        st.subheader("3. Hubungan Umur dengan Jumlah Postingan")
        if 'Age' in data.columns and data['Age'].dtype in ['float64', 'int64']:
            age_post_corr = data['Age'].corr(data['Posts_Per_Day'])
            st.write(f"Korelasi antara umur dan jumlah postingan per hari adalah *{age_post_corr:.2f}*.")
            st.write("*Penjelasan:* Nilai korelasi berkisar antara -1 hingga 1. Nilai mendekati 1 menunjukkan hubungan positif, sedangkan mendekati -1 menunjukkan hubungan negatif.")
        else:
            st.warning("Kolom 'Age' tidak tersedia atau bukan numerik.")

        # Analisis 4: Hubungan Like dan Comment dengan Emosi
        st.subheader("4. Hubungan Like dan Komentar dengan Emosi")
        if 'Likes_Received_Per_Day' in data.columns and 'Comments_Received_Per_Day' in data.columns:
            emotion_metrics = data.groupby('Dominant_Emotion')[['Likes_Received_Per_Day', 'Comments_Received_Per_Day']].mean()
            st.write("Rata-rata jumlah like dan komentar per hari berdasarkan emosi dominan:")
            st.dataframe(emotion_metrics)
            st.write("*Penjelasan:* Tabel ini menunjukkan rata-rata jumlah like dan komentar per hari yang diterima pengguna dengan emosi dominan tertentu.")
        st.subheader("Clustering Analysis")
        st.write("Metode yang digunakan: *K-Means Clustering*")

        # Pilih fitur untuk clustering
        st.write("Pilih kolom numerik untuk clustering:")
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        features = st.multiselect("Fitur untuk clustering:", numeric_columns, default=numeric_columns)

        if len(features) < 2:
            st.warning("Pilih setidaknya dua fitur untuk clustering.")
        else:
            # Standarisasi data
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(data[features])

            # Metode Elbow untuk menentukan jumlah cluster
            distortions = []
            K = range(1, 10)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_features)
                distortions.append(kmeans.inertia_)

            # Visualisasi Elbow Method
            st.subheader("1. Menentukan Jumlah Cluster")
            fig, ax = plt.subplots()
            ax.plot(K, distortions, 'bo-')
            ax.set_title("Metode Elbow")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("Distorsi (Inertia)")
            st.pyplot(fig)

            # Pilihan jumlah cluster
            num_clusters = st.slider("Pilih jumlah cluster:", 2, 10, 3)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            data['Cluster'] = kmeans.fit_predict(scaled_features)

            # Hasil centroid
            st.write(f"Centroid cluster untuk {num_clusters} cluster:")
            st.write(pd.DataFrame(kmeans.cluster_centers_, columns=features))

            # Save clustering result
            st.session_state['data'] = data


def visualizations():
    st.header("Visualisasi Data")
    if st.session_state['data'] is None:
        st.warning("Harap unggah atau hasilkan data terlebih dahulu.")
    else:
        data = st.session_state['data']

        st.subheader("1. Proporsi Emosi Berdasarkan Platform Media Sosial")
        platform_emotion = data.groupby('Platform')['Dominant_Emotion'].value_counts(normalize=True).unstack()
        fig, ax = plt.subplots(figsize=(8, 5))
        platform_emotion.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
        ax.set_title("Proporsi Emosi Berdasarkan Platform Media Sosial")
        ax.set_ylabel("Proporsi")
        st.pyplot(fig)

        st.subheader("2. Jumlah Postingan Berdasarkan Gender")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Gender', y='Posts_Per_Day', data=data, ci=None, ax=ax, palette="coolwarm")
        ax.set_title("Rata-Rata Postingan Berdasarkan Gender")
        ax.set_ylabel("Jumlah Postingan Per Hari")
        st.pyplot(fig)

        st.subheader("3. Hubungan Umur dengan Jumlah Postingan")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x='Age', y='Posts_Per_Day', data=data, ax=ax)
        ax.set_title("Hubungan Umur dengan Jumlah Postingan")
        ax.set_xlabel("Umur")
        ax.set_ylabel("Jumlah Postingan Per Hari")
        st.pyplot(fig)

        st.subheader("4. Hubungan Like dan Komentar dengan Emosi")
        fig, ax = plt.subplots(figsize=(8, 5))
        emotion_metrics = data.groupby('Dominant_Emotion')[['Likes_Received_Per_Day', 'Comments_Received_Per_Day']].mean().reset_index()
        emotion_metrics_melted = pd.melt(emotion_metrics, id_vars='Dominant_Emotion', 
                                         value_vars=['Likes_Received_Per_Day', 'Comments_Received_Per_Day'])
        sns.barplot(x='Dominant_Emotion', y='value', hue='variable', data=emotion_metrics_melted, ax=ax, palette="muted")
        ax.set_title("Rata-Rata Like dan Komentar Per Hari Berdasarkan Emosi")
        ax.set_ylabel("Rata-Rata")
        st.pyplot(fig)


def about_us():
    st.header("Tentang Kami")
    st.write("""
    **Pengembang**: Dibuat oleh kelompok 5 Mata Kuliah Akuisisi Data yang beranggotakan :

    - **Triana Zahara Nurhaliza	(2211522008)**
    - **Fadli Hidayat			(2111522010)**
    - **Hatta Asri Rahman 		(2211522012)**
    - **Nabila R Dzakira		(2211523036)** 

    **Tujuan**: Membantu menganalisis pengaruh media sosial terhadap emosi.  

    **Teknologi yang digunakan**: Streamlit, Pandas, Scikit-learn, Seaborn, Plotly.
    """)

# Render selected menu
if selected == "Home":
    home()
elif selected == "Input Data":
    input_data()
elif selected == "Preprocessing":
    preprocessing()
elif selected == "Analysis":
    analysis()
elif selected == "Visualizations":
    visualizations()
elif selected == "About Us":
    about_us()
