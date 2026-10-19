import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 설정 (앱 제목, 아이콘, 브라우저 탭 제목)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

# 화면 상단 제목 및 아이콘
st.title("🎬 영화 유형 나누기")
st.write("KOBIS 영화 데이터를 활용하여 군집화(K-Means Clustering)를 수행하고 결과를 시각화합니다.")

# 2. 데이터 불러오기 및 전처리
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    total_raw_count = len(df)
    
    # 네 속성 결측치 제거 및 first_week_audi > 0 조건 적용
    df_clean = df.dropna(subset=['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']).copy()
    df_clean = df_clean[df_clean['first_week_audi'] > 0].copy()
    
    # 속성 변환
    # 1) 스크린 수 상용로그 (log10(first_scrn))
    df_clean['log_first_scrn'] = np.log10(df_clean['first_scrn'])
    # 2) 누적 관객 상용로그 (log10(total_audi))
    df_clean['log_total_audi'] = np.log10(df_clean['total_audi'])
    # 3) 10위권 일수 (days_in_top10) - 그대로 유지
    
    # 4) 롱런 지수 (total_audi / first_week_audi, 20 초과 시 20으로 클리핑)
    df_clean['long_run_index'] = (df_clean['total_audi'] / df_clean['first_week_audi']).clip(upper=20)
    
    valid_count = len(df_clean)
    
    return df_clean, total_raw_count, valid_count

df, total_count, valid_count = load_and_preprocess_data()

# 전체 편수와 묶은 편수를 한 줄로 표시
st.info(f"**전체 데이터 편수:** {total_count}편 | **군집 분석에 사용된 편수(유효 데이터):** {valid_count}편")

# 3. 군집화 속성 선택
feature_options = {
    "스크린 수 (상용로그)": "log_first_scrn",
    "누적 관객 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index"
}

st.subheader("⚙️ 군집 분석 설정")
selected_feature_names = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (최소 2개 이상, 기본 전체 선택):",
    options=list(feature_options.keys()),
    default=list(feature_options.keys())
)

if len(selected_feature_names) < 2:
    st.warning("⚠️ 군집화를 위해 속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_options[name] for name in selected_feature_names]

# 4. 데이터 표준화 및 K-평균 군집화 (난수 고정)
X = df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
initial_clusters = kmeans.fit_predict(X_scaled)

df['cluster_raw'] = initial_clusters

# 5. 군집 라벨 재정의 (누적
