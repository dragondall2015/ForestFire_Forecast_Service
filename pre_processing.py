# 모듈용 필수 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# 1. 데이터 불러오기 및 전처리 함수
def load_and_preprocess_data(filepath):
    """
    CSV 파일을 읽고 burned_area 컬럼에 로그 변환을 적용한다.
    """
    fires = pd.read_csv(filepath, sep=',')
    fires['burned_area'] = np.log(fires['burned_area'] + 1)
    return fires

# 2. 데이터 기본 정보 출력 함수
def explore_data(fires):
    """
    데이터프레임의 head(), info(), describe(), value_counts(month, day)를 출력한다.
    """
    print("### fires.head()")
    print(fires.head())
    
    print("\n### fires.info()")
    print(fires.info())
    
    print("\n### fires.describe()")
    print(fires.describe())
    
    print("\n### month value_counts()")
    print(fires['month'].value_counts())
    
    print("\n### day value_counts()")
    print(fires['day'].value_counts())

# 3. 특성별 히스토그램 시각화 함수
def plot_feature_histograms(fires):
    """
    주요 특성(avg_temp, avg_wind, burned_area 등)의 히스토그램을 출력한다.
    """
    columns_to_plot = ['avg_temp', 'avg_wind', 'burned_area', 
                       'latitude', 'longitude', 'max_temp', 'max_wind_speed']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    plt.gcf().canvas.manager.set_window_title('2020810041/손영준')

    for i, col in enumerate(columns_to_plot):
        fires[col].hist(bins=30, ax=axes[i])
        axes[i].set_title(col)

    for i in range(len(columns_to_plot), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# 4. burned_area 로그 변환 전후 비교 시각화 함수
def plot_burned_area_transformation(fires):
    """
    burned_area의 로그 변환 전후 분포를 히스토그램으로 비교한다.
    """
    # 복원된 원본 burned_area 계산 (exp(burned_area) - 1)
    burned_area_original = np.exp(fires['burned_area']) - 1

    plt.figure(figsize=(10, 4))

    # 변환 전 (복원값) 히스토그램
    plt.subplot(1, 2, 1)
    plt.hist(burned_area_original, bins=30)
    plt.title('burned_area (202081041/sonyoungjun)')

    # 변환 후 히스토그램
    plt.subplot(1, 2, 2)
    plt.hist(fires['burned_area'], bins=30)
    plt.title('burned_area (로그 변환 후)')

    plt.tight_layout()
    plt.show()

# 5. 데이터 분리 함수
def split_data(fires, test_size=0.2, random_state=42):
    """
    train_test_split과 StratifiedShuffleSplit을 사용하여 데이터를 나누고,
    월(month) 기준 분포가 유지되는지 확인한다.
    """
    # (1) 일반적인 랜덤 train/test 분리
    train_set, test_set = train_test_split(fires, test_size=test_size, random_state=random_state)
    print("=== Random Split Test Set Preview ===")
    print(test_set.head())
    
    # 월별 히스토그램 (전체 fires 데이터 기준)
    plt.figure()
    plt.gcf().canvas.manager.set_window_title('2020810041/손영준')  # 창 이름 설정
    fires["month"].hist()
    plt.title('Month Distribution (Original)')
    plt.show()

    # (2) StratifiedShuffleSplit 사용 (월 기준 비율 유지)
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(fires, fires["month"]):
        strat_train_set = fires.loc[train_index]
        strat_test_set = fires.loc[test_index]

    # Stratified 결과 확인
    print("\n=== Month category proportion in Stratified Test Set ===")
    print(strat_test_set["month"].value_counts() / len(strat_test_set))

    print("\n=== Overall month category proportion in Whole Dataset ===")
    print(fires["month"].value_counts() / len(fires))

    return strat_train_set, strat_test_set

# 6. scatter matrix 출력 함수
def plot_scatter_matrix(fires):
    """
    burned_area, max_temp, avg_temp, max_wind_speed 특성 간 관계를 scatter_matrix로 시각화한다.
    """
    attributes = ['burned_area', 'max_temp', 'avg_temp', 'max_wind_speed']

    plt.figure()
    plt.gcf().canvas.manager.set_window_title('2020810041/손영준')  # 창 제목 설정

    scatter_matrix(fires[attributes], figsize=(12, 8), alpha=0.2)

    # 전체 figure에 제목 추가
    plt.suptitle('2020810041/syj: Scatter Matrix', fontsize=16)
    plt.show()

# 7. 지역별 burned_area 시각화 함수
def plot_geographic_burned_area(fires):
    """
    longitude/latitude 기준으로 산불 범위(burned_area)를 색으로, max_temp를 크기로 표시하는 산점도 그린다.
    """
    plt.figure(figsize=(10, 7))
    plt.gcf().canvas.manager.set_window_title('2020810041/손영준')  # 창 제목 설정

    fires.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,
        s=fires["max_temp"],  # 원의 크기: max_temp
        label="max_temp",
        c="burned_area",  # 원의 색: burned_area
        cmap=plt.get_cmap("jet"),
        colorbar=True
    )

    plt.title('2020810041/syj: Geographic Burned Area')
    plt.legend()
    plt.show()

# 8. 범주형 인코딩 및 분리 함수
def encode_categorical_features(strat_train_set):
    """
    strat_train_set에서 month, day를 OneHotEncoder로 인코딩할 준비를 한다.
    'burned_area'를 분리하고, 수치형/범주형 피처를 분리한다.
    """
    print("2020810041/손영준")

    # (1) 타겟 라벨 분리
    fires = strat_train_set.drop(["burned_area"], axis=1)
    fires_labels = strat_train_set["burned_area"].copy()

    # (2) 수치형 특성만 추출 (month, day 제거)
    fires_num = fires.drop(["month", "day"], axis=1)

    return fires, fires_num, fires_labels

# 9. 파이프라인 구성 및 데이터 전처리 함수
def preprocess_with_pipeline(fires):
    """
    수치형 특성은 StandardScaler로 정규화, 범주형 특성(month, day)은 OneHot 인코딩하는 파이프라인을 만든다.
    """
    print("\n\n########################################################################")
    print("Now let's build a pipeline for preprocessing the numerical attributes:")
    print("2020810041/손영준")

    # (1) 수치형/범주형 특성 구분
    num_attribs = fires.drop(["month", "day"], axis=1).columns.tolist()
    cat_attribs = ["month", "day"]

    # (2) 수치형 파이프라인
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

    # (3) 전체 파이프라인 (수치형 + 범주형 결합)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])

    # (4) 전체 데이터 전처리 실행
    fires_prepared = full_pipeline.fit_transform(fires)

    return fires_prepared

# ==============================================================
# 사용 예시 (메인 코드)
if __name__ == "__main__":
    # 1. 데이터 로딩 및 변환
    fires = load_and_preprocess_data('./sanbul2district-divby100.csv')

    # 2. 데이터 탐색
    print("2020810041/손영준")
    explore_data(fires)

    # 3. 주요 특성 히스토그램 출력
    plot_feature_histograms(fires)

    # 4. burned_area 변환 전후 비교
    plot_burned_area_transformation(fires)

    # 5. 데이터 분리 (train/test)
    strat_train_set, strat_test_set = split_data(fires)

    # 6
    plot_scatter_matrix(fires)

    # 7
    plot_geographic_burned_area(fires)

    # 8
    fires_train, fires_train_num, fires_train_labels = encode_categorical_features(strat_train_set)
    fires_prepared = preprocess_with_pipeline(fires_train)

    # 9
    fires_test, fires_test_num, fires_test_labels = encode_categorical_features(strat_test_set)
    fires_test_prepared = preprocess_with_pipeline(fires_test)

    np.save('./fires_prepared.npy', fires_prepared)
    np.save('./fires_labels.npy', fires_train_labels)
    np.save('./fires_test_prepared.npy', fires_test_prepared)
    np.save('./fires_test_labels.npy', fires_test_labels)

    print("✅ 2020810041/손영준: 전처리 결과 저장 완료")
