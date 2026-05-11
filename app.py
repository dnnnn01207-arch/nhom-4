import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split


# Load biến môi trường từ file .env nếu có
load_dotenv()


DATA_FILE = Path(__file__).resolve().parent / "cleaned_financial_data.csv"
FEATURE_COLUMNS = [
    "Total_Assets",
    "Total_Liabilities",
    "Revenue",
    "Operating_Expenses",
    "Net_Income",
    "Cash_Flow_Operating",
    "Cash_Flow_Investing",
    "Cash_Flow_Financing",
    "Current_Ratio",
    "Debt_to_Equity",
    "Gross_Margin",
    "Return_on_Assets",
    "Return_on_Equity",
]
TARGET_COLUMN = "Financial_Status"
LABEL_MAPPING = {
    0: "Bình thường",
    1: "Cảnh báo",
    2: "Có sai lệch",
}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Đọc dữ liệu CSV từ file và trả về DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    data = pd.read_csv(path)
    return data


@st.cache_data
def train_model(data: pd.DataFrame):
    """Huấn luyện mô hình RandomForest sau khi xử lý mất cân bằng với SMOTE."""
    # Tách features và target từ dữ liệu
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    # Chia dữ liệu thành tập Train và Test theo tỷ lệ 80/20
    # Sử dụng random_state cố định để kết quả có thể tái lập
    # Đồng thời giữ tỉ lệ nhãn trong Train và Test bằng stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Áp dụng SMOTE chỉ trên tập Train để cân bằng dữ liệu
    # TUYỆT ĐỐI không được áp dụng SMOTE lên tập Test để tránh rò rỉ dữ liệu
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Khởi tạo và huấn luyện mô hình RandomForestClassifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_resampled, y_train_resampled)

    # Dự đoán trên tập Test để đánh giá hiệu năng
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    return model, X_test, y_test, accuracy, report, cm


def plot_confusion_matrix(matrix: np.ndarray):
    """Vẽ ma trận nhầm lẫn bằng seaborn và matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["0", "1", "2"],
        yticklabels=["0", "1", "2"],
        ax=ax,
    )
    ax.set_title("Ma trận nhầm lẫn")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Giá trị thực tế")
    plt.tight_layout()
    return fig
    """Vẽ ma trận nhầm lẫn bằng seaborn và matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["0", "1", "2"],
        yticklabels=["0", "1", "2"],
        ax=ax,
    )
    ax.set_title("Ma trận nhầm lẫn")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Giá trị thực tế")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Dự đoán Sai lệch Báo Cáo Tài Chính",
        page_icon="📊",
        layout="wide",
    )

    st.title("Phát hiện sai lệch trong báo cáo tài chính")
    st.write(
        "Sử dụng mô hình học máy Random Forest và SMOTE để dự đoán trạng thái tài chính của công ty."
    )

    # Load và huấn luyện mô hình khi ứng dụng chạy
    data = load_data(DATA_FILE)
    model, X_test, y_test, accuracy, report, cm = train_model(data)

    # Phần tải lên file dữ liệu
    st.header("Tải lên file dữ liệu để dự đoán")
    uploaded_file = st.file_uploader("Chọn file CSV chứa dữ liệu tài chính", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Đọc file CSV
            input_data = pd.read_csv(uploaded_file)
            st.write("Dữ liệu tải lên:")
            st.dataframe(input_data.head())
            
            # Kiểm tra các cột cần thiết
            missing_cols = [col for col in FEATURE_COLUMNS if col not in input_data.columns]
            if missing_cols:
                st.error(f"File thiếu các cột sau: {missing_cols}")
            else:
                # Dự đoán
                predictions = model.predict(input_data[FEATURE_COLUMNS])
                input_data['Predicted_Status'] = predictions
                input_data['Label'] = [LABEL_MAPPING.get(p, 'Không xác định') for p in predictions]
                
                st.success("Dự đoán hoàn thành!")
                st.write("Kết quả dự đoán:")
                st.dataframe(input_data)
                
                # Cho phép tải xuống kết quả
                csv = input_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tải xuống kết quả CSV",
                    data=csv,
                    file_name="predicted_financial_status.csv",
                    mime="text/csv",
                    key="download-csv"
                )
        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {e}")

    # Phần đánh giá hiệu năng mô hình
    st.header("Đánh giá hiệu năng mô hình")
    st.markdown(f"**Độ chính xác (Accuracy):** {accuracy:.4f}")
    st.subheader("Báo cáo phân loại")
    st.text(report)

    st.subheader("Ma trận nhầm lẫn")
    fig = plot_confusion_matrix(cm)
    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
