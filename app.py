import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ thống Chẩn đoán Gian lận Tài chính", layout="wide")

class FraudDetectionApp:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.features = [
            'Total_Assets', 'Total_Liabilities', 'Revenue', 'Operating_Expenses', 
            'Net_Income', 'Cash_Flow_Operating', 'Cash_Flow_Investing', 
            'Cash_Flow_Financing', 'Current_Ratio', 'Debt_to_Equity', 
            'Gross_Margin', 'Return_on_Assets', 'Return_on_Equity'
        ]

    @st.cache_resource
    def train_model(_self):
        """Huấn luyện mô hình với class_weight='balanced'"""
        try:
            df = pd.read_csv(_self.data_path)
            X = df[_self.features]
            y = df['Financial_Status']

            class_counts = y.value_counts()

            # Chỉ dùng class_weight='balanced' — không cần SMOTE.
            # SMOTE tạo quá nhiều mẫu tổng hợp (306→2253 cho class 2)
            # khiến model học ranh giới nhiễu, dự đoán sai.
            # class_weight='balanced' tính w = n/(n_classes*n_j) từ dữ liệu thực,
            # đủ để bù imbalance mà không cần dữ liệu giả.
            rf = RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',
                max_depth=12,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)
            return rf, df, class_counts
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện mô hình: {e}")
            return None, None, None

    def run(self):
        st.title("🛡️ Hệ thống Phát hiện Gian lận Báo cáo Tài chính")
        st.markdown("---")

        # Khởi tạo mô hình
        with st.spinner("Đang huấn luyện mô hình thực tế..."):
            self.model, full_data, class_counts = self.train_model()

        if self.model is None:
            return

        # Hiển thị thống kê dữ liệu huấn luyện trong sidebar
        with st.sidebar:
            st.header("📈 Thông tin mô hình")
            st.write("**Phân phối class gốc:**")
            dist_df = pd.DataFrame({
                'Class': ['0 - Bình thường', '1 - Nghi vấn', '2 - Rủi ro cao'],
                'Số mẫu': [class_counts.get(0, 0), class_counts.get(1, 0), class_counts.get(2, 0)],
            })
            st.dataframe(dist_df, hide_index=True)
            st.caption("Sử dụng class_weight='balanced' để bù imbalance.")

        # Giao diện chính chia làm 2 cột
        col1, col2 = st.columns([1, 2])

        with col1:
            st.header("📋 Nhập dữ liệu tài chính")
            input_data = {}

            # Tạo các ô nhập liệu cho 13 chỉ số
            for feature in self.features:
                val_str = st.text_input(f"{feature}", value="0.0")
                
                # Ép kiểu từ chữ sang số thực (float)
                try:
                    input_data[feature] = float(val_str)
                except ValueError:
                    st.error(f"⚠️ Vui lòng chỉ nhập số cho trường {feature}!")
                    input_data[feature] = 0.0
            
            predict_btn = st.button("🔍 Thực hiện chẩn đoán", use_container_width=True)

        with col2:
            st.header("📊 Kết quả dự báo")
            if predict_btn:
                # Chuyển dữ liệu nhập vào thành DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Thực hiện dự đoán
                prediction = self.model.predict(input_df)[0]
                probability = self.model.predict_proba(input_df)

                # Hiển thị kết quả trực quan
                if prediction == 0:
                    st.success("✅ Kết quả: Báo cáo Tài chính Bình thường")
                elif prediction == 1:
                    st.warning("⚠️ Kết quả: Có dấu hiệu Gian lận (Loại 1)")
                else:
                    st.error("🚨 Kết quả: Nghi vấn Gian lận Nghiêm trọng (Loại 2)")

                # Hiển thị xác suất
                st.write("### Xác suất chi tiết:")
                prob_df = pd.DataFrame(
                    probability, 
                    columns=["Bình thường (0)", "Nghi vấn (1)", "Rủi ro cao (2)"]
                )
                st.dataframe(prob_df.style.highlight_max(axis=1))

            else:
                st.info("Vui lòng nhập các thông số bên trái và ấn nút Chẩn đoán.")


if __name__ == "__main__":
    # Đảm bảo file train_data.csv nằm cùng thư mục với app.py
    app = FraudDetectionApp("train_data.csv")
    app.run()
    