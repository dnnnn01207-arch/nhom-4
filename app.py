import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

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
            # KIỂM TRA FILE TRƯỚC KHI ĐỌC
            if not os.path.exists(_self.data_path):
                return None, None, None
                
            df = pd.read_csv(_self.data_path)
            X = df[_self.features]
            y = df['Financial_Status']
            class_counts = y.value_counts()

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
            st.error(f"Lỗi hệ thống: {e}")
            return None, None, None

    def run(self):
        st.title("🛡️ Hệ thống Phát hiện Gian lận Báo cáo Tài chính")
        st.markdown("---")

        # 1. KHỞI TẠO MÔ HÌNH
        self.model, full_data, class_counts = self.train_model()

        if self.model is None:
            st.error(f"❌ KHÔNG TÌM THẤY FILE '{self.data_path}'! Em hãy kéo file này vào thư mục dự án nhé.")
            return

        # 2. SIDEBAR - THÔNG TIN & HƯỚNG DẪN
        with st.sidebar:
            st.header("📈 Thông tin mô hình")
            dist_df = pd.DataFrame({
                'Trạng thái': ['Bình thường', 'Nghi vấn', 'Rủi ro cao'],
                'Số mẫu': [class_counts.get(0, 0), class_counts.get(1, 0), class_counts.get(2, 0)],
            })
            st.dataframe(dist_df, hide_index=True)
            
            st.markdown("---")
            st.header("💡 Giải thích chỉ số")
            st.info("- **Revenue:** Doanh thu\n- **Net Income:** Lợi nhuận ròng\n- **Total Assets:** Tổng tài sản")

        # 3. GIAO DIỆN CHÍNH
        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.header("📋 Nhập dữ liệu")
            input_data = {}
            # Dùng number_input để chuyên nghiệp và tránh lỗi nhập chữ
            for feature in self.features:
                input_data[feature] = st.number_input(f"Chỉ số {feature}", value=0.0, format="%.2f")
            
            predict_btn = st.button("🔍 Thực hiện chẩn đoán", use_container_width=True)

        with col2:
            st.header("📊 Kết quả dự báo")
            if predict_btn:
                input_df = pd.DataFrame([input_data])
                prediction = self.model.predict(input_df)[0]
                probability = self.model.predict_proba(input_df)

                # Hiển thị thông báo kết quả
                if prediction == 0:
                    st.success("✅ Kết quả: Báo cáo Tài chính Bình thường")
                elif prediction == 1:
                    st.warning("⚠️ Kết quả: Có dấu hiệu Nghi vấn (Loại 1)")
                else:
                    st.error("🚨 Kết quả: Rủi ro Gian lận Cao (Loại 2)")

                # BIỂU ĐỒ TRỰC QUAN
                st.write("### 📉 Phân tích xác suất:")
                prob_data = pd.DataFrame({
                    'Nhãn': ["Bình thường", "Nghi vấn", "Rủi ro"],
                    'Phần trăm': probability[0] * 100
                })
                st.bar_chart(prob_data.set_index('Nhãn'))

                # NÚT TẢI BÁO CÁO
                res_txt = f"CHẨN ĐOÁN GIAN LẬN\nKết quả: {prediction}\nXác suất Rủi ro: {probability[0][2]*100:.2f}%"
                st.download_button("📥 Tải kết quả (.txt)", res_txt, file_name="ket_qua.txt")
            else:
                st.info("Nhập thông số bên trái và nhấn nút để bắt đầu phân tích.")

if __name__ == "__main__":
    app = FraudDetectionApp("train_data.csv")
    app.run()

if __name__ == "__main__":
    # Đảm bảo file train_data.csv nằm cùng thư mục với app.py
    app = FraudDetectionApp("train_data.csv")
    app.run()
    
