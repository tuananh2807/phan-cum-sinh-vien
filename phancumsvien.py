import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Cấu hình font chữ để hiển thị tiếng Việt trên biểu đồ
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans']

# --- 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU ---
print("\nPHÂN CỤM SINH VIÊN BẰNG K-MEANS CLUSTERING")

# Đọc toàn bộ file không bỏ dòng nào để tự động dò tìm
df_raw = pd.read_csv('diemhe4_sinhvien.csv', header=None, dtype=str)

# Tự động tìm vị trí dòng chứa "Tên Sinh Viên" và dòng bắt đầu danh sách môn (STT = 1)
idx_name = df_raw[df_raw[2] == 'Tên Sinh Viên'].index[0]
idx_subject = df_raw[df_raw[0] == '1'].index[0]

# Lấy danh sách tên sinh viên (từ cột 3 trở đi của dòng Tên Sinh Viên)
raw_names = df_raw.iloc[idx_name, 3:].tolist()
raw_names = [str(x).strip().strip('"') for x in raw_names if pd.notna(x) and str(x).strip() != 'nan']

# XỬ LÝ TRÙNG TÊN: Thêm hậu tố _1, _2 cho các bạn trùng tên để Pandas không bị lỗi
student_names = []
name_counts = {}
for name in raw_names:
    if name in name_counts:
        name_counts[name] += 1
        student_names.append(f"{name}_{name_counts[name]}")
    else:
        name_counts[name] = 1
        student_names.append(name)

# Lấy danh sách 52 môn học (từ dòng idx_subject trở đi, cột 2)
subject_names = df_raw.iloc[idx_subject:, 2].tolist()

# Lấy riêng vùng dữ liệu điểm
scores_raw = df_raw.iloc[idx_subject:, 3:3+len(student_names)].copy()

# Chuẩn hóa dấu phẩy thành dấu chấm
scores_raw = scores_raw.apply(lambda col: col.map(lambda x: str(x).replace(',', '.') if pd.notna(x) else ''))

# Ghép lại thành DataFrame hoàn chỉnh
df_scores = pd.DataFrame(scores_raw.values, columns=student_names)
df_scores.insert(0, 'Môn học', subject_names)

# Loại bỏ hàng không có tên môn
df_scores = df_scores[df_scores['Môn học'].notna()]
df_scores = df_scores[df_scores['Môn học'].astype(str).str.strip() != '']
df_scores = df_scores.reset_index(drop=True)

print(f"\nDữ liệu đã tải: {len(student_names)} sinh viên (ban đầu), {df_scores.shape[0]} môn học")

# Chuyển đổi điểm thành số thực
for col in student_names:
    df_scores[col] = pd.to_numeric(df_scores[col], errors='coerce')

# Loại bỏ sinh viên thiếu quá 50% dữ liệu
print("\nKiểm tra dữ liệu sinh viên:")
null_counts = df_scores[student_names].isnull().sum()
threshold = len(df_scores) * 0.5
valid_students = null_counts[null_counts <= threshold].index.tolist()

removed_students = len(student_names) - len(valid_students)
print(f"  - Bỏ qua sinh viên thiếu dữ liệu (> 50%): {removed_students} người")

# Giữ lại sinh viên hợp lệ
df_scores = df_scores[['Môn học'] + valid_students]

# Điền giá trị NULL bằng trung bình môn học
for col in valid_students:
    mean_val = df_scores[col].mean()
    df_scores[col] = df_scores[col].fillna(mean_val)

print(f"Dữ liệu cuối cùng: {len(valid_students)} sinh viên, {df_scores.shape[0]} môn học")

# --- 2. PHÂN LOẠI CÁC MÔN HỌC ---
print("\nPHÂN LOẠI CÁC MÔN HỌC")

technical_keywords = [
    'kỹ thuật', 'lập trình', 'trí tuệ nhân tạo', 'học máy', 'xử lý ảnh',
    'công nghệ', 'python', 'quản trị mạng', 'cơ sở dữ liệu', 'mạng máy tính',
    'hệ điều hành', 'cấu trúc dữ liệu', 'giải thuật', 'hướng đối tượng',
    'ứng dụng web', 'logic số', 'tín hiệu', 'toán rời rạc', 'giải tích',
    'xác suất', 'thống kê', 'đại số tuyến tính', 'vật lý', 'kiến trúc',
    'vi xử lý', 'vi điều khiển', 'nhúng', 'thiết bị truyền thông',
    'tin học', 'giao tiếp kỹ thuật', 'an toàn', 'bảo mật', 'quản lý dự án'
]

theory_keywords = [
    'tư tưởng', 'triết học', 'lịch sử', 'tiếng anh', 'pháp luật',
    'marketing', 'kinh tế', 'phương pháp', 'quản lý chất lượng',
    'môi trường', 'thể chất', 'bóng', 'thể dục', 'giáo dục'
]

def classify_subject(subject_name):
    name_lower = str(subject_name).lower()
    if any(keyword in name_lower for keyword in technical_keywords):
        return 'Kỹ Thuật'
    elif any(keyword in name_lower for keyword in theory_keywords):
        return 'Lý Thuyết'
    else:
        return 'Khác'

df_scores['Loại Môn'] = df_scores['Môn học'].apply(classify_subject)

print("\nPhân loại môn học:")
print(df_scores['Loại Môn'].value_counts())

# --- 3. TÍNH TOÁN CÁC ĐẠC TRƯNG PHÂN CỤM ---
print("\nTÍNH TOÁN CÁC ĐẶC TRƯNG PHÂN CỤM")

scores_data = df_scores[valid_students].values

features = pd.DataFrame()
features['Tên SV'] = valid_students
features['GPA'] = np.nanmean(scores_data, axis=0)
features['StdDev'] = np.nanstd(scores_data, axis=0)

tech_mask = df_scores['Loại Môn'] == 'Kỹ Thuật'
tech_data = scores_data[tech_mask.values]
if len(tech_data) > 0:
    features['Tech_Avg'] = np.nanmean(tech_data, axis=0)
else:
    features['Tech_Avg'] = features['GPA']

theory_mask = df_scores['Loại Môn'] == 'Lý Thuyết'
theory_data = scores_data[theory_mask.values]
if len(theory_data) > 0:
    features['Theory_Avg'] = np.nanmean(theory_data, axis=0)
else:
    features['Theory_Avg'] = features['GPA']

features['Tech_Theory_Ratio'] = features['Tech_Avg'] - features['Theory_Avg']

print(f"\nĐã tính các đặc trưng phân cụm cho {len(valid_students)} sinh viên")
features.to_csv('dactrung.csv', index=False, encoding='utf-8')
print(f"Lưu đặc trưng vào: dactrung.csv")

# --- 4. CHUẨN BỊ DỮ LIỆU CHO CLUSTERING ---
print("\nCHUẨN BỊ DỮ LIỆU CHO CLUSTERING")

X = features[['GPA', 'StdDev', 'Tech_Avg', 'Theory_Avg']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dữ liệu đã chuẩn hóa: {X_scaled.shape[0]} sinh viên, {X_scaled.shape[1]} đặc trưng")

# --- 5. ÁP DỤNG K-MEANS CLUSTERING ---
print("\nÁP DỤNG K-MEANS CLUSTERING (K=3)")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
cluster_labels = {0: 'Cụm 1', 1: 'Cụm 2', 2: 'Cụm 3'}
features['Cluster'] = [cluster_labels[c] for c in clusters]

print(f"\nPhân bố sinh viên:")
print(features['Cluster'].value_counts().sort_index())

# --- 6. PHÂN TÍCH CÁC CỤM ---
print("\nTHỐNG KÊ CHI TIẾT TỪNG CỤM")

for cluster_name in ['Cụm 1', 'Cụm 2', 'Cụm 3']:
    cluster_features = features[features['Cluster'] == cluster_name]
    if len(cluster_features) == 0:
        continue
    
    print(f"\n{cluster_name.upper()}:")
    print(f"  Số sinh viên: {len(cluster_features)}")
    print(f"  GPA: {cluster_features['GPA'].mean():.2f} (±{cluster_features['GPA'].std():.2f})")
    print(f"  Độ lệch chuẩn: {cluster_features['StdDev'].mean():.2f} (±{cluster_features['StdDev'].std():.2f})")

# --- 7. LƯU KẾT QUẢ ---
print("\nLƯU KẾT QUẢ PHÂN CỤM VÀO FILE CSV")

result_df = features[['Tên SV', 'GPA', 'StdDev', 'Tech_Avg', 'Theory_Avg', 'Tech_Theory_Ratio', 'Cluster']].copy()
result_df = result_df.sort_values('Cluster')
result_df.to_csv('kqphancum.csv', index=False, encoding='utf-8')

print(f"Kết quả phân cụm đã lưu: kqphancum.csv")

# --- 8. TẠO BIỂU ĐỒ ---
print("\nTẠO BIỂU ĐỒ PHÂN TÍCH")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
fig = plt.figure(figsize=(16, 12))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
cluster_order = ['Cụm 1', 'Cụm 2', 'Cụm 3']

# Biểu đồ 1: GPA vs Độ lệch chuẩn (StdDev)
# Ý nghĩa: Xem sinh viên điểm cao có học đều không (độ lệch chuẩn thấp = điểm đều, cao = học lệch)
ax1 = plt.subplot(2, 3, 1)
for i, cluster in enumerate(cluster_order):
    cluster_points = features[features['Cluster'] == cluster]
    ax1.scatter(cluster_points['GPA'], cluster_points['StdDev'], label=cluster, s=100, alpha=0.7, color=colors[i])
ax1.set_xlabel('GPA (Điểm trung bình)', fontweight='bold')
ax1.set_ylabel('Độ lệch chuẩn', fontweight='bold')
ax1.set_title('GPA vs Độ Lệch Chuẩn', fontweight='bold')
ax1.legend()

# Biểu đồ 2: Điểm Kỹ thuật vs Lý thuyết
# Ý nghĩa: Phân tách nhóm sinh viên có thiên hướng thực hành giỏi hay học bài lý thuyết giỏi
ax2 = plt.subplot(2, 3, 2)
for i, cluster in enumerate(cluster_order):
    cluster_points = features[features['Cluster'] == cluster]
    ax2.scatter(cluster_points['Tech_Avg'], cluster_points['Theory_Avg'], label=cluster, s=100, alpha=0.7, color=colors[i])
ax2.set_xlabel('Điểm Kỹ Thuật', fontweight='bold')
ax2.set_ylabel('Điểm Lý Thuyết', fontweight='bold')
ax2.set_title('Điểm Kỹ Thuật vs Lý Thuyết', fontweight='bold')
ax2.legend()

# Biểu đồ 3: Phân bố số lượng sinh viên
# Ý nghĩa: Đếm số lượng thực tế sinh viên bị xếp vào mỗi cụm
ax3 = plt.subplot(2, 3, 3)
cluster_counts = [len(features[features['Cluster'] == c]) for c in cluster_order]
bars = ax3.bar(cluster_order, cluster_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Số sinh viên', fontweight='bold')
ax3.set_title('Phân Bố Sinh Viên Theo Cụm', fontweight='bold')
for bar, count in zip(bars, cluster_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, count + 0.5, str(count), ha='center', fontweight='bold')

# Biểu đồ 4: Boxplot phân bố GPA
# Ý nghĩa: So sánh dải điểm tổng kết trung bình (GPA) giữa 3 nhóm
ax4 = plt.subplot(2, 3, 4)
data_to_plot = [features[features['Cluster'] == c]['GPA'].values for c in cluster_order]
bp = ax4.boxplot(data_to_plot, tick_labels=cluster_order, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('GPA', fontweight='bold')
ax4.set_title('Phân Bố GPA Theo Cụm', fontweight='bold')

# Biểu đồ 5: Boxplot phân bố Độ lệch chuẩn
# Ý nghĩa: So sánh mức độ học lệch của các nhóm
ax5 = plt.subplot(2, 3, 5)
data_to_plot = [features[features['Cluster'] == c]['StdDev'].values for c in cluster_order]
bp = ax5.boxplot(data_to_plot, tick_labels=cluster_order, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_ylabel('Độ Lệch Chuẩn', fontweight='bold')
ax5.set_title('Phân Bố Độ Lệch Chuẩn Theo Cụm', fontweight='bold')

# Biểu đồ 6: Chỉ số Thiên về Kỹ thuật
# Ý nghĩa: Giá trị dương biểu thị sinh viên điểm môn Kỹ thuật cao hơn Lý thuyết và ngược lại
ax6 = plt.subplot(2, 3, 6)
for i, cluster in enumerate(cluster_order):
    cluster_points = features[features['Cluster'] == cluster]
    ax6.scatter([i]*len(cluster_points), cluster_points['Tech_Theory_Ratio'], s=100, alpha=0.7, color=colors[i], label=cluster)
ax6.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Cân bằng')
ax6.set_xticks([0, 1, 2])
ax6.set_xticklabels(cluster_order)
ax6.set_ylabel('Chỉ số Tech/Theory', fontweight='bold')
ax6.set_title('Chỉ Số Thiên Về Kỹ Thuật', fontweight='bold')

plt.tight_layout()
plt.savefig('bieudo_phancum.png', dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ: bieudo_phancum.png")
plt.close()