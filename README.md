# phan-cum-sinh-vien<br>
# Lê Tuấn Anh-K225480106001<br>
# lớp: k58KTP<br>
# môn: Khoa Học Dữ LIệu<br>

# 1. Giới thiệu dự án<br>
Dự án ứng dụng thuật toán K-Means Clustering để phân loại sinh viên thành các nhóm có chung đặc điểm học tập. Bằng cách phân tích điểm số các môn học hệ 4, hệ thống giúp giảng viên và nhà quản lý giáo dục nhận diện được năng lực, thế mạnh và điểm yếu của từng nhóm sinh viên, từ đó đưa ra phương pháp giảng dạy và định hướng phát triển phù hợp.<br>

**- Phương Pháp Tiếp Cận Và Tiêu Chí Phân Cụm<br>**
Để đảm bảo việc phân cụm phản ánh đúng năng lực toàn diện và thiên hướng của sinh viên, dự án không chỉ dựa vào điểm tổng kết mà còn trích xuất 5 đặc trưng quan trọng sau:<br>
+ GPA (Điểm trung bình tích lũy):** Đánh giá năng lực học tập tổng thể của sinh viên dựa trên điểm trung bình tất cả các môn học.<br>
+ StdDev (Độ lệch chuẩn điểm số):** Đo lường mức độ đồng đều trong kết quả học tập. Độ lệch chuẩn thấp cho thấy sinh viên học đều các môn; ngược lại, độ lệch chuẩn cao biểu hiện sự chênh lệch lớn về điểm số (có dấu hiệu học lệch).<br>
+ Tech_Avg (Điểm trung bình khối Kỹ thuật):** Điểm trung bình của các môn chuyên ngành kỹ thuật, lập trình, công nghệ (AI, Cơ sở dữ liệu, Mạng máy tính, Hệ điều hành, v.v.).<br>
+ Theory_Avg (Điểm trung bình khối Lý thuyết):** Điểm trung bình của các môn cơ sở, khoa học xã hội và lý thuyết (Triết học, Lịch sử, Ngoại ngữ, Kinh tế, v.v.).<br>
+ Tech_Theory_Ratio (Chỉ số thiên hướng):** Sự chênh lệch giữa khối lượng kiến thức Kỹ thuật và Lý thuyết (`Tech_Avg - Theory_Avg`). Chỉ số dương thể hiện sinh viên có thiên hướng hoặc thực hành tốt môn Kỹ thuật hơn, trong khi chỉ số âm thể hiện thế mạnh về các môn Lý thuyết.<br>

# 2. Mô Tả Dữ Liệu Và Quy Trình Tiền Xử Lý<br>
**- Dữ liệu thô (diemhe4_sinhvien.csv)<br>**
+ Nguồn dữ liệu: Bảng điểm hệ 4 của sinh viên được lưu dưới dạng file định dạng CSV (`diemhe4_sinhvien.csv`).<br>
+ Cấu trúc: Dữ liệu có cấu trúc không đồng nhất, bao gồm dòng tiêu đề chứa "Tên Sinh Viên" trải dài ở các cột, và các dòng bên dưới chứa danh sách môn học kèm điểm số tương ứng.<br>
+ Đặc điểm: Dữ liệu thực tế thường chưa hoàn thiện, có thể chứa giá trị trống (do sinh viên chưa học môn đó), tên sinh viên trùng lặp hoặc lỗi định dạng số thập phân (sử dụng dấu phẩy thay vì dấu chấm theo chuẩn tính toán).<br>

**- Tiền xử lý dữ liệu (Data Cleaning & Imputation)<br>**
Để thuật toán Machine Learning hoạt động ổn định và chính xác, dữ liệu thô trải qua một quy trình làm sạch tự động (Data Cleaning) như sau:<br>
+ Đọc và định vị cấu trúc linh hoạt: Tự động dò tìm (index) vị trí dòng chứa "Tên Sinh Viên" và dòng bắt đầu danh sách các môn học để trích xuất đúng vùng dữ liệu.<br>
+ Xử lý trùng lặp tên: Thuật toán tự động kiểm tra và thêm các hậu tố đánh số (ví dụ: `_1`, `_2`) cho các sinh viên trùng tên nhau để đảm bảo tên mỗi cột trong Pandas là duy nhất.<br>
+ Chuẩn hóa định dạng số: Toàn bộ điểm số được quét để thay thế các dấu phẩy `,` thành dấu chấm `.` và ép kiểu dữ liệu từ chuỗi (string) sang số thực (float).<br>
+ Loại bỏ dữ liệu lỗi: Lọc bỏ các dòng trắng hoặc các dòng không chứa tên môn học hợp lệ.<br>
+ Loại bỏ ngoại lệ (Outliers): Những sinh viên bị khuyết điểm quá 50% tổng số môn học (có thể do bỏ học, bảo lưu) sẽ bị loại bỏ khỏi danh sách phân tích nhằm tránh làm nhiễu kết quả phân cụm.<br>
+ Điền khuyết dữ liệu (Imputation): Những môn chưa có điểm (giá trị NULL) của các sinh viên hợp lệ sẽ được điền tự động bằng điểm trung bình môn học (Mean) của chính sinh viên đó. Điều này giúp giữ vững phân phối dữ liệu gốc.<br>
+ Phân loại môn học: Phân tách tự động các môn học thành 2 nhóm `Kỹ Thuật` và `Lý Thuyết` dựa trên các bộ từ khóa (keywords) cấu hình sẵn.<br>


