# Tóm tắt ràng buộc chính thức Task 1

Trạng thái bằng chứng: thông báo của BTC do thành viên nhóm cung cấp ngày
27-08-2026 và contract công khai của Codabench. Nếu hai nguồn khác nhau, thông báo
BTC mới và cụ thể hơn được ưu tiên; điểm chưa rõ phải hỏi lại BTC.

## Contract đã khóa

- Đầu vào là mã câu hỏi và câu hỏi pháp luật tiếng Việt.
- Submission là tệp ZIP chỉ chứa submission.json.
- Mỗi câu hỏi trả tối đa 5 document ID hợp lệ, không trùng. Trả quá 5 làm Recall
  và Precision của câu đó bằng 0.
- Macro Recall là tiêu chí xếp hạng chính; Macro Precision là tiêu chí phân hạng
  khi Recall bằng nhau.
- Task 1 chỉ được dùng dữ liệu Task 1 do BTC cung cấp. Không dùng dữ liệu Task 2,
  dữ liệu ngoài hoặc tăng cường dữ liệu.
- Tổng số tham số của mọi thành phần chạy trong một hệ thống Task 1 phải nhỏ hơn
  4 tỷ, kể cả embedding. LoRA và quantization không thay đổi số tham số danh nghĩa.
- Không gọi API trong phương pháp dự thi. Mô hình phải mở, cầm được và do đội kiểm soát.

## Tiền xử lý bắt buộc cho train

BTC xác nhận train Task 1 có passage rỗng và passage trùng; public/private test
không có đáp án context rỗng hoặc trùng. Vì vậy pipeline train phải:

1. Giữ nguyên bản raw bất biến và checksum.
2. Chuẩn hóa Unicode NFC và khoảng trắng chỉ trong bản dẫn xuất.
3. Gắn cờ passage rỗng sau chuẩn hóa; không biến nó thành negative hợp lệ.
4. Nhóm passage trùng bằng nội dung chuẩn hóa nhưng giữ toàn bộ document ID và
   provenance để không phá contract nhãn.
5. Báo số bản ghi rỗng, nhóm trùng, ID xung đột và thay đổi trước/sau xử lý.
6. Chia train/validation theo nhóm trùng để cùng một passage không rò sang hai split.

Không áp dụng kết luận này cho Task 2 nếu chưa có audit riêng.

## Precision và Recall

Chưa có bằng chứng công khai để kết luận các đội đang “Recall cao, Precision thấp”
hoặc ngược lại: endpoint leaderboard chi tiết trả HTTP 403 khi không đăng nhập.
Điều xác minh được là cơ chế metric tạo trade-off:

- tăng k thường không làm giảm Recall trên cùng candidate ordering, nhưng có thể
  giảm Precision;
- vì Recall là tiêu chí chính, fixed top-5 là anchor an toàn hơn adaptive-k;
- chỉ dùng threshold hoặc adaptive-k khi paired held-out cho thấy không mất Recall
  và có Precision tốt hơn trên nhiều seed/fold.

Không dùng F2 của ALQAC để thay metric UIT.

## Nguồn

- [UIT-DSC 2026 LegalIR trên Codabench](https://www.codabench.org/competitions/17715/)
- Thông báo BTC do người dùng cung cấp: giới hạn tham số, API, dữ liệu, đóng gói,
  passage rỗng/trùng và cấm dùng chéo Task 1/Task 2.
- [ViDRILL](https://aclanthology.org/2025.vlsp-1.17/): prior art cho hybrid
  retrieval và hard-negative; không phải luật hay kết quả UIT.

