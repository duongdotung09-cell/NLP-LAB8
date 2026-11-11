# Reflection

Sau LAB 07, chúng ta biết cách lắp một Transformer từ self-attention, positional encoding, feed-forward, residual và LayerNorm. Nhưng kiến trúc chỉ là cái khung, giá trị nằm phần lớn ở trọng số đã học.

Trước hết là vấn đề data. Một model train from scratch phải học đồng thời nghĩa của từ, quan hệ ngữ cảnh và bộ phân loại chỉ từ dataset của bài toán. Vài nghìn review không đủ để học tốt vài triệu tham số embedding, nhất là với các từ hiếm.

Pretrained representation giải quyết đúng điểm này. Nhờ pretraining trên một corpus rất lớn, chẳng hạn bằng Masked Language Modeling, model đã học được cách biểu diễn từ trong ngữ cảnh mà không cần nhãn của bài toán cụ thể. Khi đó, dữ liệu downstream không còn phải gánh việc dạy ngôn ngữ từ đầu mà chủ yếu chỉ dùng để điều chỉnh theo task. Trong thí nghiệm, fine-tuning với 500 review đã đạt accuracy 0.812, cao hơn model from scratch dùng 2000 review (0.690).

Về computational cost, cần phân biệt hai loại chi phí. Chi phí pretraining rất lớn nhưng đã được người khác trả một lần và dùng lại được nhiều lần. Chi phí fine-tuning không hề nhỏ, vì model pretrained có nhiều tham số hơn hẳn một Transformer nhỏ, và attention có chi phí tăng theo bình phương độ dài chuỗi. Ở bài này, mỗi epoch fine-tuning trên CPU mất khoảng 400 giây, gấp khoảng 15 lần model from scratch.

Với downstream task, phần cần xây mới thường rất nhỏ. Ở bài này chỉ cần một lớp linear trên biểu diễn [CLS] là có classifier cho sentiment. Cùng encoder đó có thể gắn head khác cho task khác, như token classification ở LAB 09. Đây là lý do pretrained Transformer trở thành nền tảng chung của nhiều ứng dụng NLP.

Fine-tuning là cách điều chỉnh nền tảng đó cho task. Feature extraction với encoder frozen rẻ và ít nguy cơ overfit, nhưng biểu diễn không được tối ưu cho task. Fine-tuning cập nhật cả encoder với learning rate nhỏ nên linh hoạt hơn, đổi lại tốn thời gian hơn.

Cuối cùng là limitation. Pretrained không có nghĩa là hiểu mọi domain. Trong bài này, model train trên review phim vẫn đạt accuracy 0.884 trên review sản phẩm Amazon, nhưng đó là hai domain gần nhau, và khi so cùng độ dài thì kết quả trên Amazon vẫn thấp hơn IMDb một chút. Với domain xa hơn như tin tức hay văn bản khoa học, chất lượng cần được đo lại thay vì giả định.

Vì vậy, người ta thường không train Transformer từ đầu không phải vì kiến thức ở LAB 07 vô ích. Chính kiến thức đó giúp ta hiểu model pretrained đang làm gì, đọc được chi phí của nó và biết khi nào nó có thể sai.
