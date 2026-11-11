# Prediction Before Experiment

File này được viết trước khi chạy bất kỳ thí nghiệm nào và không sửa lại sau khi có kết quả. Việc đối chiếu với kết quả thực tế nằm ở README.md.

## Prediction 1

Prediction: Có. DistilBERT fine-tuned trên IMDb sẽ đạt kết quả tốt hơn Transformer nhỏ train from scratch trên cùng dữ liệu, chênh lệch F1 khoảng 5 đến 10 điểm phần trăm.

Reason: Model pretrained đã có sẵn embedding và các tầng encoder học từ một corpus rất lớn nên chỉ cần điều chỉnh cho bài toán sentiment. Model from scratch phải học đồng thời biểu diễn từ, ngữ cảnh và bộ phân loại chỉ từ vài nghìn review, trong đó nhiều token hiếm gần như không được học.

Expected evidence: Accuracy và F1 trên test của pretrained cao hơn ở cả 3 seed và khoảng dao động giữa các seed không chồng lấn. Pretrained hội tụ chỉ sau 1 đến 2 epoch, còn from scratch cần nhiều epoch hơn và có khoảng cách train/validation lớn hơn.

## Prediction 2

Prediction: C. Frozen encoder sẽ kém hơn fine-tuning toàn bộ, F1 thấp hơn khoảng 3 đến 8 điểm phần trăm.

Reason: Với encoder frozen, biểu diễn [CLS] là đặc trưng chung và chỉ một lớp linear được học, nên model chỉ tạo được một ranh giới tuyến tính trên đặc trưng không được tối ưu cho sentiment. Kết quả vẫn phải tốt hơn hẳn đoán ngẫu nhiên vì biểu diễn pretrained đã chứa thông tin ngữ nghĩa. Fine-tuning cho phép toàn bộ encoder điều chỉnh theo task nên thường tốt hơn, với chi phí thời gian cao hơn.

Expected evidence: F1 của frozen khoảng 0.78 đến 0.84, thấp hơn fine-tuning ở cả 3 seed, nhưng thời gian train ngắn hơn nhiều.

## Prediction 3

Prediction: Model train from scratch sẽ giảm performance mạnh hơn khi giảm từ 100% xuống 10% training data.

Reason: Pretraining đã cung cấp sẵn biểu diễn ngôn ngữ từ dữ liệu bên ngoài, nên dữ liệu downstream chủ yếu chỉ cần dùng để điều chỉnh theo task. Model from scratch không có nguồn thông tin nào khác ngoài training set, nên khi còn 10% dữ liệu thì cả biểu diễn từ lẫn bộ phân loại đều thiếu dữ liệu để học. Mỗi tham số trong số vài triệu tham số embedding cũng được cập nhật ít hơn nhiều.

Expected evidence: Đường cong của pretrained nằm trên đường from scratch tại mọi mức dữ liệu, và khoảng cách giữa hai đường lớn nhất ở mức 10%. Mức giảm F1 từ 100% xuống 10% của from scratch lớn hơn của pretrained.
