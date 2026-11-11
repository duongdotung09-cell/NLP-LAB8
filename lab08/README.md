# LAB 08 - Pretrained Transformers and BERT Fine-Tuning

Model chính: distilbert-base-uncased fine-tuned trên IMDb. Mọi số liệu lấy từ results.csv và fine_tuning.ipynb, mỗi cấu hình chạy 1 lần với seed 42.

## Cách chạy

1. Chạy tokenization.ipynb.
2. Chạy fine_tuning.ipynb. Notebook gọi pretrained_classification.py cho từng cấu hình, ghi results.csv và tạo data_efficiency.png, confusion_matrix.png, confidence_analysis.png. Cấu hình đã có trong results.csv sẽ được bỏ qua.

Lưu ý khi chạy lại: mỗi lần train ghi dự đoán từng mẫu ra file .jsonl trong thư mục đặt bằng biến môi trường LAB08_RUNS (mặc định là thư mục lab08_runs trong thư mục tạm của hệ thống). Các file này không nằm trong bài nộp. Vì vậy nếu giữ nguyên results.csv và chạy notebook trên máy khác, các lần train sẽ được bỏ qua nhưng các cell confusion matrix, confidence và error analysis sẽ báo không tìm thấy file. Để chạy lại toàn bộ, xóa results.csv rồi chạy fine_tuning.ipynb từ đầu, mất khoảng 1 đến 1.5 giờ trên CPU 8 threads khi máy không chạy việc khác.

## Mục 12.1 và 13 - Tokenization và classification head

Kiểm tra tokens.keys() cho thấy tokenizer trả về input_ids, token_type_ids và attention_mask.

- input_ids là chỉ số token trong vocabulary.
- attention_mask đánh dấu token thật (1) và padding (0).
- DistilBERT không dùng token_type_ids.

Classification head là lớp Dropout + Linear đặt trên biểu diễn [CLS], ở class PretrainedClassifier trong pretrained_classification.py. Head là thành phần task-specific: encoder được dùng lại, còn head được khởi tạo mới cho 2 lớp positive và negative.

## Mục 14 - Experiment 1: Pretrained vs From Scratch

Hai model dùng chung tokenizer, dataset, split, metric và T = 128. Số epoch khác nhau có chủ đích: from scratch được train tối đa 10 epoch vì 2 epoch là không đủ để nó hội tụ. Cả hai đều giữ epoch có validation accuracy tốt nhất.

| Model | Accuracy | F1 | Training time | # Parameters |
|---|---|---|---|---|
| Transformer from scratch | 0.690 | 0.719 | 265.9 s (10 epoch) | 4,453,634 |
| Pretrained Transformer | 0.866 | 0.865 | 810.5 s (2 epoch) | 66,364,418 |

1. Representation khác nhau như thế nào?
   - From scratch bắt đầu với embedding và encoder ngẫu nhiên, chỉ học từ 2000 review. Model này overfit sớm: ở mức 50% dữ liệu, train loss giảm từ 0.71 xuống 0.21, còn validation loss thấp nhất ở epoch 4 (0.634) rồi tăng lên 1.009.
   - Pretrained bắt đầu từ biểu diễn theo ngữ cảnh đã học trên corpus lớn. Ngay cả khi frozen hoàn toàn, biểu diễn [CLS] của nó vẫn đạt accuracy 0.781, cao hơn from scratch.

2. Training data cần thiết khác nhau như thế nào? Pretrained chỉ với 500 review đã đạt 0.812, cao hơn from scratch dùng 2000 review (0.690). Chi tiết ở Experiment 3.

3. Chi phí tính toán khác nhau như thế nào?
   - Pretrained có số tham số gấp khoảng 15 lần.
   - Mỗi epoch tốn khoảng 405 giây, so với khoảng 27 giây của from scratch, cũng gấp khoảng 15 lần.
   - Peak memory là 2980 MB so với 845 MB.
   - Tổng thời gian train chỉ chênh khoảng 3 lần vì pretrained cần ít epoch hơn.
   - Chi phí pretraining ban đầu không tính ở đây vì đã được trả trước.

4. Kết quả có phù hợp với prediction không?
   - Về hướng: đúng.
   - Về độ lớn: Prediction 1 dự đoán chênh 5 đến 10 điểm F1, thực tế chênh 14.6 điểm F1 và 17.6 điểm accuracy, tức tôi đã đánh giá thấp mức chênh.
   - Về hội tụ: đúng như dự đoán. Pretrained đạt validation accuracy 0.830 ngay sau epoch 1 (mức 50% dữ liệu), còn from scratch cần nhiều epoch và overfit.
   - Phần "ở cả 3 seed" trong prediction không kiểm tra được vì chỉ chạy 1 seed.

## Mục 15 - Experiment 2: Feature Extraction vs Fine-Tuning

Frozen dùng learning rate 1e-3 và tối đa 30 epoch cho head, vì learning rate 2e-5 của fine-tuning là quá nhỏ để một lớp linear hội tụ. Thời gian của frozen đã gồm cả bước tính [CLS] cho train và validation.

| Mode | Encoder trainable? | Accuracy | F1 | Training time |
|---|---|---|---|---|
| Frozen | No | 0.781 | 0.758 | 110.8 s |
| Fine-tuning | Yes | 0.866 | 0.865 | 810.5 s |

Vì sao frozen encoder vẫn có thể thực hiện classification? Biểu diễn [CLS] pretrained đã chứa thông tin ngữ nghĩa của cả sequence, nên một lớp linear chỉ với 1,538 tham số vẫn tìm được ranh giới tách hai lớp. Kết quả 0.781 vượt xa mức đoán ngẫu nhiên 0.5 và vượt cả model from scratch 4.4 triệu tham số.

Vì sao fine-tuning có thể cải thiện kết quả? Encoder được cập nhật theo task loss, nên biểu diễn được điều chỉnh để làm nổi bật thông tin sentiment. Frozen có precision 0.849 nhưng recall chỉ 0.684, tức ranh giới trên đặc trưng cố định bị lệch về negative. Fine-tuning cân bằng hơn với precision 0.872 và recall 0.858.

Đối chiếu Prediction 2: Chọn C (kém hơn) là đúng. Tôi dự đoán F1 của frozen khoảng 0.78 đến 0.84 và chênh 3 đến 8 điểm, thực tế F1 là 0.758 và chênh 10.7 điểm (8.5 điểm accuracy). Dự đoán frozen train nhanh hơn nhiều cũng đúng: 110.8 s so với 810.5 s.

## Mục 16 - Experiment 3: Amount of Training Data

Validation (500) và test (1000) giữ cố định. Tập con nhỏ nằm trong tập con lớn hơn và cân bằng 2 lớp. Ngoài pretrained, from scratch cũng được chạy ở cả 4 mức để trả lời Prediction 3 và câu Q1 ở mục 28. Đường cong ở data_efficiency.png gồm 2 panel accuracy và F1, mỗi panel có đường tham chiếu.

| Training data | Accuracy pretrained | F1 pretrained | Accuracy from scratch | F1 from scratch |
|---|---|---|---|---|
| 10% (200) | 0.675 | 0.707 | 0.566 | 0.605 |
| 25% (500) | 0.812 | 0.815 | 0.609 | 0.517 |
| 50% (1000) | 0.833 | 0.834 | 0.663 | 0.700 |
| 100% (2000) | 0.866 | 0.865 | 0.690 | 0.719 |

Pretraining có giúp model sử dụng dữ liệu downstream hiệu quả hơn không? Có, dựa trên toàn bộ đường cong:

- Đường pretrained nằm trên from scratch ở cả 4 mức.
- Pretrained với 25% dữ liệu đã vượt from scratch với 100%, tức cần khoảng 1/4 lượng dữ liệu để đạt kết quả tốt hơn.
- From scratch ở 10% chỉ đạt 0.566, gần mức đoán ngẫu nhiên.

Chỉ nhìn F1 sẽ dễ kết luận sai. F1 của from scratch đi xuống từ 0.605 ở 10% còn 0.517 ở 25%, trong khi accuracy tăng đều. Lý do là model from scratch dự đoán lệch lớp (recall 0.418 ở mức 25%), nên F1 của một lớp dao động theo độ lệch đó. Vì vậy đường accuracy được dùng làm căn cứ chính.

Đối chiếu Prediction 3: chỉ đúng một phần.

- Đúng: pretrained luôn nằm trên from scratch.
- Không đúng như dự đoán: khi giảm từ 100% xuống 10%, pretrained mất 0.191 accuracy, còn from scratch chỉ mất 0.124 vì đã sát sàn 0.5.
- Khoảng cách giữa hai model lớn nhất ở 25% (0.203), không phải ở 10% như dự đoán.
- Ở 10%, fine-tuning chỉ có 26 bước cập nhật, nên điểm này chưa ổn định.

## Mục 17 - Experiment 4: Maximum Sequence Length

| Max length | Accuracy | F1 | Training time | CPU memory | Thời gian 1 bước train |
|---|---|---|---|---|---|
| 64 | 0.799 | 0.808 | 747.1 s | 2197 MB | 2.67 s |
| 128 | 0.866 | 0.865 | 810.5 s | 2980 MB | 5.05 s |
| 256 | 0.892 | 0.891 | 2852.2 s | 4593 MB | 11.10 s |

Cách đo:

- Training time là thời gian thực của cả lần chạy.
- Lần chạy T = 64 bị chậm do máy đang chạy việc khác cùng lúc, nên cột này không dùng để so sánh tỉ lệ.
- Cột cuối được đo lại có kiểm soát trong fine_tuning.ipynb: cùng tiến trình, batch 16, 3 giá trị T xen kẽ qua 5 vòng, lấy trung vị.
- CPU memory là peak working set của tiến trình.

Liên hệ O(T^2):

- Ma trận attention có kích thước T x T, nên khi T tăng gấp đôi, số điểm attention tăng 4 lần.
- Tuy vậy, thời gian một bước chỉ tăng khoảng 2.2 lần mỗi khi T gấp đôi (0.53, 1.00, 2.20 so với T = 128).
- Lý do: với d = 768, phần tính điểm attention chỉ chiếm khoảng 1.4%, 2.7% và 5.3% số phép tính ở T = 64, 128, 256. Phần projection và feed-forward tăng tuyến tính theo T vẫn chiếm phần lớn.
- Phần O(T^2) chỉ áp đảo khi T lớn hơn nhiều, cụ thể khi 2T^2d > 12Td^2, tức T > 6d.
- Memory tăng nhanh hơn: từ 2980 MB lên 4593 MB khi T từ 128 lên 256.

Tăng sequence length luôn làm model tốt hơn không? Không phải luôn luôn.

- Trên IMDb, tăng T có cải thiện: accuracy tăng 6.7 điểm từ 64 lên 128, nhưng chỉ tăng 2.6 điểm từ 128 lên 256, trong khi thời gian mỗi bước tăng 2.2 lần.
- Lợi ích có được là vì review IMDb dài: trung vị 223 token và 88.3% review bị cắt ở 128 token.
- Với Amazon, trung vị chỉ 89 token. Code này pad mọi chuỗi đến max_length, nên tăng T lên 256 ở đó chủ yếu là thêm padding và chi phí mà không thêm thông tin.

## Mục 18 - Experiment 5: Domain Shift

Domain A là IMDb movie reviews, domain B là Amazon product reviews (1000 review cân bằng từ split test của amazon_polarity). Model train trên A, test trên A và B, không train thêm trên B.

| Test domain | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| IMDb (in-domain) | 0.866 | 0.872 | 0.858 | 0.865 |
| Amazon (out-of-domain) | 0.884 | 0.903 | 0.860 | 0.881 |

Accuracy so trong cùng nhóm độ dài:

| | IMDb | Amazon |
|---|---|---|
| Trung vị số token | 223 | 89 |
| Tỉ lệ bị cắt ở 128 token | 88.3% | 29.2% |
| Accuracy, review không bị cắt | 0.915 | 0.898 |
| Accuracy, review bị cắt | 0.860 | 0.849 |

Pretrained model có nghĩa là model đã "hiểu mọi domain" không? Không. Kết quả cho thấy:

- Performance trên Amazon không giảm mà còn cao hơn, nhưng lý do chủ yếu là review Amazon ngắn hơn và ít bị cắt.
- Trong cùng nhóm độ dài, Amazon lại thấp hơn IMDb 1 đến 2 điểm. Mức chênh này nhỏ và chưa có ý nghĩa thống kê với cỡ mẫu này.
- Movie review và product review là hai domain gần nhau, cùng là văn bản đánh giá với từ vựng sentiment tương tự. Thí nghiệm này chỉ cho thấy model chuyển giao tốt giữa hai domain gần nhau.
- Nó không cho biết gì về các domain xa hơn như tin tức hay văn bản khoa học.

## Mục 19 - Application: Sentiment Classification

Model chính: fine-tuning, seed 42, T = 128, test trên 1000 review IMDb. Confusion matrix ở confusion_matrix.png.

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 0.866 | 0.872 | 0.858 | 0.865 |

| | Pred negative | Pred positive |
|---|---|---|
| True negative | 437 | 63 |
| True positive | 71 | 429 |

Hai loại lỗi gần cân bằng, false negatives nhiều hơn một chút. Phân tích lỗi ở error_analysis.md.

## Mục 22 - Ablation

Chọn A: Frozen vs Fine-tuned (Experiment 2).

- Biến thay đổi: encoder có được train hay không, kèm learning rate và số epoch phù hợp với từng chế độ.
- Biến giữ nguyên: model, tokenizer, split, T = 128 và seed.
- Kết quả: fine-tuning tốt hơn 8.5 điểm accuracy và 10.7 điểm F1, đổi lại thời gian train tăng khoảng 7.3 lần (810.5 s so với 110.8 s).

Các thí nghiệm B, C, D cũng đã được chạy ở các mục 14, 16 và 17.

## Mục 23 - Reproducibility

Model: distilbert-base-uncased, thêm Dropout 0.1 và Linear trên [CLS]. Model from scratch: 4 layers, 4 heads, d_model 128, d_ff 256, learned positional embedding.

Tokenizer: distilbert-base-uncased.

Dataset: stanfordnlp/imdb, đã bỏ thẻ br và review trùng lặp. Domain B: fancyzhx/amazon_polarity, 1000 review từ split test, ghép title và content.

Train split: 2000 review lấy từ IMDb train, cân bằng 2 lớp.

Validation split: 500 review lấy từ IMDb train, không trùng với train.

Test split: 1000 review lấy từ IMDb test, cân bằng 2 lớp, không trùng với train.

Max sequence length: 128 (Experiment 4 thêm 64 và 256).

Batch size: 16.

Learning rate: fine-tuning 2e-5, frozen head 1e-3, from scratch 5e-4. Fine-tuning và from scratch giảm tuyến tính về 0.

Epochs: fine-tuning 2, frozen head tối đa 30, from scratch tối đa 10. Giữ trọng số của epoch có validation accuracy tốt nhất.

Optimizer: AdamW, gradient clipping 1.0.

Random seed: 42 cho model và tập con train. Seed chia dữ liệu cố định là 2026.

Hardware: Intel Core i9-13900H, không dùng GPU, 8 CPU threads. Python 3.13.12, torch 2.12.0, transformers 5.14.1.

Training time: ghi cho từng lần chạy ở cột train_time_s của results.csv. Thời gian đã gồm đánh giá validation sau mỗi epoch. Máy là laptop dùng chung nên thời gian tuyệt đối có dao động.

Quy mô dữ liệu nhỏ hơn IMDb đầy đủ vì chạy trên CPU, nên kết quả tuyệt đối sẽ thấp hơn khi train trên 25000 review.

## Mục 27 - Individual Learning Check

1. Pretraining huấn luyện model trên corpus lớn để học biểu diễn ngôn ngữ, thường không gắn với task cụ thể. Fine-tuning tiếp tục huấn luyện model đó trên dataset của task cụ thể.

2. MLM che một số token trong câu và yêu cầu model dự đoán token bị che từ ngữ cảnh, nhờ đó model học biểu diễn mà không cần nhãn do người gán.

3. Model chỉ xử lý số, nên tokenizer chuyển raw text thành các token ID theo đúng vocabulary mà model đã được huấn luyện.

4. Subword token giúp xử lý từ hiếm hoặc từ mới bằng cách tách thành các phần có trong vocabulary, nhờ đó giảm token [UNK] và giữ vocabulary ở kích thước hữu hạn.

5. [CLS] được đặt ở đầu sequence, và vector đầu ra của nó thường được dùng làm đại diện cho toàn bộ sequence để đưa vào classification head.

6. Attention mask đánh dấu token thật và padding để padding không tham gia attention như nội dung thực của sequence.

7. Frozen encoder là encoder có trọng số giữ nguyên trong khi train, chỉ classification head được cập nhật.

8. Fine-tuning toàn bộ cập nhật mọi tham số trainable: embedding, attention, feed-forward, LayerNorm trong encoder và classification head.

9. Vì biểu diễn ngôn ngữ đã được học từ corpus lớn, dữ liệu downstream chỉ cần dùng để điều chỉnh theo task thay vì phải học ngôn ngữ từ đầu.

10. Không. Pretrained model vẫn bị ảnh hưởng khi phân phối dữ liệu lúc test khác với dữ liệu fine-tuning.

11. Self-attention tính điểm giữa mọi cặp token nên ma trận attention có kích thước T x T. Khi T tăng gấp đôi, số điểm cần tính tăng gấp bốn, tức O(T^2).

12. Không. Confidence chỉ là xác suất lớn nhất model xuất ra, model vẫn có thể rất tự tin mà sai.

## AI assistance statement

- AI-assisted code: Claude Code
- Purpose: đọc documentation, tra cứu API của Hugging Face transformers, datasets và PyTorch, debugging, giải thích runtime errors, viết boilerplate và visualization, đúng phạm vi cho phép ở mục 24 của W8.pdf. Không dùng AI để thay thế MLM calculation, prediction trước experiment, giải thích pretraining/fine-tuning, phân tích kết quả, error analysis và reflection
- What was generated: boilerplate code (tải IMDb và Amazon bằng datasets, chia split cân bằng lớp và bỏ review trùng, tokenize, vòng lặp train/evaluate, ghi results.csv và file dự đoán từng mẫu, chạy mỗi cấu hình trong một tiến trình riêng để đo peak memory), code vẽ hình (data_efficiency.png, confusion_matrix.png, confidence_analysis.png), cell đo thời gian một bước train ở Experiment 4, định dạng bảng trong các file .md
- What was modified: đổi tiêu chí chọn epoch từ validation F1 sang validation accuracy sau khi phát hiện model yếu đoán toàn positive vẫn có F1 0.667, rồi chạy lại toàn bộ thí nghiệm; thêm panel accuracy và đường tham chiếu vào data_efficiency.png vì chỉ nhìn F1 dễ kết luận sai; giảm quy mô dữ liệu và chỉ chạy seed 42 vì chạy trên CPU; chọn 8 threads và eager attention theo kết quả đo tốc độ trên máy; chạy lại T = 64 và T = 256 và thêm phép đo có kiểm soát khi thời gian bị nhiễu do máy dùng chung; thêm phân tích độ dài và truncation cho Experiment 5; các con số trong README.md, error_analysis.md, reflection.md được cập nhật theo output của notebook (chỉ thay số liệu, không thay phần phân tích); prediction.md không sửa sau khi chạy
- What was verified: chạy thử cả 3 chế độ finetune, frozen, scratch trên dữ liệu rất nhỏ trước khi chạy thật; kiểm tra train, validation, test cân bằng lớp và không có review trùng nhau; kiểm tra tập Amazon giống hệt nhau giữa các lần chạy; kiểm tra chế độ frozen chỉ có 1,538 tham số trainable; tính lại các giá trị -ln(p) ở calculations.md bằng Python; chạy tokenization.ipynb và fine_tuning.ipynb từ đầu đến cuối và đối chiếu các con số trong README.md, error_analysis.md, reflection.md với output của notebook và results.csv; với mỗi lỗi trong error_analysis.md, đọc cả phần 128 token model thực sự thấy và phần bị cắt; đọc lại các file .md để đảm bảo đúng yêu cầu từng mục trong W8.pdf
