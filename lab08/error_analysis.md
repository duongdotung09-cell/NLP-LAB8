# Error Analysis

Model phân tích: DistilBERT fine-tuning, T = 128, seed 42, IMDb test 1000 review. Số liệu lấy từ fine_tuning.ipynb.

## Mục 20 - Phân tích lỗi

Tổng số lỗi: 63 false positives và 71 false negatives. Với mỗi lỗi, tôi đọc cả phần model thực sự thấy (128 token đầu) và phần bị cắt.

| Text | True | Pred | Confidence | Error type | Hypothesis |
|---|---|---|---|---|---|
| "As an avid Gone With the Wind fan... the film on its own was a wonderful piece... I consider the six hours I spent watching the "Scarlett" miniseries to be some of the worst-spent hours of my life" | negative | positive | 0.973 | long-range dependency, tokenization (truncation) | Review dài 391 từ. 128 token đầu chủ yếu khen bản phim gốc và cuốn sách, còn câu chê miniseries bị cắt đúng sau "to be some of", nên model không thấy kết luận. |
| "This show lasted for most of the 1980s, and had its moments, but plots were usually dishwater thin and often painfully unfunny... many people remember it fondly, but I prefer to remember the decade for alternative comedy" | negative | positive | 0.946 | ambiguous sentiment | Review vừa khen vừa chê. Đoạn cuối mà model thấy là "fond memories", "smiling", "really is clever", còn mệnh đề "but I prefer..." bị cắt. |
| "his incredible movie is absolutely in his image... it's very pleasant to laugh at Zep (mentally retarded persons are not funny, except in this movie.)" | negative | positive | 0.931 | sarcasm, context | "incredible", "pleasant", "Absolute Love" được dùng để mỉa mai. Phần model thấy chủ yếu là tóm tắt cốt truyện, không có đánh giá trực tiếp, nên model bám vào từ tích cực bề mặt. |
| "Forget everything that you have ever read... most reviews... say... the acting was amateurish... the entire thing amounted to a hideous waste of time" | positive | negative | 0.980 | context, long-range dependency | Người viết trích lời chê của người khác để phản bác. Model không phân biệt lời trích với quan điểm của người viết, và câu kết luận "I think it's better than Snakes on a Plane" nằm cuối review 674 từ, ngoài 128 token. |
| "The visuals are dynamite... The DVD loses points, however, for the pointless addition of new music... uniformly uninteresting... rather bad" | positive | negative | 0.970 | ambiguous sentiment, long-range dependency | Phần model thấy kết thúc bằng một loạt lời chê nhạc mới. Lời khen tổng kết "This disc really deserves your attention" nằm ở cuối và bị cắt. |
| "These days, you rarely come by a kid's show that does not involve... they all lack good acting, originality, and good story telling" | positive | negative | 0.960 | context, long-range dependency | Đoạn mở đầu chê các show khác để làm nền so sánh. Show được khen và điểm "9/10" chỉ xuất hiện sau 128 token. |
| "This has to be one of, if not THE greatest Mob/Crime films of all time... Master P's acting skills make you actually believe he is Italian!... and I have the brain capacity of an earth worm." | negative | positive | 0.983 | sarcasm | Lỗi tự tin nhất. Review chỉ có 65 từ nên model thấy toàn bộ, không phải lỗi truncation. Bề mặt toàn từ khen, còn tín hiệu mỉa mai cần kiến thức ngoài văn bản (Master P là rapper) và câu tự giễu ở cuối. |
| "Every episode I saw when I was an innocent child was stupid. There were some funny looking puppets called "The Mitts"..." | positive | negative | 0.968 | label ambiguity | Lỗi tự tin thứ hai. Câu đánh giá rõ nhất là "was stupid", phần còn lại chỉ mô tả chương trình. Đọc toàn bộ review cũng khó khẳng định nó positive, nên nhãn có thể dựa trên điểm số người viết chấm mà text không thể hiện. |

Hàng 1 đến 3 là false positive, hàng 4 đến 6 là false negative, hàng 7 và 8 là 2 lỗi có confidence cao nhất.

Nhận xét chung:

- Trong 8 lỗi trên, 6 review dài hơn 128 token và có phần quyết định sentiment nằm ngoài đoạn model thấy.
- Accuracy giảm rõ theo độ dài review: 0.915 với review từ 100 từ trở xuống, 0.902 với 101-200 từ, 0.844 với 201-400 từ và 0.721 với review trên 400 từ.
- Kết quả này phù hợp với Experiment 4: khi tăng T lên 256, accuracy tăng từ 0.866 lên 0.892.
- Các lỗi còn lại đến từ sarcasm, lời trích của người khác và nhãn mơ hồ. Các lỗi này không giải quyết được chỉ bằng cách tăng T.

## Mục 21 - Confidence analysis

confidence = max(softmax(logits)). Ngưỡng: Low từ 0.5 đến dưới 0.7, Medium từ 0.7 đến dưới 0.9, High từ 0.9 trở lên. Biểu đồ ở confidence_analysis.png.

| Nhóm | Số mẫu | Số sai | Mean confidence | Accuracy |
|---|---|---|---|---|
| Low | 115 | 41 | 0.604 | 0.643 |
| Medium | 227 | 63 | 0.816 | 0.722 |
| High | 658 | 30 | 0.965 | 0.954 |

Confidence cao có đồng nghĩa với prediction đúng không? Không.

- Nhóm High có accuracy cao nhất (0.954), nên confidence vẫn có ích để xếp hạng mức tin cậy giữa các dự đoán.
- Tuy vậy, nhóm High vẫn có 30 lỗi, chiếm 22% trong tổng số 134 lỗi, và 2 lỗi trong bảng trên có confidence trên 0.96.
- Nhóm Medium bị over-confident: confidence trung bình 0.816 nhưng accuracy thực tế chỉ 0.722.
- Vì vậy confidence không phải là xác suất đúng đã được hiệu chỉnh.

## Mục 28 - Câu hỏi bắt buộc

### Q1. Model nào hưởng lợi nhiều hơn khi training data ít?

Observation: Ở mọi mức dữ liệu, pretrained đều có accuracy cao hơn from scratch.

| Training data | Pretrained | From scratch | Chênh lệch |
|---|---|---|---|
| 10% | 0.675 | 0.566 | 0.109 |
| 25% | 0.812 | 0.609 | 0.203 |
| 50% | 0.833 | 0.663 | 0.170 |
| 100% | 0.866 | 0.690 | 0.176 |

Evidence:

- Pretrained chỉ với 500 review (25%) đã đạt 0.812, cao hơn from scratch dùng toàn bộ 2000 review (0.690).
- Khi giảm từ 100% xuống 10%, pretrained mất 0.191 accuracy, from scratch mất 0.124. Tuy nhiên from scratch ở 10% chỉ còn 0.566, sát mức đoán ngẫu nhiên 0.5, nên nó gần như không còn gì để mất.
- Ở 10%, fine-tuning chỉ có 26 bước cập nhật (200 review, batch 16, 2 epoch) và validation accuracy giảm từ 0.710 ở epoch 1 xuống 0.648 ở epoch 2, nên điểm này chưa ổn định.

Interpretation:

- Model pretrained hưởng lợi nhiều hơn khi dữ liệu ít: nó cần khoảng 1/4 lượng dữ liệu để vượt model from scratch dùng đủ dữ liệu.
- Prediction 3 chỉ đúng một phần. Nếu tính mức giảm tuyệt đối thì pretrained giảm nhiều hơn, vì from scratch đã chạm sàn gần mức ngẫu nhiên.
- Khoảng cách giữa hai model lớn nhất ở mức 25% chứ không phải 10% như đã dự đoán.
- Đây là kết quả của 1 seed, nên các chênh lệch nhỏ như giữa 50% và 100% chưa đủ để kết luận.

### Q2. Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?

Observation: Pretrained đạt accuracy 0.866 và F1 0.865. From scratch đạt accuracy 0.690 và F1 0.719.

Evidence: Pretrained tốt hơn ở cả 4 mức dữ liệu. Model from scratch bị overfit sớm: ở mức 50% dữ liệu, train loss giảm từ 0.71 xuống 0.21 trong khi validation loss thấp nhất ở epoch 4 (0.634) rồi tăng lên 1.009.

Interpretation:

- Điều đó chỉ chứng minh rằng với dataset, split và cấu hình này, model pretrained cho kết quả tốt hơn.
- Kết quả phù hợp với giả thuyết representation pretrained có ích, nhưng chưa tách riêng được tác động của pretraining. Hai model còn khác nhau về kích thước (66 triệu so với 4.4 triệu tham số), learning rate và số epoch.
- Nó không chứng minh model hiểu ngôn ngữ. Các lỗi sarcasm và lời trích ở mục 20 cho thấy model vẫn dựa nhiều vào từ ngữ bề mặt.

### Q3. Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?

Observation: Trong thí nghiệm này, frozen không gần bằng fine-tuned: accuracy 0.781 so với 0.866, F1 0.758 so với 0.865.

Evidence: Frozen chỉ train 1,538 tham số nhưng vẫn vượt model from scratch train 4,453,634 tham số (accuracy 0.781 so với 0.690).

Interpretation:

- Nếu frozen gần bằng fine-tuned, điều đó có nghĩa biểu diễn pretrained đã đủ để tách các lớp bằng một ranh giới tuyến tính. Khi đó frozen là lựa chọn hợp lý vì rẻ hơn nhiều.
- Ở đây điều kiện đó không xảy ra. Tuy vậy, việc frozen vượt from scratch cho thấy biểu diễn [CLS] pretrained đã chứa nhiều thông tin sentiment ngay cả khi chưa được điều chỉnh.

### Q4. Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?

Observation: Fine-tuning tốt hơn frozen 8.5 điểm accuracy và 10.7 điểm F1.

Evidence:

- Frozen có precision 0.849 nhưng recall chỉ 0.684, tức ranh giới tuyến tính trên đặc trưng cố định bị lệch về phía negative. Fine-tuning cân bằng hơn: precision 0.872, recall 0.858.
- Đổi lại, fine-tuning tốn 810.5 giây, còn frozen chỉ 110.8 giây.

Interpretation:

- Biểu diễn pretrained chung chưa làm nổi bật đủ thông tin mà task cần. Khi được cập nhật, encoder điều chỉnh được biểu diễn theo sentiment.
- F1 của frozen bị kéo xuống một phần do ngưỡng lệch, nên nên xem cả accuracy (chênh 8.5 điểm) chứ không chỉ F1.

### Q5. Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?

Observation: Performance không giảm. Model train trên IMDb đạt accuracy 0.884 trên Amazon, cao hơn 0.866 trên chính IMDb.

Evidence:

- Hai tập khác nhau rõ về độ dài. IMDb có trung vị 223 token và 88.3% review bị cắt ở 128 token. Amazon có trung vị 89 token và chỉ 29.2% bị cắt.
- Khi so trong cùng nhóm độ dài, Amazon lại thấp hơn IMDb một chút. Với review không bị cắt, accuracy là 0.898 trên Amazon so với 0.915 trên IMDb. Với review bị cắt, là 0.849 so với 0.860.
- Nhóm không bị cắt của IMDb chỉ có 117 review, nên các chênh lệch 1 đến 2 điểm này chưa có ý nghĩa thống kê.

Interpretation:

- Accuracy cao hơn trên Amazon chủ yếu do review Amazon ngắn hơn và ít bị cắt, không phải vì model làm tốt hơn trên domain mới.
- Movie review và product review là hai domain gần nhau, cùng là văn bản đánh giá với từ vựng sentiment tương tự, nên domain shift ở đây nhỏ.
- Nếu performance giảm mạnh trên một domain xa hơn, nguyên nhân có thể là:
  - từ vựng và cách diễn đạt sentiment khác;
  - cùng một từ mang cực tính khác nhau giữa các domain;
  - khác biệt về độ dài làm thay đổi tác động của truncation;
  - model đã học các dấu hiệu riêng của review phim;
  - nhãn được tạo theo quy tắc khác.

### Q6. Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?

Observation: Trên IMDb, tình huống này chưa xảy ra. Accuracy tăng 0.799, 0.866, 0.892 khi T lần lượt là 64, 128, 256, nhưng mức tăng nhỏ dần: 6.7 điểm từ 64 lên 128 và 2.6 điểm từ 128 lên 256.

Evidence:

- Thời gian một bước train tăng gấp 2.2 lần, peak memory tăng từ 2980 MB lên 4593 MB khi T từ 128 lên 256.
- IMDb có trung vị 223 token và 88.3% review bị cắt ở 128, nên T lớn hơn vẫn đem thêm nội dung thật.
- Error analysis ở mục 20 cũng cho thấy nhiều lỗi do kết luận nằm sau token 128.

Interpretation: Không nên, trừ khi error analysis cho thấy lỗi tập trung ở các văn bản dài bị cắt. Nên chọn T nhỏ nhất nằm trên vùng performance đã bão hòa, vì chi phí attention tăng theo O(T^2) còn lợi ích thì không tăng tương ứng.
