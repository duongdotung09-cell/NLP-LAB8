# Calculations

## Exercise 1 - MLM Loss

Dùng logarit tự nhiên. Token đúng là eats, P(eats) = 0.70:

L = -ln(0.70) = 0.3567

1. Nếu eats = 0.95: L = -ln(0.95) = 0.0513. Loss giảm 0.3054 so với 0.3567, tức còn khoảng 1/7 giá trị ban đầu.

2. Nếu eats = 0.01: L = -ln(0.01) = 4.6052. Loss tăng 4.2485 so với 0.3567, tức gấp khoảng 12.9 lần.

3. MLM là bài toán phân loại nhiều lớp trên vocabulary tại mỗi vị trí bị mask, model xuất ra một phân phối xác suất và nhãn đúng là một token. Cross-entropy chính là negative log-likelihood của token đúng nên việc giảm loss tương đương tăng xác suất của token đúng. Loss bằng 0 khi xác suất đạt 1 và tăng rất mạnh khi model gán xác suất rất thấp cho token đúng, như ví dụ 0.01 ở trên. Loss chỉ phụ thuộc xác suất của token đúng, không phụ thuộc cách phân bổ xác suất cho các token còn lại.

## Exercise 2 - Tokenization

Câu: "The cats are playing."

Tokenizer: distilbert-base-uncased. Kết quả chạy ở tokenization.ipynb.

Token sequence: [CLS], the, cats, are, playing, ., [SEP]

Token IDs: 101, 1996, 8870, 2024, 2652, 1012, 102

Special tokens: [CLS] (ID 101) và [SEP] (ID 102). Câu đơn không cần padding nên không có [PAD].

Attention mask: 1, 1, 1, 1, 1, 1, 1

| Đại lượng | Số lượng | Ghi chú |
|---|---|---|
| Words | 4 | The, cats, are, playing (không tính dấu câu) |
| Tokens | 7 | 4 words + dấu chấm + [CLS] + [SEP] |
| Subword tokens | 0 | Không có token nào bắt đầu bằng ## |

Với câu này, cả 4 từ đều có sẵn trong vocabulary nên không bị tách, kể cả cats và playing. Số tokens lớn hơn số words là do dấu chấm và 2 special tokens, không phải do subword.
