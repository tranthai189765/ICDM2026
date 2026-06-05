# Phương pháp sinh Ambiguous Buyer Intent bằng Dialogue-Level NILC

## 1. Mục tiêu

Mục tiêu của phương pháp là khám phá và sinh các mẫu huấn luyện cho bài toán nhận diện **ambiguous buyer intent** trong hội thoại mua bán, đặc biệt trong các kịch bản **bargaining** và **recommendation**.

Khác với cách xem từng câu nói riêng lẻ như một đơn vị intent, phương pháp này xem **toàn bộ đoạn hội thoại buyer-seller** là đơn vị phân tích. Intent của buyer thường không thể được suy ra chính xác từ một utterance đơn lẻ, vì nó phụ thuộc vào:

- mục tiêu tiềm ẩn của buyer;
- ràng buộc về giá, nhu cầu, ngân sách, độ phù hợp;
- phản hồi và chiến lược của seller;
- trạng thái ra quyết định của buyer trong quá trình tương tác;
- sự chồng lấn giữa nhiều mục tiêu như mặc cả, so sánh, xin gợi ý, trì hoãn quyết định, hoặc kiểm tra độ phù hợp.

Do đó, embedding trong phương pháp này không được xây dựng từ utterance riêng lẻ, mà từ **dialogue-level intent representation**.

## 2. Ý tưởng chính

Gọi:

\[
D_i = \{u_1, u_2, ..., u_T\}
\]

là đoạn hội thoại thứ \(i\), gồm nhiều lượt nói của buyer và seller.

Thay vì mã hóa từng utterance \(u_t\), ta dùng LLM để trích xuất một intent frame:

\[
F_i = LLM\_Extract(D_i)
\]

Sau đó mới mã hóa intent frame thành embedding:

\[
z_i = PTE(F_i)
\]

Trong đó:

- \(D_i\): toàn bộ đoạn hội thoại buyer-seller;
- \(F_i\): thông tin có cấu trúc do LLM trích xuất từ hội thoại;
- \(PTE(\cdot)\): prompt/text encoder;
- \(z_i\): embedding biểu diễn intent ở cấp hội thoại.

Sau khi có tập embedding \(\{z_i\}_{i=1}^{N}\), ta phân cụm để khám phá các nhóm buyer intent mới. Với mỗi cụm, LLM đọc các dialogue hoặc intent frame tiêu biểu để dựng ra intent label, intent description, ambiguity reason và candidate related intents. Các intent này sau đó được dùng để phát hiện hoặc sinh thêm ambiguous training samples.

## 3. Dialogue-Level Intent Frame

LLM được dùng để biến một đoạn hội thoại thô thành một representation có cấu trúc. Representation này nên giữ lại những thông tin cần thiết để suy ra intent của buyer, đồng thời dùng seller response như ngữ cảnh phụ trợ.

Một intent frame có thể có dạng:

```json
{
  "dialogue_type": "bargaining_or_recommendation",
  "buyer_latent_intent": "short intent phrase",
  "buyer_goal": "what the buyer is ultimately trying to achieve",
  "buyer_constraints": [
    "price concern",
    "uncertain preference",
    "budget limit",
    "comparison with alternatives"
  ],
  "buyer_decision_stage": "exploring | comparing | negotiating | hesitating | ready_to_buy",
  "buyer_signals": [
    "evidence from buyer turns"
  ],
  "seller_intent_or_strategy": [
    "discount offer",
    "value justification",
    "recommendation",
    "clarification question",
    "urgency creation"
  ],
  "seller_context_for_buyer_intent": "how seller responses reveal or constrain the buyer's intent",
  "candidate_buyer_intents": [
    "intent_a",
    "intent_b"
  ],
  "ambiguity_reason": "why this dialogue can belong to more than one buyer intent"
}
```

Điểm quan trọng là `seller_intent_or_strategy` không phải nhãn chính của bài toán, nhưng nó giúp LLM suy luận buyer intent chính xác hơn. Ví dụ, nếu seller liên tục đưa discount, bảo hành, hoặc lý do về chất lượng, điều đó cho thấy buyer có thể đang mặc cả, so sánh giá, hoặc chưa bị thuyết phục về giá trị sản phẩm.

## 4. Xây dựng embedding từ intent frame

Có hai lựa chọn để tạo embedding:

### 4.1. Embed trực tiếp toàn bộ hội thoại

\[
z_i = PTE(D_i)
\]

Cách này đơn giản nhưng dễ nhiễu vì hội thoại có thể chứa chào hỏi, câu xã giao, thông tin lặp lại, hoặc chi tiết không liên quan đến intent.

### 4.2. Embed intent frame do LLM trích xuất

\[
z_i = PTE(Serialize(F_i))
\]

Đây là lựa chọn phù hợp hơn cho bài toán này. Phần được embed nên bao gồm:

```text
buyer_latent_intent
buyer_goal
buyer_constraints
buyer_decision_stage
buyer_signals
seller_intent_or_strategy
seller_context_for_buyer_intent
candidate_buyer_intents
ambiguity_reason
dialogue_type
```

Ví dụ text dùng để embedding:

```text
The buyer is interested in buying but is price-sensitive.
The buyer compares the current product with cheaper alternatives and uses this comparison as leverage for negotiation.
The seller responds by justifying product quality, offering a small discount, and emphasizing warranty.
The buyer intent is ambiguous between price negotiation and alternative comparison.
Dialogue type: bargaining.
```

Cách này giúp embedding tập trung vào latent intent thay vì bề mặt ngôn ngữ của từng câu.

## 5. Phân cụm bằng K-Means

Sau khi có dialogue-level embeddings:

\[
Z = \{z_1, z_2, ..., z_N\}
\]

ta áp dụng K-Means:

\[
C_1, C_2, ..., C_K = KMeans(Z)
\]

Mỗi cụm \(C_k\) biểu diễn một nhóm hội thoại có cấu trúc intent tương tự nhau.

Ví dụ trong dữ liệu bargaining/recommendation, các cụm có thể tương ứng với:

- buyer muốn mặc cả trước khi quyết định mua;
- buyer muốn so sánh sản phẩm hiện tại với lựa chọn khác;
- buyer chưa rõ nhu cầu và cần seller gợi ý;
- buyer đã có nhu cầu nhưng còn nghi ngại về giá trị sản phẩm;
- buyer muốn xác nhận thông tin quan trọng trước khi mua;
- buyer từ chối mềm nhưng vẫn để ngỏ khả năng mua;
- buyer dùng recommendation như một bước trước khi bargain;
- buyer vừa hỏi gợi ý vừa đặt giới hạn ngân sách.

## 6. Chọn exemplar dialogues trong mỗi cụm

Với mỗi cụm \(C_k\), ta chọn \(m\) đoạn hội thoại tiêu biểu để đưa vào LLM. Đơn vị chọn là **dialogue hoặc intent frame**, không phải utterance.

Có thể dùng K-Means++ hoặc MMR. Với MMR:

\[
D^* = \arg\max_{D_i \in C_k}
\lambda \cdot sim(z_i, \mu_k)
-
(1-\lambda) \cdot \max_{D_j \in S} sim(z_i, z_j)
\]

Trong đó:

- \(\mu_k\): centroid của cụm \(C_k\);
- \(S\): tập exemplar dialogues đã chọn;
- \(z_i\): embedding của intent frame từ dialogue \(D_i\);
- \(sim(\cdot)\): cosine similarity;
- \(\lambda\): hệ số cân bằng giữa tính đại diện và tính đa dạng.

Nếu \(\lambda\) cao, các exemplar sẽ gần centroid hơn. Nếu \(\lambda\) thấp, các exemplar sẽ đa dạng hơn. Trong bài toán ambiguous intent, nên ưu tiên một mức cân bằng, ví dụ \(\lambda = 0.6\) hoặc \(\lambda = 0.7\), để vừa giữ được chủ đề chính của cụm vừa không làm mất các biến thể mơ hồ.

## 7. LLM dựng buyer intent từ cụm

Sau khi chọn exemplar dialogues hoặc exemplar frames, LLM được dùng để tóm tắt cụm thành một buyer intent mới.

Prompt gợi ý:

```text
You are analyzing buyer intent in buyer-seller conversations.

Below are representative conversations or extracted intent frames from the same cluster.
Focus on the buyer's latent goal, constraints, decision stage, and how seller responses reveal the buyer's intent.

For this cluster, return:
1. intent_name
2. intent_description
3. buyer_goal
4. buyer_constraints
5. buyer_decision_stage
6. seller_context
7. positive_dialogue_patterns
8. confusing_or_related_intents
9. ambiguity_reason
10. recommended_training_label

The intent must describe the buyer, not the seller.
The label should be concise and suitable for training an intent classifier.
```

Ví dụ output:

```json
{
  "intent_name": "price_sensitive_purchase_negotiation",
  "intent_description": "The buyer is interested in purchasing but tries to obtain a better price before committing.",
  "buyer_goal": "Reduce the final cost or increase perceived value before making a purchase decision.",
  "buyer_constraints": [
    "limited budget",
    "price comparison with alternatives",
    "uncertainty about whether the product is worth the price"
  ],
  "buyer_decision_stage": "negotiating",
  "seller_context": "The seller responds by offering discounts, justifying quality, emphasizing warranty, or creating urgency.",
  "positive_dialogue_patterns": [
    "buyer asks whether the price can be reduced",
    "buyer compares the item with a cheaper alternative",
    "buyer says they may buy if the price becomes lower"
  ],
  "confusing_or_related_intents": [
    "compare_alternatives_before_purchase",
    "purchase_hesitation",
    "budget_constrained_recommendation"
  ],
  "ambiguity_reason": "The buyer may be genuinely comparing alternatives, but may also be using comparison as a negotiation strategy.",
  "recommended_training_label": "price_sensitive_purchase_negotiation"
}
```

## 8. Tạo intent prototype

Với summary của cụm \(k\), ta tạo prototype intent:

\[
\theta_k = PTE(LLM\_summary_k)
\]

Trong đó `LLM_summary_k` nên là chuỗi kết hợp:

```text
intent_name
intent_description
buyer_goal
buyer_constraints
buyer_decision_stage
seller_context
confusing_or_related_intents
ambiguity_reason
```

Prototype \(\theta_k\) đại diện cho một buyer intent ở cấp hội thoại.

## 9. Phát hiện ambiguous buyer intent

Với mỗi dialogue \(D_i\), ta đã có intent frame \(F_i\) và embedding:

\[
z_i = PTE(F_i)
\]

Tính độ tương đồng với các intent prototype:

\[
s_{ik} = cos(z_i, \theta_k)
\]

Lấy hai intent gần nhất:

\[
\theta_a, \theta_b = Top2(s_{i1}, s_{i2}, ..., s_{iK})
\]

Dialogue được xem là ambiguous nếu:

\[
|s_{ia} - s_{ib}| < \delta
\]

và:

\[
\max_k s_{ik} > \tau
\]

Điều kiện thứ nhất đảm bảo dialogue nằm giữa ít nhất hai intent. Điều kiện thứ hai đảm bảo dialogue vẫn liên quan đến không gian intent, không phải nhiễu.

Có thể dùng thêm entropy:

\[
H(p_i) = - \sum_k p_{ik} \log p_{ik}
\]

Trong đó \(p_{ik}\) là phân phối xác suất sau khi chuẩn hóa similarity scores. Dialogue có entropy cao thường là dialogue có intent mơ hồ hoặc chồng lấn.

## 10. Sinh ambiguous training samples

Sau khi xác định các cặp hoặc bộ intent dễ nhầm, LLM được dùng để sinh thêm dữ liệu huấn luyện. Điểm quan trọng là sample sinh ra phải là **đoạn hội thoại buyer-seller**, không phải một câu đơn lẻ.

Prompt gợi ý:

```text
Generate buyer-seller conversations for training a buyer intent classifier.

The buyer intent should be ambiguous between the following intents:

Intent A:
{intent_a_name}
{intent_a_description}

Intent B:
{intent_b_name}
{intent_b_description}

Requirements:
- The output must be a multi-turn buyer-seller dialogue.
- The buyer's intent must not clearly belong to only one intent.
- Seller responses should help reveal the buyer's latent goal, constraints, or decision stage.
- The dialogue should be natural for e-commerce bargaining or product recommendation.
- Avoid making the buyer intent too explicit.
- Return both the dialogue and structured annotation.

Return JSON with:
- dialogue
- buyer_intent_label
- candidate_intents
- ambiguity_reason
- soft_label
- evidence_turns
```

Ví dụ:

```json
{
  "dialogue": [
    {
      "speaker": "buyer",
      "text": "Mẫu này tôi cũng thích, nhưng bên khác đang có loại gần giống mà giá thấp hơn."
    },
    {
      "speaker": "seller",
      "text": "Dạ mẫu bên em chất liệu tốt hơn và được bảo hành 2 năm."
    },
    {
      "speaker": "buyer",
      "text": "Nếu giá mềm hơn chút thì tôi có thể chọn bên bạn luôn."
    },
    {
      "speaker": "seller",
      "text": "Em có thể giảm thêm một ít nếu anh đặt hôm nay."
    }
  ],
  "buyer_intent_label": "ambiguous",
  "candidate_intents": [
    "price_sensitive_purchase_negotiation",
    "compare_alternatives_before_purchase"
  ],
  "ambiguity_reason": "The buyer compares alternatives while also using the comparison as leverage for price negotiation.",
  "soft_label": {
    "price_sensitive_purchase_negotiation": 0.52,
    "compare_alternatives_before_purchase": 0.48
  },
  "evidence_turns": [
    "buyer mentions a cheaper similar product",
    "buyer says they may choose the seller if the price is softened"
  ]
}
```

## 11. Lọc và kiểm tra chất lượng mẫu sinh

Mỗi synthetic dialogue sau khi sinh cần được đưa lại qua pipeline:

\[
F'_i = LLM\_Extract(D'_i)
\]

\[
z'_i = PTE(F'_i)
\]

Sau đó kiểm tra:

\[
\max_k cos(z'_i, \theta_k) > \tau
\]

và:

\[
|s'_{ia} - s'_{ib}| < \delta
\]

Nếu dialogue quá gần một intent duy nhất, nó không còn là ambiguous sample tốt. Nếu dialogue không gần bất kỳ prototype nào, nó có thể là nhiễu và nên bị loại.

Các tiêu chí lọc bổ sung:

- buyer intent phải xuất hiện qua nhiều lượt hội thoại, không chỉ một câu;
- seller response phải có vai trò làm rõ ngữ cảnh hoặc tạo điều kiện để buyer bộc lộ intent;
- dialogue không được gắn nhãn quá hiển nhiên;
- candidate intents phải hợp lý với evidence turns;
- soft label không nên quá lệch nếu mục tiêu là ambiguous sample, ví dụ không nên là 0.95 và 0.05.

## 12. Nhãn huấn luyện

Tùy bài toán, có thể tạo ba kiểu nhãn:

### 12.1. Hard label

Dùng khi muốn classifier dự đoán một intent chính:

```json
{
  "label": "price_sensitive_purchase_negotiation"
}
```

### 12.2. Multi-label

Dùng khi một dialogue có thể đồng thời thuộc nhiều intent:

```json
{
  "labels": [
    "price_sensitive_purchase_negotiation",
    "compare_alternatives_before_purchase"
  ]
}
```

### 12.3. Soft label

Phù hợp nhất cho ambiguous intent:

```json
{
  "soft_label": {
    "price_sensitive_purchase_negotiation": 0.52,
    "compare_alternatives_before_purchase": 0.48
  }
}
```

Soft label giúp mô hình học vùng ranh giới giữa các intent thay vì ép một dialogue mơ hồ vào một nhãn tuyệt đối.

## 13. Quy trình lặp của NILC

Toàn bộ framework có thể chạy theo quy trình lặp:

1. Thu thập dialogue bargaining/recommendation.
2. Dùng LLM trích xuất dialogue-level intent frame \(F_i\).
3. Tạo embedding \(z_i = PTE(F_i)\).
4. Phân cụm bằng K-Means.
5. Chọn exemplar dialogues bằng K-Means++ hoặc MMR.
6. Dùng LLM tóm tắt cụm để tạo buyer intent mới.
7. Tạo intent prototype \(\theta_k = PTE(LLM\_summary_k)\).
8. Phát hiện dialogue ambiguous dựa trên top intent similarity, margin và entropy.
9. Sinh thêm ambiguous buyer-seller dialogues bằng LLM.
10. Lọc lại synthetic dialogues bằng extraction, embedding và prototype matching.
11. Cập nhật taxonomy intent và lặp lại nếu cần.

## 14. Tóm tắt công thức

Dialogue-level extraction:

\[
F_i = LLM\_Extract(D_i)
\]

Dialogue-level embedding:

\[
z_i = PTE(Serialize(F_i))
\]

Clustering:

\[
C_1, ..., C_K = KMeans(\{z_i\}_{i=1}^{N})
\]

Cluster summarization:

\[
S_k = LLM\_Summary(Exemplars(C_k))
\]

Intent prototype:

\[
\theta_k = PTE(S_k)
\]

Dialogue-intent similarity:

\[
s_{ik} = cos(z_i, \theta_k)
\]

Ambiguity condition:

\[
|s_{ia} - s_{ib}| < \delta
\]

với \(\theta_a\) và \(\theta_b\) là hai prototype gần nhất.

## 15. Mô tả ngắn gọn để đưa vào paper

Phương pháp đề xuất sử dụng một biến thể dialogue-level của NILC để khám phá và sinh ambiguous buyer intents từ hội thoại bargaining và recommendation. Trước hết, mỗi đoạn hội thoại buyer-seller được chuyển thành một intent frame có cấu trúc bằng LLM, bao gồm buyer latent goal, constraints, decision stage, seller strategy và ambiguity reason. Intent frame này được mã hóa thành embedding và phân cụm bằng K-Means để phát hiện các nhóm buyer intent tiềm ẩn. Trong mỗi cụm, các dialogue tiêu biểu được chọn bằng K-Means++ hoặc MMR, sau đó LLM tóm tắt chúng thành intent description và prototype intent. Một dialogue được xem là ambiguous nếu embedding của intent frame gần với nhiều intent prototypes đồng thời, đặc biệt khi khoảng cách giữa hai similarity scores cao nhất nhỏ hơn một ngưỡng \(\delta\). Từ các cặp intent dễ nhầm lẫn này, LLM tiếp tục sinh các đoạn hội thoại buyer-seller mới có nhãn mềm, giúp tạo dữ liệu huấn luyện cho mô hình nhận diện intent trong các vùng chồng lấn giữa bargaining, recommendation, comparison và purchase hesitation.
