# PPDPP
**Kế Hoạch Chỉ Cho PPDPP**

**Tóm Tắt**
- Không chỉnh PADPP trong kế hoạch này.
- PPDPP không có input `w`; nó là policy tĩnh `π(a|s)` trong [PPDPP/agent.py](/Users/admin/Downloads/Data-download/PAPER/2026/ICDM2026/ICDM2026/PPDPP/agent.py:19).
- Nếu mỗi dialogue có một `w` riêng, PPDPP không thể đổi `w` ở inference. Muốn tối ưu đúng từng `w` riêng thì phải train/fine-tune riêng cho từng `w`, nhưng cách đó không thực tế.
- Cách đúng theo PADPP paper khi so sánh baseline: train PPDPP riêng theo một số objective cố định, ví dụ `uniform`, `sl_ratio`, `fairness`, `deal_rate`; không train theo từng hội thoại.

**Thay Đổi Cần Làm**
- Thêm loader cho dataset của bạn:
  - Đọc `hmod_bargain_{train,test}_scenarios.yaml`.
  - Đọc `hmod_recommendation_{train,test}_scenarios.yaml`.
  - Convert mỗi scenario thành case kiểu Craigslist Bargain mà PPDPP đang dùng: `item_name`, `buyer_price`, `seller_price`, `buyer_item_description`, `seller_item_description`, kèm metadata `scenario_id`, `drift_mode`, `buyer_intent_id`, `recommendation_domain`, `static_w`.
- Giữ action space PPDPP `cb` hiện tại trong [PPDPP/prompt.py](/Users/admin/Downloads/Data-download/PAPER/2026/ICDM2026/ICDM2026/PPDPP/prompt.py:16): `greet`, `inquire`, `propose`, `counter`, `agree`, etc.
- Thêm reward helper cho PPDPP:
  - Tính reward vector `[sl_ratio, fairness, deal_rate]`.
  - Scalar reward dùng cho REINFORCE: `r_scalar = fixed_w · reward_vector`.
  - Mapping objective:
    - `uniform = [1/3, 1/3, 1/3]`
    - `sl_ratio = [1, 0, 0]`
    - `fairness = [0, 1, 0]`
    - `deal_rate = [0, 0, 1]`
- Sửa [PPDPP/env.py](/Users/admin/Downloads/Data-download/PAPER/2026/ICDM2026/ICDM2026/PPDPP/env.py:83):
  - Cho phép nhận custom scenario list thay vì chỉ đọc `PPDPP/data/cb-*.txt`.
  - Trả về scalar reward để train PPDPP.
  - Đồng thời lưu reward vector để evaluation.
- Sửa runner [PPDPP/run.py](/Users/admin/Downloads/Data-download/PAPER/2026/ICDM2026/ICDM2026/PPDPP/run.py:15):
  - Thêm args: `--scenario_file`, `--test_scenario_file`, `--objective`, `--num_cases`, `--output_dir`, `--judge_model`.
  - Xuất `metrics.json`, `dialogues.jsonl`, `summary.csv`.

**Protocol Thực Nghiệm PPDPP**
- SFT:
  - Mặc định dùng `PPDPP/data/cb-train.txt` làm prior vì dataset generated không có label strategy đầy đủ.
  - Nếu sau này có dialogue label strategy trong dataset của bạn thì mới thêm SFT trực tiếp trên custom train.
- RL fine-tune:
  - Train 4 checkpoint PPDPP cho mỗi dataset: `uniform`, `sl_ratio`, `fairness`, `deal_rate`.
  - Mỗi checkpoint dùng cùng train scenario file, cùng seed, cùng max turn.
- Evaluation:
  - Chạy từng checkpoint trên cùng test scenario file.
  - Không dùng `static_w` để điều khiển PPDPP.
  - Có thể dùng `static_w` chỉ để tính post-hoc score: `static_w · reward_vector`, nhằm báo cáo PPDPP lệch hay khớp macro-goal đến đâu.

**Metrics Cần Trả**
- Core metrics: `SR/deal_rate`, `AvgTurn`, `sl_ratio`, `fairness`, `weighted_return`.
- Dataset metrics: `GSR`, `CVR`.
- `T2DA`: để `null` hoặc `not_applicable` cho PPDPP vì PPDPP không sinh `w_t`.
- Group metrics theo: `drift_mode`, `seller_persona`, `buyer_intent_id`; với recommendation thêm `recommendation_domain`.

**Test Plan**
- Test loader đọc đủ Bargain và Recommendation scenario YAML.
- Smoke run `--num_cases 3` cho một objective, xác nhận sinh đủ `metrics.json`, `dialogues.jsonl`, `summary.csv`.
- Test reward helper bằng case giả có deal/no-deal.
- Test 4 objective runs tạo 4 output folder/checkpoint riêng, không ghi đè nhau.

**Assumptions**
- Dataset của bạn là các file generated trong `config/scenario/generated/`.
- Recommendation-derived dataset vẫn chạy qua bargaining interface như repo hiện tại.
- PPDPP baseline chính thức sẽ là 4 model tĩnh theo objective cố định, không phải một model nhận `w`.
