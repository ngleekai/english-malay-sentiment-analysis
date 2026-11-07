# Data Audit Guide

打开 `results/data_audit/label_audit_candidates.csv`，从 `priority=1` 开始检查。

只填写以下三列：

- `your_decision`：填写 `keep`、`change` 或 `uncertain`。
- `corrected_label`：只有选择 `change` 时才填写 `0`、`1` 或 `2`。
- `notes`：用一句话记录判断理由。

标签定义：

- `0` = Negative
- `1` = Neutral
- `2` = Positive

规则：

1. `suspicious_reasons` 只是提醒，不代表原标签一定错误。
2. 不要因为 ontology 不同意就自动修改标签。
3. 意思明确才选择 `change`；不能确定就选择 `uncertain`。
4. 不要编辑 `train.csv`、`validation.csv` 或 `test.csv`。
5. Official test 不参与这次检查。

