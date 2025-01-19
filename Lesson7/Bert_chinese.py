# 匯入 Hugging Face 的 pipeline 功能
from transformers import pipeline

# 初始化問答管道 (pipeline)，使用中文多語言模型
qa_pipeline = pipeline(
    "question-answering",
    model="uer/roberta-base-chinese-extractive-qa"  # 預訓練的中文問答模型
)

# 定義上下文內容（Context）
context = """
BERT 是一種基於 Transformer 的語言模型，由 Google 提出，用於自然語言理解任務。
它的全稱是 Bidirectional Encoder Representations from Transformers。
BERT 支援多語言，包括英文和中文，廣泛應用於問答、文本分類等任務。
"""

# 使用者提問
question = "BERT 是什麼？"

# 問答推論
result = qa_pipeline(question=question, context=context)

# 顯示結果
print(f"問題: {question}")
print(f"答案: {result['answer']}")
print(f"信心分數: {result['score']}")
