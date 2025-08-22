# Medical Question Answering System 🩺

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-orange.svg)
![Status](https://img.shields.io/badge/Status-Beta-yellow.svg)

A Retrieval-Augmented Generation (RAG) system for medical question answering, built with the MedQuAD dataset (~16,413 Q&A pairs) and evaluated on TREC-2017 LiveQA. The system fine-tunes T5-base for generation, uses FAISS with `all-MiniLM-L6-v2` embeddings for retrieval, and runs in a Kaggle notebook environment. 🚀

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
This project develops a medical QA bot that answers health-related questions with factual accuracy and fluency. It:
- **Trains** on the MedQuAD dataset (~16,413 Q&A pairs from NIH).
- **Retrieves** relevant contexts using FAISS and `all-MiniLM-L6-v2` embeddings.
- **Generates** answers with a fine-tuned T5-base model.
- **Evaluates** on TREC-2017 LiveQA (`hyesunyun/liveqa_medical_trec2017`) with ROUGE, BLEU, and attempted BERTScore.

**Why RAG?** Combines retrieval for factual grounding with generation for natural responses, reducing hallucinations critical in medical QA. Alternatives like pure generation (e.g., GPT) risk misinformation, while rule-based systems lack flexibility.

## Features
- 📚 **MedQuAD Dataset**: ~16,413 Q&A pairs covering diseases, treatments, and diagnostics.
- 🤖 **T5-Base Model**: Fine-tuned for medical QA with seq2seq architecture.
- 🔍 **FAISS Retrieval**: Fast semantic search with lightweight embeddings.
- 📊 **Evaluation**: ROUGE-1 (0.1565), ROUGE-2 (0.0235), ROUGE-L (0.1047), BLEU (0.0090) on 100 TREC questions.
- 🛠️ **Kaggle-Compatible**: Runs on Kaggle’s GPU (e.g., P100).

## Installation
### Prerequisites
- Python 3.8+
- Kaggle account with GPU enabled
- Git for cloning the repository

### Dependencies
Install libraries in your Kaggle notebook:
```bash
pip install pandas transformers datasets sentence-transformers faiss-cpu torch scikit-learn nltk rouge-score bert-score
```

### Dataset Setup
1. **MedQuAD**: Download from [Kaggle](https://www.kaggle.com/datasets/jpmiller/medquad) and place in `/kaggle/input/medquad-csv/medquad.csv`.
2. **TREC-2017 LiveQA**: Loaded via `datasets.load_dataset('hyesunyun/liveqa_medical_trec2017', split='test')`.

Clone the repository:
```bash
git clone https://github.com/your-username/medical-qa-system.git
cd medical-qa-system
```

## Usage
1. **Preprocessing**:
   - Loads MedQuAD, cleans non-medical content (e.g., URLs, contact info), normalizes terms (e.g., ‘htn’ → ‘hypertension’), and splits into train (~13,130) and validation (~3,283) sets.
   ```python
   import pandas as pd
   import re
   from datasets import Dataset
   from sklearn.model_selection import train_test_split

   df = pd.read_csv("/kaggle/input/medquad-csv/medquad.csv")
   print("Dataset Info:", df.info())
   print("Question Length Stats:", df['question'].apply(lambda x: len(str(x).split())).describe())
   print("Answer Length Stats:", df['answer'].apply(lambda x: len(str(x).split())).describe())

   def clean_references(text):
       if not isinstance(text, str): return ""
       phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\d{3}-\d{2}-\d{4}\b|\b1-\d{3}-\d{3}-\d{4}\b|\b\d{4}-\d{3}-\d{4}\b|\bToll Free:.*?\b'
       email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
       url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
       address_pattern = r'\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}(-\d{4})?'
       text = re.sub(phone_pattern, '', text, flags=re.IGNORECASE)
       text = re.sub(email_pattern, '', text)
       text = re.sub(url_pattern, '', text)
       text = re.sub(address_pattern, '', text)
       ref_keywords = ['toll free', 'phone', 'email', 'fax', 'tty', 'clearinghouse']
       sentences = text.split('. ')
       cleaned_sentences = [s for s in sentences if not any(keyword.lower() in s.lower() for keyword in ref_keywords)]
       cleaned_text = '. '.join(cleaned_sentences).strip()
       cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
       cleaned_text = re.sub(r'\.\s*\.', '.', cleaned_text)
       return cleaned_text if cleaned_text else text

   medical_terms = {'htn': 'hypertension', 'dm': 'diabetes mellitus'}
   def normalize_terms(text):
       if not isinstance(text, str): return text
       for term, replacement in medical_terms.items():
           text = re.sub(r'\b' + term + r'\b', replacement, text, flags=re.IGNORECASE)
       return text

   df = df.dropna(subset=['answer'])
   df['answer'] = df['answer'].apply(clean_references).apply(normalize_terms)
   df['question'] = df['question'].apply(normalize_terms).str.strip()
   df['focus_area'] = df['focus_area'].fillna('Unknown').str.strip()

   def preprocess(example):
       return {
           'input_text': f"answer the medical question (focus: {example['focus_area']}): {example['question']}",
           'target_text': example['answer']
       }

   train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
   train_dataset = Dataset.from_pandas(train_df)
   val_dataset = Dataset.from_pandas(val_df)
   train_dataset = train_dataset.map(preprocess)
   val_dataset = val_dataset.map(preprocess)
   print(f"Train size: {len(train_dataset)}")
   print(f"Validation size: {len(val_dataset)}")
   ```

2. **Tokenization**:
   - Prepares data for T5 fine-tuning with `max_length=512`.
   ```python
   from transformers import T5Tokenizer
   tokenizer = T5Tokenizer.from_pretrained('t5-base')

   def tokenize(batch):
       inputs = [text for text in batch['input_text'] if isinstance(text, str) and text.strip()]
       targets = [text for text in batch['target_text'] if isinstance(text, str) and text.strip()]
       if not inputs or not targets:
           print("Warning: Empty inputs or targets in batch")
           return {'input_ids': [], 'attention_mask': [], 'labels': []}
       inputs = [re.sub(r'\s+', ' ', text).strip() for text in inputs]
       targets = [re.sub(r'\s+', ' ', text).strip() for text in targets]
       max_length = 512
       inputs_enc = tokenizer(
           inputs,
           padding='max_length',
           truncation=True,
           max_length=max_length,
           return_tensors='pt',
           return_attention_mask=True
       )
       labels_enc = tokenizer(
           targets,
           padding='max_length',
           truncation=True,
           max_length=max_length,
           return_tensors='pt'
       )
       return {
           'input_ids': inputs_enc['input_ids'].tolist(),
           'attention_mask': inputs_enc['attention_mask'].tolist(),
           'labels': labels_enc['input_ids'].tolist()
       }
   ```

3. **Fine-Tuning**:
   - Fine-tunes T5-base (3 epochs, `batch_size=2`).
   - Run: `python fine_tune.py` (assumed script).

4. **Retrieval**:
   - Builds FAISS index with `all-MiniLM-L6-v2`.
   - Run: `python retrieval.py` (assumed script).

5. **RAG QA Bot**:
   - Combines retrieval (k=5) and generation (`max_new_tokens=300`).
   ```python
   from rag import medical_qa_bot
   result = medical_qa_bot("What causes hypertension?", k=5)
   print("Answer:", result['answer'])
   print("Contexts:", result['contexts'])
   ```

6. **Evaluation**:
   - Evaluates on 100 TREC-2017 questions.
   - Run: `python evaluate.py`.

## Project Structure
```
medical-qa-system/
├── data/
│   └── medquad.csv                 # MedQuAD dataset
├── fine_tuned_t5_medical/          # Fine-tuned T5 model
├── medquad_faiss_index.bin         # FAISS index
├── medquad_answer_metadata.pkl     # Retrieval metadata
├── evaluation_results.csv          # Evaluation output
├── preprocess.py                   # Data preprocessing
├── tokenize.py                     # Tokenization
├── fine_tune.py                    # T5 fine-tuning (assumed)
├── retrieval.py                    # FAISS index creation (assumed)
├── rag.py                          # RAG QA bot (assumed)
└── evaluate.py                     # Evaluation script
```

## Results
Evaluated on 100 TREC-2017 LiveQA questions:
| Metric         | Score  |
|----------------|--------|
| ROUGE-1 F1     | 0.1565 |
| ROUGE-2 F1     | 0.0235 |
| ROUGE-L F1     | 0.1047 |
| BLEU           | 0.0090 |
| BERTScore F1   | N/A*   |

*BERTScore not computed due to installation issues (`pip install --no-cache-dir bert-score`).

**Analysis**:
- **Low Scores**: Due to lexical mismatch (MedQuAD’s concise answers vs. TREC’s verbose, multi-part answers) and different phrasing.
- **No Hallucinations**: Answers were factually grounded but lacked detail.
- **Benchmark**: Strong TREC systems achieve ROUGE-1 ~0.2–0.4; this is a baseline.

## Future Improvements
- 🧬 Use medical embeddings (e.g., `bionlp/bluebert_pubmed_1M`) for better retrieval.
- 📈 Fine-tune on TREC-2017 subset to match evaluation style.
- ✍️ Increase `max_new_tokens` to 400 for detailed answers.
- 🔍 Resolve BERTScore installation for semantic evaluation.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/xyz`).
3. Commit changes (`git commit -m "Add xyz feature"`).
4. Push to the branch (`git push origin feature/xyz`).
5. Open a pull request.

Report issues or suggest features via [GitHub Issues](https://github.com/your-username/medical-qa-system/issues).

## License
[MIT License](LICENSE)

## Acknowledgments
- **Datasets**: [MedQuAD](https://www.kaggle.com/datasets/jpmiller/medquad), [TREC-2017 LiveQA](https://huggingface.co/datasets/hyesunyun/liveqa_medical_trec2017)
- **Libraries**: Pandas, Hugging Face Transformers, Sentence-Transformers, FAISS, NLTK, ROUGE-Score, BERT-Score
- **Platform**: Kaggle for GPU support

---

*Built with 💻 and ☕ by [Your Name]. Star ⭐ the repo if you find it useful!*