<p align="center">
  <img src="asset/flag.png" alt="Dutch Flag" width="50"/>
  &nbsp;&nbsp;
  <img src="asset/roberta.jpg" alt="RoBERTa Logo" width="70"/>
  &nbsp;&nbsp;
  <img src="asset/doctor.png" alt="Medical Icon" width="50"/>
</p>

<h1 align="center">Dutch-RoBERTa-for-Rectal-Cancer-Reports</h1>
<h3 align="center"><em>Exploring Pretrained Language Models using Survival Prediction from Radiology Reports</em></h3>

---

## 🧠 Model Overview

This project explores the use of **Dutch RoBERTa**, a transformer-based pretrained language model, for **survival prediction** in rectal cancer patients based on free-text radiology reports.

The pipeline involves:
- Pretraining (if needed) on Dutch clinical corpora.
- Extracting feature embeddings from radiology reports.
- Training a survival model using the extracted embeddings.

---

## 📂 Repository Structure

. ├── run.sh # Script to pretrain RoBERTa ├── feature_extractor.py # Script to extract embeddings ├── survival/ │ └── train.py # Survival model training script ├── README.md # This file └── ...



---

## 🚀 How to Use

### 📦 Requirements
Install dependencies using:


`pip install -r requirements.txt`

### 1. Pretrain Dutch RoBERTa (Optional)

If you want to pretrain the model on a custom corpus:

Pre-train RoBERTa


`bash run.sh`

⚠️ Make sure to edit run.sh with your data and training parameters.

### 2. Extract Feature Embeddings
To extract embeddings from radiology reports using a pretrained RoBERTa model:

```bash
python feature_extractor.py

Output: .npy files containing sentence/sequence-level embeddings.

You can adjust model paths and input sources in the script.

### 3. Train the Survival Model
Train a survival model using extracted features:

```bash
python survival/train.py --feature_dir PATH_TO_FEATURES --output_path PATH_TO_SAVE_MODEL

--feature_dir: Path to the directory containing .npy features

--output_path: Directory to save the trained model and logs





