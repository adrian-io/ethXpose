# ETHXpose - Ethereum Wallet Fraud Detection API

## Abstract

The blockchain's decentralized nature has enabled financial innovation but also sophisticated fraud, including phishing scams and Ponzi schemes. Traditional fraud detection methods struggle with the scale and complexity of blockchain transactions.

**ETHXpose** provides a graph-based fraud detection approach using graph embeddings and machine learning models to classify illicit activity in Ethereum transactions in a computationally cost-effective way.

* Transaction graphs are constructed from XBlock data, where **nodes** represent Ethereum wallets and **edges** denote transactions.
* Instead of resource-intensive Graph Neural Networks (GNNs), **Graph2Vec** and **Feather-G** embeddings are used to convert transaction graphs into structured vectors.
* Classification is performed with **Random Forest**, **Support Vector Machines (SVMs)**, or **Gradient Boosting (GB)**.

Results show that **Feather-G embeddings with Gradient Boosting** achieve the highest accuracy (93.9%) and F1-score (93.6%), effectively detecting fraudulent wallets while remaining computationally efficient.

This project wraps the trained models into a **FastAPI web service**, enabling real-time wallet classification through a REST API.

---

## Requirements

This project uses **Python 3.9** (recommended for compatibility with `karateclub` and `scikit-learn`).

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Dataset

Data is collected from **XBlock**, an academic blockchain data platform.

* The dataset contains **1,660 phishing addresses**, **200 Ponzi-scheme addresses**, and **1,700 normal addresses**.
* Transaction networks include **first-order** (direct neighbors) and **second-order** (neighbors of neighbors) networks.
* Each transaction records: sender, receiver, value, and timestamp.

The dataset is available [here](https://xblock.pro/#/search?types=datasets&tags=Transaction+Analysis).

---

## Training and Inference

**Train and evaluate models:**

```bash
python train.py
```

**Classify a wallet address:**

```bash
python test_classify.py --wallet_address <WALLET_ADDRESS> --embedding <EMBEDDING_METHOD> --model <MODEL_NAME>
```

**Ensure the following graphs are available in `api/data/graphs` before training:**

* Normal first-order nodes
* Normal second-order nodes
* Phishing first-order nodes
* Phishing second-order nodes

---

## Command-line Options

```
--graph       STR   Order of transaction graphs (first, second).         Default: 'first'
--embedding   STR   Embedding algorithm (Feather-G, Graph2Vec, GL2Vec). Default: 'Feather-G'
--classifier  STR   Classifier (SVM, MLP, RF, GB).                       Default: 'GB'
```

**Examples:**

Train and evaluate:

```bash
python train.py
```

Classify a wallet with Feather-G embeddings and RandomForest:

```bash
python test_classify.py --wallet_address 0x123456789abcdef --embedding "Feather-G" --model "RF"
```

---

## FastAPI Web Service

The web service exposes the following endpoints:

* **POST `/api/py/classify`**
  Classify a wallet in real-time. Request JSON:

  ```json
  {
    "wallet_address": "0x0c90ddbeaf1d855e9fb6a7180b9dbd07156215b6",
    "model_name": "first_Feather-G_GB.joblib"
  }
  ```

  Response JSON:

  ```json
  {
    "fraud_probability": 0.93,
    "graph": {
      "nodes": [{"id": "1", "label": "0xabc..."}],
      "edges": [{"source": "1", "target": "2", "value": 1.5, "timestamp": "2025-09-03 12:34:56"}]
    }
  }
  ```

* **GET `/api/health`**
  Check if the service is running:

  ```json
  {"status": "ok"}
  ```

**Run locally:**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Notes

* Recommended Python version: **3.9** for full package compatibility.
* For reproducibility and deployment, use **pinned versions** in `requirements.txt`.
