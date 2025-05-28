# 🛡️ Machine Learning Based Web Application Firewall (WAF)

This project implements a Web Application Firewall (WAF) using Machine Learning techniques to detect and classify potentially malicious HTTP requests. It parses raw HTTP logs, extracts features, and uses ML algorithms to classify the traffic as malicious or benign.

---

## 📌 Purpose

The objective of this project is to:
- Automatically detect web-based attacks such as SQL Injection or command injection.
- Convert raw HTTP request logs into feature vectors.
- Train and evaluate a machine learning model for classification.
- Use these insights to protect web applications in real-time or batch scenarios.

---

## 🧱 Project Structure

```
├── bad_req.log             # Raw HTTP requests (Burp Suite format)
├── bad_req_http.csv        # Feature-extracted version of bad_req.log
├── httplogs.csv            # Merged dataset (benign + malicious)
├── kmeans_result.csv       # Results from KMeans clustering
├── test.csv                # Dataset used for testing ML model
├── log_parser.py           # Python script to extract features from logs
├── jupyter_file.ipynb      # ML training and evaluation notebook
```

---

## ⚙️ How It Works

### 1. Log Parsing (`log_parser.py`)
- `parse_log()` reads Burp Suite-style XML logs.
- Each HTTP request and response is decoded from Base64 and parsed.
- `parseRawHTTPReq()` breaks down each request into method, path, body, and headers.

### 2. Feature Extraction
- Extracted features include:
  - Number of quotes, dashes, braces, spaces
  - Presence of suspicious keywords like `select`, `union`, `sleep`, etc.
- Label (`class`) is hardcoded as `bad` in this version.
- Outputs to `httplogbad.csv`.

### 3. Dataset Preparation
- Feature vectors are stored in `httplogs.csv` with `good` and `bad` labels.
- These are used for ML training and testing.

### 4. Machine Learning
- Models are trained using `jupyter_file.ipynb` (Logistic Regression, Decision Tree, etc.).
- Unsupervised KMeans clustering is applied for exploratory analysis (`kmeans_result.csv`).

### 5. Evaluation
- Model is tested on unseen data from `test.csv`.
- Performance metrics: Accuracy, Precision, Recall, F1 Score.

---

## 🚀 Getting Started

### Requirements
- Python 3.8+
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

### Steps to Run

1. Parse and extract features:
```bash
python log_parser.py
```

2. Open the notebook and train ML model:
```bash
jupyter notebook jupyter_file.ipynb
```

3. Test the model:
```python
model.predict(test_data)
```

---

## 🧠 Features Extracted

| Feature        | Description                                  |
|----------------|----------------------------------------------|
| `method`       | HTTP method (GET, POST)                      |
| `path`         | Request path (URL-encoded)                   |
| `body`         | Request body (decoded)                       |
| `single_q`     | Count of single quotes `'`                   |
| `double_q`     | Count of double quotes `"`                   |
| `dashes`       | Count of `--`                                |
| `braces`       | Count of `(`                                 |
| `spaces`       | Number of spaces                             |
| `badwords`     | Count of known bad words (e.g., select, drop)|
| `class`        | `bad` or `good`                              |

---

## 📊 Results

Model performance and clustering outputs are:
- `kmeans_result.csv`: Clustering results (malicious vs benign).
- Jupyter Notebook: Accuracy, confusion matrix, classification report.

---

## 📬 Contact

Feel free to reach out if you’d like to contribute or discuss improvements.

---

## 📌 Future Work

- Live HTTP traffic filtering (via proxy/web server middleware).
- Expand feature engineering (n-grams, deep packet inspection).
- Deep learning integration.
- Web interface for real-time alerts and logging.

---
