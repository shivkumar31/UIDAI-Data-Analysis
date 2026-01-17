# UIDAI: Smart Resource Allocation & Anomaly Detection System


> **Hackathon Submission 2026** | **Theme:** Data-Driven Innovation for Aadhaar

A data analytics and machine learning dashboard designed to optimize resource allocation across 1,100+ Aadhaar districts and detect operational anomalies using a hybrid AI approach.

---



## 💡 The Problem
The Aadhaar ecosystem has shifted from **Enrolment Phase** to **Maintenance Phase**.
- **Challenge 1:** Administrative metrics often treat all transactions equally, masking the true workload. (e.g., A 15-minute Enrolment is counted the same as a 5-minute Update).
- **Challenge 2:** High-volume update centers often hide fraudulent activities or process deviations that are hard to spot with simple threshold checks.

## 🚀 Our Solution
We built a **Smart Resource Allocation System** that:
1.  **Calculates a Weighted Burden Score:** Re-ranks districts based on actual effort `(Enrolment*3 + BioUpdate*2 + DemoUpdate*1)`.
2.  **Detects Anomalies:** Uses **Isolation Forest (Unsupervised ML)** combined with domain-specific **Business Rules** to flag suspicious activity (e.g., >75% Demographic Updates).
3.  **Segments Districts:** Clusters districts into "Growth Zones" vs. "Maintenance Zones" using **K-Means Clustering**.

---

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Frontend:** [Streamlit](https://streamlit.io/) (for interactive dashboarding)
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Isolation Forest, K-Means, StandardScaler)
* **Visualization:** Plotly Express (Interactive 3D/2D charts)

---
