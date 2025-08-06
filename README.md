# 🥁 DRUM: Real-Time Regime Shift Detection in Multivariate Streams

<div align="center">
  <img src="https://img.shields.io/badge/paper-arxiv%2F2023-blue?logo=arxiv" alt="arXiv Paper"/>
  <img src="https://img.shields.io/badge/python-3.7%2B-green?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/unsupervised-yes-success" alt="Unsupervised"/>
  <img src="https://img.shields.io/badge/real--time-yes-blueviolet" alt="Real-Time"/>
</div>

> **DRUM** is a blazing-fast, unsupervised, multivariate regime shift (change point) detector built for real-time, streaming data.  
> Designed for minimal compute, no retraining, and deployment anywhere—from edge IoT to cloud pipelines.

---

## 🚀 **Why DRUM?**

- **Online:** Detects changes as data arrives—no need to store all history
- **Unsupervised:** Works without labels or human intervention
- **Multivariate:** Handles dozens of features, not just one
- **Lightweight:** Runs on low-power devices, IoT, and embedded systems
- **SOTA accuracy:** Outperforms many classical & deep learning baselines

---

## 📖 **How Does DRUM Work?**

1. **Sliding and Disjoint Windows:**  
   DRUM divides your data stream into *windows*—some overlap (sliding), some don’t (disjoint).

2. **For each pair of windows, DRUM computes:**  
   - 🟣 **Mean Shift (\(\Delta m\))**
   - 🟢 **Std Shift (\(\Delta s\))**
   - 🔵 **Fluctuation Across Running Mean (\(\Delta frm\))**  
     (counts how "jagged" or "noisy" the signal is vs. its own mean)

3. **Aggregate into a Change Score (LCS):**
   $$
   \text{LCS} = \alpha \sum_{i} \Delta m_{i} + \beta \sum_{i} \Delta s_{i} + \gamma \sum_{i} \Delta frm_{i}
   $$
   <sup>*(α, β, γ are weighting parameters, sum to 1)*</sup>

4. **Detect Change Points:**  
   If LCS jumps by a threshold (e.g. 5%), it’s a candidate regime shift!  
   DRUM then pinpoints the *exact* timestamp using a local sliding window.

---

## ✨ **DRUM Algorithm (Pseudocode)**

```text
Input: Data stream S, window size d, weights α, β, γ, threshold
Output: Change points

1. For each new chunk of data:
    a. Split into windows (disjoint and sliding)
    b. For every variable:
        i.   Compute mean, std, and running mean crossings
    c. Calculate LCS for each window pair:
       LCS = α * sum(Δm) + β * sum(Δs) + γ * sum(Δfrm)
    d. If LCS jumps > threshold:
        - Mark as candidate change region
        - Use sliding windows in this region to find exact change point (max LCS)
2. Repeat for new data in stream!
