## DRUM: Change Point Detection Algorithm

**Input:**  
- Multivariate data stream `S`
- Window size `d`
- Weights `α, β, γ`
- Threshold (e.g. 0.05)

**Output:**  
- Detected change points (indices)

### Algorithm Steps

1. **Initialize:** Set window pointers `k = 1`, `j = d+1`
2. **For** each position `t` in the data stream:
    1. For each variable `i`, extract disjoint windows `w_{i,[k, k+d]}` and `w_{i,[j, j+d]}`
    2. Compute:
        - $\Delta m_i = \left| \text{mean}(w_{i,[k, k+d]}) - \text{mean}(w_{i,[j, j+d]}) \right|$
        - $\Delta s_i = \left| \text{std}(w_{i,[k, k+d]}) - \text{std}(w_{i,[j, j+d]}) \right|$
        - $\Delta frm_i = \left| \text{crossings}(w_{i,[k, k+d]}) - \text{crossings}(w_{i,[j, j+d]}) \right|$
    3. Compute the LCS score:
        - $LCS_j = \alpha \sum_i \Delta m_i + \beta \sum_i \Delta s_i + \gamma \sum_i \Delta frm_i$
    4. **If** $|LCS_j - LCS_{j-1}| > \text{threshold}$:
        - Mark window `j` as candidate change region.
        - Within this region, use **sliding windows** to compute LCS at each position.
        - Report the timestamp with max LCS as the change point.
    5. Move window pointers: `k ← j`, `j ← j + d + 1`
3. **Repeat** until end of data stream.

---

### Summary Table

| Statistic   | Formula                                                       | Description                        |
|-------------|---------------------------------------------------------------|------------------------------------|
| $\Delta m$  | $|\text{mean}(w_1) - \text{mean}(w_2)|$                       | Mean shift between windows         |
| $\Delta s$  | $|\text{std}(w_1) - \text{std}(w_2)|$                         | Std deviation shift                |
| $\Delta frm$| $|\text{crossings}(w_1) - \text{crossings}(w_2)|$             | Change in number of mean crossings |
| **LCS**     | $\alpha \sum \Delta m + \beta \sum \Delta s + \gamma \sum \Delta frm$ | Total change score        |

---

**Note:**  
- For inline math on GitHub, use single dollar signs: `$LCS = ...$`
- For block math, double dollar signs: `$$LCS = ...$$` (works on nbviewer/Jupyter, **not** GitHub web)
- For native GitHub README rendering, use code blocks for formulas if you want them to always look decent.

---

#### **References**
- Bashir, A., & Estrada, T. (2023). DRUM: A Real Time Detector for Regime Shifts in Data Streams via an Unsupervised, Multivariate Framework. ([PDF](your-paper-link.pdf))
