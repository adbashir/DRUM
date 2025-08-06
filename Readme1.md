**Algorithm: DRUM Change Point Detection**

**Input:**  
- Multivariate data stream \( S \)  
- Window size \( d \)  
- Weights \( \alpha, \beta, \gamma \)  
- Threshold (e.g., 0.05)

**Output:**  
- Detected change points (indices)

1. **Initialize:** Set window pointers \( k = 1, j = d+1 \)
2. **For** each position \( t \) in the data stream:
    1. For each variable \( i \), extract disjoint windows \( w_{i,[k, k+d]} \) and \( w_{i,[j, j+d]} \)
    2. Compute:
        - \( \Delta m_i = |\text{mean}(w_{i,[k, k+d]}) - \text{mean}(w_{i,[j, j+d]})| \)
        - \( \Delta s_i = |\text{std}(w_{i,[k, k+d]}) - \text{std}(w_{i,[j, j+d]})| \)
        - \( \Delta frm_i = |\text{crossings}(w_{i,[k, k+d]}) - \text{crossings}(w_{i,[j, j+d]})| \)
    3. Compute the LCS score:
        - \( LCS_j = \alpha \sum_i \Delta m_i + \beta \sum_i \Delta s_i + \gamma \sum_i \Delta frm_i \)
    4. **If** \( |LCS_j - LCS_{j-1}| > \text{threshold} \):
        - Mark window \( j \) as candidate change region.
        - Within this region, use **sliding windows** to compute LCS at each position.
        - Report the timestamp with max LCS as the change point.
    5. Move window pointers: \( k \leftarrow j \), \( j \leftarrow j + d + 1 \)
3. **Repeat** until end of data stream.
