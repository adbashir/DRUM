# DRUM: Unsupervised Change Point Detection for Streaming Data

DRUM is a Python toolkit for real-time detection of regime shifts and anomalies in multivariate data streams, with a focus on operational robustness and low-latency inference.

## Features
- Unsupervised detection of change points in high-dimensional data
- Streaming and batch modes
- Integration-ready for industrial applications

## Example
```python
from drum.change_point import DrumDetector

detector = DrumDetector(method="bayesian")
events = detector.detect(stream)
