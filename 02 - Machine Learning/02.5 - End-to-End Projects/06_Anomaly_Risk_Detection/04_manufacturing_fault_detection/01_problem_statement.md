# Manufacturing Fault Detection — Problem Statement

## Business Context
A semiconductor fab runs 24/7 with 500+ sensors per production tool. Faulty wafers cost ~$2K each in materials + lost tool time. Current rule-based alarms trigger 200+ times/shift with 90% false positive rate. Operators ignore alarms. Need a model that flags real faults with high recall and fewer false alarms.

## Problem Type
- **Primary**: Time-series anomaly detection (sequential sensor readings)
- **Secondary**: Binary classification for confirmed faults

## Success Metrics
- **Recall** (must catch >90% of actual faults)
- Precision (reduce false alarms by >50%)
- Mean lead time: minutes before fault detected vs. current system
- False alarm rate per shift (< 10 acceptable)
