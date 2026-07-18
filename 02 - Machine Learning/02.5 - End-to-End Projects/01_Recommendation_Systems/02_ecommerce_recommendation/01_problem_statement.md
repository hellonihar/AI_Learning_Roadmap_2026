# Problem Statement — E‑Commerce Recommendation (Hybrid CF + Content-Based)

## Business Context
An online retailer wants to increase average order value by recommending products a user is likely to purchase. The catalog spans millions of SKUs. The system must mix collaborative signals (what similar users bought) with content signals (product category, price, brand).

## Problem Type
**Top‑N ranking (implicit + explicit feedback).**  
- **Input:** User ID, Product ID, purchase history, product attributes.  
- **Output:** Ranked list of candidate products for each user.  
- **Approach:** 2‑stage pipeline — candidate generation via CF → ranking via content‑aware lightGBM.

## Success Metrics
- **Recall@k** (k = 20) — fraction of purchased items in test set that appear in the top‑20. Target ≥ 25%.  
- **Hit Rate@k** — whether the user’s next purchase is in the top‑20. Target ≥ 40%.  
- Business proxy: conversion rate lift vs. popularity baseline.
