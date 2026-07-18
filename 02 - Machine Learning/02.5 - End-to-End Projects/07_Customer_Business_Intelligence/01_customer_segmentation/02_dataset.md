# Customer Segmentation — Dataset

## Source
[UCI Online Retail](https://archive.ics.uci.edu/ml/datasets/Online+Retail) — transaction data from a UK-based online gift retailer (Dec 2010 – Dec 2011).

## Size & Shape
- **Total**: 541K invoice lines, 8 columns
- **Raw features**: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- **Engineered RFM features**:
  - Recency: days since last purchase
  - Frequency: number of purchases (invoice count)
  - Monetary: total spend
  - Additional: avg order value, days since first purchase, unique product categories purchased

## Challenges
- **Cancelled orders** — Invoice codes starting with 'C' are cancellations; need to filter/handle separately
- **Quantity outliers** — Some orders have 80K units (probably wholesale); caps at 99th percentile needed
- **Zero-value invoices** — Some invoices have total = 0 (free samples); exclude
- **New vs. churned** — Customers who joined in last month have artificially low recency and frequency
