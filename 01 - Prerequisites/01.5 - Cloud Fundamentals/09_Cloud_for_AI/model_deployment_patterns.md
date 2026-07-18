# Model Deployment Patterns

## 1. Real-time Endpoint (SageMaker / Vertex AI)
```
Client → API Gateway → Load Balancer → Model Container(s) → Response
```
- **Best for**: chatbots, real-time predictions, low-latency APIs
- **Scaling**: auto-scale based on requests or GPU utilization

## 2. Serverless Inference
```
Client → API Gateway → Cloud Function → Model (small) → Response
```
- **Best for**: lightweight models (<500MB), sporadic traffic
- **Pros**: no infra management, pay per invocation
- **Cons**: cold starts, GPU not available (most providers)

## 3. Batch Transform
```
S3 (input data) → Batch Job (GPU instances) → S3 (predictions)
```
- **Best for**: bulk processing, periodic scoring
- **Cost**: use spot instances for 60-90% savings

## 4. Edge / On-device
```
Model → ONNX / TensorRT → Edge device → Local inference
```
- **Best for**: offline, low-latency, privacy-sensitive use cases

## 5. Multi-model Endpoint
- Host multiple models behind one endpoint
- Route requests to the appropriate model
- Saves cost vs separate endpoints per model
