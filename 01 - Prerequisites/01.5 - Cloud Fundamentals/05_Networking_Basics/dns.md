# DNS (Route53 / Cloud DNS)

DNS translates human-readable domain names (`api.myservice.com`) to IP addresses.

## Key Concepts
- **A record**: maps domain → IPv4 address
- **AAAA record**: maps domain → IPv6 address
- **CNAME**: maps domain → another domain (alias)
- **Alias record**: maps domain → AWS resource (LB, CloudFront)
- **TTL**: how long clients cache the DNS response

## Routing Policies
| Policy | Use Case |
|---|---|
| Simple | Single endpoint |
| Weighted | A/B testing, canary deploys |
| Latency-based | Route to lowest-latency region |
| Geolocation | Route based on user location |
| Failover | Route to secondary if primary is unhealthy |

## For AI Engineers
- Set up custom domain for inference endpoint: `api.myservice.com → LB DNS`
- Use latency-based routing to direct users to nearest inference region
- Use weighted routing for A/B testing model versions
