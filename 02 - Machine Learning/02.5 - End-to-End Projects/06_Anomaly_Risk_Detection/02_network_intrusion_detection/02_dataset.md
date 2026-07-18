# Network Intrusion Detection — Dataset

## Source
[KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) — DARPA intrusion detection evaluation. Despite its age, the feature set (basic TCP features, content features, traffic statistics) remains the standard benchmark.

## Size & Shape
- **Train**: 4.9M connection records, 41 features
- **Test** (corrected): 311K records with 14 additional attack types unseen in training
- **Features**: 9 basic (duration, protocol_type, service, flag, src_bytes, dst_bytes, etc.), 13 content (hot, num_failed_logins, root_shell, etc.), 19 traffic-based (count, serror_rate, etc.)
- **Target**: 23 attack types mapped to 4 categories — DOS, Probe, R2L, U2R

## Challenges
- **Class imbalance** — Normal traffic is ~80%, DOS is ~18%, other attack types <1%
- **Novel attacks** — Test set contains 14 attack types not present in training
- **Feature skew** — `num_outbound_cmds` is always 0, `is_host_login` is nearly constant
- **Categorical encoding** — protocol_type (3), service (70), flag (11) need careful handling
