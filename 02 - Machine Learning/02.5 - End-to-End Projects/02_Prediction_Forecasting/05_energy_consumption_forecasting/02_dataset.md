# Dataset — Energy Consumption Forecasting

## Source
[UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption). One household in Sceaux, France (2006–2010). Augmented with local weather data.

## Size
- **Rows:** 2,075,259 (hourly readings)
- **Time range:** Dec 2006 – Nov 2010 (~4 years)
- **Features:** 12 (after feature engineering)

## Raw Features
| Column | Description |
|--------|-------------|
| `date`, `time` | Timestamp |
| `global_active_power` | Household avg active power (kW) — **target** |
| `global_reactive_power` | Reactive power (kVAR) |
| `voltage` | Voltage (V) |
| `global_intensity` | Current (A) |
| `sub_metering_1/2/3` | Specific circuits (kitchen, laundry, water heater) |

## Weather Features (Joined)
| Feature | Source |
|---------|--------|
| `temperature` | OpenWeather API (nearest station) |
| `humidity` | OpenWeather API |
| `cloud_cover` | OpenWeather API |
| `hour_of_day`, `day_of_week`, `month` | Derived from timestamp |

## Known Challenges
- **Missing periods** — ~2% of timestamps missing (meter downtime). Forward-fill acceptable for gaps < 2 h
- **Seasonal patterns** — Winter consumption is 2× summer (electric heating dominates)
- **Daily seasonality** — Morning (7–9 AM) and evening (6–9 PM) peaks
- **Holiday effects** — Weekends and holidays shift load profiles significantly
- **Meter precision** — Sub-metering data has quantization noise (0.001 kWh steps)
