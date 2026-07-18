# Dataset — Microsoft News Dataset (MIND)

## Source
[MS MIND](https://msnews.github.io/) — 1M users, 160k news articles, 15M impression logs.

## Size
| Split    | Impressions | Users  | Articles |
|----------|-------------|--------|----------|
| Train    | ~12M        | 800k   | 120k     |
| Val/Test | ~3M         | 200k   | 40k      |

## Features
- **User:** `user_id`, `history` (clicked article IDs), `device`, `session_hour`.  
- **Article:** `news_id`, `category`, `subcategory`, `title`, `abstract`, `url`.  
- **Impression:** `impression_id`, `user_id`, `clicked` (list of 0/1 per article), `timestamp`.  
- **Derived:** article popularity (last 1h clicks), recency (minutes since publish), user session length.

## Target
Binary `click` label — 1 if user clicked the article in the impression.

## Known Challenges
- **Implicit = noisy:** a click doesn't always mean "liked" (clickbait).  
- **Position bias:** articles at the top get clicked more regardless of relevance.  
- **Temporal non‑stationarity:** breaking news shifts interest distributions hourly.  
- **Explore‑exploit dilemma:** greedy ranking collapses diversity.
