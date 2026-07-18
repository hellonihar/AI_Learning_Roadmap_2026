# Dataset — Spotify Million Playlist Dataset (MPD)

## Source
[Spotify MPD](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset) — 1M playlists, 2.2M unique tracks, 66M track occurrences.

## Size
| Entity       | Count        |
|--------------|--------------|
| Playlists    | 1,000,000    |
| Unique tracks| 2,262,292    |
| Track occurrences | 66,346,598 |

## Features
- **Playlist:** `pid`, `name`, `tracks` (ordered list of track URIs), `num_samples`, `num_segments`.  
- **Track metadata:** `track_uri`, `track_name`, `artist_name`, `album_uri`, `duration_ms`.  
- **Derived:** session length (number of tracks so far), position in playlist, gap from last skip.

## Target
`next_track` — the track that immediately follows the current window in the playlist sequence.

## Known Challenges
- **Gigantic item space:** 2.2M tracks — softmax over all items is infeasible.  
- **Short sessions:** 40% of playlists have ≤ 10 tracks — sequence signal is weak.  
- **Artist repeats:** many playlists repeat the same artist — model can overfit to artist bias.  
- **Playlist naming noise:** "workout" and "gym" playlists overlap but share no tracks.
