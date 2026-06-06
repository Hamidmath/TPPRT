# Raw_preprocess

Reads the raw probe GPS CSVs and writes `compressed_routes.parquet`, the
per-trip trajectories used as input to map-matching.

## Map-matching

Map-matching assigns each GPS fix to the road link the vehicle was on, turning
a noisy trajectory into a sequence of network links. We use the Newson-Krumm
hidden Markov matcher (Leuven `DistanceMatcher`): candidate links near each fix
are the hidden states, the GPS-to-link distance is a zero-mean Gaussian with
standard deviation `sigma`, and a Viterbi pass picks the most likely link
sequence. `sigma` is the only parameter, refit on the SLC corpus to 3.5 m by
self-consistency, with candidate radius `max_dist = 400 m`. Matching runs on
the MATSim car network.

## Preprocessing

`preprocess_raw.py` cleans the raw CSVs: drops invalid rows, drops fixes slower
than 2 m/s, and splits into trips on a device change or a gap longer than 5
minutes. Input CSVs have no header and four columns
(`device_id, timestamp, lat, lon`); output Parquet columns are
`route_id, time_sec, lat, lon`.

## Run

    python preprocess_raw.py --input-dir Combined --output compressed_routes.parquet
    python map_match.py --network slc_network.xml --routes compressed_routes.parquet --output matched_routes.json
