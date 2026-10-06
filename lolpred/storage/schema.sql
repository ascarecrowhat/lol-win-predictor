-- Raw layer: exactly what Riot returned, gzipped, so features can be re-derived
-- later without re-crawling.
CREATE TABLE IF NOT EXISTS raw_matches (
    match_id   VARCHAR PRIMARY KEY,
    fetched_at TIMESTAMP NOT NULL,
    patch      VARCHAR,
    payload    BLOB NOT NULL          -- gzip(json)
);

-- Clean layer.
CREATE TABLE IF NOT EXISTS matches (
    match_id     VARCHAR PRIMARY KEY,
    platform     VARCHAR,
    queue_id     INTEGER,
    patch        VARCHAR,
    game_version VARCHAR,
    game_start   TIMESTAMP,
    duration_s   INTEGER,
    winner       SMALLINT               -- 100 = blue, 200 = red
);

CREATE TABLE IF NOT EXISTS participants (
    match_id     VARCHAR,
    puuid        VARCHAR,
    team_id      SMALLINT,
    position     VARCHAR,
    champion_id  INTEGER,
    win          BOOLEAN,
    kills        INTEGER,
    deaths       INTEGER,
    assists      INTEGER,
    gold_earned  INTEGER,
    cs           INTEGER,
    dmg_champs   INTEGER,
    dmg_taken    INTEGER,
    vision_score INTEGER,
    wards_placed INTEGER,
    turret_kills INTEGER,
    team_pos_idx SMALLINT,
    PRIMARY KEY (match_id, puuid)
);

-- Rank snapshots observed while seeding; a time-stamped series per player so
-- features can look up the rank known *before* a given match.
CREATE TABLE IF NOT EXISTS player_ranks (
    puuid       VARCHAR,
    queue       VARCHAR,
    tier        VARCHAR,
    division    VARCHAR,
    lp          INTEGER,
    wins        INTEGER,
    losses      INTEGER,
    observed_at TIMESTAMP,
    PRIMARY KEY (puuid, queue, observed_at)
);

-- Crawl frontier. Everything needed to resume after a crash or Ctrl-C.
CREATE TABLE IF NOT EXISTS crawl_players (
    puuid        VARCHAR PRIMARY KEY,
    tier         VARCHAR,
    division     VARCHAR,
    lp           INTEGER,
    source       VARCHAR,              -- 'seed' or 'snowball'
    discovered_at TIMESTAMP,
    processed_at TIMESTAMP,            -- NULL = still in the frontier
    n_match_ids  INTEGER
);

CREATE TABLE IF NOT EXISTS crawl_matches (
    match_id        VARCHAR PRIMARY KEY,
    state           VARCHAR,           -- pending | stored | rejected | error
    reason          VARCHAR,
    discovered_at   TIMESTAMP,
    resolved_at     TIMESTAMP,
    discovery_count INTEGER DEFAULT 0, -- processed participants known to be in it
    attempts        INTEGER DEFAULT 0  -- transient failures so far
);

-- Discovery map: "player P appears in match M", learned from match-v5 ids calls
-- without fetching M itself. Because we only record an edge for a player we have
-- fully processed, the number of edges on a match is a lower bound on how many of
-- its ten participants we hold deep history for - known before spending a request
-- on the match.
CREATE TABLE IF NOT EXISTS match_discoveries (
    match_id VARCHAR,
    puuid    VARCHAR,
    PRIMARY KEY (match_id, puuid)
);
