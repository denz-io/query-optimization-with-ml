-- Synthetic data. Substituted by scripts/setup_db.py with the row counts from .env.
-- price is generated as a linear function of the four feature columns plus noise,
-- so a linear regression trained on it recovers weights close to the paper's 3 / 2 / 1 / 0.5.
--
-- Feature ranges (chosen so every feature contributes a similar span to the prediction):
--   apartment.room_num       1 .. 6     weight 3.0
--   building.building_age    2 .. 10    weight 2.0
--   district.hospital_num    3 .. 15    weight 1.0
--   landlord.rating_score    2.0 .. 10  weight 0.5
SELECT setseed(0.42);

INSERT INTO district
SELECT g,
       3 + floor(random() * 13)::int,
       (ARRAY['north','south','east','west'])[1 + floor(random() * 4)::int]
FROM generate_series(1, {DISTRICT_ROWS}) g;

INSERT INTO landlord
SELECT g, round((2 + random() * 8)::numeric, 1)
FROM generate_series(1, {LANDLORD_ROWS}) g;

INSERT INTO building
SELECT g,
       1 + floor(random() * {DISTRICT_ROWS})::int,
       2 + floor(random() * 9)::int,
       (ARRAY['north','south','east','west'])[1 + floor(random() * 4)::int]
FROM generate_series(1, {BUILDING_ROWS}) g;

-- Apartments: join the dimension rows so price can be computed from the real feature values.
INSERT INTO apartment (aid, bid, lid, room_num, price)
SELECT r.aid, r.bid, r.lid, r.room_num,
       round((3.0 * r.room_num
            + 2.0 * b.building_age
            + 1.0 * d.hospital_num
            + 0.5 * l.rating_score
            + 2.0 * (random() + random() + random() - 1.5))::numeric, 2)   -- noise, sd ~1
FROM (
    SELECT g AS aid,
           1 + floor(random() * {BUILDING_ROWS})::int AS bid,
           1 + floor(random() * {LANDLORD_ROWS})::int AS lid,
           CASE WHEN u < 0.10 THEN 1
                WHEN u < 0.40 THEN 2
                WHEN u < 0.70 THEN 3
                WHEN u < 0.85 THEN 4
                WHEN u < 0.95 THEN 5
                ELSE 6 END AS room_num
    FROM (SELECT g, random() AS u FROM generate_series(1, {APARTMENT_ROWS}) g) gu
) r
JOIN building b ON b.bid = r.bid
JOIN district d ON d.did = b.did
JOIN landlord l ON l.lid = r.lid;

-- A fixed 1% sample of the joined space, used by the rewriter to estimate predicate
-- selectivity (the paper's cost model consumes estimates; we take them from a sample).
DROP TABLE IF EXISTS apartment_sample;
CREATE TABLE apartment_sample AS
SELECT a.aid, a.room_num, b.building_age, d.hospital_num, l.rating_score, a.price
FROM apartment a TABLESAMPLE SYSTEM (1) REPEATABLE (42)
JOIN building b ON a.bid = b.bid
JOIN district d ON b.did = d.did
JOIN landlord l ON a.lid = l.lid;

ANALYZE district; ANALYZE landlord; ANALYZE building; ANALYZE apartment; ANALYZE apartment_sample;
