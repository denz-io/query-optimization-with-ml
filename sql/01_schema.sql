-- The paper's running example (Figure 1): four tables, two join paths.
--   apartment -> building -> district
--   apartment -> landlord
DROP TABLE IF EXISTS apartment, building, landlord, district CASCADE;

CREATE TABLE district  (did INT PRIMARY KEY, hospital_num INT NOT NULL, zone TEXT NOT NULL);
CREATE TABLE landlord  (lid INT PRIMARY KEY, rating_score NUMERIC(4,1) NOT NULL);
CREATE TABLE building  (bid INT PRIMARY KEY, did INT NOT NULL, building_age INT NOT NULL, zone TEXT NOT NULL);
CREATE TABLE apartment (aid BIGINT PRIMARY KEY, bid INT NOT NULL, lid INT NOT NULL,
                        room_num INT NOT NULL, price NUMERIC(8,2) NOT NULL);
