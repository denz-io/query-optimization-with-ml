# PostgreSQL cheat sheet for the Smart demo

Everything below assumes the stack from `docker compose up -d`: PostgreSQL 16 in the
`smartdemo-pg` container, database `smartdemo`, user `postgres`, password `demo`.

## Connecting

```bash
docker compose exec db psql -U postgres -d smartdemo          # psql inside the container
psql postgresql://postgres:demo@localhost:5432/smartdemo       # from the host, if psql is installed
docker compose exec db psql -U postgres -d smartdemo -c "SELECT count(*) FROM apartment"   # one-liner
docker compose exec -T db psql -U postgres -d smartdemo < sql/03_catalog.sql               # run a file
```

## psql meta-commands

| Command | What it does |
|---|---|
| `\l` | list databases |
| `\c smartdemo` | switch database |
| `\dt` | list tables |
| `\d apartment` | describe a table (columns, indexes, constraints) |
| `\d+ apartment` | same, plus size and storage details |
| `\df` | list functions (`\df lr_*` for the demo UDFs) |
| `\sf lr_predict` | show a function's source |
| `\di` | list indexes |
| `\dv` | list views |
| `\x` | toggle expanded (vertical) output, handy for wide rows |
| `\timing` | print execution time after every query |
| `\i sql/04_baseline_udf.sql` | run a SQL file |
| `\e` | edit the last query in `$EDITOR` |
| `\watch 2` | re-run the last query every 2 s |
| `\copy (SELECT ...) TO 'out.csv' CSV HEADER` | export a result to a file on the client |
| `\q` | quit |

## Demo schema

```
apartment(aid, bid, lid, room_num, price)   3,000,000 rows
building (bid, did, building_age, zone)       300,000
district (did, hospital_num, zone)              1,000
landlord (lid, rating_score)                  100,000
sys_model  (model_name, model_category, intercept, weights JSONB, coef NUMERIC[])
sys_feature(table_name, attribute_name, min_val, max_val)