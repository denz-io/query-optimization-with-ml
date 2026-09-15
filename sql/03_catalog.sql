-- The two catalog tables the rewriter reads. Weights live here, not in code:
-- that is the paper's integration story.
DROP TABLE IF EXISTS sys_model, sys_feature CASCADE;

CREATE TABLE sys_model (
  model_name     TEXT PRIMARY KEY,
  model_category TEXT NOT NULL CHECK (model_category IN ('linear', 'logistic', 'tree')),
  intercept      NUMERIC NOT NULL,
  weights        JSONB NOT NULL,     -- [{"table":"apartment","col":"room_num","w":3.0}, ...]
  coef           NUMERIC[] NOT NULL  -- same weights as a flat array, for the inference UDF
);

CREATE TABLE sys_feature (
  table_name     TEXT,
  attribute_name TEXT,
  min_val        NUMERIC NOT NULL,
  max_val        NUMERIC NOT NULL,
  PRIMARY KEY (table_name, attribute_name)
);
