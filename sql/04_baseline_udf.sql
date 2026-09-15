-- Baseline inference: a UDF that reads the model weights from the catalog on every call.
-- This is the "run the model inside the database" baseline the paper compares against
-- (it reports the UDF baseline as faster than MADlib, so this is the conservative choice).
--
-- STABLE, not IMMUTABLE: IMMUTABLE would let the planner constant-fold or cache in ways
-- that distort the baseline timings in our favour.
-- COST tells the planner this is far more expensive than an ordinary operator; the
-- placement search (Technique 3) relies on the planner knowing that.

CREATE OR REPLACE FUNCTION lr_predict(model TEXT, feats NUMERIC[])
RETURNS NUMERIC
LANGUAGE sql STABLE COST 500 AS $$
  SELECT m.intercept + COALESCE((SELECT SUM(c * f) FROM unnest(m.coef, feats) AS t(c, f)), 0)
  FROM sys_model m
  WHERE m.model_name = model
$$;

-- Partial inference over a subset of the model's features (by 1-based feature index).
-- Used when composite predicates are evaluated "through the model" rather than as
-- inlined arithmetic (scenario 4: the expensive-predicate placement case).
CREATE OR REPLACE FUNCTION lr_partial(model TEXT, idx INT[], feats NUMERIC[])
RETURNS NUMERIC
LANGUAGE sql STABLE COST 500 AS $$
  SELECT m.intercept + COALESCE((SELECT SUM(m.coef[i] * f) FROM unnest(idx, feats) AS t(i, f)), 0)
  FROM sys_model m
  WHERE m.model_name = model
$$;

-- Logistic classification: sigmoid(z) >= 0.5  <=>  z >= 0
CREATE OR REPLACE FUNCTION logit_classify(model TEXT, feats NUMERIC[])
RETURNS BOOLEAN
LANGUAGE sql STABLE COST 500 AS $$
  SELECT (1.0 / (1.0 + exp(-lr_predict(model, feats)))) >= 0.5
$$;
