from pathlib import Path
import duckdb
import pandas as pd
import pyarrow.parquet as pq

DATA = EXP_ROOT / "data"

PARTS_DIR = DATA / "hard_negatives/parts"
POOL = DATA / "query_pool/queries.parquet"
GT = DATA / "query_pool/ground_truth.parquet"

TRAIN_INDEX = (
    VG_ROOT
    / "clip_baseline/split_v2/train_hard_negatives/image_embedding_index.parquet"
)

parts = sorted(PARTS_DIR.glob("part_*.parquet"))

assert len(parts) == 102, f"Expected 102 parts, got {len(parts)}"

con = duckdb.connect(database=":memory:")
con.execute("SET threads = 4")
con.execute("SET memory_limit = '6GB'")

def sqlpath(path):
    return str(path).replace("'", "''")

H = sqlpath(PARTS_DIR / "part_*.parquet")
Q = sqlpath(POOL)
G = sqlpath(GT)
I = sqlpath(TRAIN_INDEX)

# 1. Count, uniqueness, list lengths
stats = con.execute(f"""
    SELECT
        COUNT(*) AS total,
        COUNT(DISTINCT text_id) AS unique_queries,

        SUM(CASE
            WHEN negative_image_ids IS NULL
              OR array_length(negative_image_ids) <> 50
            THEN 1 ELSE 0 END) AS bad_k,

        SUM(CASE
            WHEN list_unique(negative_image_ids) <> 50
            THEN 1 ELSE 0 END) AS duplicate_negatives,

        SUM(CASE
            WHEN negative_scores IS NULL
              OR array_length(negative_scores) <> 50
            THEN 1 ELSE 0 END) AS bad_scores

    FROM read_parquet('{H}')
""").fetchone()

total, unique_queries, bad_k, duplicate_negatives, bad_scores = stats

pool_count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{Q}')"
).fetchone()[0]

print("Total queries:", total)
print("Unique text IDs:", unique_queries)
print("Pool queries:", pool_count)
print("Incorrect negative count:", bad_k)
print("Duplicate negatives:", duplicate_negatives)
print("Incorrect score count:", bad_scores)

assert total == unique_queries == pool_count
assert bad_k == duplicate_negatives == bad_scores == 0

# 2. Verify query and canonical positive correspondence
bad_query_mapping = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{H}') h
    LEFT JOIN read_parquet('{Q}') q USING (text_id)
    WHERE q.text_id IS NULL
       OR h.positive_image_id IS DISTINCT FROM q.positive_image_id
""").fetchone()[0]

print("Incorrect query mappings:", bad_query_mapping)
assert bad_query_mapping == 0

# 3. Exclude ALL known positives
positive_leakage = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{H}') h
    JOIN read_parquet('{G}') g USING (text_id)
    WHERE list_contains(h.negative_image_ids, g.image_id)
""").fetchone()[0]

print("Known-positive leakage:", positive_leakage)
assert positive_leakage == 0

# 4. Every negative must be a train candidate
out_of_train = con.execute(f"""
    WITH negatives AS (
        SELECT UNNEST(negative_image_ids) AS image_id
        FROM read_parquet('{H}')
    )
    SELECT COUNT(*)
    FROM negatives n
    ANTI JOIN read_parquet('{I}') i
        ON n.image_id = CAST(i.image_id AS VARCHAR)
""").fetchone()[0]

print("Negatives outside train:", out_of_train)
assert out_of_train == 0

# 5. Check shard ordering against embedding indexes
for part in parts:
    index_path = DATA / "embeddings/indexes" / part.name

    index_df = pd.read_parquet(
        index_path,
        columns=["text_id", "embedding_row"]
    )

    hard_ids = pq.read_table(
        part,
        columns=["text_id"]
    ).column("text_id").to_pylist()

    assert index_df.text_id.astype(str).tolist() == hard_ids, (
        f"Text ID order mismatch: {part.name}"
    )

    assert index_df.embedding_row.tolist() == list(
        range(len(index_df))
    ), f"Invalid embedding index: {part.name}"

    print(f"Alignment PASS: {part.name}", end="\r")

con.close()

print("\nALL HARD-NEGATIVE STRUCTURAL CHECKS PASSED")
