"""Path isolation and immutable baseline contracts. Own all writer paths in experiment_v2."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SPLIT_SHA256 = '6cdc592eb7ee52738adff612d97824bd5b42b61d1d6064d527f9d6ce3f849100'
VAL_HASHES = {'candidate_images.parquet':'1f1744a5fe3e882cef784ebb7d37c6dd11335648707ebc1e29a47d0808f35bb8',
              'queries.parquet':'06add985d6b64d367ac97733c8c9b4ec13f5e38aef838fd5663da0b73c5b978e',
              'ground_truth.parquet':'4ca5d2bc65169b617aa9c9d7facb271ada7f49cec0862d1052908b42fca6a700'}
TEST_HASHES = {'candidate_images.parquet':'ed108c49ec995af38f4dc479037fcd0636f6de27cacaeeb4e355263e53002437',
               'queries.parquet':'cfc0d47544d7f444b527ea874a7b660078475f95e6f3c4aca06d2dba786732ef',
               'ground_truth.parquet':'a630343b9ff34a0c6b85cbb33d07deb174e48204af1b08889b71227d60e43a18'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fd:
        for chunk in iter(lambda: fd.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class WriteGuard:
    def __init__(self, original_root: Path, experiment_root: Path):
        self.original = Path(original_root).resolve()
        self.root = Path(experiment_root).resolve()
        if self.root == self.original or self.root.is_relative_to(self.original):
            raise ValueError('Experiment must not be inside the immutable Visual Genome root')
        if self.original.is_relative_to(self.root):
            raise ValueError('Source must not be inside writable experiment root')
        if self.root.name != 'experiment_v2':
            raise ValueError('Writable root must be named experiment_v2')

    def writable(self, file: Path) -> Path:
        dest = Path(file).resolve()
        if not dest.is_relative_to(self.root) or dest == self.root:
            raise PermissionError(f'WRITE BLOCKED: {dest} outside experiment_v2')
        if dest.is_relative_to(self.original):
            raise PermissionError(f'WRITE BLOCKED: source {dest}')
        return dest

    def mkdir(self, path: Path):
        p = self.writable(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def assert_original(self, path: Path):
        p = Path(path).resolve()
        if not p.is_relative_to(self.original):
            raise ValueError(f'Unexpected original-source path: {p}')
        return p

    def save_json(self, path: Path, payload):
        p = self.writable(path); p.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.writable(p.with_name(p.name + '.tmp'))
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        tmp.replace(p)


def verify_frozen_val_test(vg_root: Path, full_sha=True, val_hashes=None, test_hashes=None, expected_split_sha=None) -> dict:
    """Only reads original files; full_sha=True performs exact artifact checksums."""
    root=Path(vg_root); base=root/'clip_baseline'/'split_v2'
    checks={}
    vh = val_hashes or VAL_HASHES
    th = test_hashes or TEST_HASHES
    split_sha = expected_split_sha or SPLIT_SHA256
    for split, hashes in [('val', vh), ('test', th)]:
        protocol=json.loads((base/split/'protocol.json').read_text())
        assert protocol['split_version']=='split_v2'
        assert protocol['split_sha256']==split_sha
        for name, expected in hashes.items():
            assert protocol['artifact_sha256'][{'candidate_images.parquet':'candidates','queries.parquet':'queries','ground_truth.parquet':'ground_truth'}[name]] == expected
            p=base/split/name
            assert p.is_file(), f'Missing immutable {p}'
            if full_sha:
                assert sha256(p)==expected, f'IMMUTABLE FILE HASH MISMATCH: {p}'
            checks[f'{split}/{name}']='PASS' if full_sha else 'MANIFEST_MATCH_ONLY'
    return checks


def verify_original_split(vg_root: Path, expected_counts=None, expected_sha=None) -> dict:
    """Independently check frozen image IDs, counts, duplicates, cluster placement."""
    import duckdb
    root=Path(vg_root)
    con=duckdb.connect(database=':memory:')
    p=lambda x:str(Path(x).resolve()).replace("'","''")
    splits=root/'processed/splits_v2.parquet'
    target_sha = expected_sha or SPLIT_SHA256
    target_counts = expected_counts or {'train':83540,'val':10443,'test':10443}
    if sha256(splits)!=target_sha:
        raise AssertionError('FROZEN SPLIT FILE HASH MISMATCH: processed/splits_v2.parquet')
    got=con.execute(f"SELECT split, COUNT(*) n, COUNT(DISTINCT image_id) unique_images FROM read_parquet('{p(splits)}') GROUP BY split").fetchdf()
    result={str(r.split):int(r.n) for r in got.itertuples(index=False)}
    assert result == target_counts, result
    assert all(int(r.n)==int(r.unique_images) for r in got.itertuples(index=False))
    crossings=con.execute(f"SELECT COUNT(*) FROM (SELECT cluster_id FROM read_parquet('{p(splits)}') GROUP BY cluster_id HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
    assert crossings==0,f'Cross-split duplicate clusters: {crossings}'
    for name in ('train','val','test'):
        old=root/f'clip_baseline/split_v2/{"train_hard_negatives" if name=="train" else name}/candidate_images.parquet'
        difference=con.execute(f"""WITH s AS (SELECT CAST(image_id AS VARCHAR) image_id FROM read_parquet('{p(splits)}') WHERE split='{name}'),
          c AS (SELECT CAST(image_id AS VARCHAR) image_id FROM read_parquet('{p(old)}'))
          SELECT (SELECT COUNT(*) FROM (SELECT * FROM s EXCEPT SELECT * FROM c)) +
                 (SELECT COUNT(*) FROM (SELECT * FROM c EXCEPT SELECT * FROM s))""").fetchone()[0]
        assert difference==0, f'{name} original candidate image IDs differ from split_v2'
    con.close()
    return {'status':'PASS','split_counts':result,'cross_split_clusters':0,'candidate_sets_match_original':True}
