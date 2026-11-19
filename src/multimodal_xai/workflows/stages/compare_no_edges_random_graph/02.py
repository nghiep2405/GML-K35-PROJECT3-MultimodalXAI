import pandas as pd
from IPython.display import display

metrics = ['MRR', 'Recall@1', 'Recall@5', 'Recall@10']
groups = ['combined', 'direct', 'indirect']
columns = ['ablation', 'seed', 'method', 'group', 'epoch', 'alpha_clip', 'num_queries', *metrics]

def load_result(kind, run_name):
    specified = CSV_PATHS[kind]
    if specified:
        path = Path(specified)
    else:
        matches = list((ROOT / run_name).glob('**/test_locked_*/test_metrics.csv'))
        if not matches:
            raise FileNotFoundError(f'{kind}: không tìm thấy CSV trong {ROOT / run_name}. Điền đường dẫn tại CSV_PATHS.')
        path = max(matches, key=lambda match: match.parent.name)

    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{path}: thiếu cột {sorted(missing)}')
    if set(frame['ablation']) != {kind} or set(frame['seed']) != {2027}:
        raise ValueError(f'{path}: cần kết quả {kind} của DB15K seed 2027.')
    result = frame.loc[frame['group'].isin(groups), columns].copy()
    if set(map(tuple, result[['method','group']].to_numpy())) != {
        (method, group) for method in ('clip','hybrid') for group in groups
    } or len(result) != 6:
        raise ValueError(f'{path}: cần đủ 6 dòng CLIP/Hybrid × combined/direct/indirect.')
    if result[metrics].isna().any().any():
        raise ValueError(f'{path}: thiếu giá trị chỉ số test.')
    print(f'{kind}: {path}')
    return result

results = pd.concat((load_result(kind, run) for kind, run in RUNS.items()), ignore_index=True)
display(results.style.format({name: '{:.6f}' for name in metrics}))
output = Path('db15k_no_edges_random_graph_summary.csv')
results.to_csv(output, index=False)
print(f'Đã lưu {output.resolve()}')
