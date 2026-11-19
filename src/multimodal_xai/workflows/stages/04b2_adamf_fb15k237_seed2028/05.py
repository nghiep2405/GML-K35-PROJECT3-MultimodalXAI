subprocess.run([sys.executable,'-m','pip','install','-q','scikit-learn','tqdm','pandas','pyarrow','transformers','huggingface_hub','matplotlib'],check=True)
import numpy as np, pandas as pd
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def resolve_data(value):
    p=(PHASE2_ROOT/str(value)).resolve()
    assert p.is_relative_to(PHASE2_ROOT.resolve()), f'Path nằm ngoài dataset: {p}'
    return p
MANIFEST_PATH=PHASE2_ROOT/'processed/fb15k237_img/retrieval_bridge/manifest.json'
BRIDGE=json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
assert BRIDGE['version']=='fb15k237_img_v1'
CANONICAL=resolve_data(BRIDGE['canonical_root'])
REPRODUCTION=resolve_data(BRIDGE['promoted_reproduction'])
assert CANONICAL==resolve_data('fb15k237_image_retrieval/runs/fb15k237_v1')
SOURCE_MODE='group_snapshot'
snapshot=PHASE2_ROOT/'sources/AdaMF-MAT'
# if snapshot.is_dir():
#     if not REPO.exists(): shutil.copytree(snapshot,REPO)
#     digest=hashlib.sha256()
#     for p in sorted(snapshot.rglob('*')):
#         if p.is_file() and p.suffix in ['.py','.cpp','.h','.sh']:
#             rel=p.relative_to(snapshot)
#             assert (REPO/rel).is_file() and (REPO/rel).read_bytes()==p.read_bytes(), f'Runtime source mismatch: {rel}'
#             digest.update(rel.as_posix().encode()); digest.update(p.read_bytes())
#     COMMIT='group-source-sha256:'+digest.hexdigest()

if snapshot.is_dir():
    # 1. Lấy danh sách source files cần xác minh
    source_files = sorted(
        p for p in snapshot.rglob('*')
        if p.is_file()
        and p.suffix in ['.py', '.cpp', '.h', '.sh']
    )
    assert source_files, 'Source snapshot không có source files'
    # 2. Kiểm tra runtime source
    needs_sync = (
        not REPO.is_dir()
        or any(
            not (REPO / p.relative_to(snapshot)).is_file()
            or (REPO / p.relative_to(snapshot)).read_bytes() != p.read_bytes()
            for p in source_files
        )
    )
    # 3. Chỉ đồng bộ khi runtime thiếu hoặc khác source
    if needs_sync:
        print('Runtime source incomplete or mismatched. Synchronizing...')
        # Bảo vệ tránh xóa nhầm thư mục khác
        assert REPO.resolve() == Path(
            '/content/AdaMF-MAT-fb15k237-retrieval'
        )
        if REPO.exists():
            shutil.rmtree(REPO)
        shutil.copytree(snapshot, REPO)
        print('Runtime source synchronized successfully.')
    else:
        print('Runtime source verified. No synchronization needed.')
    # 4. Xác minh toàn bộ source và tính fingerprint
    digest = hashlib.sha256()
    for p in source_files:
        rel = p.relative_to(snapshot)
        runtime_file = REPO / rel
        assert runtime_file.is_file(), (
            f'Missing runtime source: {rel}'
        )
        assert runtime_file.read_bytes() == p.read_bytes(), (
            f'Runtime source mismatch: {rel}'
        )
        digest.update(rel.as_posix().encode())
        digest.update(p.read_bytes())
    COMMIT = 'group-source-sha256:' + digest.hexdigest()


    print('Source verification passed.')
    print('Source fingerprint:', COMMIT)

else:
    SOURCE_MODE='official_repository_fallback'
    if RESUME_TRAINING:
        old=json.loads((RUN/'identity.json').read_text())['provenance']['commit']
        assert not old.startswith('group-source-'), 'Resume cần source snapshot nhóm như run cũ.'
        REPO_REF=old
    if not REPO.exists():
        subprocess.run(['git','clone','https://github.com/zjukg/AdaMF-MAT.git',str(REPO)],check=True)
    subprocess.run(['git','fetch','origin',REPO_REF],cwd=REPO,check=True)
    subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=REPO,check=True)
    COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
assert (REPO/'mmkgc/base/Base.cpp').is_file()
print('Source:',SOURCE_MODE,COMMIT)
print('CANONICAL:',CANONICAL,'\nREPRODUCTION:',REPRODUCTION)
