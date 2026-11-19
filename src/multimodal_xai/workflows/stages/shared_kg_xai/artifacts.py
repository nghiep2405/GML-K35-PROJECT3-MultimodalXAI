def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def read_mapping(path):
    with open(path, encoding="utf-8-sig") as f:
        n = int(f.readline()); rows = [x.rsplit(maxsplit=1) for x in f if x.strip()]
    df = pd.DataFrame(rows, columns=["raw", "id"]); df["id"] = df.id.astype("int64")
    assert len(df) == n
    return df.sort_values("id").raw.tolist()

def read_triples(path):
    with open(path) as f:
        n = int(f.readline()); a = np.loadtxt(f, dtype=np.int64, ndmin=2)
    assert a.shape == (n, 3)
    return a                                         # cột: head, tail, relation (định dạng OpenKE)

def rel_label(raw):
    s = raw.strip("<>")
    if "#" in s:                                   # ví dụ 22-rdf-syntax-ns#type → type, rdf-schema#seeAlso → see also
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s.rsplit("#", 1)[-1]).lower()
    if "dbpedia.org" in s or s.startswith("http"):
        s = s.rstrip("/").split("/")[-1]
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).lower()
    parts = [p for p in s.split("/") if p]
    return " / ".join(parts[-2:]).replace("_", " ") if parts else s

def latest_locked(folder, ckpt):
    best = None
    for d in sorted(Path(folder).glob("test_locked_*")):
        if not (d / "test_query_ranks.csv").exists():
            continue
        proto = json.loads((d / "test_protocol.json").read_text()) if (d / "test_protocol.json").exists() else {}
        if proto.get("checkpoint_sha256") and proto["checkpoint_sha256"] != sha256(ckpt):
            continue
        best = d
    return best

class DS:
    pass
