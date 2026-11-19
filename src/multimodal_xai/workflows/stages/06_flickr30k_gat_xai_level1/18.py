ENT_DIR = WORK / "flickr30k_entities"
if not (ENT_DIR / "Sentences").exists():
    repo = WORK / "flickr30k_entities_repo"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/BryanPlummer/flickr30k_entities", str(repo)], check=True)
    import zipfile
    with zipfile.ZipFile(repo / "annotations.zip") as z:
        members = [m for m in z.namelist() if m.startswith("Sentences/")]
        z.extractall(ENT_DIR, members=members)

import re
_PHRASE = re.compile(r"\[/EN#(\d+)/([^\s\]]+)\s([^\]]+)\]")

def parse_sentences(image_id):
    path = ENT_DIR / "Sentences" / f"{image_id}.txt"
    if not path.exists():
        return []
    sents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sents.append([(cid, types.split("/"), phrase) for cid, types, phrase in _PHRASE.findall(line)])
    return sents

def region_labels(split, image_index, query_sentence=None):
    image_id = split["image_ids"][image_index]
    sents = parse_sentences(image_id)
    labels = {}
    for r, region in enumerate(split["graph_metadata"][image_index]["regions"]):
        cats, phrase_other, phrase_query = [], None, None
        for cid in region["chain_ids"]:
            for si, sent in enumerate(sents):
                for c, types, phrase in sent:
                    if c != cid:
                        continue
                    for t in types:
                        if t not in cats:
                            cats.append(t)
                    if si != query_sentence and phrase_other is None:
                        phrase_other = phrase
                    if si == query_sentence and phrase_query is None:
                        phrase_query = phrase
        cat = "/".join(cats[:2]) if cats else "?"
        if phrase_other:
            labels[r + 1] = f"R{r}:{cat}:{phrase_other[:40]}"
        else:
            labels[r + 1] = f"R{r}:{cat}*"
    return labels

def query_sentence_index(caption_id):
    return int(caption_id.split("#")[1])

_lbl = region_labels(test, int(T_TARGET[check_q[0]]), query_sentence_index(test["caption_ids"][check_q[0]]))
print(test["captions"][check_q[0]]); print(_lbl)
