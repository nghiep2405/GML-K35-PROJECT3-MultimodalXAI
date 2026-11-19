MODEL_SOURCE = 'class Model(nn.Module):\n    def __init__(self, ent_tot, rel_tot):\n        super().__init__(); self.ent_tot=ent_tot; self.rel_tot=rel_tot; self.pi_const=3.14159265358979323846\nclass Loss(nn.Module):\n    pass\nimport torch\nimport torch.autograd as autograd\nimport torch.nn as nn\n\n\n\nclass AdvMixRotatE(Model):\n\n    def __init__(\n        self,\n        ent_tot,\n        rel_tot,\n        dim=100,\n        margin=6.0,\n        epsilon=2.0,\n        img_emb=None,\n        text_emb=None\n    ):\n\n        super(AdvMixRotatE, self).__init__(ent_tot, rel_tot)\n        assert img_emb is not None\n        assert text_emb is not None\n        self.margin = margin\n        self.epsilon = epsilon\n        self.dim_e = dim * 2\n        self.dim_r = dim\n        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)\n        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)\n        self.ent_embedding_range = nn.Parameter(\n            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),\n            requires_grad=False\n        )\n        self.img_dim = img_emb.shape[1]\n        self.text_dim = text_emb.shape[1]\n        self.img_proj = nn.Linear(self.img_dim, self.dim_e)\n        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(True)\n        self.text_proj = nn.Linear(self.text_dim, self.dim_e)\n        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)\n        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)\n        self.ent_attn.requires_grad_(True)\n        nn.init.uniform_(\n            tensor=self.ent_embeddings.weight.data,\n            a=-self.ent_embedding_range.item(),\n            b=self.ent_embedding_range.item()\n        )\n        self.rel_embedding_range = nn.Parameter(\n            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),\n            requires_grad=False\n        )\n        nn.init.uniform_(\n            tensor=self.rel_embeddings.weight.data,\n            a=-self.rel_embedding_range.item(),\n            b=self.rel_embedding_range.item()\n        )\n        self.margin = nn.Parameter(torch.Tensor([margin]))\n        self.margin.requires_grad = False\n        \n\n    def get_joint_embeddings(self, es, ev, et):\n        # es, ev, et: [num_ent, emb_dim]\n        e = torch.stack((es, ev, et), dim=1)\n        u = torch.tanh(e)\n        scores = self.ent_attn(u).squeeze(-1)\n        attention_weights = torch.softmax(scores, dim=-1)\n        context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)\n        return context_vectors\n    \n\n    def _calc(self, h, t, r, mode):\n        pi = self.pi_const\n\n        re_head, im_head = torch.chunk(h, 2, dim=-1)\n        re_tail, im_tail = torch.chunk(t, 2, dim=-1)\n\n        phase_relation = r / (self.rel_embedding_range.item() / pi)\n\n        re_relation = torch.cos(phase_relation)\n        im_relation = torch.sin(phase_relation)\n\n        re_head = re_head.view(-1,\n                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)\n        re_tail = re_tail.view(-1,\n                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)\n        im_head = im_head.view(-1,\n                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)\n        im_tail = im_tail.view(-1,\n                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)\n        im_relation = im_relation.view(\n            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)\n        re_relation = re_relation.view(\n            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)\n\n        if mode == "head_batch":\n            re_score = re_relation * re_tail + im_relation * im_tail\n            im_score = re_relation * im_tail - im_relation * re_tail\n            re_score = re_score - re_head\n            im_score = im_score - im_head\n        else:\n            re_score = re_head * re_relation - im_head * im_relation\n            im_score = re_head * im_relation + im_head * re_relation\n            re_score = re_score - re_tail\n            im_score = im_score - im_tail\n\n        score = torch.stack([re_score, im_score], dim=0)\n        score = score.norm(dim=0).sum(dim=-1)\n        return score.permute(1, 0).flatten()\n\n    def forward(self, data):\n        batch_h = data[\'batch_h\']\n        batch_t = data[\'batch_t\']\n        batch_r = data[\'batch_r\']\n        mode = data[\'mode\']\n        h = self.ent_embeddings(batch_h)\n        t = self.ent_embeddings(batch_t)\n        r = self.rel_embeddings(batch_r)\n        h_img_emb = self.img_proj(self.img_embeddings(batch_h))\n        t_img_emb = self.img_proj(self.img_embeddings(batch_t))\n        h_text_emb = self.text_proj(self.text_embeddings(batch_h))\n        t_text_emb = self.text_proj(self.text_embeddings(batch_t))\n        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb)\n        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb)\n        score = self.margin - self._calc(h_joint, t_joint, r, mode)\n        return score\n    \n    def get_batch_ent_embs(self, data):\n        return self.ent_embeddings(data)\n    \n    def get_fake_score(\n        self,\n        batch_h,\n        batch_r, \n        batch_t,\n        mode,\n        fake_hv=None, \n        fake_tv=None,\n        fake_ht=None,\n        fake_tt=None\n    ):\n        if fake_hv is None or fake_tv is None or fake_ht is None or fake_tt is None:\n            raise NotImplementedError\n        h = self.ent_embeddings(batch_h)\n        t = self.ent_embeddings(batch_t)\n        r = self.rel_embeddings(batch_r)\n        h_img_emb = self.img_proj(self.img_embeddings(batch_h))\n        t_img_emb = self.img_proj(self.img_embeddings(batch_t))\n        h_text_emb = self.text_proj(self.text_embeddings(batch_h))\n        t_text_emb = self.text_proj(self.text_embeddings(batch_t))\n        # the fake joint embedding\n        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb)\n        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb)\n        h_fake = self.get_joint_embeddings(h, fake_hv, fake_ht)\n        t_fake = self.get_joint_embeddings(t, fake_tv, fake_tt)\n        score_h = self.margin - self._calc(h_fake, t_joint, r, mode)\n        score_t = self.margin - self._calc(h_joint, t_fake, r, mode)\n        score_all = self.margin - self._calc(h_fake, t_fake, r, mode)\n        return [score_h, score_t, score_all], [h_img_emb, t_img_emb, h_text_emb, t_text_emb]\n\n    def predict(self, data):\n        score = -self.forward(data)\n        return score.cpu().data.numpy()\n\n    def regularization(self, data):\n        batch_h = data[\'batch_h\']\n        batch_t = data[\'batch_t\']\n        batch_r = data[\'batch_r\']\n        h = self.ent_embeddings(batch_h)\n        t = self.ent_embeddings(batch_t)\n        r = self.rel_embeddings(batch_r)\n        regul = (torch.mean(h ** 2) +\n                 torch.mean(t ** 2) +\n                 torch.mean(r ** 2)) / 3\n        return regul\n    \n    def get_attention(self, es, ev, et):\n        # es, ev, et: [num_ent, emb_dim]\n        e = torch.stack((es, ev, et), dim=1)\n        u = torch.tanh(e)\n        scores = self.ent_attn(u).squeeze(-1)\n        attention_weights = torch.softmax(scores, dim=-1)\n        return attention_weights\n\n    def get_attention_weight(self, h, t):\n        h = torch.LongTensor([h])\n        t = torch.LongTensor([t])\n        h_s = self.ent_embeddings(h)\n        t_s = self.ent_embeddings(t)\n        h_img_emb = self.img_proj(self.img_embeddings(h))\n        t_img_emb = self.img_proj(self.img_embeddings(t))\n        h_text_emb = self.text_proj(self.text_embeddings(h))\n        t_text_emb = self.text_proj(self.text_embeddings(t))\n        # the fake joint embedding\n        h_attn = self.get_attention(h_s, h_img_emb, h_text_emb)\n        t_attn = self.get_attention(t_s, t_img_emb, t_text_emb)\n        return h_attn, t_attn\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport numpy as np\n\n\nclass SigmoidLoss(Loss):\n\n\tdef __init__(self, adv_temperature = None):\n\t\tsuper(SigmoidLoss, self).__init__()\n\t\tself.criterion = nn.LogSigmoid()\n\t\tif adv_temperature != None:\n\t\t\tself.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))\n\t\t\tself.adv_temperature.requires_grad = False\n\t\t\tself.adv_flag = True\n\t\telse:\n\t\t\tself.adv_flag = False\n\n\tdef get_weights(self, n_score):\n\t\treturn F.softmax(n_score * self.adv_temperature, dim = -1).detach()\n\n\tdef forward(self, p_score, n_score):\n\t\tif self.adv_flag:\n\t\t\treturn -(self.criterion(p_score).mean() + (self.get_weights(n_score) * self.criterion(-n_score)).sum(dim = -1).mean()) / 2\n\t\telse:\n\t\t\treturn -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2\n\n\tdef predict(self, p_score, n_score):\n\t\tscore = self.forward(p_score, n_score)\n\t\treturn score.cpu().data.numpy()\n\nclass TrainOnlyAdaMF(AdvMixRotatE):\n    def __init__(self, availability, **kwargs):\n        super().__init__(**kwargs);self.register_buffer(\'availability\',availability.bool())\n        self.img_embeddings.weight.requires_grad_(False)\n        self.text_embeddings.weight.requires_grad_(False)\n    def joint(self,ids):\n        values=torch.stack([self.ent_embeddings(ids),self.img_proj(self.img_embeddings(ids)),self.text_proj(self.text_embeddings(ids))],dim=1)\n        flags=torch.cat([torch.ones_like(self.availability[ids,:1]),self.availability[ids]],dim=1)\n        values=values.masked_fill(~flags.unsqueeze(-1),0.)\n        logits=self.ent_attn(torch.tanh(values)).squeeze(-1).masked_fill(~flags,float(\'-inf\'))\n        return (logits.softmax(-1).unsqueeze(-1)*values).sum(1)\n    def forward(self,data):\n        h,t=data[\'batch_h\'],data[\'batch_t\']\n        ids,inverse=torch.unique(torch.cat([h,t]),sorted=True,return_inverse=True)\n        joint=self.joint(ids);hh=joint[inverse[:len(h)]];tt=joint[inverse[len(h):]]\n        return self.margin-self._calc(hh,tt,self.rel_embeddings(data[\'batch_r\']),data[\'mode\'])\n'
exec(MODEL_SOURCE)

MODEL_HASH=hashlib.sha256(MODEL_SOURCE.encode()).hexdigest()
(RUN/'model_definitions.py').write_text(MODEL_SOURCE,encoding='utf-8')
model=TrainOnlyAdaMF(availability=availability,ent_tot=N,rel_tot=R,dim=CFG['dim'],margin=CFG['margin'],epsilon=2.,img_emb=train_visual,text_emb=textual).to(DEVICE)
criterion=SigmoidLoss(adv_temperature=CFG['adv_temperature']).to(DEVICE)
optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=CFG['lr'])
rng=np.random.default_rng(CFG['seed'])
# Held-out triples are used only as a fixed exclusion set for edge swaps.
test_path = kgroot/'test2id.txt'
with test_path.open() as f:
    declared_test = int(f.readline())
    official_test = np.loadtxt(f, dtype=np.int64, ndmin=2)
assert declared_test == 9902 and official_test.shape == (9902, 3)
assert ((0 <= official_test[:,:2]) & (official_test[:,:2] < N)).all()
assert ((0 <= official_test[:,2]) & (official_test[:,2] < R)).all()
test_set = set(map(tuple, official_test.tolist()))
assert len(test_set) == declared_test
assert not (test_set & (train_set | valid_set))
held_out = valid_set | test_set
test_exclusion_sha256 = sha(test_path)
blocked_swaps = 0
original_train = splits['train'].copy()
graph_rng = np.random.default_rng(CFG['seed'] + 104729)
original_keys = set(map(tuple, original_train.tolist()))
if ABLATION == 'no_edges':
    ablated_train = np.empty((0, 3), dtype=np.int64)
elif ABLATION == 'shuffled_edges':
    ablated_train = original_train.copy()
    accepted_swaps = 0
    for rel in np.unique(original_train[:, 2]):
        positions = np.flatnonzero(original_train[:, 2] == rel)
        occupied = set(map(tuple, ablated_train[positions, :2].tolist()))
        # Tail swaps within each relation preserve exact per-relation in/out degrees.
        for _ in range(20 * len(positions)):
            if len(positions) < 2: break
            i, j = graph_rng.choice(positions, 2, replace=False)
            h1, t1 = map(int, ablated_train[i, :2]); h2, t2 = map(int, ablated_train[j, :2])
            if h1 == h2 or t1 == t2: continue
            if h1 == t2 or h2 == t1: continue
            if (h1, t2) in occupied or (h2, t1) in occupied: continue
            if (h1, t2, rel) in held_out or (h2, t1, rel) in held_out:
                blocked_swaps += 1
                continue
            occupied.remove((h1, t1)); occupied.remove((h2, t2))
            occupied.add((h1, t2)); occupied.add((h2, t1))
            ablated_train[i, 1] = t2; ablated_train[j, 1] = t1
            accepted_swaps += 1
        for col in [0, 1]:
            assert np.array_equal(np.sort(original_train[positions, col]), np.sort(ablated_train[positions, col]))
    assert accepted_swaps > 0
elif ABLATION == 'random_graph':
    ablated_train = original_train.copy()
    for rel in np.unique(original_train[:, 2]):
        positions = np.flatnonzero(original_train[:, 2] == rel)
        occupied = set()
        for pos in positions:
            for attempt in range(100000):
                h, t = map(int, graph_rng.integers(0, N, size=2))
                if h != t and (h, t) not in occupied and (h, t, int(rel)) not in original_keys:
                    occupied.add((h, t)); ablated_train[pos] = (h, t, int(rel)); break
            else: raise RuntimeError('Cannot sample unique random edge')
else:
    raise ValueError(ABLATION)
assert len(np.unique(ablated_train, axis=0)) == len(ablated_train)
if ABLATION != 'no_edges':
    assert np.array_equal(np.bincount(ablated_train[:, 2], minlength=R), np.bincount(original_train[:, 2], minlength=R))
    assert not np.array_equal(original_train, ablated_train)
overlap = len(original_keys & set(map(tuple, ablated_train.tolist())))
GRAPH_HASH = hashlib.sha256(ablated_train.tobytes()).hexdigest()
shuffled_set = set(map(tuple, ablated_train.tolist()))
validation_overlap = len(shuffled_set & valid_set)
test_overlap = len(shuffled_set & test_set)
assert validation_overlap == test_overlap == 0, (validation_overlap, test_overlap)
assert len(shuffled_set) == len(original_train) == 71300, 'Unexpected edge count or duplicate triples'
print('Graph verified:', len(shuffled_set), 'unique edges, validation overlap=0, test overlap=0')
graph_report = dict(validation_overlap=validation_overlap, test_overlap=test_overlap,
    accepted_swaps=accepted_swaps, blocked_swaps=blocked_swaps,
    validation_exclusion_sha256=HASHES['valid'], test_exclusion_sha256=test_exclusion_sha256,
    ablation=ABLATION, seed=CFG['seed'], original_edges=len(original_train),
    transformed_edges=len(ablated_train), retained_original_edges=overlap,
    graph_sha256=GRAPH_HASH, source_train_sha256=HASHES['train'],
    no_edges_policy='CLIP only; no KGC training' if ABLATION == 'no_edges' else None,
    label_policy='Fixed validation and official test triple exclusion only; no test metric used for graph or model selection')
graph_path = GRAPH_DIR/'ablation_train_triples.npy'
if graph_path.exists():
    assert np.array_equal(np.load(graph_path, allow_pickle=False), ablated_train), 'Existing graph differs'
else:
    np.save(graph_path, ablated_train, allow_pickle=False)
write_json(graph_report, GRAPH_DIR/'ablation_graph_report.json')
print(json.dumps(graph_report, indent=2))

train_array=ablated_train
def triple_keys(a):return (a[...,0]*N+a[...,1])*R+a[...,2]
known_keys=np.unique(triple_keys(train_array))
head_probability=np.full(R,.5)
for r in np.unique(train_array[:,2]):
    part=train_array[train_array[:,2]==r]
    tph=len(part)/len(np.unique(part[:,0]));hpt=len(part)/len(np.unique(part[:,1]))
    head_probability[r]=tph/(tph+hpt)
def sample_negatives(batch,rng):
    b=len(batch);k=CFG['negative_count']
    neg=np.repeat(batch[:,None,:],k,axis=1)
    corrupt_head=rng.random((b,k))<head_probability[batch[:,2],None]
    for attempt in range(100):
        if attempt==0:bad=np.ones((b,k),dtype=bool)
        i,j=np.where(bad)
        neg[i,j,np.where(corrupt_head[i,j],0,1)]=rng.integers(0,N,size=len(i))
        bad=np.isin(triple_keys(neg),known_keys)
        if not bad.any():return neg
    raise RuntimeError('Không lấy được negative sau 100 lần; kiểm tra graph/sampler')
def kgc_loss(batch,neg):
    b=len(batch);flat=np.concatenate([batch,neg.reshape(-1,3)],axis=0)
    a=torch.tensor(flat,dtype=torch.long,device=DEVICE)
    scores=model(dict(batch_h=a[:,0],batch_t=a[:,1],batch_r=a[:,2],mode='normal'))
    return criterion(scores[:b],scores[b:].reshape(b,-1))
# One smoke backward; reset random states so smoke does not change training samples.
saved_rng=rng.bit_generator.state
loss=kgc_loss(train_array[:2],sample_negatives(train_array[:2],rng));assert torch.isfinite(loss)
loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
optimizer.zero_grad(set_to_none=True);rng.bit_generator.state=saved_rng
print('KGC forward/backward PASS; trainable params:',sum(p.numel() for p in model.parameters() if p.requires_grad))
