SCORER_SOURCE = 'sys.path.insert(0,str(REPO))\nfrom mmkgc.module.model import AdvMixRotatE\nclass MaskedAdvMixRotatE(AdvMixRotatE):\n    """AdaMF adaptation: mask absent modalities before attention normalization."""\n    def __init__(self, availability, **kwargs):\n        super().__init__(**kwargs)\n        assert availability.shape == (self.ent_tot, 2)\n        self.register_buffer(\'availability\', availability.bool())\n\n    def _joint(self, ids, visual=None, textual=None):\n        structural = self.ent_embeddings(ids)\n        if visual is None:\n            visual = self.img_proj(self.img_embeddings(ids))\n        if textual is None:\n            textual = self.text_proj(self.text_embeddings(ids))\n        flags = self.availability[ids]\n        present = torch.cat([torch.ones_like(flags[..., :1]), flags], dim=-1)\n        values = torch.stack([structural, visual, textual], dim=-2)\n        # Zero masked values too: suppress projector bias and absent raw rows.\n        values = values.masked_fill(~present.unsqueeze(-1), 0.0)\n        logits = self.ent_attn(torch.tanh(values)).squeeze(-1)\n        logits = logits.masked_fill(~present, float(\'-inf\'))\n        weights = torch.softmax(logits, dim=-1)\n        return (weights.unsqueeze(-1) * values).sum(dim=-2)\n\n    def forward(self, data):\n        h = self._joint(data[\'batch_h\'])\n        t = self._joint(data[\'batch_t\'])\n        r = self.rel_embeddings(data[\'batch_r\'])\n        return self.margin - self._calc(h, t, r, data[\'mode\'])\n\n    def get_fake_score(self, batch_h, batch_r, batch_t, mode,\n                       fake_hv=None, fake_tv=None, fake_ht=None, fake_tt=None):\n        assert all(x is not None for x in [fake_hv, fake_tv, fake_ht, fake_tt])\n        h, t = self._joint(batch_h), self._joint(batch_t)\n        hf = self._joint(batch_h, fake_hv, fake_ht)\n        tf = self._joint(batch_t, fake_tv, fake_tt)\n        r = self.rel_embeddings(batch_r)\n        scores = [self.margin-self._calc(hf,t,r,mode),\n                  self.margin-self._calc(h,tf,r,mode),\n                  self.margin-self._calc(hf,tf,r,mode)]\n        real = [self.img_proj(self.img_embeddings(batch_h)),\n                self.img_proj(self.img_embeddings(batch_t)),\n                self.text_proj(self.text_embeddings(batch_h)),\n                self.text_proj(self.text_embeddings(batch_t))]\n        return scores, real\n\nclass RetrievalAdaMF(nn.Module):\n    def __init__(self,visual,textual,availability):\n        super().__init__()\n        self.kge=MaskedAdvMixRotatE(availability=availability,\n            ent_tot=N,rel_tot=R,dim=CFG[\'dim\'],margin=12.0,epsilon=2.0,\n            img_emb=visual.clone(),text_emb=textual.clone())\n        self.kge.img_embeddings.weight.requires_grad_(False)\n        self.kge.text_embeddings.weight.requires_grad_(False)\n        self.query_projection=nn.Linear(512,2*CFG[\'dim\'])\n        self.logit_scale=nn.Parameter(torch.tensor(2.6592))\n        self.register_buffer(\'image_entities\',IMAGE_ENTITY.clone())\n\n    def score_images(self,rows,text_features,image_feature_rows):\n        # image_feature_rows: [queries,candidates], map image -> all linked entities.\n        result=[]\n        query_z=F.normalize(self.query_projection(text_features),dim=-1)\n        for i,row in enumerate(rows.itertuples(index=False)):\n            ids=self.image_entities[image_feature_rows[i]]\n            present=ids>=0; flat=ids.clamp_min(0).reshape(-1)\n            direction=row.prediction_direction\n            if direction==\'direct\':\n                joint=F.normalize(self.kge._joint(flat),dim=-1)\n                score=joint@query_z[i]\n            else:\n                assert direction in [\'head\',\'tail\']\n                anchor=int(row.anchor_entity_index); relation=int(row.relation_index)\n                assert 0<=anchor<N and 0<=relation<R\n                anchor_ids=torch.full_like(flat,anchor); relation_ids=torch.full_like(flat,relation)\n                batch=dict(batch_h=anchor_ids if direction==\'tail\' else flat,\n                    batch_t=flat if direction==\'tail\' else anchor_ids,\n                    batch_r=relation_ids,mode=\'normal\')\n                # AdaMF score higher is better; scale keeps CE logits manageable.\n                score=self.kge(batch)/CFG[\'dim\']\n            score=score.reshape(ids.shape).masked_fill(~present,float(\'-inf\')).max(dim=-1).values\n            result.append(score)\n        return torch.stack(result)\n\ndef make_model(seed):\n    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)\n    return RetrievalAdaMF(train_visual,TEXTUAL,train_availability).to(DEVICE)\n\n'
sys.path.insert(0,str(REPO))
from mmkgc.module.model import AdvMixRotatE
class MaskedAdvMixRotatE(AdvMixRotatE):
    """AdaMF adaptation: mask absent modalities before attention normalization."""
    def __init__(self, availability, **kwargs):
        super().__init__(**kwargs)
        assert availability.shape == (self.ent_tot, 2)
        self.register_buffer('availability', availability.bool())

    def _joint(self, ids, visual=None, textual=None):
        structural = self.ent_embeddings(ids)
        if visual is None:
            visual = self.img_proj(self.img_embeddings(ids))
        if textual is None:
            textual = self.text_proj(self.text_embeddings(ids))
        flags = self.availability[ids]
        present = torch.cat([torch.ones_like(flags[..., :1]), flags], dim=-1)
        values = torch.stack([structural, visual, textual], dim=-2)
        # Zero masked values too: suppress projector bias and absent raw rows.
        values = values.masked_fill(~present.unsqueeze(-1), 0.0)
        logits = self.ent_attn(torch.tanh(values)).squeeze(-1)
        logits = logits.masked_fill(~present, float('-inf'))
        weights = torch.softmax(logits, dim=-1)
        return (weights.unsqueeze(-1) * values).sum(dim=-2)

    def forward(self, data):
        h = self._joint(data['batch_h'])
        t = self._joint(data['batch_t'])
        r = self.rel_embeddings(data['batch_r'])
        return self.margin - self._calc(h, t, r, data['mode'])

    def get_fake_score(self, batch_h, batch_r, batch_t, mode,
                       fake_hv=None, fake_tv=None, fake_ht=None, fake_tt=None):
        assert all(x is not None for x in [fake_hv, fake_tv, fake_ht, fake_tt])
        h, t = self._joint(batch_h), self._joint(batch_t)
        hf = self._joint(batch_h, fake_hv, fake_ht)
        tf = self._joint(batch_t, fake_tv, fake_tt)
        r = self.rel_embeddings(batch_r)
        scores = [self.margin-self._calc(hf,t,r,mode),
                  self.margin-self._calc(h,tf,r,mode),
                  self.margin-self._calc(hf,tf,r,mode)]
        real = [self.img_proj(self.img_embeddings(batch_h)),
                self.img_proj(self.img_embeddings(batch_t)),
                self.text_proj(self.text_embeddings(batch_h)),
                self.text_proj(self.text_embeddings(batch_t))]
        return scores, real

class RetrievalAdaMF(nn.Module):
    def __init__(self,visual,textual,availability):
        super().__init__()
        self.kge=MaskedAdvMixRotatE(availability=availability,
            ent_tot=N,rel_tot=R,dim=CFG['dim'],margin=12.0,epsilon=2.0,
            img_emb=visual.clone(),text_emb=textual.clone())
        self.kge.img_embeddings.weight.requires_grad_(False)
        self.kge.text_embeddings.weight.requires_grad_(False)
        self.query_projection=nn.Linear(512,2*CFG['dim'])
        self.logit_scale=nn.Parameter(torch.tensor(2.6592))
        self.register_buffer('image_entities',IMAGE_ENTITY.clone())

    def score_images(self,rows,text_features,image_feature_rows):
        # image_feature_rows: [queries,candidates], map image -> all linked entities.
        result=[]
        query_z=F.normalize(self.query_projection(text_features),dim=-1)
        for i,row in enumerate(rows.itertuples(index=False)):
            ids=self.image_entities[image_feature_rows[i]]
            present=ids>=0; flat=ids.clamp_min(0).reshape(-1)
            direction=row.prediction_direction
            if direction=='direct':
                joint=F.normalize(self.kge._joint(flat),dim=-1)
                score=joint@query_z[i]
            else:
                assert direction in ['head','tail']
                anchor=int(row.anchor_entity_index); relation=int(row.relation_index)
                assert 0<=anchor<N and 0<=relation<R
                anchor_ids=torch.full_like(flat,anchor); relation_ids=torch.full_like(flat,relation)
                batch=dict(batch_h=anchor_ids if direction=='tail' else flat,
                    batch_t=flat if direction=='tail' else anchor_ids,
                    batch_r=relation_ids,mode='normal')
                # AdaMF score higher is better; scale keeps CE logits manageable.
                score=self.kge(batch)/CFG['dim']
            score=score.reshape(ids.shape).masked_fill(~present,float('-inf')).max(dim=-1).values
            result.append(score)
        return torch.stack(result)

def make_model(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return RetrievalAdaMF(train_visual,TEXTUAL,train_availability).to(DEVICE)

smoke_model=make_model(SEEDS[0])
sample=train_q.iloc[:3]
sample_candidates=torch.stack([torch.tensor([positive_rows['train'][row.query_id][0]]+TRAIN_CACHE['negative_rows'][i][:CFG['train_negatives']].tolist()) for i,row in enumerate(sample.itertuples(index=False))]).to(DEVICE)
smoke_scores=smoke_model.score_images(sample,CLIP_QUERY['train'][:len(sample)].to(DEVICE),sample_candidates)
assert smoke_scores.shape==(len(sample),1+CFG['train_negatives']) and bool(torch.isfinite(smoke_scores).all())
smoke_loss=F.cross_entropy(smoke_scores*smoke_model.logit_scale.exp().clamp(max=100),torch.zeros(len(sample),dtype=torch.long,device=DEVICE))
smoke_loss.backward()
assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in smoke_model.parameters())
# Target label must not affect forward scores.
with torch.no_grad():
    altered=sample.copy(); altered['target_entity_index']=(altered.target_entity_index+1)%N
    assert torch.allclose(smoke_scores,smoke_model.score_images(altered,CLIP_QUERY['train'][:len(sample)].to(DEVICE),sample_candidates))
del smoke_model,smoke_scores,smoke_loss
torch.cuda.empty_cache()
print('Retrieval forward/backward smoke PASS. Train khởi tạo lại model mới.')
