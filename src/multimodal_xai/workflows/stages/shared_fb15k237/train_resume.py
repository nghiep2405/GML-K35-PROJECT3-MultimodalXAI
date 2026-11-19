ARCHITECTURE='AdaMF_RotatE_Retrieval_v2'
# Save the model definition so future test notebooks can reconstruct the scorer.
model_source='import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom mmkgc.module.model import AdvMixRotatE\n'+'class MaskedAdvMixRotatE(AdvMixRotatE):\n    """AdaMF adaptation: mask absent modalities before attention normalization."""\n    def __init__(self, availability, **kwargs):\n        super().__init__(**kwargs)\n        assert availability.shape == (self.ent_tot, 2)\n        self.register_buffer(\'availability\', availability.bool())\n\n    def _joint(self, ids, visual=None, textual=None):\n        structural = self.ent_embeddings(ids)\n        if visual is None:\n            visual = self.img_proj(self.img_embeddings(ids))\n        if textual is None:\n            textual = self.text_proj(self.text_embeddings(ids))\n        flags = self.availability[ids]\n        present = torch.cat([torch.ones_like(flags[..., :1]), flags], dim=-1)\n        values = torch.stack([structural, visual, textual], dim=-2)\n        # Zero masked values too: suppress projector bias and absent raw rows.\n        values = values.masked_fill(~present.unsqueeze(-1), 0.0)\n        logits = self.ent_attn(torch.tanh(values)).squeeze(-1)\n        logits = logits.masked_fill(~present, float(\'-inf\'))\n        weights = torch.softmax(logits, dim=-1)\n        return (weights.unsqueeze(-1) * values).sum(dim=-2)\n\n    def forward(self, data):\n        h = self._joint(data[\'batch_h\'])\n        t = self._joint(data[\'batch_t\'])\n        r = self.rel_embeddings(data[\'batch_r\'])\n        return self.margin - self._calc(h, t, r, data[\'mode\'])\n\n    def get_fake_score(self, batch_h, batch_r, batch_t, mode,\n                       fake_hv=None, fake_tv=None, fake_ht=None, fake_tt=None):\n        assert all(x is not None for x in [fake_hv, fake_tv, fake_ht, fake_tt])\n        h, t = self._joint(batch_h), self._joint(batch_t)\n        hf = self._joint(batch_h, fake_hv, fake_ht)\n        tf = self._joint(batch_t, fake_tv, fake_tt)\n        r = self.rel_embeddings(batch_r)\n        scores = [self.margin-self._calc(hf,t,r,mode),\n                  self.margin-self._calc(h,tf,r,mode),\n                  self.margin-self._calc(hf,tf,r,mode)]\n        real = [self.img_proj(self.img_embeddings(batch_h)),\n                self.img_proj(self.img_embeddings(batch_t)),\n                self.text_proj(self.text_embeddings(batch_h)),\n                self.text_proj(self.text_embeddings(batch_t))]\n        return scores, real'+'\n'
# Full notebook copy is the authoritative definition of scoring/evaluation.
MODEL_CODE_ID='adamf_workflow_04a_v3_early_stopping_1e5_v1'
source_digest=hashlib.sha256()
for p in sorted((REPO/'mmkgc').rglob('*.py')):
    source_digest.update(p.relative_to(REPO).as_posix().encode()); source_digest.update(p.read_bytes())
IDENTITY=dict(architecture=ARCHITECTURE,code_version=MODEL_CODE_ID,scorer_sha256=hashlib.sha256(SCORER_SOURCE.encode()).hexdigest(),evaluator_sha256=hashlib.sha256(EVALUATOR_SOURCE.encode()).hexdigest(),source_commit=COMMIT,source_sha256=source_digest.hexdigest(),cache_key=cache_key,protocol=PROTOCOL,text_encoder=TEXT_META,train_text_sha256=sha256(CACHE/'train_clip_text.pt'),val_text_sha256=sha256(CACHE/'val_clip_text.pt'),train_visual_logical_sha256=hashlib.sha256(train_visual.numpy().tobytes()).hexdigest(),textual_sha256=BRIDGE['artifacts']['entity_textual']['sha256'])
for name,payload in [('config.json',CFG),('identity.json',dict(provenance={'commit':COMMIT},**IDENTITY)),('evaluation_protocol.json',PROTOCOL)]:
    path=RUN/name
    if path.exists(): assert json.loads(path.read_text())==payload, f'Run mismatch: {name}. Không ghi đè run; dùng RUN_NAME mới.'
    else: write_json(payload,path)
write_json(PROVENANCE,RUN/'provenance.json')
(RUN/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
write_json(TEXT_META,RUN/'text_encoder.json')
(RUN/'model_definitions.py').write_text(SCORER_SOURCE,encoding='utf-8')
(RUN/'evaluation_definitions.py').write_text(EVALUATOR_SOURCE,encoding='utf-8')

def train_one_seed(seed):
    folder=RUN/f'seed_{seed}'; folder.mkdir(parents=True,exist_ok=True)
    last=folder/'last_checkpoint.pt'; previous=folder/'previous_checkpoint.pt'
    model=make_model(seed)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    rng=np.random.default_rng(seed+10000)
    epoch0=0; next_start=0; order=None; loss_sum=0.; correct=0; seen=0; history=[]; best_mrr=-1.
    early_best_mrr=float('-inf'); bad_epochs=0
    if not last.exists() and previous.exists(): shutil.copy2(previous,last)
    if last.exists():
        assert RESUME_TRAINING, 'Đặt RESUME_TRAINING=True để tiếp tục run cũ.'
        state=torch.load(last,map_location='cpu',weights_only=False)
        assert state['identity']==IDENTITY and state['cfg']==CFG and state['seed']==seed
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        epoch0=state['epoch']; next_start=state['next_start']; order=state['order']
        loss_sum=state['loss_sum']; correct=state['correct']; seen=state['seen']; history=state['history']; best_mrr=state['best_mrr']
        early_best_mrr=state['early_best_mrr']; bad_epochs=state['bad_epochs']
        rng.bit_generator.state=state['rng']; random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        print(f'RESUME epoch={epoch0+1}, next query={next_start:,}')
    def save(epoch,start,current_order):
        payload=dict(identity=IDENTITY,cfg=CFG,seed=seed,epoch=epoch,next_start=start,order=current_order,
            model=model.state_dict(),optimizer=optimizer.state_dict(),loss_sum=loss_sum,correct=correct,seen=seen,
            history=history,best_mrr=best_mrr,early_best_mrr=early_best_mrr,bad_epochs=bad_epochs,rng=rng.bit_generator.state,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        temporary=folder/'checkpoint_pending.pt'; atomic_torch_save(payload,temporary)
        if last.exists(): os.replace(last,previous)
        os.replace(temporary,last)
    def finish(stopped_early):
        write_json(dict(epochs=len(history),max_epochs=CFG['epochs'],seed=seed,
            best_mrr=best_mrr,stopped_early=stopped_early,
            stop_reason='early_stopping' if stopped_early else 'max_epochs',
            early_stopping_patience=CFG['early_stopping_patience'],
            early_stopping_min_delta=CFG['early_stopping_min_delta'],
            bad_epochs=bad_epochs),folder/'complete.json')
    if bad_epochs>=CFG['early_stopping_patience'] or epoch0>=CFG['epochs']:
        finish(bad_epochs>=CFG['early_stopping_patience'])
        print('Training already completed:',folder); return history
    deadline=time.time()+SESSION_TRAIN_HOURS*3600
    last_save=time.monotonic()
    for epoch in range(epoch0,CFG['epochs']):
        if order is None:
            order=rng.permutation(len(train_q)); next_start=0; loss_sum=0.; correct=0; seen=0
        model.train()
        for start in tqdm(range(next_start,len(order),CFG['query_batch_size']),desc=f'Seed {seed} epoch {epoch+1}/{CFG["epochs"]}'):
            selected=order[start:start+CFG['query_batch_size']]; rows=train_q.iloc[selected]
            batch_images=[]
            for i,q in zip(selected,rows.itertuples(index=False)):
                positives=positive_rows['train'][q.query_id]
                pos=int(rng.choice(positives))
                pool=TRAIN_CACHE['negative_rows'][int(i)].numpy()
                assert not np.isin(pool,positives).any()
                neg=rng.choice(pool,CFG['train_negatives'],replace=False)
                batch_images.append(np.concatenate([[pos],neg]))
            imgs=torch.from_numpy(np.stack(batch_images)).to(DEVICE)
            logits=model.score_images(rows,CLIP_QUERY['train'][torch.as_tensor(selected)].to(DEVICE),imgs)*model.logit_scale.exp().clamp(max=100)
            loss=F.cross_entropy(logits,torch.zeros(len(rows),dtype=torch.long,device=DEVICE))
            assert bool(torch.isfinite(loss)), f'Nonfinite loss epoch {epoch+1}'
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad_norm=nn.utils.clip_grad_norm_(model.parameters(),1.0)
            assert bool(torch.isfinite(grad_norm)), 'Nonfinite gradient'
            optimizer.step()
            loss_sum+=float(loss.detach())*len(rows); correct+=int((logits.argmax(1)==0).sum()); seen+=len(rows)
            next_start=start+len(rows)
            timed=time.monotonic()-last_save>=CHECKPOINT_MINUTES*60
            if (math.ceil(next_start/CFG['query_batch_size'])%CFG['checkpoint_every_batches']==0) or timed or time.time()>=deadline:
                save(epoch,next_start,order); last_save=time.monotonic()
            if time.time()>=deadline:
                print('SESSION PAUSED. Đặt RESUME_TRAINING=True và chạy lại.'); return history
        save(epoch,len(order),order)
        alpha,all_metrics=evaluate_validation(model)
        clip=all_metrics['1.0']; hybrid=all_metrics[str(alpha)]
        val_mrr=hybrid['combined']['MRR']
        record=dict(epoch=epoch+1,train_loss=loss_sum/max(seen,1),train_top1=correct/max(seen,1),alpha_clip=alpha,alpha_kg=1-alpha,clip_only=clip,clip_plus_graph=hybrid,
            improvement={k:hybrid['combined'][k]-clip['combined'][k] for k in ['Recall@1','Recall@5','Recall@10','Recall@50','MRR']})
        history.append(record); print(json.dumps(record,indent=2))
        write_json(all_metrics,folder/f'validation_epoch_{epoch+1:03}.json')
        if val_mrr>best_mrr:
            best_mrr=val_mrr
            atomic_torch_save(dict(model=model.state_dict(),identity=IDENTITY,cfg=CFG,seed=seed,epoch=epoch+1,alpha_clip=alpha,alpha_kg=1-alpha,val_metrics=all_metrics),folder/'best_adamf.pt')
        # Use a separate reference for significant improvement; keep the absolute
        # highest-MRR checkpoint above even when its gain is below min_delta.
        if val_mrr>early_best_mrr+CFG['early_stopping_min_delta']:
            early_best_mrr=val_mrr; bad_epochs=0
        else:
            bad_epochs+=1
        record.update(early_stopping_best_mrr=early_best_mrr,
                      early_stopping_bad_epochs=bad_epochs)
        print(f'Early stopping: {bad_epochs}/{CFG["early_stopping_patience"]}; min_delta={CFG["early_stopping_min_delta"]}')
        write_json(history,folder/'history.json')
        order=None; next_start=0; loss_sum=0.; correct=0; seen=0
        save(epoch+1,0,None); last_save=time.monotonic()
        if bad_epochs>=CFG['early_stopping_patience']:
            finish(True)
            print(f'EARLY STOP at epoch {epoch+1}; best validation MRR={best_mrr:.8f}')
            return history
        if time.time()>=deadline:
            print('SESSION PAUSED sau validation.'); return history
    finish(False)
    return history

run_histories={}
for seed in SEEDS: run_histories[str(seed)]=train_one_seed(seed)
print('RUN:',RUN)
