IDENTITY=dict(version='db15k-kgc-retrieval-v1',notebook_code_hash='e4203439ee243779c251de2b11d385525e58e81b217f1bde7caad30fc419a3e3',hashes=HASHES,model_hash=MODEL_HASH,
    evaluation_hash=hashlib.sha256(EVALUATION_SOURCE.encode()).hexdigest(),
    visual_hash=hashlib.sha256(train_visual.numpy().tobytes()).hexdigest(),cache_identity=cache_identity,
    protocol='db15k_clean_from_train_v1',retrieval_protocol='db15k_clean_retrieval_v2')
def compatible_config(old, new):
    differences={k:(old.get(k),new.get(k)) for k in set(old)|set(new)
                 if k!='epochs' and (k not in old or k not in new or old[k]!=new[k])}
    assert not differences, f'Cấu hình khác ngoài epochs: {differences}'
    assert isinstance(new['epochs'],int) and new['epochs']>=old['epochs'], 'Chỉ cho phép giữ nguyên hoặc tăng epochs.'
config_path=RUN/'config.json'
identity_path=RUN/'identity.json'
if config_path.exists(): compatible_config(json.loads(config_path.read_text()),CFG)
if identity_path.exists(): assert json.loads(identity_path.read_text())==IDENTITY, 'Data/model/evaluation identity khác run cũ.'
(RUN/'evaluation_source.py').write_text(EVALUATION_SOURCE,encoding='utf-8')
(RUN/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
val_q[['query_id','split','prediction_direction']].to_csv(RUN/'validation_query_ids.csv',index=False)
last=RUN/'last_checkpoint.pt';previous=RUN/'previous_checkpoint.pt'
SNAPSHOT_EVERY_EPOCHS = 30
snapshots = RUN/'epoch_checkpoints'
snapshots.mkdir(exist_ok=True)

def load_resume_checkpoint():
    candidates=[last, previous, RUN/'pending_checkpoint.pt']
    candidates+=sorted(snapshots.glob('epoch_*.pt'),reverse=True)
    candidates=[p for p in candidates if p.is_file()]
    best_path=None; best_progress=(-1,-1)
    for path in candidates:
        try:
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['identity']==IDENTITY, 'identity mismatch'
            compatible_config(saved['cfg'],CFG)
            required={'model','optimizer','epoch','offset','order','history','loss_sum','seen_count','best_mrr','rng','torch_rng','cuda_rng'}
            assert required<=set(saved), 'incomplete checkpoint'
            assert 0<=saved['epoch']<=CFG['epochs'] and saved['offset']>=0
            # Check that model/optimizer payloads can actually be restored.
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            rng.bit_generator.state=saved['rng']
            torch.set_rng_state(saved['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
            progress=(saved['epoch'],saved['offset'])
            if progress>best_progress:best_path=path;best_progress=progress
            del saved
        except Exception as exc:
            print('Bỏ qua checkpoint không dùng được:',path.name,type(exc).__name__,str(exc)[:180])
    if best_path is not None:
        print('Chọn checkpoint:',best_path,'| tiến độ:',best_progress)
        return torch.load(best_path,map_location='cpu',weights_only=False)
    # Never silently start over if previous training evidence exists.
    evidence=candidates or (RUN/'history.json').exists() or (RUN/'complete.json').exists() or (RUN/'best_adamf.pt').exists()
    assert not evidence, 'Không còn checkpoint resume hợp lệ. Giữ nguyên dữ liệu; kiểm tra backup Drive. Không tự train từ đầu.'
    return None
epoch0=0;offset=0;order=None;loss_sum=0.;seen_count=0;history=[];best_mrr=-1.
state=load_resume_checkpoint()
if state is not None:
    assert state['identity']==IDENTITY, 'Checkpoint identity mismatch'
    compatible_config(state['cfg'],CFG)
    assert CFG['epochs']>=state['epoch'], 'Epoch đích nhỏ hơn tiến độ đã lưu'
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    # Optimizer tensors follow parameter devices after load_state_dict.
    epoch0=state['epoch'];offset=state['offset'];order=state['order'];history=state['history']
    loss_sum=state['loss_sum'];seen_count=state['seen_count'];best_mrr=state['best_mrr']
    rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    print('RESUME:',epoch0,'completed epochs; next triple offset:',offset)
    del state
# Update run metadata only after checkpoint/config compatibility checks succeed.
if config_path.exists():
    old_cfg=json.loads(config_path.read_text())
    if old_cfg!=CFG:
        backup=RUN/f"config_before_extend_{time.time_ns()}.json"
        write_json(old_cfg,backup)
write_json(CFG,config_path)
if not identity_path.exists(): write_json(IDENTITY,identity_path)
if (RUN/'complete.json').exists() and epoch0<CFG['epochs']:
    os.replace(RUN/'complete.json',RUN/f"complete_before_extend_{time.time_ns()}.json")
print('Epoch đích:',CFG['epochs'],'| Đã hoàn thành:',epoch0)
def checkpoint(epoch,offset,order):
    payload=dict(identity=IDENTITY,cfg=CFG,model=model.state_dict(),optimizer=optimizer.state_dict(),
        epoch=epoch,offset=offset,order=order,history=history,loss_sum=loss_sum,seen_count=seen_count,
        best_mrr=best_mrr,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    pending=RUN/'pending_checkpoint.pt';save_tensor(payload,pending)
    if last.exists():os.replace(last,previous)
    os.replace(pending,last)
    # Keep independent full recovery snapshots; never rotate/delete older milestones.
    if order is None and offset==0 and epoch>0 and (epoch%SNAPSHOT_EVERY_EPOCHS==0 or epoch==CFG['epochs']):
        target=snapshots/f'epoch_{epoch:06d}.pt'
        save_tensor(payload,target)
        print('Đã lưu checkpoint mốc:',target)
def run_training():
    global order,offset,loss_sum,seen_count,best_mrr
    deadline=time.time()+CFG['session_hours']*3600
    for epoch in range(epoch0,CFG['epochs']):
        if order is None:order=rng.permutation(len(train_array));offset=0;loss_sum=0.;seen_count=0
        model.train()
        for start in tqdm(range(offset,len(order),CFG['batch_size']),desc=f"KGC epoch {epoch+1}/{CFG['epochs']}"):
            batch=train_array[order[start:start+CFG['batch_size']]]
            neg=sample_negatives(batch,rng);optimizer.zero_grad(set_to_none=True)
            loss=kgc_loss(batch,neg);assert torch.isfinite(loss),'Nonfinite loss'
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Nonfinite gradient'
            optimizer.step();loss_sum+=float(loss.detach())*len(batch);seen_count+=len(batch);offset=start+len(batch)
            if math.ceil(offset/CFG['batch_size'])%CFG['checkpoint_batches']==0 or time.time()>=deadline:checkpoint(epoch,offset,order)
            if time.time()>=deadline:print('PAUSED: chạy lại notebook để resume.');return
        checkpoint(epoch,len(order),order)
        alpha,metrics=evaluate_validation();mrr=metrics[str(alpha)]['combined']['MRR']
        record=dict(epoch=epoch+1,kgc_loss=loss_sum/seen_count,alpha_clip=alpha,
            clip=metrics['1.0'],hybrid=metrics[str(alpha)])
        history.append(record);write_json(metrics,RUN/f'validation_epoch_{epoch+1:03}.json')
        if mrr>best_mrr:
            best_mrr=mrr;save_tensor(dict(model=model.state_dict(),cfg=CFG,identity=IDENTITY,epoch=epoch+1,
                alpha_clip=alpha,metrics=metrics[str(alpha)]),RUN/'best_adamf.pt')
        write_json(history,RUN/'history.json')
        print('epoch',epoch+1,'KGC loss',record['kgc_loss'],'val retrieval MRR',mrr,'alpha',alpha)
        order=None;offset=0;loss_sum=0.;seen_count=0;checkpoint(epoch+1,0,None)
        if time.time()>=deadline and epoch+1<CFG['epochs']:print('PAUSED after validation.');return
    write_json(dict(epochs=CFG['epochs'],best_mrr=best_mrr,seed=CFG['seed']),RUN/'complete.json')
    print(f"COMPLETE: {CFG['epochs']} epochs. Best checkpoint:",RUN/'best_adamf.pt')
run_training()
