import matplotlib.pyplot as plt
for seed in SEEDS:
    folder=RUN/f'seed_{seed}'
    path=folder/'history.json'
    if not path.is_file():
        print('Chưa có epoch validation hoàn tất:',folder); continue
    history=json.loads(path.read_text())
    if not history: continue
    best=max(history,key=lambda r:r['clip_plus_graph']['combined']['MRR'])
    rows=[]
    for method in ['clip_only','clip_plus_graph']:
        for group,metrics in best[method].items(): rows.append(dict(method=method,group=group,seed=seed,epoch=best['epoch'],alpha_clip=1.0 if method=='clip_only' else best['alpha_clip'],**metrics))
    frame=pd.DataFrame(rows); display(frame); frame.to_csv(folder/'best_validation_metrics.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['clip_plus_graph']['combined']['MRR'] for r in history],'o-',label='CLIP + AdaMF')
    axes[0].axhline(history[0]['clip_only']['combined']['MRR'],linestyle='--',color='gray',label='CLIP only')
    axes[0].set_ylabel('Validation MRR'); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(epochs,[r['train_loss'] for r in history],color='orange'); axes[1].set(xlabel='Epoch',ylabel='Train loss'); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(folder/'training_validation.png',dpi=150); plt.show(); plt.close(fig)
print('Checkpoint tốt nhất chọn validation, không phải checkpoint epoch cuối.')
