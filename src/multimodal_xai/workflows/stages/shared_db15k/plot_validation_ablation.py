import matplotlib.pyplot as plt
if history:
    best=max(history,key=lambda x:x['hybrid']['combined']['MRR'])
    rows=[]
    for method in ['clip','hybrid']:
        for group,metrics in best[method].items():
            rows.append(dict(ablation=ABLATION,seed=CFG['seed'],method=method,group=group,epoch=best['epoch'],alpha_clip=1. if method=='clip' else best['alpha_clip'],**metrics))
    frame=pd.DataFrame(rows);display(frame);frame.to_csv(RESULTS_DIR/'best_validation_metrics.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].plot([r['epoch'] for r in history],[r['kgc_loss'] for r in history]);ax[0].set(xlabel='Epoch',ylabel='KGC loss')
    ax[1].plot([r['epoch'] for r in history],[r['hybrid']['combined']['MRR'] for r in history],label='CLIP + AdaMF')
    ax[1].axhline(history[0]['clip']['combined']['MRR'],linestyle='--',label='CLIP')
    ax[1].set(xlabel='Epoch',ylabel='Validation retrieval MRR');ax[1].legend()
    fig.tight_layout();fig.savefig(RESULTS_DIR/'training_validation.png',dpi=150);plt.show()
else:print('Chưa hoàn thành epoch validation; checkpoint giữ tiến độ train.')
print('OUTPUT:',RUN)
print('Direct = CLIP fallback; indirect = triple rerank. Chưa đánh giá official test.')
