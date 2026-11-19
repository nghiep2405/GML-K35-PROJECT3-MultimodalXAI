from multimodal_xai.visual_genome.guard import sha256
if RUN_TEST_EVALUATION:
    assert BEST.is_file(), f'Train the NEW v2 GAT model before final held-out test. Missing: {BEST}'
    checkpoint=torch.load(BEST,map_location='cpu',weights_only=False)
    frozen_alpha=float(checkpoint['alpha'])
    ALPHA_GRID=[frozen_alpha]
    GUARD=WriteGuard(VG_ROOT,OUTPUT_ROOT)
    reports = OUTPUT_ROOT / 'reports/final_model'
    if OUTPUT_ROOT != EXP_ROOT:
        reports = reports / RUN_NAME
    final_dir=GUARD.mkdir(reports)
    # One-time test evaluation. Do not rerun with alternate alpha or hyperparameters.
    final_report_path=final_dir/f'seed_{SEED}_test.json'
    if final_report_path.exists():
        raise FileExistsError(f'Existing final test result: {final_report_path}')
    model.edge_mode = os.environ.get("GRAPHGAT_EVAL_EDGE_MODE", "real_edges")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    _,result=evaluate_frozen_test(model,523347,SEED)
    score=result[str(frozen_alpha)]
    output={'status':'HELD_OUT_TEST','seed':SEED,'selected_alpha_from_validation':frozen_alpha,
            'source_checkpoint_sha256':sha256(BEST),'test_metrics':score,
            'split_version':'split_v2','test_tuning_permitted':False}
    GUARD.save_json(final_report_path,output)
    print(json.dumps(output,indent=2))
else:
    print('Held-out test not run. Set RUN_TEST_EVALUATION=True only after freezing model and alpha.')
