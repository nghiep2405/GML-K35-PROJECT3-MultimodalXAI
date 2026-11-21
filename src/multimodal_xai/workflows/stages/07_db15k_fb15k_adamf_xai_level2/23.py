# Giao diện web (Gradio): chọn ca trong báo cáo (chạy ngay) hoặc tự nhập truy vấn; kết quả là trang dùng chung của demo_web.
try:
    import gradio as gr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True); import gradio as gr

def _cases(name):
    return list(DEMO_CASES.get(name, {}).keys())

_d0 = DATA[DATASET_NAMES[0]]; _cs0 = _cases(DATASET_NAMES[0])
_q0 = _d0.test_q.set_index("query_id").loc[DEMO_CASES[DATASET_NAMES[0]][_cs0[0]]] if _cs0 else None
with gr.Blocks(title="Truy vấn đa phương thức có giải thích bằng đồ thị") as demo:
    gr.Markdown("### Cấp độ 2 · CLIP + AdaMF trên DB15K và FB15K-237-IMG")
    with gr.Row(equal_height=True):
        ds_in = gr.Radio(DATASET_NAMES, value=DATASET_NAMES[0], label="Tập dữ liệu", scale=1)
        case_in = gr.Dropdown(_cs0, value=_cs0[0] if _cs0 else None, label="Ca trong báo cáo (5 thành công, 5 thất bại)", scale=5)
        k_in = gr.Slider(3, 10, value=5, step=1, label="Top-k", scale=1)
    with gr.Accordion("Tự nhập truy vấn", open=False):
        with gr.Row(equal_height=True):
            mode_in = gr.Radio(["Gián tiếp (neo + quan hệ)", "Trực tiếp (câu tự do)"], value="Gián tiếp (neo + quan hệ)", label="Loại truy vấn", scale=2)
            anchor_in = gr.Dropdown(_d0.ent_choice, value=_d0.ent_choice[int(_q0.anchor_entity_index)] if _q0 is not None and _q0.prediction_direction != "direct" else _d0.ent_choice[0],
                                    label="Thực thể neo", filterable=True, scale=3)
            rel_in = gr.Dropdown(_d0.rel_choice, value=_d0.rel_choice[int(_q0.relation_index)] if _q0 is not None and _q0.prediction_direction != "direct" else _d0.rel_choice[0],
                                 label="Quan hệ", filterable=True, scale=3)
            dir_in = gr.Radio(["tail: (neo, quan hệ, ?)", "head: (?, quan hệ, neo)"],
                              value="head: (?, quan hệ, neo)" if _q0 is not None and _q0.prediction_direction == "head" else "tail: (neo, quan hệ, ?)",
                              label="Chiều", scale=2)
            text_in = gr.Textbox(label="Câu truy vấn (tiếng Anh)", placeholder="An image of ...", visible=False, scale=6)
            free_btn = gr.Button("Tìm", variant="primary", scale=1)
    out = gr.HTML()

    def _on_ds(name):
        D = DATA[name]; cs = _cases(name)
        return (gr.update(choices=cs, value=cs[0] if cs else None), gr.update(choices=D.ent_choice, value=D.ent_choice[0]),
                gr.update(choices=D.rel_choice, value=D.rel_choice[0]))
    def _on_mode(mode):
        ind = mode.startswith("Gián tiếp")
        return gr.update(visible=ind), gr.update(visible=ind), gr.update(visible=ind), gr.update(visible=not ind)
    ds_in.change(_on_ds, ds_in, [case_in, anchor_in, rel_in])
    mode_in.change(_on_mode, mode_in, [anchor_in, rel_in, dir_in, text_in])
    case_in.change(run_case, [ds_in, case_in, k_in], out); k_in.release(run_case, [ds_in, case_in, k_in], out)
    free_inputs = [ds_in, mode_in, anchor_in, rel_in, dir_in, text_in, k_in]
    free_btn.click(run_free, free_inputs, out); text_in.submit(run_free, free_inputs, out)
    demo.load(run_case, [ds_in, case_in, k_in], out)

if os.environ.get("XAI_DEMO_LAUNCH", "1") == "1":
    W.launch(demo, IN_COLAB)
