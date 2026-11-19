# Giao diện Web Demo (Gradio): hàng nhập truy vấn ở trên, 2 tab Graph | Ranking bên dưới.
try:
    import gradio as gr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True); import gradio as gr

_d0 = DATA[DATASET_NAMES[0]]
with gr.Blocks(title="Project 3: Multimodal Graph Retrieval with XAI Demo — Level 2") as demo:
    gr.Markdown("# Project 3: Multimodal Graph Retrieval with XAI Demo\nHệ thống truy vấn văn bản - hình ảnh có giải thích bằng đồ thị (AdaMF + XAI).")
    with gr.Row(equal_height=True):
        ds_in = gr.Radio(DATASET_NAMES, value=DATASET_NAMES[0], label="Tập dữ liệu", scale=1)
        mode_in = gr.Radio(["Gián tiếp (anchor + quan hệ)", "Trực tiếp (câu tự do)"], value="Gián tiếp (anchor + quan hệ)", label="Loại truy vấn", scale=2)
    with gr.Row(equal_height=True):
        anchor_in = gr.Dropdown(_d0.ent_choice, value=_d0.ent_choice[int(_q0.anchor_entity_index)], label="Thực thể anchor", filterable=True, scale=3)
        rel_in = gr.Dropdown(_d0.rel_choice, value=_d0.rel_choice[int(_q0.relation_index)], label="Quan hệ", filterable=True, scale=3)
        dir_in = gr.Radio(["tail: (anchor, quan hệ, ?)", "head: (?, quan hệ, anchor)"],
                          value="tail: (anchor, quan hệ, ?)" if _q0.prediction_direction == "tail" else "head: (?, quan hệ, anchor)", label="Chiều", scale=2)
        text_in = gr.Textbox(label="Nhập câu truy vấn (Text Query)", placeholder="An image of ...", visible=False, scale=6)
        btn = gr.Button("Tìm kiếm", variant="primary", scale=1)
    out = gr.HTML()
    def _on_ds(name):
        D = DATA[name]
        return gr.update(choices=D.ent_choice, value=D.ent_choice[0]), gr.update(choices=D.rel_choice, value=D.rel_choice[0])
    def _on_mode(mode):
        ind = mode.startswith("Gián tiếp")
        return gr.update(visible=ind), gr.update(visible=ind), gr.update(visible=ind), gr.update(visible=not ind)
    ds_in.change(_on_ds, ds_in, [anchor_in, rel_in])
    mode_in.change(_on_mode, mode_in, [anchor_in, rel_in, dir_in, text_in])
    inputs = [ds_in, mode_in, anchor_in, rel_in, dir_in, text_in]
    btn.click(run_demo, inputs, out); text_in.submit(run_demo, inputs, out); demo.load(run_demo, inputs, out)

if IN_COLAB:
    demo.launch(inline=False, share=False)
