# Giao diện Web Demo (Gradio): hàng truy vấn ở trên, bên dưới là 2 tab Graph | Ranking (rộng hết trang).
# Tiêu đề, mô tả, ô truy vấn và launch giữ đúng notebook 05 v2, ô 6.
try:
    import gradio as gr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
    import gradio as gr

with gr.Blocks(title='Project 3: Multimodal Graph Retrieval with XAI Demo') as demo:
    gr.Markdown('# Project 3: Multimodal Graph Retrieval with XAI Demo\n'
                'Hệ thống truy vấn văn bản - hình ảnh có giải thích bằng đồ thị (GAT + XAI).')
    with gr.Row(equal_height=True):
        query_box = gr.Textbox(label='Nhập câu truy vấn (Text Query)', value='a child playing football on grass',
                               placeholder='Nhập bất kỳ câu truy vấn tiếng Anh nào', scale=6)
        search_btn = gr.Button('Tìm kiếm', variant='primary', scale=1)
    result_html = gr.HTML()
    search_btn.click(gradio_demo_inference, inputs=query_box, outputs=result_html)
    query_box.submit(gradio_demo_inference, inputs=query_box, outputs=result_html)
    demo.load(gradio_demo_inference, inputs=query_box, outputs=result_html)

if IN_COLAB:
    demo.launch(inline=False, share=False)
