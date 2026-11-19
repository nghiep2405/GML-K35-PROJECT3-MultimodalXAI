# Xem nhanh: 1 ảnh đầu của 24 thực thể có ảnh đầu tiên
g = kd.groupby("entity_index").head(1).head(24).merge(ents[["entity_index", "label_vi"]], on="entity_index")
display(HTML('<div style="display:flex;flex-wrap:wrap;gap:8px">' + "".join(
    f'<div style="width:150px;font-size:12px"><img src="data:image/jpeg;base64,{thumb_b64(MMKG / r.file, 150)}" style="width:150px;height:110px;object-fit:cover"><br>{html.escape(r.label_vi)}</div>'
    for r in g.itertuples()) + "</div>"))
