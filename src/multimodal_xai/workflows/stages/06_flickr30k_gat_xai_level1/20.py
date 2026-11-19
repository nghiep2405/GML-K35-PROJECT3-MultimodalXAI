def ensure_test_images():
    global IMAGE_DIR
    if IMAGE_DIR and Path(IMAGE_DIR).exists():
        return IMAGE_DIR
    target = WORK / "flickr30k_test_images"
    need = {f"{i}.jpg" for i in test["image_ids"]}
    if target.exists() and need <= {p.name for p in target.glob("*.jpg")}:
        IMAGE_DIR = str(target); return IMAGE_DIR
    if not DOWNLOAD_TEST_IMAGES:
        print("Không có ảnh gốc → hình sẽ vẽ bố cục hộp vùng. Điền IMAGE_DIR để hiện ảnh thật.")
        return None
    try:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
            from huggingface_hub import HfApi, hf_hub_download
        repo = "nlphuji/flickr30k"
        zips = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".zip") and "image" in f]
        assert zips, "không thấy file zip ảnh trong repo"
        path = hf_hub_download(repo, zips[0], repo_type="dataset", cache_dir=str(WORK / "hf_cache"))
        import zipfile
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            for m in z.namelist():
                if Path(m).name in need:
                    (target / Path(m).name).write_bytes(z.read(m))
        # Kiểm tra đúng ảnh gốc: kích thước phải khớp width/height trong annotation Flickr30k Entities của nhóm.
        from PIL import Image
        bad = []
        for iid, meta in zip(test["image_ids"], test["graph_metadata"]):
            f = target / f"{iid}.jpg"
            if f.exists():
                with Image.open(f) as im:
                    if im.size != (meta["width"], meta["height"]):
                        bad.append(iid); f.unlink()
        got = len(need & {p.name for p in target.glob("*.jpg")})
        print(f"Đã giải nén {got}/{len(need)} ảnh test vào {target}; "
              f"{len(bad)} ảnh lệch kích thước so với annotation (đã loại, sẽ vẽ bố cục hộp).")
        if got:
            IMAGE_DIR = str(target)
        return IMAGE_DIR
    except Exception as err:
        print("Không tải được ảnh gốc (", repr(err), ") → hình sẽ vẽ bố cục hộp vùng.")
        return None

ensure_test_images()
