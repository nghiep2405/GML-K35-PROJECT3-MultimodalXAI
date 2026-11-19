for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"{p.relative_to(OUT)}  ({p.stat().st_size / 1024:.1f} KB)")
