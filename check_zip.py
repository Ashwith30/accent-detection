from zipfile import ZipFile

zip_path = r"C:\Users\ashwi\.cache\huggingface\hub\datasets--DarshanaS--IndicAccentDb\snapshots\d9e9a08f352843aa4bee542e408e4067893ead85\IndicAccentDB.zip"

with ZipFile(zip_path, "r") as zf:
    all_files = [f for f in zf.namelist() if f.endswith(".wav")]
    folders = sorted(set(f.split("/")[0] for f in all_files if "/" in f))
    print("Top-level folders found in ZIP:")
    for f in folders:
        print(" -", f)
    print(f"\nTotal audio (.wav) files found: {len(all_files)}")
