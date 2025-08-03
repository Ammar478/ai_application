import os
import pandas as pd
from pathlib import Path

def load_text_documents(folder_path):

	docs =[]
	for fname in os.listdir(folder_path):
		if fname.lower().endswith(".txt"):
			path = os.path.join(folder_path,fname)
			with open(path, encoding="utf-8", errors="ignore") as f:
				content = f.read()
			docs.append({
				"source": fname,
				"content": content.strip()
			})
	return docs

def load_excel_rows(folder_path):
    docs = []
    primary_engines = {
        ".xlsx": "openpyxl",
        ".xlsm": "openpyxl",
        ".xls": "xlrd",
        ".xlsb": "pyxlsb"
    }
    fallback_order = ["openpyxl", "pyxlsb", "xlrd"]

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".xlsx", ".xls", ".xlsm", ".xlsb")):
            continue

        path = Path(folder_path) / fname
        ext = path.suffix.lower()


        engines = []
        primary = primary_engines.get(ext)
        if primary:
            engines.append(primary)
        for eng in fallback_order:
            if eng not in engines:
                engines.append(eng)

        df = None
        last_exc = None
        for eng in engines:
            try:
                df = pd.read_excel(path, engine=eng)
                break
            except Exception as e:
                last_exc = e

        if df is None:
            print(f"[WARN] failed to load {fname}: {last_exc}")
            continue

        for idx, row in df.iterrows():
            row_text = "; ".join(f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]))
            docs.append({
                "source": f"{fname}#row{idx}",
                "content": row_text
            })

    return docs
