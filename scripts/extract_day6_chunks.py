"""Extract chunks from so_tay_sinh_vien_2026.pdf using exact Day 6 Strategy B logic."""
import json
import re
import unicodedata
from pathlib import Path
import pypdf

def get_page_category(page_num: int) -> str:
    if page_num in [2, 3] or page_num >= 24:
        return "gioi_thieu_vien_khoa"
    elif page_num in [10, 11, 14, 15, 16]:
        return "diem_ren_luyen"
    elif page_num in [17, 21, 22]:
        return "hoc_bong_an_sinh"
    else:
        return "quy_che_quy_dinh"

def normalize_text(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text)
    norm = norm.replace("\u00a0", " ").replace("\ufeff", "")
    norm = norm.replace("\r\n", "\n").replace("\r", "\n")
    norm = re.sub(r"(\w+)\s*-\s*\n\s*(\w+)", r"\1\2", norm)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in norm.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def by_paragraph(page_id: str, source: str, page_num: int, text: str, page_meta: dict, max_size: int = 800):
    if not text:
        return []
    raw_paras = [p.strip() for p in re.split(r"\n{2,}|\n(?=[A-Z0-9\-\*\•IIVX\d]+\.|\bTC\d|\bĐiều\b|\bKhoản\b)", text) if p.strip()]
    if not raw_paras:
        raw_paras = [text]
    normalized_paras = []
    for p in raw_paras:
        if len(p) <= max_size:
            normalized_paras.append(p)
        else:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
            cur_s = ""
            for s in sentences:
                if len(cur_s) + len(s) + 1 <= max_size:
                    cur_s = f"{cur_s} {s}".strip() if cur_s else s
                else:
                    if cur_s:
                        normalized_paras.append(cur_s)
                    cur_s = s
            if cur_s:
                normalized_paras.append(cur_s)
    chunks = []
    current_chunk = ""
    idx = 0
    for p in normalized_paras:
        if not current_chunk:
            current_chunk = p
        elif len(current_chunk) + len(p) + 2 <= max_size:
            current_chunk = f"{current_chunk}\n\n{p}"
        else:
            cid = f"{page_id}_p{page_num}_para_{idx}"
            meta = dict(page_meta)
            meta["chunk_type"] = "by_paragraph"
            chunks.append({
                "chunk_id": cid,
                "document_id": page_id,
                "source": source,
                "page": page_num,
                "text": current_chunk.strip(),
                "metadata": meta
            })
            idx += 1
            current_chunk = p
    if current_chunk.strip():
        cid = f"{page_id}_p{page_num}_para_{idx}"
        meta = dict(page_meta)
        meta["chunk_type"] = "by_paragraph"
        chunks.append({
            "chunk_id": cid,
            "document_id": page_id,
            "source": source,
            "page": page_num,
            "text": current_chunk.strip(),
            "metadata": meta
        })
    return chunks

def main():
    pdf_path = Path("data/so_tay_sinh_vien_2026.pdf")
    doc_id = "hust-student-handbook-2026"
    source = "data/so_tay_sinh_vien_2026.pdf"
    reader = pypdf.PdfReader(str(pdf_path))
    chunks = []
    valid_pages = 0
    for idx, pdf_page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = pdf_page.extract_text() or ""
        text = normalize_text(raw_text)
        if len(text) < 300:
            continue
        valid_pages += 1
        meta = {
            "category": get_page_category(page_num),
            "char_count": len(text),
            "physical_page": page_num
        }
        page_chunks = by_paragraph(doc_id, source, page_num, text, meta, max_size=800)
        chunks.extend(page_chunks)
    print(f"Extracted {valid_pages} valid pages, total {len(chunks)} chunks (expected 184 chunks).")
    out_file = Path("data/day6_chunks_raw.json")
    out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {out_file}")

if __name__ == "__main__":
    main()
