"""Bước 1 — dựng lại trục protein từ đầu.

Trước: MMseqs2 chỉ gom cụm các protein `protein:<UniProt>` đến từ BindingDB/ChEMBL,
       nên target của DUD-E/DEKOIS/LIT-PCBA (vốn là gene symbol) nằm ngoài mọi cụm
       -> 81,7% example mù trục protein-family.

Sau:   mọi Protein trong KG đều có chuỗi và đều được gom cụm ở 30/50/90%.

Đầu ra:
  data/processed/protein_id_map.parquet      tgt:<Corpus>:<target> -> protein:<acc>
  data/processed/all_proteins.fasta
  data/processed/protein_clusters_{30,50,90}.parquet   (protein_id, cluster_id)

HIV: protease / RT / integrase của DUD-E là ba VÙNG CHỨC NĂNG của cùng một
polyprotein Gag-Pol. Gộp làm một node sẽ báo rò rỉ quá tay (mô hình học protease
không tự biết RT). Nên tách thành node cấp domain, dùng đúng đoạn chuỗi mà UniProt
đã chú giải. DEKOIS:hiv1pr trỏ về cùng node protease -> lộ ra rò rỉ xuyên corpus.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl
import sqlite3

PROC = Path("data/processed")
DB = "data/raw/ChEMBL/extracted/chembl_35/chembl_35_sqlite/chembl_35.db"
HIV_POL = "P0C6F2"          # HIV-1 Gag-Pol polyprotein (chủng HXB2/BRU)

# node domain <- (corpus, target)
HIV_OVERRIDE = {
    ("DUD-E", "hivpr"):   ("protein:HIV1:PR", "Protease"),
    ("DEKOIS", "hiv1pr"): ("protein:HIV1:PR", "Protease"),
    ("DUD-E", "hivrt"):   ("protein:HIV1:RT", "Reverse transcriptase"),
    ("DUD-E", "hivint"):  ("protein:HIV1:IN", "Integrase"),
}


def http(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode(errors="ignore")
    except Exception as ex:
        print(f"  ! {url[:70]}: {ex}")
        return None


# ---------------------------------------------------------------- 1. tập protein
tmap = pl.read_parquet(PROC / "target_uniprot_map.parquet")
raw_nodes = pl.scan_parquet(PROC / "kg_nodes.parquet")

tgt_nodes = (raw_nodes.filter(pl.col("node_id").str.starts_with("tgt:"))
             .select("node_id").collect()["node_id"].to_list())
prot_nodes = (raw_nodes.filter(pl.col("node_id").str.starts_with("protein:"))
              .select("node_id").collect()["node_id"].to_list())
print(f"node tgt:* = {len(tgt_nodes)}   node protein:* = {len(prot_nodes)}")

def _entry_name(acc: str) -> str | None:
    """`AL1A1_HUMAN_4_501_0` -> `AL1A1_HUMAN` (id dựng từ pocket, không phải accession)."""
    parts = acc.split("_")
    # ENTRY_ORG_<start>_<end>[_TM][_<n>] — nhận dạng qua 2 token số ngay sau tên entry
    if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
        return "_".join(parts[:2])
    return None


id_map: dict[str, str] = {}          # tgt:... -> protein:...
domain_of: dict[str, str] = {}       # protein:HIV1:PR -> tên chain trong UniProt

pending_entry: dict[str, list[str]] = {}
for tid in tgt_nodes:
    _, corpus, target = tid.split(":", 2)
    if (corpus, target) in HIV_OVERRIDE:
        new, chain = HIV_OVERRIDE[(corpus, target)]
        id_map[tid] = new
        domain_of[new] = chain
        continue
    row = tmap.filter((pl.col("corpus") == corpus) & (pl.col("target") == target))
    if row.height and row["uniprot"][0]:
        id_map[tid] = f"protein:{row['uniprot'][0]}"
    elif _entry_name(target):
        pending_entry.setdefault(_entry_name(target), []).append(tid)
    elif len(target) >= 6 and target[0].isalpha() and any(c.isdigit() for c in target):
        id_map[tid] = f"protein:{target}"        # BigBind/BayesBind: vốn đã là UniProt
    else:
        print(f"  !! không ánh xạ được: {tid}")

if pending_entry:
    print(f"target dạng pocket-construct (ENTRY_ORG_start_end): {len(pending_entry)} entry name")
    keys = list(pending_entry)
    for i in range(0, len(keys), 40):
        b = keys[i:i + 40]
        js = http("https://rest.uniprot.org/uniprotkb/search?query="
                  + urllib.parse.quote(" OR ".join(f"id:{k}" for k in b))
                  + "&fields=accession,id&size=100")
        if not js:
            continue
        for r in json.loads(js).get("results", []):
            for tid in pending_entry.get(r["uniProtkbId"], []):
                id_map[tid] = f"protein:{r['primaryAccession']}"
    # Entry name có thể đã bị UniProt đổi. KHÔNG tra toàn văn: tìm "GLCM_HUMAN"
    # bằng full-text trả về GLCM1_HUMAN (Q8IVK1, "cell adhesion molecule 1") —
    # một protein hoàn toàn khác. Sai kiểu này không báo lỗi, chỉ làm hỏng kết quả.
    #
    # Chỉ chấp nhận ánh xạ có bằng chứng độc lập:
    STALE_ENTRY_NAME = {
        # GLCM_HUMAN -> GBA1_HUMAN. Bằng chứng: (1) P04062 dài đúng 536 aa, khớp
        # construct GLCM_HUMAN_40_536_0; (2) DUD-E:glcm khớp P04062 qua chuỗi cấu
        # trúc; (3) DEKOIS:gba và LIT-PCBA:GBA đều độc lập giải ra P04062.
        "GLCM_HUMAN": "P04062",
    }
    for en, tids in pending_entry.items():
        if any(t in id_map for t in tids):
            continue
        acc = STALE_ENTRY_NAME.get(en)
        if acc:
            print(f"  entry name cũ (đã kiểm chứng): {en} -> {acc}")
            for t in tids:
                id_map[t] = f"protein:{acc}"
        else:
            print(f"  !! entry name không giải được, KHÔNG đoán: {en} ({len(tids)} node)")
    done = sum(1 for v in pending_entry.values() for t in v if t in id_map)
    print(f"  giải được {done}/{sum(len(v) for v in pending_entry.values())} node")

pl.DataFrame({"old_id": list(id_map), "new_id": list(id_map.values())}) \
  .write_parquet(PROC / "protein_id_map.parquet")
print(f"ánh xạ được {len(id_map)}/{len(tgt_nodes)} node tgt:")

# --- chuẩn hoá node protein: bị hỏng (lỗi loader) ---
#   protein:AL1A1_HUMAN_4_501_0  -> entry name + khoảng residue  -> tra UniProt
#   protein:"C0L093 P09114"      -> hai accession dính nhau      -> lấy cái đầu
def _entry_name(acc: str) -> str | None:
    """`AL1A1_HUMAN_4_501_0` -> `AL1A1_HUMAN` (id dựng từ pocket, không phải accession)."""
    parts = acc.split("_")
    if len(parts) >= 5 and all(x.isdigit() for x in parts[-3:]):
        return "_".join(parts[:2])
    return None

_bad = [n for n in prot_nodes
        if " " in n or _entry_name(n.split(":", 1)[1]) is not None]
if _bad:
    print(f"node protein: hỏng cần chuẩn hoá: {len(_bad)}")
    _names = {}
    for n in _bad:
        acc = n.split(":", 1)[1]
        if " " in acc:
            id_map[n] = f"protein:{acc.split()[0]}"
        else:
            en = _entry_name(acc)
            if en:
                _names.setdefault(en, []).append(n)
    if _names:
        keys = list(_names)
        for i in range(0, len(keys), 40):
            b = keys[i:i + 40]
            js = http("https://rest.uniprot.org/uniprotkb/search?query="
                      + urllib.parse.quote(" OR ".join(f"id:{k}" for k in b))
                      + "&fields=accession,id&size=100")
            if not js:
                continue
            for r in json.loads(js).get("results", []):
                en = r["uniProtkbId"]
                for n in _names.get(en, []):
                    id_map[n] = f"protein:{r['primaryAccession']}"
    print(f"  chuẩn hoá được {sum(1 for n in _bad if n in id_map)}/{len(_bad)}")
    pl.DataFrame({"old_id": list(id_map), "new_id": list(id_map.values())})       .write_parquet(PROC / "protein_id_map.parquet")

prot_nodes = [id_map.get(n, n) for n in prot_nodes]
wanted = sorted({v for v in id_map.values()} | set(prot_nodes))
print(f"tổng protein cần chuỗi: {len(wanted):,}")

# ---------------------------------------------------------------- 2. chuỗi
seqs: dict[str, str] = {}
con = sqlite3.connect(DB)
chembl_seq = dict(con.execute(
    "select accession, sequence from component_sequences "
    "where sequence is not null and accession is not null"))
print(f"chuỗi có sẵn trong ChEMBL: {len(chembl_seq):,}")

plain = [w for w in wanted if not w.startswith("protein:HIV1:")]
for w in plain:
    acc = w.split(":", 1)[1]
    if acc in chembl_seq:
        seqs[w] = chembl_seq[acc]

missing = [w for w in plain if w not in seqs]
print(f"thiếu chuỗi, tải từ UniProt: {len(missing):,}")
for i in range(0, len(missing), 50):
    batch = [w.split(":", 1)[1] for w in missing[i:i + 50]]
    txt = http("https://rest.uniprot.org/uniprotkb/stream?format=fasta&query="
               + urllib.parse.quote(" OR ".join(f"accession:{a}" for a in batch)))
    if not txt:
        continue
    acc = None
    for line in txt.splitlines():
        if line.startswith(">"):
            parts = line[1:].split("|")
            acc = parts[1] if len(parts) >= 2 else line[1:].split()[0]
            seqs.setdefault(f"protein:{acc}", "")
        elif acc:
            seqs[f"protein:{acc}"] += line.strip()
    print(f"  ...{min(i+50, len(missing))}/{len(missing)}", flush=True)
seqs = {k: v for k, v in seqs.items() if v}

# lô nào lỗi -> tải lẻ từng accession (accession chết/obsolete sẽ lộ ra ở đây)
still = [w for w in plain if w not in seqs]
if still:
    print(f"lô lỗi, tải lẻ {len(still)} accession...", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def one(w):
        acc = w.split(":", 1)[1]
        t = http(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta")
        if not t or not t.startswith(">"):
            return w, ""
        return w, "".join(l.strip() for l in t.splitlines()[1:])

    with ThreadPoolExecutor(12) as ex:
        for w, sq in ex.map(one, still):
            if sq:
                seqs[w] = sq
    dead = [w for w in still if w not in seqs]
    print(f"  lấy thêm {len(still)-len(dead)}, chết hẳn {len(dead)}", flush=True)
    if dead[:8]:
        print("  accession chết:", dead[:8])

# HIV: cắt đúng chain mà UniProt chú giải trên polyprotein
js = http(f"https://rest.uniprot.org/uniprotkb/{HIV_POL}.json")
if js:
    d = json.loads(js)
    full = d["sequence"]["value"]
    chains = [(f["description"], f["location"]["start"]["value"], f["location"]["end"]["value"])
              for f in d.get("features", []) if f["type"] == "Chain"]
    print(f"\nchain của {HIV_POL}: {[c[0] for c in chains]}")
    for node, want in {v: k for v, k in
                       ((n, c) for n, c in domain_of.items())}.items():
        hit = next((c for c in chains if want.lower() in c[0].lower()), None)
        if hit:
            seqs[node] = full[hit[1] - 1: hit[2]]
            print(f"  {node:<18} {hit[0][:38]:<40} {hit[1]}-{hit[2]}  ({len(seqs[node])} aa)")
        else:
            print(f"  !! không tìm thấy chain {want!r} trong {HIV_POL}")

no_seq = [w for w in wanted if w not in seqs]
print(f"\ncó chuỗi: {len(seqs):,}/{len(wanted):,}   thiếu: {len(no_seq)}")
if no_seq[:10]:
    print("  thiếu:", no_seq[:10])

fasta = PROC / "all_proteins.fasta"
with fasta.open("w") as fh:
    for k, v in sorted(seqs.items()):
        fh.write(f">{k.split(':', 1)[1]}\n{v}\n")
print(f"ghi {fasta} ({len(seqs):,} chuỗi)")

# ---------------------------------------------------------------- 3. gom cụm
class _UF:
    """Union-find để ép các mức cụm lồng nhau."""

    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


tmp = Path(tempfile.mkdtemp())
raw_clusters: dict[str, pl.DataFrame] = {}
for res, minid in (("30", 0.30), ("50", 0.50), ("90", 0.90)):
    pref = tmp / f"clu{res}"
    subprocess.run(
        ["mmseqs", "easy-cluster", str(fasta), str(pref), str(tmp / f"t{res}"),
         "--min-seq-id", str(minid), "-c", "0.8", "--cov-mode", "0",
         "--cluster-mode", "0", "-v", "1"],
        check=True, capture_output=True, text=True)
    tsv = Path(f"{pref}_cluster.tsv")
    df = pl.read_csv(tsv, separator="\t", has_header=False,
                     new_columns=["cluster_id", "protein_id"])
    raw_clusters[res] = df.select("protein_id", "cluster_id")
    print(f"  {res}%: {df['cluster_id'].n_unique():>5} cụm thô, phủ {df['protein_id'].n_unique():,}")

# MMseqs chạy độc lập từng mức nên KHÔNG đảm bảo lồng nhau: đã thấy 7 cụm 90%
# bị tách ra ở mức 30/50%. Hai protein giống nhau >=90% thì đương nhiên cũng
# giống nhau >=30% — vi phạm điều đó làm split 3 nấc mất tính đơn điệu.
# Ép lồng: cụm ở mức LỎNG = thành phần liên thông của (mức đó ∪ mọi mức CHẶT hơn).
print("\nép các mức lồng nhau (90% ⊆ 50% ⊆ 30%)")
for res, tighter in (("90", []), ("50", ["90"]), ("30", ["50", "90"])):
    uf = _UF()
    for lvl in [res] + tighter:
        for cid, grp in raw_clusters[lvl].group_by("cluster_id"):
            members = grp["protein_id"].to_list()
            for m in members[1:]:
                uf.union(members[0], m)
    df = raw_clusters[res].with_columns(
        pl.col("protein_id").map_elements(uf.find, return_dtype=pl.Utf8).alias("cluster_id"))
    out = PROC / f"protein_clusters_{res}.parquet"
    df.write_parquet(out)
    n_before = raw_clusters[res]["cluster_id"].n_unique()
    print(f"  {res}%: {n_before:>5} -> {df['cluster_id'].n_unique():>5} cụm sau khi ép lồng"
          f"   phủ {df['protein_id'].n_unique():,}/{len(seqs):,}")

# ------------------------------------------------------- 4. protein_exact
# `protein_exact` được schema khai báo (trọng số 1.00) VÀ nằm trong trục protein —
# và KG chưa bao giờ có một cạnh nào. Hệ quả: hai node Protein mang trình tự đồng
# nhất 100% (cùng một protein, chỉ khác accession) không hề được nối, nên đường rò
# rỉ xuyên corpus mạnh nhất bị chấm 0.85 × 0.85 = 0.72 qua cụm 90% thay vì 1.00 —
# và số "protein dùng chung bởi ≥2 corpus" bị đếm thiếu.
#
# Đây KHÔNG phải câu hỏi về loài. Quy tắc thuần cấu trúc: đồng nhất ≥ PIDENT_MIN
# và alignment phủ ≥ COV_MIN của CẢ HAI trình tự. Phủ hai chiều là mấu chốt — nó
# giữ nguyên việc tách miền HIV: miền PR (99 aa) đồng nhất 100% với một ĐOẠN của
# polyprotein P03366 (1.447 aa), nhưng chỉ phủ 7% bên polyprotein nên không gộp.
#
# pident được ghi vào props. Ngưỡng là CHÍNH SÁCH: downstream nâng được, KG thì
# ghi rộng để không đánh mất sự thật.
PIDENT_MIN = 98.0
COV_MIN = 0.90

print("\ndựng protein_exact (cùng một protein theo trình tự)")
pe_tsv = PROC / "_protein_exact_hits.tsv"
with tempfile.TemporaryDirectory() as td:
    subprocess.run(
        ["mmseqs", "easy-search", str(fasta), str(fasta), str(pe_tsv), str(Path(td) / "pe"),
         "-s", "7.5", "--max-seqs", "300", "-e", "1e-3", "--threads", "16", "-v", "1",
         "--format-output", "query,target,pident,alnlen,qlen,tlen"],
        check=True)
hits = pl.read_csv(pe_tsv, separator="\t", has_header=False,
                   new_columns=["a", "b", "pident", "alnlen", "qlen", "tlen"])
pe = (hits.filter(pl.col("a") != pl.col("b"))
        .filter((pl.col("pident") >= PIDENT_MIN)
                & ((pl.col("alnlen") / pl.col("qlen")) >= COV_MIN)
                & ((pl.col("alnlen") / pl.col("tlen")) >= COV_MIN))
        .with_columns(pl.min_horizontal("a", "b").alias("src"),
                      pl.max_horizontal("a", "b").alias("dst"))
        .group_by("src", "dst")
        .agg(pl.col("pident").max().alias("pident"),
             pl.col("alnlen").max().alias("alnlen"))
        .sort(["src", "dst"]))
pe.write_parquet(PROC / "protein_exact.parquet")
print(f"  {pe.height} cặp (ident ≥ {PIDENT_MIN}%, phủ ≥ {COV_MIN:.0%} cả hai chiều)")

print("\n=== CỔNG KIỂM TRA ===")
ok = True

# miền HIV phải KHÔNG bị gộp vào polyprotein — đó là cả lý do chúng được tách
hiv_bad = pe.filter(
    (pl.col("src").str.contains("HIV1:") | pl.col("dst").str.contains("HIV1:"))
    & (pl.col("alnlen") > 700))
print(f"  miền HIV bị gộp vào polyprotein: {hiv_bad.height} (phải = 0)")
ok &= hiv_bad.height == 0
for res in ("30", "50", "90"):
    df = pl.read_parquet(PROC / f"protein_clusters_{res}.parquet")
    miss = len(seqs) - df["protein_id"].n_unique()
    print(f"  {res}%: {'ĐẠT' if miss == 0 else f'THIẾU {miss}'}")
    ok &= (miss == 0)
# mọi target benchmark phải có chuỗi
bench = {v for k, v in id_map.items()
         if k.split(":")[1] in ("DUD-E", "DEKOIS", "LIT-PCBA")}
nb = [b for b in bench if b not in seqs]
print(f"  target benchmark có chuỗi: {len(bench)-len(nb)}/{len(bench)}"
      + (f"  THIẾU: {nb}" if nb else ""))
ok &= not nb
print("\n" + ("TẤT CẢ ĐẠT" if ok else "CÓ CỔNG KHÔNG ĐẠT"))
sys.exit(0 if ok else 1)
