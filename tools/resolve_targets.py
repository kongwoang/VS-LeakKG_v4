"""Bước 0 (cuối) — target -> UniProt, hai đường độc lập, phải khớp nhau.

Đường A (tên):    DUD-E: mã target = UniProt entry name (gcr -> GCR_HUMAN).
                  LIT-PCBA: gene symbol -> UniProt.
Đường B (chuỗi):  rút chuỗi amino acid từ file cấu trúc rồi dóng bằng MMseqs2
                  vào thư viện UniProt. Miễn nhiễm nhầm lẫn protein anh em.
                  DUD-E: receptor.pdb tải từ dude.docking.org
                  DEKOIS: *_protein.pdb có sẵn trong data/raw
LIT-PCBA cũng có đường PDB->RCSB->UniProt (đọc MỌI polymer entity).

Chỉ chấp nhận khi hai đường khớp. Lệch hoặc thiếu -> báo ra cho người duyệt.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

RAW = Path("data/raw")
DB = "data/raw/ChEMBL/extracted/chembl_35/chembl_35_sqlite/chembl_35.db"
OUT = Path("data/processed/target_uniprot_map.parquet")
REPORT = Path("outputs/reports/target_uniprot_resolution.md")

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O", "MSE": "M"}
tmp = Path(tempfile.mkdtemp())


def uniprot(query: str, fields="accession,id", size=100):
    url = ("https://rest.uniprot.org/uniprotkb/search?query="
           + urllib.parse.quote(query) + f"&fields={fields}&size={size}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read()).get("results", [])
    except Exception as ex:
        print(f"    ! {query}: {ex}")
        return []


def fetch(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            return r.read().decode(errors="ignore")
    except Exception:
        return None


def ca_seq(pdb_text: str) -> str:
    chains: dict[str, dict[int, str]] = {}
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            res, ch = line[17:20].strip().upper(), line[21]
            if res not in AA3:
                continue
            try:
                num = int(line[22:26])
            except ValueError:
                continue
            chains.setdefault(ch, {})[num] = AA3[res]
    if not chains:
        return ""
    best = max(chains.values(), key=len)
    return "".join(best[k] for k in sorted(best))


def mmseqs(query_fa: Path, lib_fa: Path) -> pl.DataFrame:
    out = tmp / f"hits_{query_fa.stem}.tsv"
    subprocess.run(
        ["mmseqs", "easy-search", str(query_fa), str(lib_fa), str(out),
         str(tmp / f"mm_{query_fa.stem}"),
         "--format-output", "query,target,fident,alnlen,qlen,bits",
         "-s", "7.5", "--max-seqs", "50", "-v", "1"],
        check=True, capture_output=True, text=True)
    if not out.exists() or out.stat().st_size == 0:
        return pl.DataFrame({"target": [], "acc": [], "fident": [], "cov": []})
    h = pl.read_csv(out, separator="\t", has_header=False,
                    new_columns=["target", "acc", "fident", "alnlen", "qlen", "bits"])
    return (h.with_columns((pl.col("alnlen") / pl.col("qlen")).alias("cov"))
             .sort("bits", descending=True).group_by("target").first())


# ---------------------------------------------------------------- name routes
print("=== Đường A: tên ===")
dude_codes = sorted(d.name for d in (RAW / "DUD-E").iterdir()
                    if d.is_dir() and (d / "actives_final.ism").is_file())
name_acc: dict[str, str] = {}
for i in range(0, len(dude_codes), 25):
    q = " OR ".join(f"id:{c.upper()}_HUMAN" for c in dude_codes[i:i + 25])
    for r in uniprot(q):
        name_acc[r["uniProtkbId"].split("_")[0].lower()] = r["primaryAccession"]
print(f"  DUD-E entry name: {len(name_acc)}/{len(dude_codes)}")

lit_dirs = sorted(d for d in (RAW / "LIT-PCBA/extracted").iterdir() if d.is_dir())
lit_gene: dict[str, str | None] = {}
for d in lit_dirs:
    g = {"MTORC1": "MTOR"}.get(d.name.split("_")[0], d.name.split("_")[0])
    hits = uniprot(f"gene_exact:{g} AND organism_id:9606 AND reviewed:true", size=3)
    lit_gene[d.name] = hits[0]["primaryAccession"] if hits else None
print(f"  LIT-PCBA gene: {sum(v is not None for v in lit_gene.values())}/{len(lit_dirs)}")

# ---------------------------------------------------------------- sequence library
con = sqlite3.connect(DB)
lib = tmp / "uniprot_lib.fasta"
have = set()
with lib.open("w") as fh:
    for acc, seq in con.execute(
            "select accession, sequence from component_sequences "
            "where sequence is not null and accession is not null"):
        fh.write(f">{acc}\n{seq}\n")
        have.add(acc)
    # bổ sung các accession từ đường tên mà ChEMBL chưa có
    want = ({a for a in name_acc.values()} | {a for a in lit_gene.values() if a}) - have
    if want:
        txt = fetch("https://rest.uniprot.org/uniprotkb/stream?format=fasta&query="
                    + urllib.parse.quote(" OR ".join(f"accession:{a}" for a in want)))
        if txt:
            fh.write(txt if txt.endswith("\n") else txt + "\n")
    # bổ sung protein virus (HIV pol, influenza NA) để giải 4 target còn lại
    txt = fetch("https://rest.uniprot.org/uniprotkb/stream?format=fasta&query="
                + urllib.parse.quote(
                    "(taxonomy_id:11676 OR taxonomy_id:11320) AND reviewed:true"))
    if txt:
        fh.write(txt if txt.endswith("\n") else txt + "\n")
print(f"  thư viện UniProt: {len(have):,} từ ChEMBL + bổ sung")

# fasta header của UniProt là sp|ACC|NAME -> chuẩn hoá về ACC
norm = tmp / "lib.fasta"
with lib.open() as i, norm.open("w") as o:
    for line in i:
        if line.startswith(">"):
            h = line[1:].strip()
            acc = h.split("|")[1] if h.count("|") >= 2 else h.split()[0]
            o.write(f">{acc}\n")
        else:
            o.write(line)

# ---------------------------------------------------------------- seq route: DUD-E
print("\n=== Đường B: chuỗi ===")
dude_fa = tmp / "dude.fasta"


def grab(code: str):
    """receptor.pdb của DUD-E — dùng bản đã lưu, chỉ tải khi chưa có, và LƯU LẠI.

    Trước đây hàm này tải từ `dude.docking.org` mỗi lần chạy và **không lưu gì cả**.
    Nghĩa là toàn bộ trục protein — 102 target DUD-E, và qua đó `protein_id_map`, các
    cụm MMseqs, `protein_exact` — treo vào việc một website bên ngoài còn sống. Website
    đó chết, hoặc đổi đường dẫn, hoặc chặn IP của VUW: KG không dựng lại được, và không
    có thông báo nào ngoài "tải + rút chuỗi được 0/102 receptor".

    Với một bài báo thì đó không phải rủi ro vận hành, đó là lỗ hổng tái lập. Lưu file
    xuống `data/raw/DUD-E/<code>/receptor.pdb` — cùng chỗ với `actives_final.ism` —
    để lần chạy sau không cần mạng.
    """
    local = RAW / "DUD-E" / code / "receptor.pdb"
    if local.is_file() and local.stat().st_size > 0:
        return code, ca_seq(local.read_text(errors="ignore"))
    t = fetch(f"https://dude.docking.org/targets/{code}/receptor.pdb")
    if t and "ATOM" in t:
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(t)
        except OSError as ex:
            print(f"    ! không lưu được {local}: {ex}")
    return code, (ca_seq(t) if t else "")


with ThreadPoolExecutor(8) as ex:
    dude_seqs = dict(ex.map(grab, dude_codes))
got = {k: v for k, v in dude_seqs.items() if len(v) > 30}
with dude_fa.open("w") as fh:
    for k, v in got.items():
        fh.write(f">{k}\n{v}\n")
print(f"  DUD-E: tải + rút chuỗi được {len(got)}/{len(dude_codes)} receptor")
dude_hits = mmseqs(dude_fa, norm) if got else pl.DataFrame()

dk_root = RAW / "DEKOIS/extracted/DEKOIS2"
dk_fa = tmp / "dekois.fasta"
dk_targets = []
with dk_fa.open("w") as fh:
    for d in sorted(dk_root.iterdir()):
        p = next(iter(d.glob("protein/*_protein.pdb")), None)
        if not p:
            continue
        s = ca_seq(p.read_text(errors="ignore"))
        if len(s) > 30:
            fh.write(f">{d.name}\n{s}\n")
            dk_targets.append(d.name)
print(f"  DEKOIS: rút chuỗi {len(dk_targets)}/81")
dk_hits = mmseqs(dk_fa, norm)


def hit_of(df: pl.DataFrame, t: str):
    if df.is_empty():
        return None, 0.0, 0.0
    r = df.filter(pl.col("target") == t)
    if r.is_empty():
        return None, 0.0, 0.0
    return r["acc"][0], float(r["fident"][0]), float(r["cov"][0])


# ---------------------------------------------------------------- merge
rows = []
print("\n=== Hợp nhất hai đường ===")
for c in dude_codes:
    a = name_acc.get(c)
    b, ident, cov = hit_of(dude_hits, c)
    seq_ok = ident >= 0.80 and cov >= 0.5
    if a and b and a == b and seq_ok:
        rows.append(dict(corpus="DUD-E", target=c, uniprot=a, confidence=1.0,
                         method="entry_name+sequence", evidence=f"ident={ident:.2f}"))
    elif a and (not b or not seq_ok):
        rows.append(dict(corpus="DUD-E", target=c, uniprot=a, confidence=0.85,
                         method="entry_name_only",
                         evidence=f"seq hit={b} ident={ident:.2f} cov={cov:.2f}"))
    elif a and b and a != b:
        # chuỗi khớp gần như tuyệt đối nhưng khác accession => cấu trúc dùng ortholog
        # (vd DUD-E dùng trypsin bò). Đường tên vẫn là target thật.
        orth = ident >= 0.85
        rows.append(dict(corpus="DUD-E", target=c, uniprot=a,
                         confidence=0.95 if orth else 0.5,
                         method="entry_name(ortholog_structure)" if orth else "CONFLICT",
                         evidence=f"tên={a} chuỗi={b} ident={ident:.2f}"))
    elif b and seq_ok:
        rows.append(dict(corpus="DUD-E", target=c, uniprot=b, confidence=0.9,
                         method="sequence_only", evidence=f"ident={ident:.2f} cov={cov:.2f}"))
    else:
        rows.append(dict(corpus="DUD-E", target=c, uniprot=None, confidence=0.0,
                         method="UNRESOLVED", evidence=f"seq={b} ident={ident:.2f}"))

DK_GENE = {"vegfr1": "FLT1", "vegfr2": "KDR", "pim-2": "PIM2", "er-beta": "ESR2",
           "hiv1pr": None, "tpa": "PLAT", "upa": "PLAU", "pi3kg": "PIK3CG",
           "11betahsd1": "HSD11B1", "p38-alpha": "MAPK14", "rock-1": "ROCK1",
           "parp-1": "PARP1", "hmgr": "HMGCR", "dhfr": "DHFR", "ts": "TYMS",
           "gr": "NR3C1", "pr": "PGR", "thrombin": "F2", "fxa": "F10"}
for t in dk_targets:
    b, ident, cov = hit_of(dk_hits, t)
    ok = b and ident >= 0.85 and cov >= 0.5
    if ok:
        rows.append(dict(corpus="DEKOIS", target=t, uniprot=b, confidence=round(ident, 3),
                         method="sequence", evidence=f"ident={ident:.2f} cov={cov:.2f}"))
        continue
    g = DK_GENE.get(t, t.replace("-", "").upper())
    hits = uniprot(f"gene_exact:{g} AND organism_id:9606 AND reviewed:true", size=2) if g else []
    if hits:
        acc = hits[0]["primaryAccession"]
        rows.append(dict(corpus="DEKOIS", target=t, uniprot=acc, confidence=0.85,
                         method="gene_name(chuỗi yếu)",
                         evidence=f"gene={g} | seq={b} ident={ident:.2f}"))
    else:
        rows.append(dict(corpus="DEKOIS", target=t, uniprot=None, confidence=0.0,
                         method="UNRESOLVED", evidence=f"gene={g} seq={b} ident={ident:.2f}"))


def rcsb(pdb: str) -> list[str]:
    """Mọi polymer entity của một cấu trúc — đọc thật, không phải đọc 4 cái đầu.

    Bản cũ chạy `range(1, 5)` và `break` ngay khi một lần fetch hỏng. Hai hệ quả:
    một cấu trúc có 5+ chuỗi thì bị bỏ sót phần đuôi, và một lỗi mạng thoáng qua ở
    entity 1 làm mất TOÀN BỘ cấu trúc đó trong im lặng. Trong khi chính docstring của
    file này viết "đọc MỌI polymer entity" — và đó là lý do LIT-PCBA `MTORC1` từng ra
    FKBP1A: đọc nhầm đối tác đồng kết tinh vì chỉ nhìn entity đầu tiên.

    Hỏi RCSB xem cấu trúc có bao nhiêu entity, rồi đọc hết. Một entity lỗi thì bỏ qua
    entity đó, không bỏ cả cấu trúc.
    """
    accs: list[str] = []
    n_ent = 12  # trần an toàn nếu không hỏi được số thật
    j = fetch(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}")
    if j:
        try:
            ids = (json.loads(j).get("rcsb_entry_container_identifiers", {})
                                .get("polymer_entity_ids") or [])
            if ids:
                n_ent = len(ids)
        except Exception:
            pass
    for ent in range(1, n_ent + 1):
        j = fetch(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{ent}")
        if not j:
            continue          # entity này hỏng — bỏ qua nó, KHÔNG bỏ cả cấu trúc
        try:
            d = json.loads(j)
        except Exception:
            continue
        ids = (d.get("rcsb_polymer_entity_container_identifiers", {})
                .get("reference_sequence_identifiers") or [])
        accs += [i["database_accession"] for i in ids if i.get("database_name") == "UniProt"]
    return accs


for d in lit_dirs:
    pdbs = sorted({p.name.split("_")[0].lower() for p in d.glob("*_protein.mol2")})
    accs = []
    for code in pdbs[:3]:
        accs += rcsb(code)
    g = lit_gene[d.name]
    if g and g in accs:
        rows.append(dict(corpus="LIT-PCBA", target=d.name, uniprot=g, confidence=1.0,
                         method="gene+pdb", evidence=f"pdb={sorted(set(accs))}"))
    elif g:
        rows.append(dict(corpus="LIT-PCBA", target=d.name, uniprot=g, confidence=0.85,
                         method="gene_only", evidence=f"pdb={sorted(set(accs))} (không chứa {g})"))
    else:
        rows.append(dict(corpus="LIT-PCBA", target=d.name, uniprot=accs[0] if accs else None,
                         confidence=0.4 if accs else 0.0, method="pdb_only" if accs else "UNRESOLVED",
                         evidence=f"pdb={sorted(set(accs))}"))

df = pl.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.write_parquet(OUT)
shutil.rmtree(tmp, ignore_errors=True)

exp = {"DUD-E": 102, "DEKOIS": 81, "LIT-PCBA": 15}
print("\n" + "=" * 66)
lines = ["# Target → UniProt resolution\n\n"]
need = df.filter((pl.col("confidence") < 0.85) | pl.col("uniprot").is_null())
for c, e in exp.items():
    s = df.filter(pl.col("corpus") == c)
    r = s.filter(pl.col("uniprot").is_not_null()).height
    hi = s.filter(pl.col("confidence") >= 0.85).height
    print(f"  {c:<10} {r}/{e} giải được, {hi} độ tin cậy cao")
    lines.append(f"- **{c}**: {r}/{e} resolved, {hi} high-confidence\n")
print(f"\n  hai đường KHỚP NHAU: {df.filter(pl.col('method') == 'entry_name+sequence').height} target DUD-E")

if need.height:
    print(f"\n  === {need.height} TARGET CẦN DUYỆT ===")
    lines.append(f"\n## {need.height} targets needing review\n\n")
    for r in need.sort(["corpus", "target"]).iter_rows(named=True):
        line = f"{r['corpus']:<9} {r['target']:<12} -> {r['uniprot']}  [{r['method']}]  {r['evidence']}"
        print("    " + line)
        lines.append(f"- `{line}`\n")

dup = (df.filter(pl.col("uniprot").is_not_null()).group_by("uniprot")
         .agg(pl.col("corpus").n_unique().alias("nc"), pl.col("corpus"), pl.col("target"))
         .filter(pl.col("nc") > 1))
print(f"\n  *** {dup.height} protein ở ≥2 benchmark — rò rỉ xuyên corpus KG đang mù ***")
lines.append(f"\n## {dup.height} proteins shared across benchmarks (cross-corpus leakage)\n\n")
for r in dup.sort("nc", descending=True).iter_rows(named=True):
    s = f"{r['uniprot']} <- " + ", ".join(f"{c}:{t}" for c, t in zip(r["corpus"], r["target"]))
    print("    ", s)
    lines.append(f"- `{s}`\n")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("".join(lines))
print(f"\nghi {OUT} ({df.height} dòng) + {REPORT}")
sys.exit(1 if need.height else 0)
