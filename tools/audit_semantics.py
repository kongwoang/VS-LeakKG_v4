"""Does the KG mean what biology says, and can the proposal's method run on it?

`audit_kg.py` checks invariants. This asks the harder questions:
  A. do the components carry correct biology? (spot-checks against known truth)
  B. is the chemistry sound? (SMILES validity, stereo merges)
  C. can the axes actually be PARTITIONED? (proposal §3.5 leakage groups)
  D. is any axis' coverage confounded with the LABEL? (a trap for contamination scoring)

C is measured across a sweep of degree cut-offs rather than one threshold, because
"how promiscuous is too promiscuous" is a downstream policy — this tool reports the
curve and refuses to choose for you.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from vsleakkg.kg import schema

RDLogger.DisableLog("rdApp.*")

n = pl.scan_parquet("outputs/kg/canonical_nodes.parquet")
e = pl.scan_parquet("outputs/kg/canonical_edges.parquet")
issues: list[str] = []

N = n.filter(pl.col("node_type") == "Example").select(pl.len()).collect().item()
DEG = dict(n.select("node_id", "degree").collect().iter_rows())
print(f"Example: {N:,}\n" + "=" * 78)


def pairs(t: str) -> pl.DataFrame:
    return e.filter(pl.col("edge_type") == t).select("src", "dst").collect()


# ---------------------------------------------------------------- A. biology
print("A. BIOLOGY — target -> UniProt, against known truth")
TRUTH = {
    ("DUD-E", "gcr"): "P04150", ("DUD-E", "esr1"): "P03372", ("DUD-E", "esr2"): "Q92731",
    ("DUD-E", "pgh1"): "P23219", ("DUD-E", "pgh2"): "P35354", ("DUD-E", "try1"): "P07477",
    ("DUD-E", "thrb"): "P00734", ("DUD-E", "aces"): "P22303", ("DUD-E", "ace"): "P12821",
    ("DUD-E", "nos1"): "P29475", ("DUD-E", "akt1"): "P31749", ("DUD-E", "akt2"): "P31751",
    ("DUD-E", "egfr"): "P00533", ("DUD-E", "hs90a"): "P07900",
    ("LIT-PCBA", "ADRB2"): "P07550", ("LIT-PCBA", "MAPK1"): "P28482",
    ("LIT-PCBA", "PPARG"): "P37231", ("LIT-PCBA", "MTORC1"): "P42345",
    ("LIT-PCBA", "TP53"): "P04637", ("LIT-PCBA", "GBA"): "P04062",
}
# Hỏi ĐỒ THỊ, không hỏi file trung gian.
#
# Bản cũ đọc `data/processed/target_uniprot_map.parquet` — tức là nó xác minh một
# artifact trung gian rồi tuyên bố "sinh học đúng". Nhưng KG không được dựng từ file
# đó: nó đi qua `protein_id_map`, qua HIV_OVERRIDE, qua `_normalize_protein_ids`, qua
# `split_example_protein_relation`. Bất kỳ bước nào trong số đó lệch khỏi map là đồ
# thị trỏ vào protein khác — và bài kiểm này sẽ vẫn báo xanh, vì nó không hề nhìn vào
# đồ thị. Một bài kiểm không nhìn thứ nó tuyên bố kiểm thì không phải là bài kiểm.
ehp_all = pairs("example_has_protein").rename({"src": "ex", "dst": "prot"})
wrong, split_tgt = [], []
for (c, t), want in TRUTH.items():
    got = (ehp_all.filter(pl.col("ex").str.starts_with(f"ex:{c}:{t}:"))["prot"]
           .unique().to_list())
    if not got:
        wrong.append(f"{c}:{t} (KHÔNG có Example nào)")
    elif len(got) > 1:
        split_tgt.append(f"{c}:{t} → {got}")
    elif got[0] != f"protein:{want}":
        wrong.append(f"{c}:{t} → {got[0]}, phải là protein:{want}")
print(f"   {len(TRUTH) - len(wrong) - len(split_tgt)}/{len(TRUTH)} đúng (đọc TỪ ĐỒ THỊ)"
      + (f"  SAI: {wrong}" if wrong else ""))
if wrong:
    issues.append(f"{len(wrong)} target trỏ SAI protein trong KG: {wrong}")
if split_tgt:
    issues.append(f"{len(split_tgt)} target có Example trỏ vào NHIỀU protein: {split_tgt}")

hiv = ["protein:HIV1:PR", "protein:HIV1:RT", "protein:HIV1:IN"]
for r in ("90", "50", "30"):
    k = pairs(f"protein_cluster_{r}").filter(pl.col("src").is_in(hiv))["dst"].n_unique()
    if k < 3:
        issues.append(f"ở mức {r}%, 3 domain HIV bị gom vào {k} cụm — tách domain mất tác dụng")
print(f"   HIV protease/RT/integrase tách riêng ở cả 3 mức: "
      f"{'OK' if not any('HIV' in i for i in issues) else 'SAI'}")

ehp = pairs("example_has_protein").rename({"src": "ex", "dst": "prot"})
shared = (ehp.with_columns(pl.col("ex").str.split(":").list.get(1).alias("corpus"))
             .group_by("prot").agg(pl.col("corpus").n_unique().alias("nc"))
             .filter(pl.col("nc") > 1))
print(f"   protein dùng chung bởi >=2 corpus: {shared.height:,} "
      f"(rò rỉ xuyên corpus — v3 hoàn toàn mù)")

print("\n   cụm protein LỒNG NHAU (90% ⊆ 50% ⊆ 30%)?")
c = {r: pairs(f"protein_cluster_{r}").rename({"src": "p", "dst": f"c{r}"}) for r in ("30", "50", "90")}
m = c["90"].join(c["50"], on="p").join(c["30"], on="p")
bad = (m.group_by("c90").agg(pl.col("c30").n_unique().alias("a"), pl.col("c50").n_unique().alias("b"))
        .filter((pl.col("a") > 1) | (pl.col("b") > 1)).height)
print(f"      cụm 90% bị tách ở mức lỏng hơn: {bad}")
if bad:
    issues.append(f"{bad} cụm 90% không nằm gọn trong 1 cụm 30/50% — split 3 nấc mất đơn điệu")

# ---------------------------------------------------------------- B. chemistry
print("\nB. CHEMISTRY")
sc = n.filter(pl.col("node_type") == "Scaffold").select("node_id", "label").collect()
samp = sc.sample(min(20000, sc.height), seed=42)
badsmi = [l for l in samp["label"].to_list() if Chem.MolFromSmiles(l) is None]
print(f"   nhãn Scaffold không parse được: {len(badsmi)}/{samp.height}"
      f"  (nền RDKit round-trip; KG v3 cũng ~3/20000)")
if len(badsmi) > 5:
    issues.append(f"{len(badsmi)}/{samp.height} nhãn Scaffold hỏng — vượt nền RDKit")

# Ngưỡng lấy từ MỘT nguồn duy nhất. Trước đây 0.80 được gõ tay ở đây, 0.9995 gõ tay ở
# `fixes.py` và lại gõ tay lần nữa ở `ligand_similarity.py` — ba bản sao của một hằng
# số, và bản chạy thật (0.85) không khớp bản nào.
T_MIN, T_FP = 0.80, 0.9995

ls = e.filter(pl.col("edge_type") == "ligand_similar").select("props").collect()
vals = []
bad_props = 0
for p in ls["props"].to_list():
    try:
        d = json.loads(p) if p else {}
    except Exception:
        bad_props += 1
        continue
    for k in ("tanimoto", "similarity", "sim"):
        if k in d:
            vals.append(float(d[k]))
            break

# `if vals:` là cách bài kiểm này từng TỰ TẮT mình: không cạnh nào mang `tanimoto` thì
# nó im lặng bỏ qua và không báo gì cả. Không có dữ liệu để kiểm là một THẤT BẠI của
# bài kiểm, không phải một cái cớ để bỏ qua.
if ls.height and len(vals) < ls.height:
    issues.append(f"{ls.height - len(vals):,}/{ls.height:,} cạnh ligand_similar KHÔNG "
                  f"ghi Tanimoto ({bad_props} props hỏng) — không kiểm được ngưỡng")
if vals:
    v = pl.Series(vals)
    out = int(((v < T_MIN) | (v >= T_FP)).sum())
    print(f"   ligand_similar Tanimoto: min={v.min():.3f} max={v.max():.3f} | "
          f"ngoài [{T_MIN}, {T_FP}): {out}")
    if out:
        issues.append(f"{out} cạnh ligand_similar có Tanimoto ngoài khoảng khai báo")
elif ls.height:
    issues.append("ligand_similar không có cạnh nào ghi Tanimoto — ngưỡng không kiểm được")

lig = pairs("example_has_ligand").rename({"src": "ex", "dst": "lig"})
ghost = (lig.with_columns(pl.col("ex").str.split(":").list.get(1).alias("c"),
                          pl.col("ex").str.split(":").list.get(2).alias("t"))
            .group_by(["c", "t", "lig"]).agg(pl.col("ex").n_unique().alias("k"))
            .filter(pl.col("k") > 1).height)
# Ngưỡng cũ là `if ghost > 324`, với 324 = "mức nền của KG kế thừa". Đồ thị kế thừa
# (`outputs/kg_v0/`) KHÔNG CÒN TỒN TẠI — nó bị xoá cùng v1–v3 — nên con số 324 không
# đối chiếu lại được với bất cứ thứ gì. Một hằng số ma thuật dung thứ cho 324 bản ghi
# trùng lặp mà không ai còn kiểm chứng được là chính sách, không phải bài kiểm.
# Trùng lặp (corpus, target, ligand) nghĩa là CÙNG một hợp chất được thử trên CÙNG một
# target, trong CÙNG một corpus, hai lần — nếu chúng khác nhãn thì đó là mâu thuẫn.
print(f"   bộ ba (corpus, target, ligand) trùng: {ghost}")
if ghost:
    lab_ex = (n.filter(pl.col("node_type") == "Example")
                .select(pl.col("node_id").alias("ex"),
                        pl.col("props").str.json_path_match("$.label").alias("y"))
                .collect())
    conflict = (lig.with_columns(pl.col("ex").str.split(":").list.get(1).alias("c"),
                                 pl.col("ex").str.split(":").list.get(2).alias("t"))
                   .join(lab_ex, on="ex")
                   .group_by(["c", "t", "lig"]).agg(pl.col("y").n_unique().alias("ny"))
                   .filter(pl.col("ny") > 1).height)
    print(f"      trong đó MÂU THUẪN NHÃN (cùng chất, cùng target, hai nhãn): {conflict}")
    if conflict:
        issues.append(f"{conflict} bộ ba (corpus, target, ligand) mang HAI nhãn khác nhau "
                      f"— cùng một hợp chất vừa active vừa decoy trên cùng target")

# ---------------------------------------------------------------- C. usability
print("\nC. USABILITY — trục nào PHÂN HOẠCH được?")
et = set(e.select("edge_type").unique().collect()["edge_type"].to_list())
print("   độ phủ:")
for axis, types in schema.AXIS_EDGE_TYPES.items():
    live = [t for t in types if t in et]
    if not live:
        print(f"     {axis:<10}  0.0%   << TRỤC RỖNG >>")
        issues.append(f"trục '{axis}' rỗng (proposal có hứa)")
        continue
    anchor = [t for t in live if t.startswith("example_")]
    cov = (e.filter(pl.col("edge_type").is_in(anchor)).select("src").unique()
            .select(pl.len()).collect().item())
    print(f"     {axis:<10} {100*cov/N:5.1f}%  ({cov:,})")


def groups(ex_mid: pl.DataFrame, rels: list[pl.DataFrame], label: str, cut: int | None):
    if cut is not None:
        ex_mid = ex_mid.filter(pl.col("mid").map_elements(
            lambda x: DEG.get(x, 0) <= cut, return_dtype=pl.Boolean))
    if not ex_mid.height:
        print(f"     {label:<22} cut={str(cut):<8} (rỗng)")
        return
    exs = ex_mid["ex"].unique().to_list()
    ix = {v: i for i, v in enumerate(exs)}
    off = len(ix)
    mids = set(ex_mid["mid"].to_list())
    for d in rels:
        mids |= set(d["src"].to_list()) | set(d["dst"].to_list())
    mix = {v: i + off for i, v in enumerate(sorted(mids))}
    r = np.fromiter((ix[x] for x in ex_mid["ex"].to_list()), np.int64)
    cc = np.fromiter((mix[x] for x in ex_mid["mid"].to_list()), np.int64)
    for d in rels:
        # Lọc src và dst THEO CẶP.
        #
        # Bản cũ lọc hai đầu ĐỘC LẬP rồi ghép theo vị trí:
        #     a = [mix[x] for x in d["src"] if x in mix]
        #     b = [mix[x] for x in d["dst"] if x in mix]
        #     k = min(len(a), len(b));  r += a[:k];  cc += b[:k]
        # Nếu có dù một cạnh chỉ có một đầu nằm trong `mix`, hai mảng LỆCH NHAU và mọi
        # cặp phía sau bị ghép sai — nó BỊA ra cạnh giữa các node chẳng liên quan gì,
        # rồi connected_components gộp chúng thành một khối. Toàn bộ bảng "khối nguyên
        # tử lớn nhất" — con số quyết định trục nào chia được — sẽ sai mà không có dấu
        # hiệu nào. Hiện tại nó chưa nổ vì `mids` được nạp đủ cả hai đầu của mọi `rel`,
        # nhưng đó là may, không phải thiết kế: thêm một `rel` nào đó không được nạp vào
        # `mids` là mìn phát nổ.
        ss, tt = d["src"].to_list(), d["dst"].to_list()
        keep = [(mix[x], mix[y]) for x, y in zip(ss, tt) if x in mix and y in mix]
        if not keep:
            continue
        a = np.fromiter((p[0] for p in keep), np.int64, len(keep))
        b = np.fromiter((p[1] for p in keep), np.int64, len(keep))
        r = np.concatenate([r, a]); cc = np.concatenate([cc, b])
    g = coo_matrix((np.ones(len(r), np.int8), (r, cc)), shape=(off + len(mix),) * 2)
    _, lab = connected_components(g, directed=False)
    _, cnt = np.unique(lab[:off], return_counts=True)
    big, tot = int(cnt.max()), int(cnt.sum())
    print(f"     {label:<22} cut={str(cut):<8} {len(cnt):>9,} group | "
          f"khối lớn nhất {big:>9,} = {100*big/N:5.1f}% TOÀN CORPUS")


print("\n   khối nguyên tử lớn nhất (phải nằm trọn một bên train/test).")
print("   `cut` = bỏ node trung gian có bậc > cut. KG KHÔNG chọn ngưỡng hộ bạn — đây là đường cong:")
ehl = pairs("example_has_ligand").rename({"src": "ex", "dst": "mid"})

# Trục ligand: quét theo NGƯỠNG TANIMOTO, không chỉ một con số.
#
# KG ghi `ligand_similar` từ 0.80 trở lên và cất giá trị Tanimoto vào `props` — ngưỡng
# là CHÍNH SÁCH của downstream. Nhưng ghi ở 0.80 làm đồ thị ligand dày hơn hẳn so với
# 0.85 (bản cũ vô tình chạy ở 0.85), và cộng thêm 234K cạnh `ligand_parent_exact` vừa
# được cứu, khối liên thông có thể phình to tới mức trục ligand KHÔNG chia được nữa.
# Đó không phải lý do để ghi ít đi — đó là lý do để BÁO RA ĐƯỜNG CONG, y như trục
# assay đã làm với `cut`. Ai đọc bảng này thì tự chọn ngưỡng, và nhìn thấy cái giá.
_sim = (e.filter(pl.col("edge_type") == "ligand_similar")
         .select("src", "dst",
                 pl.col("props").str.json_path_match("$.tanimoto")
                   .cast(pl.Float64).alias("t")).collect())
_idt = [pairs(t) for t in ("ligand_exact", "ligand_parent_exact",
                           "ligand_fingerprint_exact")]
print("   trục ligand — khối lớn nhất theo NGƯỠNG TANIMOTO:")
for tmin in (0.80, 0.85, 0.90, 0.95, 1.01):   # 1.01 = chỉ các quan hệ ĐỒNG NHẤT
    sub = _sim.filter(pl.col("t") >= tmin).select("src", "dst")
    lbl = "ligand (chỉ identity)" if tmin > 1.0 else f"ligand T>={tmin:.2f}"
    groups(ehl, _idt + ([sub] if sub.height else []), lbl, None)
lsc = pairs("ligand_scaffold")
ex_sc = ehl.join(lsc, left_on="mid", right_on="src").select(
    "ex", pl.col("dst").alias("mid")).unique()
for cut in (None, 10000, 1000):
    groups(ex_sc, [], "scaffold", cut)
ehp2 = ehp.rename({"prot": "mid"})
# `protein_exact` PHẢI nằm trong nhóm rò rỉ của trục protein: nó nói hai node Protein
# là CÙNG MỘT protein. Bỏ nó ra thì DUD-E:fgfr1 (người) và DEKOIS:fgfr1 (chuột) rơi vào
# hai group khác nhau — nghĩa là split "sạch protein" vẫn đặt cùng một protein ở hai bên
# train/test, đúng thứ trục này sinh ra để chặn. Nó nằm trong AXIS_EDGE_TYPES["protein"]
# từ đầu; đồ thị chỉ chưa bao giờ có cạnh nào của nó.
pex = pairs("protein_exact")
print(f"     (protein_exact: {pex.height} cạnh — hai node, một protein)")
for res in ("90", "30"):
    groups(ehp2, [pairs(f"protein_cluster_{res}"), pex], f"protein @{res}%", None)
groups(ehp2, [pex], "protein (chỉ exact)", None)
asy = pairs("example_from_assay").rename({"src": "ex", "dst": "mid"})
pub = pairs("example_from_publication").rename({"src": "ex", "dst": "mid"})
for cut in (None, 100000, 10000, 1000, 100):
    groups(asy, [], "assay", cut)
for cut in (None, 10000, 1000, 100):
    groups(pub, [], "publication", cut)

# ---------------------------------------------------------------- D. label confound
print("\nD. ĐỘ PHỦ CÓ TƯƠNG QUAN VỚI NHÃN KHÔNG? (bẫy khi chấm contamination)")
src = e.filter(pl.col("edge_type") == "example_from_source").select(
    pl.col("src").alias("ex"), pl.col("dst").alias("corpus"))
lab = (n.filter(pl.col("node_type") == "Example")
        .select(pl.col("node_id").alias("ex"),
                pl.col("props").str.json_path_match("$.label").alias("y")))
base = src.join(lab, on="ex")
for t, nm in (("example_from_assay", "assay"), ("example_from_publication", "publication")):
    has = e.filter(pl.col("edge_type") == t).select(pl.col("src").alias("ex")).unique()
    j = (base.join(has.with_columns(pl.lit(1).alias("h")), on="ex", how="left")
             .group_by(["corpus", "y"]).agg(pl.len().alias("n"), pl.col("h").sum().alias("c"))
             .collect().with_columns((100 * pl.col("c") / pl.col("n")).alias("pct")))
    print(f"\n   {nm}:")
    worst = 0.0
    for corpus in sorted(j["corpus"].unique().to_list()):
        s = j.filter(pl.col("corpus") == corpus)
        a = s.filter(pl.col("y") == "1")["pct"].to_list()
        d = s.filter(pl.col("y") == "0")["pct"].to_list()
        if a and d:
            gap = abs(a[0] - d[0])
            worst = max(worst, gap)
            print(f"     {corpus:<20} active {a[0]:5.1f}%  decoy {d[0]:5.1f}%   chênh {gap:5.1f}pp")
    if worst > 50:
        issues.append(f"trục {nm}: độ phủ chênh tới {worst:.0f}pp giữa active và decoy — "
                      f"'có provenance hay không' gần như đoán được nhãn; điểm contamination "
                      f"trên trục này sẽ lẫn với nhãn")

print("\n" + "=" * 78)
print(f"{len(issues)} vấn đề")
for i, x in enumerate(issues, 1):
    print(f"  {i}. {x}")
sys.exit(len(issues))
