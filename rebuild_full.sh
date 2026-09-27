#!/bin/bash
# REBUILD END-TO-END de "cho chac": build_kg (assembly) -> reattach similarity -> consolidate
# -> so byte voi ban ship. Chung minh ca pipeline tai lap dung KG da ship.
# Chay MOT MINH, threads=1, retry moi tang (chong polars corruption duoi tai).
# Giu cache (guard da commit+test rieng, khong cham canonical).
cd /vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v4
export PYTHONPATH=src PYTHONNOUSERSITE=1
export PATH="/vol/dl-nguyenb5-solar/users/hoangpc/bin:$PATH"
PY=/vol/dl-nguyenb5-solar/users/hoangpc/envs/vsleak2/bin/python
L=outputs/logs
B=data/processed/_rebuild_bak
mkdir -p $B

echo "### REBUILD START $(date) ###"
uptime | grep -o "load average.*"

# 0. Sao luu de so sanh
cp outputs/kg/canonical_nodes.parquet $B/SHIP_nodes.parquet
cp outputs/kg/canonical_edges.parquet $B/SHIP_edges.parquet
cp data/processed/kg_nodes.parquet    $B/OLD_kg_nodes.parquet
echo "backup xong"

# 1. build_kg — xoa raw graph, giu moi cache (test assembly determinism)
rm -f data/processed/kg_nodes.parquet data/processed/kg_edges.parquet
ok=0
for a in 1 2 3 4 5 6; do
  echo "### build_kg attempt $a $(date +%H:%M:%S) ###"
  POLARS_MAX_THREADS=1 $PY -m vsleakkg.build_kg > $L/rb_build.log 2>&1
  if [ -f data/processed/kg_edges.parquet ]; then ok=1; break; fi
  echo "  build_kg fail $a:"; grep -aiE "error|traceback|NUL|assert|corrupt|zeroed" $L/rb_build.log | tail -4
done
[ $ok -eq 1 ] || { echo "!!! BUILD_KG FAILED ALL 6"; echo "REBUILD-ALL-DONE"; exit 1; }
grep -aE "tasks OK|validation|records=|unique_ligands" $L/rb_build.log | tail -3

# 1b. So raw graph moi vs cu (assembly determinism) — so NOI DUNG (build_kg co the khong sort)
echo "### SO RAW GRAPH (moi vs cu) ###"
POLARS_MAX_THREADS=2 $PY - <<'PY'
import polars as pl, hashlib
def sig(p):
    d=pl.read_parquet(p); d=d.sort(d.columns)
    h=hashlib.sha256()
    for c in d.columns:
        for v in d.get_column(c).cast(pl.Utf8).fill_null("<n>").to_list(): h.update(v.encode())
        h.update(b"|")
    return d.height, h.hexdigest()[:20]
a=sig("data/processed/_rebuild_bak/OLD_kg_nodes.parquet")
b=sig("data/processed/kg_nodes.parquet")
print(f"  kg_nodes: cu {a[0]:,} h={a[1]} | moi {b[0]:,} h={b[1]} -> {'GIONG HET' if a==b else 'KHAC (build_kg khong tat dinh?)'}")
PY

# 2. reattach similarity (hash khop -> vai giay)
echo "### reattach similarity ###"
$PY tools/reattach_ligand_similar.py --kg-nodes data/processed/kg_nodes.parquet --kg-edges data/processed/kg_edges.parquet > $L/rb_reattach.log 2>&1
rc=$?; echo "  reattach rc=$rc"; tail -4 $L/rb_reattach.log
[ $rc -eq 0 ] || { echo "!!! REATTACH REFUSED (ligand set doi) — can chay lai ligand_similarity 4.5h"; }

# 3. consolidate -> outputs/kg (retry)
ok=0
for a in 1 2 3 4 5 6; do
  echo "### consolidate attempt $a $(date +%H:%M:%S) ###"
  POLARS_MAX_THREADS=1 $PY -m vsleakkg.kg.consolidate --output-dir outputs/kg --corpus all > $L/rb_consol.log 2>&1 && { ok=1; break; }
  echo "  consol fail $a:"; grep -aiE "enrichment incomplete|NUL|REFUSING" $L/rb_consol.log | tail -3
done
[ $ok -eq 1 ] || { echo "!!! CONSOLIDATE FAILED ALL 6"; echo "REBUILD-ALL-DONE"; exit 1; }
grep -aE "canonical KG:|validation passed" $L/rb_consol.log | tail -2

# 4. SO BYTE canonical moi vs ban ship (BANG CHUNG CHINH)
echo "### SO CANONICAL (moi vs ship) — md5 ###"
md5sum outputs/kg/canonical_nodes.parquet $B/SHIP_nodes.parquet
md5sum outputs/kg/canonical_edges.parquet $B/SHIP_edges.parquet

# 5. audit
echo "### audit_kg ###"
$PY tools/audit_kg.py > $L/rb_audit.log 2>&1
grep -aE "failed check|PASS.*populated|0 failed" $L/rb_audit.log | tail -5
tail -3 $L/rb_audit.log

echo "REBUILD-ALL-DONE $(date)"
