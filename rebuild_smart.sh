#!/bin/bash
# REBUILD BIET-CHO-TAI: doi load < 45 truoc MOI attempt (khoi dot cycle o load 80).
# Tu hoan tat khi may ranh (co the qua dem). build_kg -> reattach -> consolidate -> so byte.
cd /vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v4
export PYTHONPATH=src PYTHONNOUSERSITE=1
export PATH="/vol/dl-nguyenb5-solar/users/hoangpc/bin:$PATH"
PY=/vol/dl-nguyenb5-solar/users/hoangpc/envs/vsleak2/bin/python
L=outputs/logs
B=data/processed/_rebuild_bak
mkdir -p $B
THRESH=45

wait_load() {
  while true; do
    ld=$(awk '{print int($1)}' /proc/loadavg)
    if [ "$ld" -lt "$THRESH" ]; then echo "  load=$ld < $THRESH -> chay $(date +%H:%M:%S)"; return; fi
    echo "  load=$ld >= $THRESH, cho 5 phut... $(date +%H:%M:%S)"
    sleep 300
  done
}

echo "### SMART REBUILD START $(date) ###"
# backup ship (neu chua co)
[ -f $B/SHIP_nodes.parquet ] || cp outputs/kg/canonical_nodes.parquet $B/SHIP_nodes.parquet
[ -f $B/SHIP_edges.parquet ] || cp outputs/kg/canonical_edges.parquet $B/SHIP_edges.parquet

# 1. build_kg — chi chay khi load thap
rm -f data/processed/kg_nodes.parquet data/processed/kg_edges.parquet
ok=0
for a in 1 2 3 4 5 6 7 8; do
  echo "### build_kg attempt $a ###"; wait_load
  POLARS_MAX_THREADS=1 $PY -m vsleakkg.build_kg > $L/rb_build.log 2>&1
  if [ -f data/processed/kg_edges.parquet ]; then ok=1; echo "  build_kg OK attempt $a"; break; fi
  echo "  fail $a:"; grep -aiE "duplicate node_id|no ':'|Segmentation|NUL" $L/rb_build.log | tail -2
done
[ $ok -eq 1 ] || { echo "!!! BUILD_KG FAIL 8 attempt"; echo "SMART-REBUILD-DONE"; exit 1; }
grep -aE "tasks OK|validation|unique_ligands" $L/rb_build.log | tail -2

# 2. reattach similarity
echo "### reattach ###"
$PY tools/reattach_ligand_similar.py --kg-nodes data/processed/kg_nodes.parquet --kg-edges data/processed/kg_edges.parquet > $L/rb_reattach.log 2>&1
echo "  reattach rc=$?"; tail -3 $L/rb_reattach.log

# 3. consolidate — chi chay khi load thap
ok=0
for a in 1 2 3 4 5 6; do
  echo "### consolidate attempt $a ###"; wait_load
  POLARS_MAX_THREADS=1 $PY -m vsleakkg.kg.consolidate --output-dir outputs/kg_new --corpus all > $L/rb_consol.log 2>&1 && { ok=1; break; }
  echo "  fail $a:"; grep -aiE "enrichment incomplete|NUL|REFUSING|Segmentation" $L/rb_consol.log | tail -2
done
[ $ok -eq 1 ] || { echo "!!! CONSOLIDATE FAIL 6"; echo "SMART-REBUILD-DONE"; exit 1; }
grep -aE "canonical KG:|validation passed" $L/rb_consol.log | tail -2

# 4. SO BYTE: canonical MOI (outputs/kg_new) vs ban SHIP (outputs/kg, KHONG dung toi)
echo "### SO CANONICAL md5 (kg_new vs kg ship) ###"
md5sum outputs/kg_new/canonical_nodes.parquet outputs/kg/canonical_nodes.parquet
md5sum outputs/kg_new/canonical_edges.parquet outputs/kg/canonical_edges.parquet
echo "  (ban ship outputs/kg KHONG bi ghi de — chi doi chieu)"

# 5. Neu byte-identical -> kg_new CHINH LA graph da audit sach -> khong can audit lai.
#    Neu KHAC -> can dieu tra (bat thuong). So noi dung de chac chan.
POLARS_MAX_THREADS=2 $PY - <<'PY'
import polars as pl, hashlib
def sig(p):
    d=pl.read_parquet(p); d=d.sort(d.columns); h=hashlib.sha256()
    for c in d.columns:
        for v in d.get_column(c).cast(pl.Utf8).fill_null("<n>").to_list(): h.update(v.encode())
        h.update(b"|")
    return d.height, h.hexdigest()[:20]
for nm,a,b in [("nodes","outputs/kg/canonical_nodes.parquet","outputs/kg_new/canonical_nodes.parquet"),
               ("edges","outputs/kg/canonical_edges.parquet","outputs/kg_new/canonical_edges.parquet")]:
    x=sig(a); y=sig(b)
    print(f"  {nm}: ship {x[0]:,} h={x[1]} | moi {y[0]:,} h={y[1]} -> {'*** GIONG HET ***' if x==y else '!!! KHAC — DIEU TRA'}")
PY
echo "  (ban ship outputs/kg giu nguyen. kg_new la ban tai tao de doi chieu.)"
echo "SMART-REBUILD-DONE $(date)"
