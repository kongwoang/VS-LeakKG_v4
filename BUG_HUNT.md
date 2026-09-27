# Săn lỗi KG — vòng đối kháng (2026-07-15)

Rà soát đối kháng, **độc lập với `audit_kg`/`audit_semantics`** (vì "ngộ nhận" là thứ vượt
mọi check tự viết). Chỉ TÌM, chưa sửa (theo yêu cầu). Canonical: 8,575,007 node /
92,040,201 cạnh, RDKit 2026.03.2.

## Kết luận

**Không tìm thấy lỗi nào làm SAI dữ liệu đã ghi.** Mọi kiểm cấu trúc/hóa học/thời gian/
split/nhãn/weights/dedup đều sạch; wire provenance truy ngược ChEMBL 300/300 THẬT; code
trục time (tôi tự viết) **đúng hoàn toàn**.

Phát hiện (không cái nào là "dữ liệu sai", đều là gap/rủi ro/cách-dùng):
- **M1** (TB, đã có guard): BindingDB 0.52% SMILES lệch InChIKey — tác động <0.2% provenance.
- **M2** (TB): confidence phân giải protein không được truyền vào KG (768K example/15% trên
  protein conf<1.0, 6 CONFLICT ở 0.5).
- **M3** (môi trường, KHÔNG phải determinism): TÍNH TẤT ĐỊNH **đã chứng minh byte-identical**
  (re-consolidate = KG ship, md5 khớp). Chỉ RELIABILITY mong manh: corrupt khi có polars nặng
  chạy ĐỒNG THỜI; chạy một mình threads=1 thì sạch. Guard luôn bắt được.
- **C1** (completeness): trục assay **chỉ ChEMBL** (0 BindingDB); BindingDB tới example chỉ
  qua publication — "assay 48%" là ChEMBL-only.
- **D1** (cách-dùng): trục ligand — 5,245 cặp cùng-chất proton/tautomer (T<0.80) chỉ
  `ligand_exact` bắt được; downstream phải gộp cả nó, không chỉ Tanimoto.

Chi tiết + bảng "đã kiểm & pass" (~45 kiểm) dưới.

---

## MỨC TRUNG BÌNH (đáng xử lý, đã bounded)

### M1. BindingDB: 0.52% SMILES lệch InChIKey — ĐÃ CÓ GUARD (commit abf30ef)
- 6,599/1,271,034 ligand BindingDB có SMILES chỉ phân tử **khác hẳn** InChIKey (lỗi
  curation nguồn BindingDB). Loader cũ không kiểm.
- **Không lọt vào canonical** (BindingDBLigand bị drop; ligand benchmark giữ SMILES đúng).
- Phép nối provenance dùng InChIKey (đúng). Rủi ro còn lại: nếu BindingDB lệch cả
  PMID/target thì <0.2% provenance nguồn-BindingDB (26% example_from_publication, 18%
  ligand_measured_protein) có thể sai đích. **Chưa rebuild** (không đáng cho <0.2%).
- Provenance **ChEMBL**: đã verify **3000/3000 full InChIKey** — 100% đúng phân tử.

### M3. Reliability (KHÔNG phải determinism) mong manh dưới polars nặng ĐỒNG THỜI
- **TÍNH TẤT ĐỊNH: ĐÃ CHỨNG MINH.** Re-consolidate độc lập (18:34) vs KG ship (14:10) cho
  **parquet GIỐNG HỆT TỪNG BYTE** (md5 nodes `b181cbde…`, edges `fc8efd56…` khớp cả hai).
  Claim "rebuild reproduces exactly" của docs là ĐÚNG.
- **Cái mong manh là RELIABILITY, không phải determinism:** lần thử đầu (đồng thời với 3
  probe nặng adv6/7/8 nạp 66M-cạnh) → corrupt 3/3 ở threads=1. Lần thử sau (chạy MỘT MÌNH,
  threads=1, load vẫn ~78) → **sạch ngay attempt 1 + byte-identical**.
- **Bản chất:** corruption bị kích bởi **polars memory-nặng chạy ĐỒNG THỜI**, không phải
  threads=1 yếu. Chạy consolidate MỘT MÌNH ở threads=1 → tin cậy + tất định.
- **Guard hoạt động đúng:** từ chối ghi cả 3 lần corrupt (KG hỏng không bao giờ lọt ra).
- **Khuyến nghị vận hành:** rebuild → chạy consolidate MỘT MÌNH (không job polars nặng khác),
  threads=1, có retry. Không phải rủi ro dữ liệu; KG ship (outputs/kg) đã byte-identical với
  bản tái tạo độc lập.

### M2. Confidence phân giải protein KHÔNG được truyền vào KG — KHOẢNG TRỐNG
- `target_uniprot_map.parquet` ghi confidence honest cho 198 target:
  - 88 target conf=1.0 (chắc)
  - ~103 target conf 0.85-0.998 (sequence, ident cao — nhiều khả năng đúng)
  - **6 CONFLICT (conf=0.5)**: route tên & route sequence bất đồng — `aces, akt1, braf,
    fgfr1, nos1, prgr` (đều "tên=human, cấu trúc=ortholog khác loài"; map chọn human)
  - 1 "UNRESOLVED" (hivrt) → thực ra map ĐÚNG vào `protein:HIV1:RT` (HIV-domain-split);
    "unresolved" chỉ vì không phải UniProt chuẩn — KHÔNG phải lỗi.
- **Vấn đề:** Protein node props chỉ `{"source":...}`, `example_has_protein` props `{}` —
  confidence **không có trong graph**. Downstream không phân biệt được protein 50/50
  (CONFLICT) với protein chắc chắn. **768,329 example (15%)** trên protein conf<1.0.
- Trái nguyên tắc "KG ghi sự thật khôi phục được": confidence là fact nhưng bị bỏ.
- Bản thân các phân giải **nhất quán** (mỗi target → 1 protein node) và phù hợp nguyên
  tắc "chỉ xét cấu trúc không xét loài" của bạn (MMseqs gộp ortholog). Nên đây là gap
  về TÍNH MINH BẠCH, không phải phân giải sai.
- **Đề xuất (chưa làm):** ghi `resolution_confidence` + `resolution_method` vào
  `example_has_protein` props (hoặc Protein node), lấy từ target_uniprot_map. Không cần
  rebuild build_kg — chỉ sửa consolidate + re-consolidate (~22 phút).

---

## CẢNH BÁO DOWNSTREAM (không phải lỗi KG — cách DÙNG)

### D1. Trục ligand có rò rỉ proton/tautomer mà Tanimoto BỎ SÓT
- **5,245 cặp `ligand_exact`** (cùng full InChIKey = CÙNG hợp chất) có ECFP4 Tanimoto
  **0.48-0.74** — dưới ngưỡng 0.80 — vì chúng khác **proton hóa** (`[NH+]` vs `N`) hoặc
  **tautomer** (`[nH]c(=O)` vs `nc(O)`). InChIKey chuẩn hóa nên thấy giống; ECFP4 nhạy nên
  thấy khác.
- Ligand node key theo md5(canonical SMILES) → hai dạng proton/tautomer = HAI node khác nhau,
  nối bởi `ligand_exact`.
- **Hệ quả:** split chỉ dựa `ligand_similar` (T≥0.80) sẽ để 5,245 cặp cùng-chất lọt hai bên
  train/test → RÒ RỈ ẩn. **Downstream PHẢI gộp cả `ligand_exact` (và `ligand_parent_exact`),
  không chỉ Tanimoto.** KG ghi đủ (cả 4 quan hệ trong AXIS_EDGE_TYPES["ligand"]); đây là
  điểm MẠNH (bắt được cái fingerprint bỏ sót), nhưng dễ bị dùng sai.

---

## COMPLETENESS — bất đối xứng ChEMBL/BindingDB (cần xác nhận)

### C1. Trục assay chỉ có ChEMBL; BindingDB tới example chỉ qua publication
- Code (build_kg.py ~1234): BindingDB record → node "Assay" `bdb_rec:<id>`, chỉ nối tới
  `bdb_lig` (BindingDBLigand). BindingDBLigand **bị drop khỏi canonical** → cạnh treo →
  prune → node Assay BindingDB bị orphan-drop.
- Hệ quả: `example_from_assay` **100% ChEMBL** (adv5 xác nhận: 52,920,513 ChEMBL, 0 BindingDB).
  Nhưng `example_from_publication` có 26% BindingDB.
- **Bất đối xứng:** một measurement chỉ-có-ở-BindingDB tới được example qua trục PUBLICATION
  nhưng KHÔNG qua trục ASSAY. "Assay coverage 48%" thực ra là "ChEMBL-assay coverage".
- **XÁC NHẬN:** 857,115/857,115 Assay canonical = `chembl_asy:`, **0** BindingDB.
- Không mất tín hiệu rò rỉ (vẫn có trên publication), nhưng trục assay **hụt** provenance
  BindingDB. Đây là **implementation chưa hoàn chỉnh**: raw graph dựng `bdb_rec` Assay +
  `bindingdb_record_has_ligand/protein`, nhưng `_wire_reference_provenance` chỉ làm chuỗi
  ChEMBL cho `example_from_assay` → hạ tầng assay BindingDB **chết** (orphan-drop). Nếu muốn
  trục assay đầy đủ, cần wire thêm chuỗi BindingDB record (như đã làm cho publication).
  KHÔNG phải lỗi dữ liệu; là gap độ phủ + code thừa. **Nên ghi rõ "assay = ChEMBL-only" trong
  docs** (hiện KG_REPORT nói "assay 48%" dễ hiểu nhầm là gồm cả hai nguồn).

## INFO — đã điều tra, KHÔNG phải lỗi

- **3,939 nhóm (corpus,ligand,protein,label_type) trùng → 7,880 example:**
  - 324 nhóm = trùng thật (cùng target-name, khác row_idx) — artifact benchmark, 0.006%.
  - 3,615 nhóm = LIT-PCBA cùng chất ở HAI assay (`ESR1_ago` vs `ESR1_ant`), đúng là cùng
    protein ESR1 — example khác nhau hợp lệ (khác assay), KG biểu diễn đúng.
  - Không tạo rò rỉ ẩn (gộp đúng khi chia theo ligand/protein).
- **2 DatasetSource degree 0** = ChEMBL + BindingDB (reference DB, pinned có chủ đích —
  consolidate giữ DatasetSource dù 0 cạnh).
- **1 protein không cluster** = A0A8C0LZB8 (accession lỗi thời, không sequence) — đã ghi
  docs; 0 example chạm nó (coverage protein vẫn 100%).
- **TimeBin thiếu năm 1975** = không ChEMBL document nào trong provenance có năm 1975.
- **protein_exact pident 98-100%** (không strict 100%) — ngưỡng "exact" là ≥98% (isoform/
  biến thể nhỏ). Đặt tên "exact" hơi rộng, nhưng có chủ đích.

---

## ĐÃ KIỂM & PASS (các bất biến audit KHÔNG kiểm)

| Kiểm | Kết quả |
|---|---|
| Ligand id == md5(canonical SMILES) | 0 sai / 2,013,247 |
| Ligand không label rỗng | 0 rỗng |
| label ⟺ label_type ⟺ corpus | 0 vi phạm; mapping đúng từng corpus |
| ligand_exact = cùng full InChIKey | 0/3000 lệch |
| ligand_parent_exact = cùng parent InChIKey | 0/2000 lệch |
| ligand_similar Tanimoto ghi = tính lại | 0/3000 lệch, max\|dT\|=0.00005 |
| ligand_fingerprint_exact đều T≥0.9995 | 0/3000 |
| Scaffold node == Murcko stereo-free | 0/4000 lệch |
| bit-bound pruning không bỏ sót T≥0.80 | 0 thiếu (mẫu + thuật toán đúng) |
| Provenance ChEMBL = cùng full InChIKey | 3000/3000 (100%) |
| **Trục time: mọi (example,năm-provenance) có timebin** | **0 thiếu** |
| **Trục time: mọi timebin có provenance (không bịa năm)** | **0 bịa** |
| example_has_timebin → TimeBin hợp lệ | 0 treo |
| Mỗi example đúng 1 split | 0 nhiều |
| props.source ⟺ example_from_source | khớp |
| Mọi example → protein có cluster_30 | 0 mắc kẹt |
| Mọi decoy/random có source_decoy_protocol | 0 thiếu |
| Không self-loop (mọi loại cạnh) | sạch |
| Mọi edge type có DEFAULT_WEIGHT, weight ∈[0,1] | sạch |
| protein_exact: Protein→Protein, pident≥98% | sạch |
| ligand_measured_protein: Ligand→Protein | sạch |
| props cạnh JSON hợp lệ | sạch (mẫu 20k) |
| **example_from_assay truy ngược ChEMBL activity** | **300/300 khớp (wire THẬT)** |
| **ligand_measured_protein truy ngược ChEMBL** | **295/300 (5 còn lại BindingDB)** |
| dedup: (src,dst) trùng trong provenance | 0 mỗi loại |
| fingerprint_exact ∩ similar (phải rời) | 0 (đúng — 1 ngưỡng) |
| exact ∩ parent (phải rời) | 0 |
| global (src,dst,edge_type) trùng | 0 |
| mọi split có cả 2 nhãn (active+neg) | sạch (9/9 split) |
| ligand SMILES parse được | 30,000/30,000 |

---

## CHƯA KIỂM ĐƯỢC (thành thật)

- ~~Tính tất định consolidate~~ — **ĐÃ XÁC NHẬN byte-identical** (re-consolidate = KG ship,
  md5 khớp cả nodes lẫn edges). Node id ligand cũng 5000/5000 khớp RDKit 2026.03.2.
- **Cơ chế chính xác lệch BindingDB** (chỉ SMILES hay cả PMID/target) — cần TSV gốc; code cho
  thấy cạnh key theo InChIKey per-row → rủi ro <0.2% (không phải 0).
- **Tính đúng phân giải cho 88 target conf=1.0** — tin theo evidence (ident cao) nhưng chưa
  đối chiếu cấu trúc độc lập từng cái (đã spot-check 20/20 sinh học + phân tích 110 target
  conf<1.0).

## ĐỀ XUẤT (nếu bạn muốn sửa sau)

1. **M2 (nên làm):** ghi `resolution_confidence`+`method` vào `example_has_protein` props —
   sửa consolidate + re-consolidate (~22 phút), không cần rebuild build_kg.
2. **D1 (tài liệu):** ghi rõ downstream phải gộp `ligand_exact`+`ligand_parent_exact`, không
   chỉ Tanimoto.
3. **M1:** guard đã có (abf30ef); rebuild chỉ khi cần provenance BindingDB sạch 100%.
