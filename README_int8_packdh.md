# INT8 for Packed Distance+Heading (DetDistHeading)

## Current artifacts
- INT8 engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_int8.engine`
- Calib table: `/home/nvidia/Documents/DeepStream-Yolo/calib_dh.table`
- Configs:
  - INT8: `config_infer_primary_dh_pack_int8.txt`
  - FP16: `config_infer_primary_dh_pack_fp16.txt`
  - QAT noid: `config_infer_primary_dh_pack_int8_qat_noid.txt`
  - QAT-lite PTQ (letterbox+minmax): `config_infer_primary_dh_pack_int8_qatlite_ptq_5k_letterbox_minmax.txt`
- Comparison/overlay helpers:
  - `compare_dh_int8_fp16.py` (FPS + sample detections, batch=1)
  - `overlay_dh_video.py` (renders mp4 with text: boat/other, base, dist, heading)

## Engine naming map (to avoid confusion)
| Friendly name | Actual engine file |
|---|---|
| `dh_headingcls16_soft_10ep_8gpu_best_packdh_8b.engine` | `/home/nvidia/Documents/DeepStream-Yolo/dh_headingcls16_soft_10ep_8gpu_best_packdh_8b.engine` |
| `epoch_298.engine` | `/home/nvidia/Documents/DeepStream-Yolo/epoch_298.engine` |
| `int8_qatlite_v2_ptq_5k_lb_minmax.engine` | `/home/nvidia/Documents/DeepStream-Yolo/dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_qatlite_v2_ptq_int8_5k_letterbox_minmax.engine` |
| `int8_qatlite_v2_ptq_5k.engine` | no exact v2 non-letterbox engine; closest is `/home/nvidia/Documents/DeepStream-Yolo/dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_qatlite_ptq_int8_5k.engine` |

## Training checkpoints on device (2026-01-09 drop)
| Label | Path | Notes |
|---|---|---|
| `init_weights` | `/home/nvidia/Downloads/yolov7_distance_checkpoints/init_weights.pt` | base weights, no DH fine-tune |
| `dh_headingcls16_soft_10ep_8gpu_best` | `/home/nvidia/Downloads/yolov7_distance_checkpoints/dh_headingcls16_soft_10ep_8gpu_best.pt` | main float checkpoint (packdh baseline) |
| `qat_symact_ft10ep_od_dh_bs8_foreground_best` | `/home/nvidia/Downloads/yolov7_distance_checkpoints/qat_symact_ft10ep_od_dh_bs8_foreground_best.pt` | QAT fine-tune used for QDQ noid exports |
| `distance_heading_finetune_multi_gpu_best` | `/home/nvidia/Downloads/yolov7_distance_checkpoints/distance_heading_finetune_multi_gpu_best.pt` | newer fine-tune, not exported on device yet |
| `checksums` | `/home/nvidia/Downloads/yolov7_distance_checkpoints/checksums.sha256` | source checksums from training |

## Export variants (what changes speed/accuracy most)
| Variant | Source `.pt` | ONNX (export) | Q/DQ | TRT 8.5 tweaks | Engine(s) | Notes |
|---|---|---|---|---|---|---|
| QDQ noid (explicit INT8) | `qat_symact_ft10ep_od_dh_bs8_foreground_best.pt` | `best_packdh_8b_qdq_ds736x1280_fixfuse_trt85_noid.onnx` | Yes | fixfuse + remove Identity on INT8 initializers | `best_packdh_8b_qdq_ds736x1280_fixfuse_trt85_noid.engine` | Best accuracy, slower (explicit precision blocks fusion) |
| Float packdh (implicit PTQ) | `dh_headingcls16_soft_10ep_8gpu_best.pt` | `dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_fused.onnx` | No | none | PTQ engines like `dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_ptq_int8_*.engine` | Fast INT8 when calibrated; accuracy depends on calib set |
| QAT-lite float v2 (implicit PTQ) | distill pipeline (see `EXPORT_ARGS.txt`) | `qatlite_v2_packdh_float_trt85_f32const.onnx` | No | f32-const export (TRT 8.5 friendly) | `dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_qatlite_v2_ptq_int8_5k_letterbox_minmax.engine` | Best speed/accuracy trade so far |
| Float packdh (ds736x1280) | `dh_headingcls16_soft_10ep_8gpu_best.pt` | `float_best_packdh_8b_ds736x1280.onnx` | No | from onnx bundle (correct H/W) | `dh_packdh_float_best_fp16.engine`, `dh_packdh_float_best_ptq_int8_1k_letterbox_minmax.engine` | Clean landscape export; good speed with PTQ |
| Best packdh ds736x1280 (maritimo-style) | `dh_headingcls16_soft_10ep_8gpu_best.pt` | `dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_ds736x1280_trt85_f32const.onnx` (or `_simplify.onnx`) | No | trt85 f32const + simplify | `dh_packdh_best_ds736x1280_trt85_fp16.engine`, `dh_packdh_best_ds736x1280_trt85_ptq_int8_1k_letterbox_minmax.engine`, `dh_packdh_best_ds736x1280_simplify_fp16.engine`, `dh_packdh_best_ds736x1280_simplify_ptq_int8_1k_letterbox_minmax.engine` | New export; faster INT8 vs float_best |
| Normal YOLO (simplified) | n/a (maritimo) | `maritimo.onnx` (onnx-simplifier used) | No | onnx-simplified | `maritimo_fp16.engine`, `maritimo_int8.engine` | No DH outputs; used to compare pipeline cost |
| New DH fine-tune | `distance_heading_finetune_multi_gpu_best.pt` | not exported | n/a | n/a | none yet | candidate for future exports |

Note:
- `dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_fused.onnx` input shape is `1x3x1280x736` (H/W swapped). Any engine built from it will not match a `3x736x1280` DeepStream config.
- The bundle ONNX `float_best_packdh_8b_ds736x1280.onnx` is the correct landscape export (1x3x736x1280).
- The training re-export also provides `dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_ds736x1280_trt85_f32const.onnx` and `_simplify.onnx` (preferred for TRT 8.5 landscape builds).

## Calibration used
- Calibration list (5k images): `/home/nvidia/Documents/DeepStream-Yolo/calibration_dh_5k.txt` (sampled from ~15k collected under `/mnt/gs/TempRecordings_COCO_v1/datasets`).
- Build command (batch=1 app, batch_size=8 calibration):
  ```bash
  INT8_CALIB_IMG_PATH=/home/nvidia/Documents/DeepStream-Yolo/calibration_dh_5k.txt \
  INT8_CALIB_BATCH_SIZE=8 \
  deepstream-app -c deepstream_app_config_dh_compare_int8.txt
  ```
  (writes engine + calib table to the paths above)

## Float-best (ds736x1280) PTQ path
Artifacts:
- ONNX: `/home/nvidia/Downloads/yolov7_distance_onnx_bundle_20260111/float_best_packdh_8b_ds736x1280.onnx`
- FP16 engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_packdh_float_best_fp16.engine`
- INT8 engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_packdh_float_best_ptq_int8_1k_letterbox_minmax.engine`
- Calib table: `/home/nvidia/Documents/DeepStream-Yolo/calib_dh_packdh_float_best_1k_letterbox_minmax.table`
- Build log: `/home/nvidia/Documents/DeepStream-Yolo/outputs/trtexec_build_packdh_float_best_fp16_20260111.log`
- PTQ log: `/home/nvidia/Documents/DeepStream-Yolo/outputs/calib_packdh_float_best_ptq_1k_letterbox_minmax_20260111_v2.log`

Build (INT8 PTQ, 1k letterbox + minmax):
```bash
INT8_CALIB_IMG_PATH=/home/nvidia/Documents/DeepStream-Yolo/calibration_dh_1k_local.txt \
INT8_CALIB_BATCH_SIZE=1 \
INT8_CALIB_LETTERBOX=1 \
INT8_CALIB_METHOD=minmax \
deepstream-app -c /home/nvidia/Documents/DeepStream-Yolo/deepstream_app_config_dh_calib_int8_float_best_ptq_1k_letterbox_minmax.txt
```

## Best ds736x1280 (trt85 f32const) PTQ path
Artifacts:
- ONNX: `/home/nvidia/Downloads/yolov7_distance_onnx_bundle_20260111/dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_ds736x1280_trt85_f32const.onnx`
- FP16 engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_packdh_best_ds736x1280_trt85_fp16.engine`
- INT8 engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_packdh_best_ds736x1280_trt85_ptq_int8_1k_letterbox_minmax.engine`
- Calib table: `/home/nvidia/Documents/DeepStream-Yolo/calib_dh_packdh_best_ds736x1280_trt85_1k_letterbox_minmax.table`
- Build log: `/home/nvidia/Documents/DeepStream-Yolo/outputs/trtexec_build_packdh_best_ds736x1280_trt85_fp16_20260111_v2.log`
- PTQ log: `/home/nvidia/Documents/DeepStream-Yolo/outputs/calib_packdh_best_ds736x1280_trt85_ptq_1k_letterbox_minmax_20260111.log`

Build (INT8 PTQ, 1k letterbox + minmax):
```bash
INT8_CALIB_IMG_PATH=/home/nvidia/Documents/DeepStream-Yolo/calibration_dh_1k_local.txt \
INT8_CALIB_BATCH_SIZE=1 \
INT8_CALIB_LETTERBOX=1 \
INT8_CALIB_METHOD=minmax \
deepstream-app -c /home/nvidia/Documents/DeepStream-Yolo/deepstream_app_config_dh_calib_int8_best_ds736x1280_trt85_ptq_1k_letterbox_minmax.txt
```

## Best ds736x1280 (simplify) PTQ path
Artifacts:
- ONNX: `/home/nvidia/Downloads/yolov7_distance_onnx_bundle_20260111/dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_ds736x1280_simplify.onnx`
- FP16 engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_packdh_best_ds736x1280_simplify_fp16.engine`
- INT8 engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_packdh_best_ds736x1280_simplify_ptq_int8_1k_letterbox_minmax.engine`
- Calib table: `/home/nvidia/Documents/DeepStream-Yolo/calib_dh_packdh_best_ds736x1280_simplify_1k_letterbox_minmax.table`
- Build log: `/home/nvidia/Documents/DeepStream-Yolo/outputs/trtexec_build_packdh_best_ds736x1280_simplify_fp16_20260111.log`
- PTQ log: `/home/nvidia/Documents/DeepStream-Yolo/outputs/calib_packdh_best_ds736x1280_simplify_ptq_1k_letterbox_minmax_20260111.log`

Build (INT8 PTQ, 1k letterbox + minmax):
```bash
INT8_CALIB_IMG_PATH=/home/nvidia/Documents/DeepStream-Yolo/calibration_dh_1k_local.txt \
INT8_CALIB_BATCH_SIZE=1 \
INT8_CALIB_LETTERBOX=1 \
INT8_CALIB_METHOD=minmax \
deepstream-app -c /home/nvidia/Documents/DeepStream-Yolo/deepstream_app_config_dh_calib_int8_best_ds736x1280_simplify_ptq_1k_letterbox_minmax.txt
```

## Ablations: rawdet3 vs packdh (export format vs DH heads)
Artifacts:
- rawdet3 ONNX: `/home/nvidia/Downloads/yolov7_distance_onnx_bundle_20260111/dh_headingcls16_soft_10ep_8gpu_best_rawdet3_ds736x1280_simplify.onnx`
- rawdet3+dh ONNX: `/home/nvidia/Downloads/yolov7_distance_onnx_bundle_20260111/dh_headingcls16_soft_10ep_8gpu_best_rawdet3_plus_dh_ds736x1280_simplify.onnx`
- rawdet3+dh fusedgroup ONNX: `/home/nvidia/Downloads/yolov7_distance_onnx_bundle_20260111/dh_headingcls16_soft_10ep_8gpu_best_rawdet3_plus_dh_fusedgroup_ds736x1280_simplify.onnx`
- rawdet3 FP16 engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_packdh_best_rawdet3_ds736x1280_simplify_fp16.engine`
- rawdet3+dh FP16 engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_packdh_best_rawdet3_plus_dh_ds736x1280_simplify_fp16.engine`
- rawdet3+dh fusedgroup FP16 engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_packdh_best_rawdet3_plus_dh_fusedgroup_ds736x1280_simplify_fp16.engine`
- CUDA parser config: `/home/nvidia/Documents/DeepStream-Yolo/config_infer_primary_yoloV7_rawdet3_fp16.txt`
- DeepStream tracker config: `/home/nvidia/Documents/DeepStream-Yolo/deepstream_app_config_tracker_rawdet3_fp16.txt`

TRT perf (FP16, batch=1):
- packdh (trt85 f32const): 12.72 qps, mean 79.01 ms (`outputs/trtexec_perf_packdh_best_ds736x1280_trt85_fp16_20260111.log`)
- packdh (simplify): 13.18 qps, mean 76.24 ms (`outputs/trtexec_perf_packdh_best_ds736x1280_simplify_fp16_20260111.log`)
- rawdet3 (no DH): 24.88 qps, mean 40.94 ms (`outputs/trtexec_perf_packdh_best_rawdet3_ds736x1280_simplify_fp16_20260111.log`)
- rawdet3+dh (logits): 13.27 qps, mean 76.15 ms (`outputs/trtexec_perf_packdh_best_rawdet3_plus_dh_ds736x1280_simplify_fp16_20260111.log`)
- rawdet3+dh fusedgroup: 13.39 qps, mean 75.26 ms (`outputs/trtexec_perf_packdh_best_rawdet3_plus_dh_fusedgroup_ds736x1280_simplify_fp16_20260111.log`)

DeepStream tracker FPS (boat_port.mp4):
- packdh FP16 (trt85): ~11.48 FPS (`outputs/bench_tracker_packdh_best_ds736x1280_trt85_fp16_20260111.txt`)
- packdh FP16 (simplify): ~11.89 FPS (`outputs/bench_tracker_packdh_best_ds736x1280_simplify_fp16_20260111.txt`)
- rawdet3 FP16: ~22.95 FPS (`outputs/bench_tracker_rawdet3_fp16_20260111.txt`)
- maritimo FP16 (reference): ~21.96 FPS (`outputs/bench_tracker_maritimo_fp16_20260109.txt`)

Interpretation:
- rawdet3 matches maritimo in DeepStream, while rawdet3+dh matches packdh.
- The fusedgroup DH heads do not materially change TRT speed (only ~1% improvement vs rawdet3+dh).
- This points to in-graph DH heads + pack/decode export style as the main latency, not the onnxsim simplify step.

## Quick compare (INT8 vs FP16, batch=1)
```bash
# boat_port
python3 compare_dh_int8_fp16.py --config config_infer_primary_dh_pack_int8.txt \
  --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/boat_port.mp4 \
  --sample-json outputs/int8_boat_port.json
python3 compare_dh_int8_fp16.py --config config_infer_primary_dh_pack_fp16.txt \
  --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/boat_port.mp4 \
  --sample-json outputs/fp16_boat_port.json

# galat_bridge
python3 compare_dh_int8_fp16.py --config config_infer_primary_dh_pack_int8.txt \
  --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/galat_bridge.mp4 \
  --sample-json outputs/int8_galat_bridge.json
python3 compare_dh_int8_fp16.py --config config_infer_primary_dh_pack_fp16.txt \
  --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/galat_bridge.mp4 \
  --sample-json outputs/fp16_galat_bridge.json
```
Observed (sampled detections):
- boat_port: INT8 base mean ~0.48 vs FP16 ~0.68 (INT8 ~2× FPS). Dist/heading bins differ noticeably.
- galat_bridge: INT8 base mean ~0.47 vs FP16 ~0.68 (INT8 ~2× FPS). Dist/heading bins closer but still different.

## Overlays with dist/heading text
```bash
python3 overlay_dh_video.py --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/boat_port.mp4 \
  --config config_infer_primary_dh_pack_int8.txt --output outputs/overlay_int8_boat_port.mp4
python3 overlay_dh_video.py --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/boat_port.mp4 \
  --config config_infer_primary_dh_pack_fp16.txt --output outputs/overlay_fp16_boat_port.mp4
python3 overlay_dh_video.py --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/galat_bridge.mp4 \
  --config config_infer_primary_dh_pack_int8.txt --output outputs/overlay_int8_galat_bridge.mp4
python3 overlay_dh_video.py --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/galat_bridge.mp4 \
  --config config_infer_primary_dh_pack_fp16.txt --output outputs/overlay_fp16_galat_bridge.mp4
```

## Learnings / notes
- INT8 throughput is ~2× FP16 on the tested videos (batch=1), but base confidences are lower and decoded dist/heading bins diverge more.
- The packed payload (<0.1) is sensitive to quantization; thresholds should stay on 0.1 steps. Expect more jitter in INT8 unless postprocess layers are kept in FP16/FP32.
- TRT logs “missing scale/zero-point” on some postprocess tensors; those layers fall back to higher precision, but mixed precision still differs from FP16.
- If you need closer parity: increase/curate calibration images, keep postprocess layers FP16/FP32, and clamp payload carefully when decoding to avoid off-by-one binning.
- For speed, use the CUDA parser + keep output on GPU:
  - `parse-bbox-func-name=NvDsInferParseYoloCuda`
  - `disable-output-host-copy=1`
  This avoids CPU decode overhead and matches the yolo_deepstream guidance.
  Note: the local DS 6.2 config parser flagged `disable-output-host-copy` as unknown, so configs here only set the CUDA parser.

## New (best) INT8: QDQ “noid” engine (TRT 8.5-friendly)
- Engine: `/home/nvidia/Documents/DeepStream-Yolo/best_packdh_8b_qdq_ds736x1280_fixfuse_trt85_noid.engine`
- ONNX (do not simplify): `/home/nvidia/Documents/DeepStream-Yolo/best_packdh_8b_qdq_ds736x1280_fixfuse_trt85_noid.onnx`
- Config: `config_infer_primary_dh_pack_int8_qat_noid.txt` (no calib file).
- Built by removing Identity on INT8 initializers (fixes “Int8 constant only allowed before DQ” on TRT 8.5.2). Keep the graph untouched afterward.

### Accuracy (appsrc image eval, Downloads_BoatTourTurkey_with_heading)
- INT8 QDQ noid: AP 0.438 / AP50 0.902, dist MAE 12.08 m, heading MAE 20.62° (1,252 matches).
- FP16: AP 0.209 / AP50 0.474, dist MAE 11.18 m, heading MAE 44.22°.
- FP32: AP 0.206 / AP50 0.466, dist MAE 11.02 m, heading MAE 43.67°.
- Older PTQ INT8 (5k): AP ≈0.086 / AP50 ≈0.195 (much worse).

Run eval:
```bash
cd /opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps-1.1.6/apps/deepstream-imagedata-multistream
python3 eval_dh_int8_fp16_turkey_appsrc_ptq5k_letterbox.py  # includes int8_qat_noid mode
```

### Quick video checks (DeepStream pipeline)
```bash
# Compare FPS/samples on a video
python3 compare_dh_int8_fp16.py --config /home/nvidia/Documents/DeepStream-Yolo/config_infer_primary_dh_pack_int8_qat_noid.txt \
  --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/boat_port.mp4 \
  --sample-json /home/nvidia/Documents/DeepStream-Yolo/outputs/noid_boat_port_samples.json

# Overlay detections with dist/heading text
python3 overlay_dh_video.py --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/boat_port.mp4 \
  --config /home/nvidia/Documents/DeepStream-Yolo/config_infer_primary_dh_pack_int8_qat_noid.txt \
  --output /home/nvidia/Documents/DeepStream-Yolo/outputs/overlay_noid_boat_port.mp4
```

Notes:
- Keep `pre-cluster-threshold` at 0.1 for packed scores unless you explicitly want more recall.
- Avoid onnx-simplifier/graph optimizers on the QDQ ONNX; they can reintroduce TRT 8.5-incompatible constants.
- QDQ is accurate but can be slower than FP16 because explicit-precision scheduling limits fusion. CUDA parse helps.
- QAT noid weights are from a QAT fine-tune and are not the same as the FP32/FP16 baseline weights, so higher AP than FP32 is plausible (not necessarily a bug).
- On-device provenance: the QAT-noid ONNX/engine has no embedded metadata, so the exact `.pt` origin can't be verified here without export args.

## New (fast + good) INT8: QAT-lite PTQ (float ONNX + TRT calibration)
This path uses a float QAT-lite ONNX (no Q/DQ) so TensorRT can do implicit INT8 calibration and keep fusion (faster than QDQ).

Artifacts (v1):
- ONNX: `/home/nvidia/Downloads/jetson_bundle_qatlite_20260106/jetson_bundle_qatlite_20260106/qat_distill_packdh_float_trt85_f32const.onnx`
- Engine: `/home/nvidia/Documents/DeepStream-Yolo/dh_headingcls16_soft_10ep_8gpu_best_packdh_8b_qatlite_ptq_int8_5k_letterbox_minmax.engine`
- Calib table: `/home/nvidia/Documents/DeepStream-Yolo/calib_dh_qatlite_5k_letterbox_minmax.table`
- Config: `config_infer_primary_dh_pack_int8_qatlite_ptq_5k_letterbox_minmax.txt`
- App config: `deepstream_app_config_dh_calib_int8_qatlite_ptq_5k_letterbox_minmax.txt`

Build (letterbox + minmax, batch=1):
```bash
INT8_CALIB_IMG_PATH=/home/nvidia/Documents/DeepStream-Yolo/calibration_dh_5k_local.txt \
INT8_CALIB_BATCH_SIZE=1 \
INT8_CALIB_LETTERBOX=1 \
INT8_CALIB_METHOD=minmax \
deepstream-app -c /home/nvidia/Documents/DeepStream-Yolo/deepstream_app_config_dh_calib_int8_qatlite_ptq_5k_letterbox_minmax.txt
```

Accuracy (Turkey appsrc eval):
- INT8 QAT-lite PTQ 5k letterbox+minmax: AP 0.172 / AP50 0.400, dist MAE 15.51 m, heading MAE 51.93°.
- FPS (appsrc eval run): ~19.7 fps (batch=1).

Quick overlays:
```bash
python3 overlay_dh_video.py --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/boat_port.mp4 \
  --config /home/nvidia/Documents/DeepStream-Yolo/config_infer_primary_dh_pack_int8_qatlite_ptq_5k_letterbox_minmax.txt \
  --output /home/nvidia/Documents/DeepStream-Yolo/outputs/overlay_int8_qatlite_ptq_5k_lb_minmax_boat_port.mp4
python3 overlay_dh_video.py --uri file:///home/nvidia/Documents/DeepStream-Yolo/videos/galat_bridge.mp4 \
  --config /home/nvidia/Documents/DeepStream-Yolo/config_infer_primary_dh_pack_int8_qatlite_ptq_5k_letterbox_minmax.txt \
  --output /home/nvidia/Documents/DeepStream-Yolo/outputs/overlay_int8_qatlite_ptq_5k_lb_minmax_galat_bridge.mp4
```

Notes:
- QAT-lite v2 ONNX is in `/home/nvidia/Downloads/qatlite_v2_float_only_20260107/qat_distill_packdh_float_trt85_f32const.onnx` (after untar).
- Export args for the QAT-lite v2 float ONNX are in `/home/nvidia/Downloads/EXPORT_ARGS.txt`.
- Use the same calibration flow above with 2k/5k lists to build a v2 engine and compare.
- All current ONNX inputs are fixed at 1x3x736x1280. Configs set `infer-dims=3;736;1280` to enforce landscape input.

QAT-lite v2 (letterbox + minmax, appsrc eval):
- 2k: AP 0.209 / AP50 0.428, dist MAE 13.06 m, heading MAE 56.33°, FPS ~21.1.
- 5k: AP 0.206 / AP50 0.439, dist MAE 13.77 m, heading MAE 59.42°, FPS ~21.0.
Log: `/home/nvidia/Documents/DeepStream-Yolo/outputs/bench_turkey_appsrc_cuda_parse_20260107.txt`.

Re-test (CUDA parser configs, 2026-01-07, landscape apples-to-apples):
| mode | AP | AP50 | FPS | dist MAE | heading MAE |
|---|---:|---:|---:|---:|---:|
| int8_qat_noid | 0.438 | 0.902 | 4.07 | 12.08 | 20.62 |
| fp16_qatlite_v2_float | 0.228 | 0.464 | 11.48 | 12.37 | 56.55 |
| fp32_qatlite_v2_float | 0.228 | 0.464 | 5.50 | 12.13 | 58.01 |

Notes:
- The FP16/FP32 baselines above are built from the same landscape float ONNX (QAT-lite v2), so they are shape-matched to QAT noid.
- Log: `/home/nvidia/Documents/DeepStream-Yolo/outputs/bench_turkey_appsrc_qatlite_v2_fp16fp32_20260107.txt`.

## Tracker + interval FPS (file source, local video)
Script:
- `/opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps-1.1.6/apps/deepstream-imagedata-multistream/bench_dh_tracker_interval.py`

Video (local copy):
- `/home/nvidia/Documents/DeepStream-Yolo/videos_local/boat_port.mp4` (593 frames)

Log:
- `/home/nvidia/Documents/DeepStream-Yolo/outputs/bench_tracker_interval_boat_port_20260108.txt`

Results (avg FPS):
| mode | interval=0 | interval=1 | interval=2 |
|---|---:|---:|---:|
| int8_qat_noid | 11.05 | 21.33 | 28.03 |
| int8_qatlite_v2_ptq_5k | 21.84 | 38.54 | 43.79 |
| fp16_qatlite_v2_float | 12.08 | 22.87 | 29.87 |

## Multi-head vs normal YOLO (tracker pipeline, interval=0)
Log:
- `/home/nvidia/Documents/DeepStream-Yolo/outputs/bench_tracker_interval_multihead_vs_normal_20260109.txt`

Results (avg FPS on boat_port.mp4):
| model | avg FPS |
|---|---:|
| packdh int8 (qatlite v2, 5k lb minmax) | 21.92 |
| packdh fp16 (qatlite v2 float) | 12.26 |
| packdh best_packdh_8b.engine | 13.01 |
| normal YOLO (epoch_298, no DH) | 24.17 |
| normal YOLO (maritimo.engine) | 23.85 |

Notes:
- Normal YOLO runs are faster than packdh FP16; packdh INT8 is closer to normal YOLO.
- Input aspect differs across models (portrait 1280x736 vs landscape 736x1280), so treat these as approximate deltas.

## Normal YOLO vs packed-DH (TRT + tracker, same video)
All runs use `videos_local/boat_port.mp4`, NvDCF perf tracker, batch=1, streammux 1280x736.
Note: `maritimo.onnx` was exported with onnx-simplifier.

Logs:
- TRT: `outputs/trtexec_perf_maritimo_fp16_20260110.log`, `outputs/trtexec_perf_maritimo_int8_20260110.log`, `outputs/trtexec_perf_packdh_qatlite_v2_fp16_20260110.log`, `outputs/trtexec_perf_packdh_qatlite_v2_int8_ptq_5k_lb_minmax_20260110.log`, `outputs/trtexec_perf_packdh_qdq_noid_int8_20260110.log`
- DeepStream tracker: `outputs/bench_tracker_maritimo_fp16_20260109.txt`, `outputs/bench_tracker_maritimo_int8_20260109.txt`, `outputs/bench_tracker_packdh_fp16_20260109.txt`, `outputs/bench_tracker_packdh_int8_20260109.txt`, `outputs/bench_tracker_packdh_int8_qdq_20260110.txt`

| model | TRT qps (mean ms) | tracker avg FPS | Notes |
|---|---:|---:|---|
| maritimo FP16 | 25.22 (40.43 ms) | 21.96 | normal YOLO |
| maritimo INT8 | 40.28 (25.45 ms) | 35.56 | PTQ with 500-image calib list |
| packdh FP16 (qtalite v2 float) | 13.77 (72.95 ms) | 11.88 | packed DH head |
| packdh INT8 (qtalite v2 PTQ 5k lb minmax) | 24.50 (41.29 ms) | 23.05 | packed DH head |
| packdh INT8 (QDQ noid) | 12.20 (82.00 ms) | 11.21 | accurate, but slow (explicit precision) |
| packdh FP16 (float_best ds736x1280) | 13.87 (72.34 ms) | 12.86 | from onnx bundle |
| packdh INT8 (float_best PTQ 1k lb minmax) | 24.16 (41.87 ms) | 21.73 | from onnx bundle |
| packdh FP16 (best ds736x1280 trt85) | 12.72 (79.01 ms) | 11.48 | maritimo-style export |
| packdh INT8 (best ds736x1280 trt85 PTQ 1k) | 26.52 (38.24 ms) | 22.85 | maritimo-style export |
| packdh FP16 (best ds736x1280 simplify) | 13.18 (76.24 ms) | 11.89 | simplify export |
| packdh INT8 (best ds736x1280 simplify PTQ 1k) | 26.54 (38.22 ms) | 23.01 | simplify export |

## COCO 10k OD eval (SeaShips7k + coco_plus_other_0)
Merged COCO annotations (boat/other only, 10k images sampled with seed=0):
- `/home/nvidia/Documents/DeepStream-Yolo/datasets/coco_10k_merged/annotations.json`
- Source datasets:
  - `/mnt/gs/TempRecordings_COCO_v1/datasets/coco_plus_other_SeaShips7k`
  - `/mnt/gs/TempRecordings_COCO_v1/datasets/coco_plus_other_0`
Eval script:
- `/opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps-1.1.6/apps/deepstream-imagedata-multistream/eval_dh_int8_fp16_coco_appsrc.py`
Log:
- `/home/nvidia/Documents/DeepStream-Yolo/outputs/bench_coco10k_appsrc_20260108.txt`
- `/home/nvidia/Documents/DeepStream-Yolo/outputs/bench_coco10k_appsrc_additional_20260109.txt`
- `/home/nvidia/Documents/DeepStream-Yolo/outputs/bench_coco10k_appsrc_best_packdh_8b_engine_20260109.txt`

Run:
```bash
python3 eval_dh_int8_fp16_coco_appsrc.py \
  --ann /home/nvidia/Documents/DeepStream-Yolo/datasets/coco_10k_merged/annotations.json \
  --max-images 10000 --seed 0
```

Results (AP/AP50, dist/heading n/a since dataset has no DH labels):
| mode | AP | AP50 | FPS |
|---|---:|---:|---:|
| best_packdh_8b_engine | 0.097 | 0.169 | 6.08 |
| int8_qat_noid | 0.076 | 0.139 | 0.89 |
| int8_qatlite_v2_ptq_5k_lb_minmax | 0.101 | 0.154 | 1.26 |
| fp16_qatlite_v2_float | 0.107 | 0.153 | 5.78 |
| int8_base | 0.077 | 0.144 | 6.05 |
| epoch_298 | 0.097 | 0.169 | 6.12 |
| int8_qatlite_ptq_5k | 0.089 | 0.135 | 6.13 |

Notes:
- FPS here is appsrc throughput and is I/O-bound (cv2 reads from gcsfuse); it is not a pure inference benchmark.
- The merged COCO file uses absolute image paths pointing to the GCS mount; copy locally if you want stable throughput numbers.
- Engines differ in input aspect (portrait 1280x736 vs landscape 736x1280), so accuracy here reflects both weights and input shape.
- `epoch_298.engine` does not carry packed distance/heading in scores; DH metrics are n/a for it.
