==============================================================================
eval report  mode=model run-id=1004-s2-decision-sft-739e commit=fb70c7939acd exempt=False
             device=mps dtype=float32 backend=tilelang n=60
             chain={'calibration_run_id': '1004-s3-calibrate-5e08', 'calibration_status': 'fitted', 'upstream_ckpt': '$WORKSPACE_ROOT/deep-mutimodal-laya-by-tilelang/runs/1004-s1-smoke-tiny-ca2f-4/model'}
------------------------------------------------------------------------------
qtype   k     n    acc     base    Δ        within1  mae     超基线  超参考线(+10.0pp)
choice  3   25   0.6400  0.3333  +30.67pp -        -       是      是     
noul    2   35   0.7429  0.5000  +24.29pp -        -       是      是     
分桶呈现：任何两个 k 都未合并；无跨 k 加权总分（合并由读者按 n 自行加权）
------------------------------------------------------------------------------
qtype     n    倍数(出处)              ECE前   ECE后   Δ         改善
choice    25   1.4000(已调,decision_config) 0.0478  0.1164  +0.0686  否
noul      35   3.9000(已调,decision_config) 0.0318  0.1643  +0.1325  否
pooled    60   (all rows              ) 0.0385  0.1444  +0.1059
口径 top-label/weighted × 15 桶；口径固定为 top-label + 质量加权（与硬命中口径不混报）；before 恒取倍数按一的那一份，逐类倍数出处随报告带出；overall 是全体行合并的一份，与逐类数不可互推
------------------------------------------------------------------------------
speed     serial n=30 (warmup-3)  P50=1.36ms P95=3.01ms P99=3.36ms [readout P50=0.98 P95=2.31]
          device=mps dtype=float32 backend=tilelang  tok/s=56824.0 (aggregate input-only, output_tokens=0)
------------------------------------------------------------------------------
parity  argmax 一致 12/12 (100.0%)  passed=True
        device=mps backend=tilelang max|err|=1.878e-04 compiles=+9 blockers=0 external_ops=gelu
------------------------------------------------------------------------------
verdict    passed=True  criteria=溯源不是豁免来的（--allow-missing-run-id 出的数按规则不算通过）; 被点名的轴都真的出了数（没有空轴）; 一致轴 argmax 名次未换人
==============================================================================
