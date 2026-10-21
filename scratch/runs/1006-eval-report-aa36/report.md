==============================================================================
eval report  mode=model run-id=1006-s2-decision-sft-1e95 commit=6c63ce3a9867 exempt=False
             device=mps dtype=float32 backend=tilelang n=100
             chain={'calibration_run_id': '1006-s3-calibrate-962f', 'calibration_status': 'fitted', 'upstream_ckpt': 'runs/1005-s1-mps-main-dc8c/model'}
------------------------------------------------------------------------------
qtype   k     n    acc     base    Δ        within1  mae     超基线  超参考线(+10.0pp)
choice  3   40   0.6500  0.3333  +31.67pp -        -       是      是     
noul    2   60   0.7500  0.5000  +25.00pp -        -       是      是     
分桶呈现：任何两个 k 都未合并；无跨 k 加权总分（合并由读者按 n 自行加权）
------------------------------------------------------------------------------
qtype     n    倍数(出处)              ECE前   ECE后   Δ         改善
choice    40   5.0000(已调,decision_config) 0.1175  0.2267  +0.1092  否
noul      60   1.1500(已调,decision_config) 0.1562  0.1682  +0.0120  否
pooled    100  (all rows              ) 0.1407  0.1916  +0.0509
口径 top-label/weighted × 15 桶；口径固定为 top-label + 质量加权（与硬命中口径不混报）；before 恒取倍数按一的那一份，逐类倍数出处随报告带出；overall 是全体行合并的一份，与逐类数不可互推
------------------------------------------------------------------------------
speed     serial n=30 (warmup-3)  P50=5.82ms P95=6.20ms P99=6.35ms [readout P50=5.36 P95=5.63]
          device=mps dtype=float32 backend=tilelang  tok/s=16360.9 (aggregate input-only, output_tokens=0)
------------------------------------------------------------------------------
parity  argmax 一致 20/20 (100.0%)  passed=True
        device=mps backend=tilelang max|err|=2.604e-04 compiles=+9 blockers=0 external_ops=gelu
------------------------------------------------------------------------------
verdict    passed=True  criteria=溯源不是豁免来的（--allow-missing-run-id 出的数按规则不算通过）; 被点名的轴都真的出了数（没有空轴）; 一致轴 argmax 名次未换人
==============================================================================
