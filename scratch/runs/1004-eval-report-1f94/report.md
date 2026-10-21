==============================================================================
eval report  mode=model run-id=1004-s2-decision-sft-17ac commit=fb70c7939acd exempt=False
             device=mps dtype=float32 backend=tilelang n=100
             chain={'calibration_run_id': '1004-s3-calibrate-a3a2', 'calibration_status': 'fitted', 'upstream_ckpt': 'bench/p1-09/pretrain_tiny'}
------------------------------------------------------------------------------
qtype   k     n    acc     base    Δ        within1  mae     超基线  超参考线(+10.0pp)
choice  3   40   0.6500  0.3333  +31.67pp -        -       是      是     
noul    2   60   0.7500  0.5000  +25.00pp -        -       是      是     
分桶呈现：任何两个 k 都未合并；无跨 k 加权总分（合并由读者按 n 自行加权）
------------------------------------------------------------------------------
qtype     n    倍数(出处)              ECE前   ECE后   Δ         改善
choice    40   0.8000(已调,decision_config) 0.0374  0.0256  -0.0119  是
noul      60   0.7500(已调,decision_config) 0.0537  0.0014  -0.0523  是
pooled    100  (all rows              ) 0.0472  0.0111  -0.0361
口径 top-label/weighted × 15 桶；口径固定为 top-label + 质量加权（与硬命中口径不混报）；before 恒取倍数按一的那一份，逐类倍数出处随报告带出；overall 是全体行合并的一份，与逐类数不可互推
------------------------------------------------------------------------------
speed     serial n=30 (warmup-3)  P50=1.46ms P95=2.34ms P99=2.39ms [readout P50=1.08 P95=1.93]
          device=mps dtype=float32 backend=tilelang  tok/s=58275.1 (aggregate input-only, output_tokens=0)
------------------------------------------------------------------------------
parity  argmax 一致 12/12 (100.0%)  passed=True
        device=mps backend=tilelang max|err|=7.647e-05 compiles=+9 blockers=0 external_ops=gelu
------------------------------------------------------------------------------
verdict    passed=True  criteria=溯源不是豁免来的（--allow-missing-run-id 出的数按规则不算通过）; 被点名的轴都真的出了数（没有空轴）; 一致轴 argmax 名次未换人
==============================================================================
