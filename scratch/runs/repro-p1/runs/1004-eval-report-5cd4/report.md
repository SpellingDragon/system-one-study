==============================================================================
eval report  mode=model run-id=1004-s2-decision-sft-c0e7 commit=fb70c7939acd exempt=False
             device=cpu dtype=float32 backend=torch_eager n=60
             chain={'calibration_run_id': '1004-s3-calibrate-6d33', 'calibration_status': 'fitted', 'upstream_ckpt': '$WORKSPACE_ROOT/deep-mutimodal-laya-by-tilelang/runs/1004-s1-smoke-tiny-ca2f-2/model'}
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
speed     serial n=27 (warmup-3)  P50=1.16ms P95=1.27ms P99=1.52ms [readout P50=1.05 P95=1.17]  ⚠ n<30
          device=cpu dtype=float32 backend=torch_eager  tok/s=81578.7 (aggregate input-only, output_tokens=0)
------------------------------------------------------------------------------
parity  argmax 一致 8/8 (100.0%)  passed=True
        device=cpu backend=torch_eager max|err|=1.732e-04 compiles=+0 blockers=0 external_ops=gelu
------------------------------------------------------------------------------
verdict    passed=True  criteria=溯源不是豁免来的（--allow-missing-run-id 出的数按规则不算通过）; 被点名的轴都真的出了数（没有空轴）; 一致轴 argmax 名次未换人
           ⚠ 本轮一致轴未走 Metal 方言（回退路径），方言验收以 @mps 用例为准
==============================================================================
