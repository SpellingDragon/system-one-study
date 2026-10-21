"""p2-13 昇腾运行时：910B/950 上的自研 TileLang 方言件底座（三栈与推理共用）。

【做什么】把"在昇腾上跑我们自己的算子"这件事拆成三层：环境（env_setup.sh + selfcheck.py）、
方言件（kernels/）、以及对拍测试（../tests/test_ascend_gradcheck.py）。
【怎么做】kernels/ 里每个算子两件：`<op>_asc.py` 放方言正文与 plan/run 判据，`<op>_kernel.py`
放与 sys1/kernels 同名同签名的对外入口和 torch 回退；分发前先问 ascend_env（环境与缓存自证）。
【为什么】接口与 P1 的 Metal 侧一一对齐，训练侧换后端只改 import 路径、不改调用行。被否方案：
把入口与方言混在一个文件里——回退口径会被方言细节污染，两处代码互相同步的代价更高。
"""
