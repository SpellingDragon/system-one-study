#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR Q_handle,  GM_ADDR K_handle,  GM_ADDR V_handle,  GM_ADDR G_handle,  GM_ADDR Beta_handle,  GM_ADDR H0_handle,  GM_ADDR DO_handle,  GM_ADDR SWS_handle,  GM_ADDR LAM_handle,  GM_ADDR DQ_handle,  GM_ADDR DKG_handle,  GM_ADDR DVG_handle,  GM_ADDR DBeta_handle,  GM_ADDR DG_handle,  GM_ADDR DH0_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> Q;
  Q.SetGlobalBuffer((__gm__ float*)Q_handle);
  AscendC::GlobalTensor<float> K;
  K.SetGlobalBuffer((__gm__ float*)K_handle);
  AscendC::GlobalTensor<float> V;
  V.SetGlobalBuffer((__gm__ float*)V_handle);
  AscendC::GlobalTensor<float> G;
  G.SetGlobalBuffer((__gm__ float*)G_handle);
  AscendC::GlobalTensor<float> Beta;
  Beta.SetGlobalBuffer((__gm__ float*)Beta_handle);
  AscendC::GlobalTensor<float> H0;
  H0.SetGlobalBuffer((__gm__ float*)H0_handle);
  AscendC::GlobalTensor<float> DO;
  DO.SetGlobalBuffer((__gm__ float*)DO_handle);
  AscendC::GlobalTensor<float> SWS;
  SWS.SetGlobalBuffer((__gm__ float*)SWS_handle);
  AscendC::GlobalTensor<float> LAM;
  LAM.SetGlobalBuffer((__gm__ float*)LAM_handle);
  AscendC::GlobalTensor<float> DQ;
  DQ.SetGlobalBuffer((__gm__ float*)DQ_handle);
  AscendC::GlobalTensor<float> DKG;
  DKG.SetGlobalBuffer((__gm__ float*)DKG_handle);
  AscendC::GlobalTensor<float> DVG;
  DVG.SetGlobalBuffer((__gm__ float*)DVG_handle);
  AscendC::GlobalTensor<float> DBeta;
  DBeta.SetGlobalBuffer((__gm__ float*)DBeta_handle);
  AscendC::GlobalTensor<float> DG;
  DG.SetGlobalBuffer((__gm__ float*)DG_handle);
  AscendC::GlobalTensor<float> DH0;
  DH0.SetGlobalBuffer((__gm__ float*)DH0_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto io_s = ascend_ub.GetWithOffset<float>(64, 0);
  auto s_ub = ascend_ub.GetWithOffset<float>(64, 256);
  auto g_one = ascend_ub.GetWithOffset<float>(1, 512);
  auto b_one = ascend_ub.GetWithOffset<float>(1, 544);
  auto io_k = ascend_ub.GetWithOffset<float>(8, 576);
  auto k_ub = ascend_ub.GetWithOffset<float>(8, 608);
  auto a_ub = ascend_ub.GetWithOffset<float>(64, 640);
  auto t_ub = ascend_ub.GetWithOffset<float>(64, 896);
  auto pred_ub = ascend_ub.GetWithOffset<float>(64, 1280);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(128, 1152);
  auto io_v = ascend_ub.GetWithOffset<float>(8, 1312);
  auto v_ub = ascend_ub.GetWithOffset<float>(8, 1344);
  auto u_ub = ascend_ub.GetWithOffset<float>(8, 1376);
  auto lam_ub = ascend_ub.GetWithOffset<float>(64, 1408);
  auto q_ub = ascend_ub.GetWithOffset<float>(64, 1664);
  auto do_ub = ascend_ub.GetWithOffset<float>(8, 1696);
  auto do_ub_broadcast_0_data = ascend_ub.GetWithOffset<float>(64, 1728);
  auto broadcast_workspace_1_data = ascend_ub.GetWithOffset<uint8_t>(128, 1984);
  auto q_ub_broadcast_2_data = ascend_ub.GetWithOffset<float>(64, 2112);
  auto dv_ub = ascend_ub.GetWithOffset<float>(64, 2368);
  auto scol_ub = ascend_ub.GetWithOffset<float>(1, 2400);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::copy_gm_to_ub<float, 8, 8>(io_s[0], H0[0], 8, 8, 8, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
    tl::ascend::copy_ub_to_ub<float, float, 64>(s_ub[0], io_s[0], 8, 8, 8, 8, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    tl::ascend::copy_ub_to_gm<float, 8, 8>(SWS[0], s_ub[0], 8, 8, 8);
    tl::ascend::copy_gm_to_ub<float, 1>(g_one[0], G[0], 1, 1, 1, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)1);
    AscendC::Exp<float, false>(g_one[0], g_one[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    tl::ascend::copy_gm_to_ub<float, 1>(b_one[0], Beta[0], 1, 1, 1, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<float, 8>(io_k[0], K[0], 8, 1, 8, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
    tl::ascend::copy_ub_to_ub<float, float, 8>(k_ub[0], io_k[0], 1, 8, 8, 1, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(5);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
    tl::ascend::copy_gm_to_ub<float, 8, 8>(s_ub[0], SWS[0], 8, 8, 8, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto g_one_scalar = g_one.GetValue(0);
    AscendC::Muls<float, false>(a_ub[0], s_ub[0], g_one_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    for (int32_t outer_broadcast_idx = 0; outer_broadcast_idx < 8; ++outer_broadcast_idx) {
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
      AscendC::Mul<float, false>(t_ub[(outer_broadcast_idx * 8)], a_ub[(outer_broadcast_idx * 8)], k_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 8, 8, -1, 8, true>(pred_ub[0], t_ub[0], tmp_ub.ReinterpretCast<float>()[0]);
    tl::ascend::copy_gm_to_ub<float, 8>(io_v[0], V[0], 8, 1, 8, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::copy_ub_to_ub<float, float, 8>(v_ub[0], io_v[0], 1, 8, 8, 1, 8, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
    AscendC::Sub<float, false>(u_ub[0], v_ub[0], pred_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto b_one_scalar = b_one.GetValue(0);
    AscendC::Muls<float, false>(u_ub[0], u_ub[0], b_one_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    for (int32_t j = 0; j < 8; ++j) {
      for (int32_t i = 0; i < 8; ++i) {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(1);
        AscendC::PipeBarrier<PIPE_ALL>();
        t_ub.SetValue(((j * 8) + i), 0.000000e+00f);
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(2);
    tl::ascend::copy_ub_to_gm<float, 8, 8>(LAM[64], t_ub[0], 8, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
    tl::ascend::copy_gm_to_ub<float, 1>(g_one[0], G[0], 1, 1, 1, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)1);
    AscendC::Exp<float, false>(g_one[0], g_one[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_ub<float, 1>(b_one[0], Beta[0], 1, 1, 1, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(5);
    tl::ascend::copy_gm_to_ub<float, 8, 8>(lam_ub[0], LAM[64], 8, 8, 8, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<float, 8, 8>(s_ub[0], SWS[0], 8, 8, 8, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<float, 8>(io_k[0], K[0], 8, 1, 8, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(6);
    tl::ascend::copy_ub_to_ub<float, float, 8>(k_ub[0], io_k[0], 1, 8, 8, 1, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(7);
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_ub<float, 8>(io_k[0], Q[0], 8, 1, 8, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::copy_ub_to_ub<float, float, 8>(q_ub[0], io_k[0], 1, 8, 8, 1, 8, 8);
    tl::ascend::copy_gm_to_ub<float, 8>(io_v[0], V[0], 8, 1, 8, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
    tl::ascend::copy_ub_to_ub<float, float, 8>(v_ub[0], io_v[0], 1, 8, 8, 1, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_ub<float, 8>(io_v[0], DO[0], 8, 1, 8, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
    tl::ascend::copy_ub_to_ub<float, float, 8>(do_ub[0], io_v[0], 1, 8, 8, 1, 8, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto g_one_scalar = g_one.GetValue(0);
    AscendC::Muls<float, false>(a_ub[0], s_ub[0], g_one_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    for (int32_t outer_broadcast_idx_1 = 0; outer_broadcast_idx_1 < 8; ++outer_broadcast_idx_1) {
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
      AscendC::Mul<float, false>(t_ub[(outer_broadcast_idx_1 * 8)], a_ub[(outer_broadcast_idx_1 * 8)], k_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 8, 8, -1, 8, true>(pred_ub[0], t_ub[0], tmp_ub.ReinterpretCast<float>()[0]);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
    AscendC::Sub<float, false>(u_ub[0], v_ub[0], pred_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto b_one_scalar = b_one.GetValue(0);
    AscendC::Muls<float, false>(u_ub[0], u_ub[0], b_one_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::Broadcast<float, 2, 1, false>(do_ub_broadcast_0_data[0],do_ub[0],broadcast_workspace_1_data[0],(uint32_t[]){8, 8}, (uint32_t[]){8, 1});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::Broadcast<float, 2, 0, false>(q_ub_broadcast_2_data[0],q_ub[0],broadcast_workspace_1_data[0],(uint32_t[]){8, 8}, (uint32_t[]){1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto q_ub_broadcast_2_data_scalar = q_ub_broadcast_2_data.GetValue(0);
    AscendC::Muls<float, false>(q_ub[0], do_ub_broadcast_0_data[0], q_ub_broadcast_2_data_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float, false>(do_ub_broadcast_0_data[0], lam_ub[0], q_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    for (int32_t outer_broadcast_idx_2 = 0; outer_broadcast_idx_2 < 8; ++outer_broadcast_idx_2) {
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
      AscendC::Mul<float, false>(t_ub[(outer_broadcast_idx_2 * 8)], do_ub_broadcast_0_data[(outer_broadcast_idx_2 * 8)], k_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 8, 8, -1, 8, true>(broadcast_workspace_1_data.ReinterpretCast<float>()[0], t_ub[0], tmp_ub.ReinterpretCast<float>()[0]);
    for (int32_t j_1 = 0; j_1 < 8; ++j_1) {
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(4);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(4);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(5);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(5);
      AscendC::PipeBarrier<PIPE_ALL>();
      q_ub_broadcast_2_data.SetValue(j_1, (0.000000e+00f - (broadcast_workspace_1_data.GetValue(j_1) * b_one.GetValue(0))));
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto b_one_scalar = b_one.GetValue(0);
    AscendC::Muls<float, false>(dv_ub[0], broadcast_workspace_1_data.ReinterpretCast<float>()[0], b_one_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, float, 8>(io_v[0], dv_ub[0], 1, 8, 8, 1, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
    tl::ascend::copy_ub_to_gm<float, 8>(DVG[0], io_v[0], 8, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(dv_ub[0], v_ub[0], pred_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(io_v[0], broadcast_workspace_1_data.ReinterpretCast<float>()[0], dv_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 1, 8, -1, 8, true>(scol_ub[0], io_v[0], tmp_ub.ReinterpretCast<float>()[0]);
    if ASCEND_IS_AIV {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      AscendC::PipeBarrier<PIPE_ALL>();
      DBeta.SetValue(0, scol_ub.GetValue(0));
    }
    tl::ascend::Broadcast<float, 2, 1, false>(dv_ub[0],u_ub[0],broadcast_workspace_1_data[0],(uint32_t[]){8, 8}, (uint32_t[]){8, 1});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::Broadcast<float, 2, 0, false>(pred_ub[0],k_ub[0],broadcast_workspace_1_data[0],(uint32_t[]){8, 8}, (uint32_t[]){1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto pred_ub_scalar = pred_ub.GetValue(0);
    AscendC::Muls<float, false>(q_ub[0], dv_ub[0], pred_ub_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    for (int32_t outer_broadcast_idx_3 = 0; outer_broadcast_idx_3 < 8; ++outer_broadcast_idx_3) {
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
      AscendC::Add<float, false>(v_ub[0], a_ub[(outer_broadcast_idx_3 * 8)], q_ub[(outer_broadcast_idx_3 * 8)], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::PipeBarrier<PIPE_ALL>();
      auto do_ub_scalar = do_ub.GetValue(outer_broadcast_idx_3);
      AscendC::Muls<float, false>(t_ub[(outer_broadcast_idx_3 * 8)], v_ub[0], do_ub_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
      }
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::reduce_sum<float,  8,  8,  0>(v_ub[0], t_ub[0], tmp_ub[0], true);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, float, 8>(io_k[0], v_ub[0], 1, 8, 8, 1, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
    tl::ascend::copy_ub_to_gm<float, 8>(DQ[0], io_k[0], 8, 1, 8);
    for (int32_t outer_broadcast_idx_4 = 0; outer_broadcast_idx_4 < 8; ++outer_broadcast_idx_4) {
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)255);
      AscendC::Mul<float, false>(q_ub[(outer_broadcast_idx_4 * 8)], u_ub[outer_broadcast_idx_4], do_ub_broadcast_0_data[(outer_broadcast_idx_4 * 8)], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::reduce_sum<float,  8,  8,  0>(do_ub[0], q_ub[0], tmp_ub[0], true);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, float, 8>(io_k[0], do_ub[0], 1, 8, 8, 1, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
    tl::ascend::copy_ub_to_gm<float, 8>(DKG[0], io_k[0], 8, 1, 8);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::Broadcast<float, 2, 0, false>(dv_ub[0],u_ub[0],tmp_ub[0],(uint32_t[]){8, 8}, (uint32_t[]){1, 8});
    AscendC::SetFlag<AscendC::HardEvent::S_V>(4);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(4);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::Broadcast<float, 2, 0, false>(pred_ub[0],q_ub_broadcast_2_data[0],tmp_ub[0],(uint32_t[]){8, 8}, (uint32_t[]){1, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    AscendC::Mul<float, false>(q_ub[0], do_ub_broadcast_0_data[0], dv_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(t_ub[0], a_ub[0], pred_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float, false>(t_ub[0], t_ub[0], q_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::reduce_sum<float,  8,  8,  0>(do_ub[0], t_ub[0], tmp_ub[0], true);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, float, 8>(io_k[0], do_ub[0], 1, 8, 8, 1, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<float, 8>(DKG[0], io_k[0], 8, 1, 8);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::Broadcast<float, 2, 0, false>(dv_ub[0],k_ub[0],broadcast_workspace_1_data[0],(uint32_t[]){8, 8}, (uint32_t[]){1, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::Broadcast<float, 2, 1, false>(pred_ub[0],q_ub_broadcast_2_data[0],broadcast_workspace_1_data[0],(uint32_t[]){8, 8}, (uint32_t[]){8, 1});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto pred_ub_scalar = pred_ub.GetValue(0);
    AscendC::Muls<float, false>(q_ub[0], dv_ub[0], pred_ub_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float, false>(t_ub[0], do_ub_broadcast_0_data[0], q_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(a_ub[0], s_ub[0], t_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 8, 8, -1, 8, true>(v_ub[0], a_ub[0], tmp_ub.ReinterpretCast<float>()[0]);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, float, 8>(io_v[0], v_ub[0], 1, 8, 8, 1, 8, 8);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 1, 8, -1, 8, true>(scol_ub[0], io_v[0], tmp_ub.ReinterpretCast<float>()[0]);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)1);
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto g_one_scalar = g_one.GetValue(0);
    AscendC::Muls<float, false>(scol_ub[0], scol_ub[0], g_one_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    if ASCEND_IS_AIV {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(7);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(7);
      AscendC::PipeBarrier<PIPE_ALL>();
      DG.SetValue(0, scol_ub.GetValue(0));
    }
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto g_one_scalar = g_one.GetValue(0);
    AscendC::Muls<float, false>(lam_ub[0], t_ub[0], g_one_scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::copy_ub_to_gm<float, 8, 8>(LAM[0], lam_ub[0], 8, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
    tl::ascend::copy_gm_to_ub<float, 8, 8>(lam_ub[0], LAM[0], 8, 8, 8, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
    tl::ascend::copy_ub_to_ub<float, float, 64>(io_s[0], lam_ub[0], 8, 8, 8, 8, 8, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
    tl::ascend::copy_ub_to_gm<float, 8, 8>(DH0[0], io_s[0], 8, 8, 8);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* Q_handle, uint8_t* K_handle, uint8_t* V_handle, uint8_t* G_handle, uint8_t* Beta_handle, uint8_t* H0_handle, uint8_t* DO_handle, uint8_t* SWS_handle, uint8_t* LAM_handle, uint8_t* DQ_handle, uint8_t* DKG_handle, uint8_t* DVG_handle, uint8_t* DBeta_handle, uint8_t* DG_handle, uint8_t* DH0_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<1, nullptr, stream>>>(Q_handle, K_handle, V_handle, G_handle, Beta_handle, H0_handle, DO_handle, SWS_handle, LAM_handle, DQ_handle, DKG_handle, DVG_handle, DBeta_handle, DG_handle, DH0_handle, fftsAddr);
}
