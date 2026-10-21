/* 910B port stub (system-one p2-13 B1): c_api/asc_simd.h is a 950-generation
 * CANN header absent on 910B/CANN 8.5.x images (e.g. ModelArts
 * pytorch_2.7.1-cann_8.5.2-...-snt9b). This stub lets AIC-path template
 * translation units compile; any symbol actually referenced by generated
 * AIC code will surface at compile time and get its 910B equivalent here.
 * Purely additive - no upstream file modified. */
#pragma once
