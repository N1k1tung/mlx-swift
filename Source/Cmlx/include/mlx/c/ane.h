/* Copyright © 2026 Apple Inc. */

#ifndef MLX_ANE_H
#define MLX_ANE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/string.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup ane ANE diagnostics
 */
/**@{*/

typedef struct mlx_ane_diagnostics_snapshot_ {
  uint64_t total_ops;
  uint64_t supported_ops;
  uint64_t ane_dispatches;
  uint64_t gpu_fallbacks;
  uint64_t cpu_fallbacks;
  uint64_t compile_cache_hits;
  uint64_t compile_cache_misses;
  uint64_t partition_boundaries;
} mlx_ane_diagnostics_snapshot;

int mlx_ane_get_diagnostics(mlx_ane_diagnostics_snapshot* res);
int mlx_ane_reset_diagnostics(void);
int mlx_ane_runtime_is_available(bool* res);
int mlx_ane_runtime_unavailable_reason(mlx_string* res);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
