// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

public struct ANEDiagnosticsSnapshot: Sendable, CustomStringConvertible {
    public let totalOps: UInt64
    public let supportedOps: UInt64
    public let aneDispatches: UInt64
    public let gpuFallbacks: UInt64
    public let cpuFallbacks: UInt64
    public let compileCacheHits: UInt64
    public let compileCacheMisses: UInt64
    public let partitionBoundaries: UInt64

    public var description: String {
        "total_ops=\(totalOps) supported_ops=\(supportedOps) ane_dispatches=\(aneDispatches) gpu_fallbacks=\(gpuFallbacks) cpu_fallbacks=\(cpuFallbacks) compile_cache_hits=\(compileCacheHits) compile_cache_misses=\(compileCacheMisses) partition_boundaries=\(partitionBoundaries)"
    }
}

public enum ANE {
    public static func diagnostics() -> ANEDiagnosticsSnapshot {
        var snapshot = mlx_ane_diagnostics_snapshot()
        mlx_ane_get_diagnostics(&snapshot)
        return ANEDiagnosticsSnapshot(
            totalOps: snapshot.total_ops,
            supportedOps: snapshot.supported_ops,
            aneDispatches: snapshot.ane_dispatches,
            gpuFallbacks: snapshot.gpu_fallbacks,
            cpuFallbacks: snapshot.cpu_fallbacks,
            compileCacheHits: snapshot.compile_cache_hits,
            compileCacheMisses: snapshot.compile_cache_misses,
            partitionBoundaries: snapshot.partition_boundaries
        )
    }

    public static func resetDiagnostics() {
        mlx_ane_reset_diagnostics()
    }

    public static var runtimeIsAvailable: Bool {
        var available = false
        mlx_ane_runtime_is_available(&available)
        return available
    }

    public static var runtimeUnavailableReason: String {
        var s = mlx_string_new()
        mlx_ane_runtime_unavailable_reason(&s)
        defer { mlx_string_free(s) }
        guard let c = mlx_string_data(s) else { return "" }
        return String(cString: c)
    }

    @discardableResult
    public static func pinToSurface(_ array: MLXArray) -> Bool {
        mlx_ane_pin_to_surface(array.ctx) == 0
    }
}
