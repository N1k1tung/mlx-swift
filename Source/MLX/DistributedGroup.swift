//
import Cmlx
import Foundation

/// A Swift wrapper for the mlx_distributed_group C API.
public final class DistributedGroup {
    /// The underlying C distributed group.
    private var group: mlx_distributed_group

    /// Initialize with an existing C distributed group.
    init(group: mlx_distributed_group) {
        self.group = group
    }

    deinit { // this requires slight update for cmlx which I rather avoid, commented out for now
//        mlx_distributed_group_free(group)
    }

    /// Returns the rank of the current process within the distributed group.
    public var rank: Int32 {
        return mlx_distributed_group_rank(group)
    }

    /// Returns the size of the distributed group.
    public var size: Int32 {
        return mlx_distributed_group_size(group)
    }

    /// Returns whether the distributed API is available.
    public static var isAvailable: Bool {
        mlx_distributed_is_available(nil)
    }

    /// Initializes the distributed group.
    /// - Parameter strict: Whether to initialize in strict mode.
    /// - Returns: A new instance of `DistributedGroup`.
    public static func initialize(strict: Bool) -> DistributedGroup {
        let group = mlx_distributed_init(strict, nil)
        return DistributedGroup(group: group)
    }

    /// Splits the current distributed group into subgroups.
    /// - Parameters:
    ///   - color: The color used to split the group.
    ///   - key: The key used to order ranks in the new group.
    /// - Returns: A new `DistributedGroup` representing the subgroup.
    public func split(color: Int32, key: Int32) -> DistributedGroup {
        let newGroup = mlx_distributed_group_split(group, color, key)
        return DistributedGroup(group: newGroup)
    }

    /// Performs an all-gather operation on the input array.
    /// - Parameters:
    ///   - x: The input MLXArray to gather from all ranks.
    ///   - stream: An optional mlx_stream for the operation.
    /// - Returns: The gathered mlx_array.
    public func allGather(_ x: MLXArray, stream: StreamOrDevice = .cpu) -> MLXArray {
        var out = mlx_array_new()
        _ = mlx_distributed_all_gather(&out, x.ctx, group, stream.ctx)        
        return MLXArray(out)
    }

    /// Performs an all-max operation on the input array.
    /// - Parameters:
    ///   - x: The input MLXArray to perform max reduction.
    ///   - stream: An optional mlx_stream for the operation.
    /// - Returns: The mlx_array containing the max values.
    public func allMax(_ x: MLXArray, stream: StreamOrDevice = .cpu) -> MLXArray {
        var out = mlx_array_new()
        _ = mlx_distributed_all_max(&out, x.ctx, group, stream.ctx)
        return MLXArray(out)
    }

    /// Performs an all-min operation on the input array.
    /// - Parameters:
    ///   - x: The input MLXArray to perform min reduction.
    ///   - stream: An optional mlx_stream for the operation.
    /// - Returns: The mlx_array containing the min values.
    public func allMin(_ x: MLXArray, stream: StreamOrDevice = .cpu) -> MLXArray {
        var out = mlx_array_new()
        _ = mlx_distributed_all_min(&out, x.ctx, group, stream.ctx)
        return MLXArray(out)
    }

    /// Performs an all-sum operation on the input array.
    /// - Parameters:
    ///   - x: The input MLXArray to perform sum reduction.
    ///   - stream: An optional mlx_stream for the operation.
    /// - Returns: The mlx_array containing the summed values.
    public func allSum(_ x: MLXArray, stream: StreamOrDevice = .cpu) -> MLXArray {
        var out = mlx_array_new()
        _ = mlx_distributed_all_sum(&out, x.ctx, group, stream.ctx)
        return MLXArray(out)
    }

    /// Receives an mlx_array from a specified rank.
    /// - Parameters:
    ///   - source: The rank to receive from.
    ///   - stream: An optional mlx_stream for the operation.
    /// - Returns: The received mlx_array.
    public func recv<T: HasDType>(_ shape: [Int], type: T.Type = Float.self, source: Int32, stream: StreamOrDevice = .cpu) -> MLXArray {
        var out = mlx_array_new()
        _ = mlx_distributed_recv(&out, shape.map { Int32($0) }, shape.count, T.dtype.cmlxDtype, source, group, stream.ctx)
        return MLXArray(out)
    }

    /// Receives an mlx_array from a specified rank, matching the shape/type of a reference mlx_array.
    /// - Parameters:
    ///   - x: The reference MLXArray to match.
    ///   - source: The rank to receive from.
    ///   - tag: The message tag to receive.
    ///   - stream: An optional mlx_stream for the operation.
    /// - Returns: The received mlx_array.
    public func recvLike(_ x: MLXArray, source: Int32, stream: StreamOrDevice = .cpu) -> MLXArray {
        var out = mlx_array_new()
        _ = mlx_distributed_recv_like(&out, x.ctx, source, group, stream.ctx)
        return MLXArray(out)
    }

    /// Sends an mlx_array to a specified rank.
    /// - Parameters:
    ///   - x: The MLXArray to send.
    ///   - dest: The destination rank.
    ///   - stream: An optional mlx_stream for the operation.
    /// - Returns: The received mlx_array.
    public func send(_ x: MLXArray, dest: Int32, stream: StreamOrDevice = .cpu) -> MLXArray {
        var out = mlx_array_new()
        _ = mlx_distributed_send(&out, x.ctx, dest, group, stream.ctx)
        return MLXArray(out)
    }

    /// Performs a sum-scatter operation on the input array.
    /// - Parameters:
    ///   - x: The MLXArray to sum and scatter.
    ///   - stream: An optional mlx_stream for the operation.
    /// - Returns: The mlx_array containing the scattered sums.
    public func sumScatter(_ x: MLXArray, stream: StreamOrDevice = .cpu) -> MLXArray {
        var out = mlx_array_new()
        _ = mlx_distributed_sum_scatter(&out, x.ctx, group, stream.ctx)        
        return MLXArray(out)
    }
}

