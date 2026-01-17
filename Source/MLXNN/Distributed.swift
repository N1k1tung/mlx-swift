// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - Gradient Aggregation

/// Cache for sum gradients functions
nonisolated(unsafe) private var sumGradientsCache: [Int32: (MLXArray) -> MLXArray] = [:]

/// Returns a function that aggregates gradients across the distributed group.
/// If the group size is 1, returns identity function.
public func sumGradients(group: DistributedGroup) -> (MLXArray) -> MLXArray {
    let groupSize = group.size
    
    if groupSize == 1 {
        return { x in x }
    }
    
    // Use cached function if available
    if let cached = sumGradientsCache[groupSize] {
        return cached
    }
    
    // Create custom function for gradient aggregation
    // We need to capture the group in the closure
    let aggregatorFn: (MLXArray) -> MLXArray = { x in
        let gradientAggregator = CustomFunction {
            Forward { inputs in
                inputs  // Forward pass is identity
            }
            
            VJP { primals, cotangents in
                // Aggregate gradients in backward pass
                [group.allSum(cotangents[0])]
            }
        }
        return gradientAggregator([x])[0]
    }
    
    sumGradientsCache[groupSize] = aggregatorFn
    return aggregatorFn
}

// MARK: - Utility Functions

/// Split weight allowing for fractional segments (equivalent to mx.split but allows fractional segments)
private func split(weight: MLXArray, segments: Either<Int, [Float]>, axis: Int) -> [MLXArray] {
    switch segments {
    case .left(let intSegments):
        return weight.split(parts: intSegments, axis: axis)
    case .right(let floatSegments):
        let N = weight.shape[axis]
        let indices = floatSegments.map { Int($0 * Float(N)) }
        return weight.split(indices: indices, axis: axis)
    }
}

// MARK: - Utility Types

/// Either type for handling int or float array segments
public enum Either<Left, Right> {
    case left(Left)
    case right(Right)
}

/// Sharding predicate function type
public typealias ShardingPredicate = (String, MLXArray) -> ShardingSpec?

/// Sharding specification
public struct ShardingSpec {
    let axis: Int
    let segments: Either<Int, [Float]>
    
    public init(axis: Int) {
        self.axis = axis
        self.segments = .left(1)
    }
    
    public init(axis: Int, segments: Int) {
        self.axis = axis
        self.segments = .left(segments)
    }
    
    public init(axis: Int, segments: [Float]) {
        self.axis = axis
        self.segments = .right(segments)
    }

    public init(axis: Int, segments: Either<Int, [Float]>) {
        self.axis = axis
        self.segments = segments
    }
}

/// Shard parameters according to the sharding predicate
private func shard(
    parameters: ModuleParameters,
    shardingPredicate: ShardingPredicate,
    group: DistributedGroup
) -> ModuleParameters {
    let N = Int(group.size)
    let r = Int(group.rank)
    
    func shardFn(weight: MLXArray) -> MLXArray {
        guard let spec = shardingPredicate("", weight) else {
            return weight
        }
        
        let parts = split(weight: weight, segments: spec.segments, axis: spec.axis)
        let shardedParts = parts.map { part in
            split(weight: part, segments: .left(N), axis: spec.axis)[r]
        }
        
        return concatenated(shardedParts, axis: spec.axis).contiguous()
    }
    
    return parameters.mapValues { weight in
        shardFn(weight: weight)
    }
}

/// Simple predicate to shard fully connected layers (all-to-sharded)
public func allToShardedPredicate(segments: Either<Int, [Float]>) -> ShardingPredicate {
    return { path, weight in
        let axis = max(weight.ndim - 2, 0)
        return ShardingSpec(axis: axis, segments: segments)
    }
}

/// Simple predicate to shard fully connected layers (sharded-to-all)
public func shardedToAllPredicate(segments: Either<Int, [Float]>) -> ShardingPredicate {
    return { path, weight in
        if path.hasSuffix("bias") {
            return nil
        }
        return ShardingSpec(axis: weight.ndim - 1, segments: segments)
    }
}

/// Check if sharding type is valid
private func checkSharding(_ sharding: String) throws {
    guard sharding == "all-to-sharded" || sharding == "sharded-to-all" else {
        throw DistributedError.invalidSharding(sharding)
    }
}

// MARK: - Public API

/// Shard a module in-place by updating its parameter dictionary
public func shardInPlace(
    module: Module,
    sharding: Either<String, ShardingPredicate>,
    segments: Either<Int, [Float]> = .left(1),
    group: DistributedGroup? = nil
) throws {
    let group = group ?? DistributedGroup.initialize(strict: false)
    
    let predicate: ShardingPredicate
    switch sharding {
    case .left(let shardingType):
        try checkSharding(shardingType)
        predicate = shardingType == "all-to-sharded" 
            ? allToShardedPredicate(segments: segments)
            : shardedToAllPredicate(segments: segments)
    case .right(let customPredicate):
        predicate = customPredicate
    }
    
    let shardedParams = shard(
        parameters: module.parameters(),
        shardingPredicate: predicate,
        group: group
    )
    
    module.update(parameters: shardedParams)
}

/// Create a new linear layer with sharded parameters and distributed communication
public func shardLinear(
    module: Module,
    sharding: String,
    segments: Either<Int, [Float]> = .left(1),
    group: DistributedGroup? = nil
) throws -> Module {
    try checkSharding(sharding)
    
    let group = group ?? DistributedGroup.initialize(strict: false)
    
    if let linear = module as? Linear {
        switch sharding {
        case "all-to-sharded":
            return AllToShardedLinear.fromLinear(linear, segments: segments, group: group)
        case "sharded-to-all":
            return ShardedToAllLinear.fromLinear(linear, segments: segments, group: group)
        default:
            throw DistributedError.invalidSharding(sharding)
        }
    } else if let quantized = module as? QuantizedLinear {
        switch sharding {
        case "all-to-sharded":
            return QuantizedAllToShardedLinear.fromQuantizedLinear(quantized, segments: segments, group: group)
        case "sharded-to-all":
            return QuantizedShardedToAllLinear.fromQuantizedLinear(quantized, segments: segments, group: group)
        default:
            throw DistributedError.invalidSharding(sharding)
        }
    } else {
        throw DistributedError.unsupportedModuleType
    }
}

// MARK: - Error Types

public enum DistributedError: Error {
    case invalidSharding(String)
    case unsupportedModuleType
    case shardingMismatch(String)
}

// MARK: - AllToShardedLinear

/// Each member of the group applies part of the affine transformation such
/// that the result is sharded across the group.
///
/// The gradients are automatically aggregated from each member of the group.
open class AllToShardedLinear: Module, UnaryLayer {
    
    public let weight: MLXArray
    public let bias: MLXArray?
    public let group: DistributedGroup
    
    /// Initialize AllToShardedLinear layer
    public init(
        inputDimensions: Int,
        outputDimensions: Int,
        bias: Bool = true,
        group: DistributedGroup? = nil
    ) throws {
        self.group = group ?? DistributedGroup.initialize(strict: false)
        
        let scale = sqrt(1.0 / Float(inputDimensions))
        let N = Int(self.group.size)
        
        guard outputDimensions % N == 0 else {
            throw DistributedError.shardingMismatch(
                "Cannot shard output of size \(outputDimensions) across \(N) devices"
            )
        }
        
        self.weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outputDimensions / N, inputDimensions]
        )
        
        if bias {
            self.bias = MLXRandom.uniform(
                low: -scale, high: scale,
                [outputDimensions / N]
            )
        } else {
            self.bias = nil
        }
        
        super.init()
    }
    
    /// Initialize with existing weight and bias
    public init(weight: MLXArray, bias: MLXArray?, group: DistributedGroup) {
        self.weight = weight
        self.bias = bias
        self.group = group
        super.init()
    }
    
    open override func describeExtra(_ indent: Int) -> String {
        let (outDims, inDims) = weight.shape2
        let N = Int(group.size)
        let totalOutDims = outDims * N
        return "(inputDimensions=\(inDims), outputDimensions=\(totalOutDims), bias=\(bias != nil))"
    }
    
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Aggregate gradients coming from each shard
        let x = sumGradients(group: group)(x)
        
        // Compute the affine projection
        if let bias = bias {
            return addMM(bias, x, weight.T)
        } else {
            return matmul(x, weight.T)
        }
    }
    
    /// Create AllToShardedLinear from existing Linear layer
    public static func fromLinear(
        _ linearLayer: Linear,
        segments: Either<Int, [Float]> = .left(1),
        group: DistributedGroup? = nil
    ) -> AllToShardedLinear {
        let group = group ?? DistributedGroup.initialize(strict: false)
        let (outputDimensions, inputDimensions) = linearLayer.weight.shape2
        
        let shardedLinear = try! AllToShardedLinear(
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions,
            bias: linearLayer.bias != nil,
            group: group
        )
        
        let shardedParams = shard(
            parameters: linearLayer.parameters(),
            shardingPredicate: allToShardedPredicate(segments: segments),
            group: group
        )
        
        shardedLinear.update(parameters: shardedParams)
        return shardedLinear
    }
}

// MARK: - ShardedToAllLinear

/// Each member of the group applies part of the affine transformation and
/// then aggregates the results.
///
/// All nodes will have the same exact result after this layer.
open class ShardedToAllLinear: Module, UnaryLayer {
    
    public let weight: MLXArray
    public let bias: MLXArray?
    public let group: DistributedGroup
    
    /// Initialize ShardedToAllLinear layer
    public init(
        inputDimensions: Int,
        outputDimensions: Int,
        bias: Bool = true,
        group: DistributedGroup? = nil
    ) throws {
        self.group = group ?? DistributedGroup.initialize(strict: false)
        
        let scale = sqrt(1.0 / Float(inputDimensions))
        let N = Int(self.group.size)
        
        guard inputDimensions % N == 0 else {
            throw DistributedError.shardingMismatch(
                "Input of size \(inputDimensions) cannot be sharded across \(N) devices"
            )
        }
        
        self.weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outputDimensions, inputDimensions / N]
        )
        
        if bias {
            self.bias = MLXRandom.uniform(
                low: -scale, high: scale,
                [outputDimensions]
            )
        } else {
            self.bias = nil
        }
        
        super.init()
    }
    
    /// Initialize with existing weight and bias
    public init(weight: MLXArray, bias: MLXArray?, group: DistributedGroup) {
        self.weight = weight
        self.bias = bias
        self.group = group
        super.init()
    }
    
    open override func describeExtra(_ indent: Int) -> String {
        let N = Int(group.size)
        let (outDims, inDims) = weight.shape2
        let totalInDims = inDims * N
        return "(inputDimensions=\(totalInDims), outputDimensions=\(outDims), bias=\(bias != nil))"
    }
    
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = matmul(x, weight.T)
        
        // Aggregate results across all devices
        x = group.allSum(x)
        
        if let bias = bias {
            x = x + bias
        }
        
        return x
    }
    
    /// Create ShardedToAllLinear from existing Linear layer
    public static func fromLinear(
        _ linearLayer: Linear,
        segments: Either<Int, [Float]> = .left(1),
        group: DistributedGroup? = nil
    ) -> ShardedToAllLinear {
        let group = group ?? DistributedGroup.initialize(strict: false)
        let (outputDimensions, inputDimensions) = linearLayer.weight.shape2
        
        let shardedLinear = try! ShardedToAllLinear(
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions,
            bias: linearLayer.bias != nil,
            group: group
        )
        
        let shardedParams = shard(
            parameters: linearLayer.parameters(),
            shardingPredicate: shardedToAllPredicate(segments: segments),
            group: group
        )
        
        shardedLinear.update(parameters: shardedParams)
        return shardedLinear
    }
}

// MARK: - QuantizedAllToShardedLinear

/// Each member of the group applies part of the affine transformation with
/// a quantized matrix such that the result is sharded across the group.
///
/// It is the quantized equivalent of AllToShardedLinear.
open class QuantizedAllToShardedLinear: Module, UnaryLayer, Quantized {
    
    public let weight: MLXArray
    public let bias: MLXArray?
    public let scales: MLXArray
    public let biases: MLXArray?
    public let group: DistributedGroup
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    
    /// Initialize QuantizedAllToShardedLinear layer
    public init(
        inputDimensions: Int,
        outputDimensions: Int,
        bias: Bool = true,
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine,
        group: DistributedGroup? = nil
    ) throws {
        self.group = group ?? DistributedGroup.initialize(strict: false)
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        
        let scale = sqrt(1.0 / Float(inputDimensions))
        let N = Int(self.group.size)
        
        guard outputDimensions % N == 0 else {
            throw DistributedError.shardingMismatch(
                "Cannot shard output of size \(outputDimensions) across \(N) devices"
            )
        }
        
        let weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outputDimensions / N, inputDimensions]
        )
        
        let (quantizedWeight, scales, biases) = quantized(
            weight, groupSize: groupSize, bits: bits, mode: mode
        )
        
        self.weight = quantizedWeight
        self.scales = scales
        self.biases = biases
        
        if bias {
            self.bias = MLXArray.zeros([outputDimensions / N])
        } else {
            self.bias = nil
        }
        
        super.init()
        self.freeze()
    }
    
    /// Initialize with existing quantized parameters
    public init(
        weight: MLXArray, bias: MLXArray?, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode, group: DistributedGroup
    ) {
        self.weight = weight
        self.bias = bias
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.group = group
        super.init()
        self.freeze()
    }
    
    public override func unfreeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        try super.unfreeze(recursive: recursive, keys: keys, strict: strict)
        self.freeze(recursive: false)
    }
    
    open override func describeExtra(_ indent: Int) -> String {
        let (outDims, inDims) = weight.shape2
        let actualInDims = inDims * 32 / bits
        let N = Int(group.size)
        let totalOutDims = outDims * N
        return "(inputDimensions=\(actualInDims), outputDimensions=\(totalOutDims), bias=\(bias != nil), groupSize=\(groupSize), bits=\(bits))"
    }
    
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Aggregate gradients coming from each shard
        let x = sumGradients(group: group)(x)
        
        var result = quantizedMatmul(
            x, weight,
            scales: scales, biases: biases,
            transpose: true,
            groupSize: groupSize, bits: bits, mode: mode
        )
        
        if let bias = bias {
            result = result + bias
        }
        
        return result
    }
    
    /// Create QuantizedAllToShardedLinear from existing QuantizedLinear layer
    public static func fromQuantizedLinear(
        _ quantizedLinear: QuantizedLinear,
        segments: Either<Int, [Float]> = .left(1),
        group: DistributedGroup? = nil
    ) -> QuantizedAllToShardedLinear {
        let group = group ?? DistributedGroup.initialize(strict: false)
        let (outputDimensions, inputDimensions) = quantizedLinear.weight.shape2
        let actualInputDimensions = inputDimensions * 32 / quantizedLinear.bits
        
        let shardedLinear = try! QuantizedAllToShardedLinear(
            inputDimensions: actualInputDimensions,
            outputDimensions: outputDimensions,
            bias: quantizedLinear.bias != nil,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits,
            mode: quantizedLinear.mode,
            group: group
        )
        
        let shardedParams = shard(
            parameters: quantizedLinear.parameters(),
            shardingPredicate: allToShardedPredicate(segments: segments),
            group: group
        )
        
        shardedLinear.update(parameters: shardedParams)
        return shardedLinear
    }
}

// MARK: - QuantizedShardedToAllLinear

/// Each member of the group applies part of the affine transformation using
/// the quantized matrix and then aggregates the results.
///
/// All nodes will have the same exact result after this layer.
open class QuantizedShardedToAllLinear: Module, UnaryLayer, Quantized {
    
    public let weight: MLXArray
    public let bias: MLXArray?
    public let scales: MLXArray
    public let biases: MLXArray?
    public let group: DistributedGroup
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    
    /// Initialize QuantizedShardedToAllLinear layer
    public init(
        inputDimensions: Int,
        outputDimensions: Int,
        bias: Bool = true,
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine,
        group: DistributedGroup? = nil
    ) throws {
        self.group = group ?? DistributedGroup.initialize(strict: false)
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        
        let scale = sqrt(1.0 / Float(inputDimensions))
        let N = Int(self.group.size)
        
        guard inputDimensions % N == 0 else {
            throw DistributedError.shardingMismatch(
                "Input of size \(inputDimensions) cannot be sharded across \(N) devices"
            )
        }
        
        let weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outputDimensions, inputDimensions / N]
        )
        
        let (quantizedWeight, scales, biases) = quantized(
            weight, groupSize: groupSize, bits: bits, mode: mode
        )
        
        self.weight = quantizedWeight
        self.scales = scales
        self.biases = biases
        
        if bias {
            self.bias = MLXArray.zeros([outputDimensions])
        } else {
            self.bias = nil
        }
        
        super.init()
        self.freeze()
    }
    
    /// Initialize with existing quantized parameters
    public init(
        weight: MLXArray, bias: MLXArray?, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode, group: DistributedGroup
    ) {
        self.weight = weight
        self.bias = bias
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.group = group
        super.init()
        self.freeze()
    }
    
    public override func unfreeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        try super.unfreeze(recursive: recursive, keys: keys, strict: strict)
        self.freeze(recursive: false)
    }
    
    open override func describeExtra(_ indent: Int) -> String {
        let (outDims, inDims) = weight.shape2
        let actualInDims = inDims * (32 / bits) * Int(group.size)
        return "(inputDimensions=\(actualInDims), outputDimensions=\(outDims), bias=\(bias != nil), groupSize=\(groupSize), bits=\(bits))"
    }
    
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = quantizedMatmul(
            x, weight,
            scales: scales, biases: biases,
            transpose: true,
            groupSize: groupSize, bits: bits, mode: mode
        )
        
        // Aggregate results across all devices
        result = group.allSum(result)
        
        if let bias = bias {
            result = result + bias
        }
        
        return result
    }
    
    /// Create QuantizedShardedToAllLinear from existing QuantizedLinear layer
    public static func fromQuantizedLinear(
        _ quantizedLinear: QuantizedLinear,
        segments: Either<Int, [Float]> = .left(1),
        group: DistributedGroup? = nil
    ) -> QuantizedShardedToAllLinear {
        let group = group ?? DistributedGroup.initialize(strict: false)
        let (outputDimensions, inputDimensions) = quantizedLinear.weight.shape2
        let actualInputDimensions = inputDimensions * 32 / quantizedLinear.bits
        
        let shardedLinear = try! QuantizedShardedToAllLinear(
            inputDimensions: actualInputDimensions,
            outputDimensions: outputDimensions,
            bias: quantizedLinear.bias != nil,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits,
            mode: quantizedLinear.mode,
            group: group
        )
        
        let shardedParams = shard(
            parameters: quantizedLinear.parameters(),
            shardingPredicate: shardedToAllPredicate(segments: segments),
            group: group
        )
        
        shardedLinear.update(parameters: shardedParams)
        return shardedLinear
    }
}
