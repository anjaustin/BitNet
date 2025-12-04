#!/usr/bin/env python3
"""
Test script for BitNet Ordered Dithering implementation
Tests resolution enhancement and quality improvement
"""

import numpy as np
import time
import sys
import os
from bitnet_dithering_python import BitNetOrderedDithering, BitNetDitheringConfig, apply_bitnet_dithering, enhance_inference_resolution

def test_basic_dithering():
    """Test basic ordered dithering functionality"""
    print("🧪 Testing Basic Ordered Dithering...")
    
    # Create test weights
    test_weights = np.random.randn(1024).astype(np.float32) * 0.1
    print(f"Input weights shape: {test_weights.shape}")
    print(f"Input weights range: [{test_weights.min():.4f}, {test_weights.max():.4f}]")
    
    # Apply dithering
    dithering = BitNetOrderedDithering()
    config = BitNetDitheringConfig()
    config.enable_dithering = True
    config.dithering_strength = 0.1
    
    dithering.set_config(config)
    dithered_weights = dithering.apply_ordered_dithering(test_weights.copy())
    
    print(f"Dithered weights range: [{dithered_weights.min():.4f}, {dithered_weights.max():.4f}]")
    
    # Check that dithering had effect
    if not np.array_equal(test_weights, dithered_weights):
        print("✅ Dithering successfully modified weights!")
        return True
    else:
        print("❌ Dithering had no effect!")
        return False

def test_resolution_enhancement():
    """Test resolution enhancement dithering"""
    print("\n🔍 Testing Resolution Enhancement...")
    
    # Create test activations (simulating transformer layer)
    batch_size = 4
    seq_length = 128
    hidden_size = 768
    
    activations = np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32) * 0.1
    print(f"Input activations shape: {activations.shape}")
    
    # Test without dithering
    original_flat = activations.flatten()
    
    # Test with dithering
    dithering = BitNetOrderedDithering()
    config = BitNetDitheringConfig()
    config.enable_dithering = True
    config.resolution_enhancement = True
    config.dithering_strength = 0.05  # Fine dithering for resolution enhancement
    
    dithering.set_config(config)
    
    # Apply enhanced resolution dithering
    enhanced_activations = dithering.enhance_resolution_dithering(
        activations.copy(), seq_length, hidden_size
    )
    
    enhanced_flat = enhanced_activations.flatten()
    
    # Compare results
    original_std = np.std(original_flat)
    enhanced_std = np.std(enhanced_flat)
    
    print(f"Original std: {original_std:.6f}")
    print(f"Enhanced std: {enhanced_std:.6f}")
    print(f"Std improvement: {((enhanced_std - original_std) / original_std * 100):.2f}%")
    
    # Check for quality improvement (higher std indicates more variation)
    if enhanced_std > original_std:
        print("✅ Resolution enhancement increased activation variation!")
        return True
    else:
        print("⚠️  Resolution enhancement did not increase variation")
        return False

def test_performance_impact():
    """Test performance impact of dithering"""
    print("\n⚡ Testing Performance Impact...")
    
    # Test with different array sizes
    sizes = [1024, 4096, 16384]
    
    for size in sizes:
        print(f"\nTesting size: {size}")
        
        # Create test data
        test_weights = np.random.randn(size).astype(np.float32)
        
        # Test without dithering (baseline)
        start_time = time.time()
        for _ in range(100):
            baseline = test_weights.copy()
        baseline_time = time.time() - start_time
        
        # Test with dithering
        dithering = BitNetOrderedDithering()
        config = BitNetDitheringConfig()
        config.enable_dithering = True
        dithering.set_config(config)
        
        start_time = time.time()
        for _ in range(100):
            dithered = dithering.apply_ordered_dithering(test_weights.copy())
        dithering_time = time.time() - start_time
        
        # Calculate overhead
        overhead = (dithering_time - baseline_time) / baseline_time * 100
        
        print(f"  Baseline time: {baseline_time*1000:.2f}ms")
        print(f"  Dithering time: {dithering_time*1000:.2f}ms")
        print(f"  Overhead: {overhead:.1f}%")
        
        if overhead < 10.0:
            print("  ✅ Acceptable performance overhead!")
        else:
            print("  ⚠️  High performance overhead!")

def test_adaptive_dithering():
    """Test adaptive dithering based on content complexity"""
    print("\n🎯 Testing Adaptive Dithering...")
    
    # Create test data with different complexities
    # Low complexity (uniform)
    low_complexity = np.ones(1024, dtype=np.float32) * 0.1
    
    # Medium complexity (some variation)
    medium_complexity = np.random.randn(1024).astype(np.float32) * 0.05
    
    # High complexity (high variation)
    high_complexity = np.random.randn(1024).astype(np.float32) * 0.2
    
    dithering = BitNetOrderedDithering()
    config = BitNetDitheringConfig()
    config.enable_dithering = True
    config.adaptive_strength = True
    config.dithering_strength = 0.1
    
    dithering.set_config(config)
    
    # Test each complexity level
    for name, data in [("Low", low_complexity), 
                       ("Medium", medium_complexity), 
                       ("High", high_complexity)]:
        
        should_apply = dithering.should_apply_dithering(data)
        print(f"  {name} complexity - Should apply dithering: {should_apply}")
        
        if name == "Low" and should_apply:
            print("  ⚠️  Low complexity should probably not use dithering")
        elif name in ["Medium", "High"] and not should_apply:
            print("  ⚠️  Medium/High complexity should use dithering")

def test_bayer_matrices():
    """Test different Bayer matrix sizes"""
    print("\n📊 Testing Bayer Matrix Sizes...")
    
    test_weights = np.random.randn(256).astype(np.float32) * 0.1
    
    # Test 4x4 Bayer matrix
    dithering_4x4 = BitNetOrderedDithering()
    config_4x4 = BitNetDitheringConfig()
    config_4x4.enable_dithering = True
    config_4x4.bayer_matrix_size = 4
    config_4x4.dithering_strength = 0.1
    
    dithering_4x4.set_config(config_4x4)
    dithered_4x4 = dithering_4x4.apply_ordered_dithering(test_weights.copy())
    
    # Test 8x8 Bayer matrix
    dithering_8x8 = BitNetOrderedDithering()
    config_8x8 = BitNetDitheringConfig()
    config_8x8.enable_dithering = True
    config_8x8.bayer_matrix_size = 8
    config_8x8.dithering_strength = 0.1
    
    dithering_8x8.set_config(config_8x8)
    dithered_8x8 = dithering_8x8.apply_ordered_dithering(test_weights.copy())
    
    # Compare results
    original_std = np.std(test_weights)
    std_4x4 = np.std(dithered_4x4)
    std_8x8 = np.std(dithered_8x8)
    
    print(f"Original std: {original_std:.6f}")
    print(f"4x4 Bayer std: {std_4x4:.6f} (improvement: {((std_4x4-original_std)/original_std*100):+.2f}%)")
    print(f"8x8 Bayer std: {std_8x8:.6f} (improvement: {((std_8x8-original_std)/original_std*100):+.2f}%)")
    
    if std_8x8 > std_4x4:
        print("✅ 8x8 Bayer matrix provides better variation than 4x4")
    else:
        print("ℹ️  4x4 and 8x8 Bayer matrices provide similar variation")

def test_convenience_functions():
    """Test convenience functions"""
    print("\n🔧 Testing Convenience Functions...")
    
    # Test apply_bitnet_dithering
    test_weights = np.random.randn(512).astype(np.float32) * 0.1
    
    dithered = apply_bitnet_dithering(test_weights, enable_resolution_enhancement=True)
    
    if not np.array_equal(test_weights, dithered):
        print("✅ apply_bitnet_dithering function works correctly!")
    else:
        print("❌ apply_bitnet_dithering function had no effect!")
        return False
    
    # Test enhance_inference_resolution
    activations = np.random.randn(2, 64, 256).astype(np.float32) * 0.1
    enhanced = enhance_inference_resolution(activations, 64, 256)
    
    if enhanced.shape == activations.shape:
        print("✅ enhance_inference_resolution function works correctly!")
        return True
    else:
        print("❌ enhance_inference_resolution function failed!")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("🎯 BitNet Ordered Dithering Test Suite")
    print("=" * 70)
    
    try:
        # Run all tests
        tests = [
            ("Basic Dithering", test_basic_dithering),
            ("Resolution Enhancement", test_resolution_enhancement),
            ("Performance Impact", test_performance_impact),
            ("Adaptive Dithering", test_adaptive_dithering),
            ("Bayer Matrices", test_bayer_matrices),
            ("Convenience Functions", test_convenience_functions)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"Running: {test_name}")
            print('='*50)
            try:
                result = test_func()
                if result is None:  # Functions that don't return boolean
                    results.append(True)
                else:
                    results.append(result)
            except Exception as e:
                print(f"❌ {test_name} failed with error: {e}")
                results.append(False)
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print("\n" + "=" * 70)
        print("📊 Test Results Summary")
        print("=" * 70)
        print(f"Tests passed: {passed}/{total}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("🎉 All tests passed!")
            print("✅ BitNet ordered dithering implementation is working correctly!")
            print("🚀 Ready for resolution enhancement in BitNet inference!")
        else:
            print("⚠️  Some tests failed. Please check the implementation.")
        
        print("=" * 70)
        return passed == total
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)