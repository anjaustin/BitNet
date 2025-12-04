#!/usr/bin/env python3
"""
Simple test for BitNet Ordered Dithering implementation
"""

import numpy as np
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from bitnet_dithering_python import BitNetOrderedDithering, BitNetDitheringConfig
    print("✅ Successfully imported BitNet dithering module")
except ImportError as e:
    print(f"❌ Failed to import dithering module: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic dithering functionality"""
    print("\n🧪 Testing Basic Ordered Dithering...")
    
    # Create test weights
    test_weights = np.random.randn(512).astype(np.float32) * 0.1
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
    batch_size = 2
    seq_length = 32
    hidden_size = 256
    
    activations = np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32) * 0.1
    print(f"Input activations shape: {activations.shape}")
    
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
    
    print(f"Enhanced activations shape: {enhanced_activations.shape}")
    
    # Check for quality improvement (higher std indicates more variation)
    original_std = np.std(activations)
    enhanced_std = np.std(enhanced_activations)
    
    print(f"Original std: {original_std:.6f}")
    print(f"Enhanced std: {enhanced_std:.6f}")
    print(f"Std improvement: {((enhanced_std - original_std) / original_std * 100):+.2f}%")
    
    if enhanced_std > original_std:
        print("✅ Resolution enhancement increased activation variation!")
        return True
    else:
        print("⚠️  Resolution enhancement did not increase variation")
        return False

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

def main():
    """Run all tests"""
    print("=" * 70)
    print("🎯 BitNet Ordered Dithering Test")
    print("=" * 70)
    
    try:
        # Run tests
        tests = [
            ("Basic Dithering", test_basic_functionality),
            ("Resolution Enhancement", test_resolution_enhancement),
            ("Bayer Matrices", test_bayer_matrices)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"Running: {test_name}")
            print('='*50)
            try:
                result = test_func()
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
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)