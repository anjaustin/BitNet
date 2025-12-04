#pragma once

#include <cstdint>
#include <cstddef>

// BitNet dithering configuration
struct BitNetDitheringConfig
{
    bool enable_dithering = true;       // Enable/disable dithering
    float dithering_strength = 0.1f;    // Dithering noise strength
    int bayer_matrix_size = 4;          // 4x4 Bayer matrix
    bool adaptive_strength = true;      // Adapt strength based on content
    bool resolution_enhancement = true; // Enable resolution enhancement
};

// Performance metrics for dithering
struct BitNetDitheringMetrics
{
    float inference_speed_ratio;     // Speed vs baseline (1.0 = same)
    float quality_improvement_ratio; // Quality improvement (0.0 = no improvement)
    float memory_overhead;           // Memory overhead (0.0 = no overhead)
    float perplexity_improvement;    // Perplexity improvement (negative is better)
};

// Main API functions

// Initialize BitNet dithering system
void bitnet_dithering_init();

// Cleanup BitNet dithering system
void bitnet_dithering_cleanup();

// Apply ordered dithering to weights before quantization
void bitnet_apply_ordered_dithering(float *weights, int size, int layer_idx, const BitNetDitheringConfig *config);

// Apply resolution enhancement dithering
void bitnet_apply_resolution_dithering(float *weights, int size, int layer_idx, float scale);

// Get dithering performance metrics
BitNetDitheringMetrics bitnet_get_dithering_metrics();

// Set dithering configuration
void bitnet_set_dithering_config(const BitNetDitheringConfig *config);

// Get current dithering configuration
BitNetDitheringConfig bitnet_get_dithering_config();

// Assess if dithering should be applied based on content
bool bitnet_should_apply_dithering(const float *weights, int size);

// Apply adaptive dithering strength based on content complexity
float bitnet_calculate_adaptive_strength(const float *weights, int size);

// Create optimized Bayer matrix for ordered dithering
void bitnet_create_bayer_matrix(float *matrix, int size);

// Apply ordered dithering using Bayer matrix
void bitnet_ordered_dither(float *data, int size, const float *bayer_matrix, int matrix_size, float strength);

// Enhanced resolution dithering for inference quality improvement
void bitnet_enhance_resolution_dithering(float *activations, int size, int sequence_length, int hidden_size);