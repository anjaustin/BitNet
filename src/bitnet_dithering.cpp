#include "bitnet_dithering.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <immintrin.h>

// Global dithering configuration
static BitNetDitheringConfig g_dithering_config;
static BitNetDitheringMetrics g_dithering_metrics;
static bool g_dithering_initialized = false;

// Optimized 4x4 Bayer matrix for ordered dithering
alignas(64) constexpr float BAYER_4X4[16] = {
    0.0f / 16.0f, 8.0f / 16.0f, 2.0f / 16.0f, 10.0f / 16.0f,
    12.0f / 16.0f, 4.0f / 16.0f, 14.0f / 16.0f, 6.0f / 16.0f,
    3.0f / 16.0f, 11.0f / 16.0f, 1.0f / 16.0f, 9.0f / 16.0f,
    15.0f / 16.0f, 7.0f / 16.0f, 13.0f / 16.0f, 5.0f / 16.0f};

// Enhanced 8x8 Bayer matrix for higher resolution
alignas(64) constexpr float BAYER_8X8[64] = {
    0.0f / 64.0f, 32.0f / 64.0f, 8.0f / 64.0f, 40.0f / 64.0f, 2.0f / 64.0f, 34.0f / 64.0f, 10.0f / 64.0f, 42.0f / 64.0f,
    48.0f / 64.0f, 16.0f / 64.0f, 56.0f / 64.0f, 24.0f / 64.0f, 50.0f / 64.0f, 18.0f / 64.0f, 58.0f / 64.0f, 26.0f / 64.0f,
    12.0f / 64.0f, 44.0f / 64.0f, 4.0f / 64.0f, 36.0f / 64.0f, 14.0f / 64.0f, 46.0f / 64.0f, 6.0f / 64.0f, 38.0f / 64.0f,
    60.0f / 64.0f, 28.0f / 64.0f, 52.0f / 64.0f, 20.0f / 64.0f, 62.0f / 64.0f, 30.0f / 64.0f, 54.0f / 64.0f, 22.0f / 64.0f,
    3.0f / 64.0f, 35.0f / 64.0f, 11.0f / 64.0f, 43.0f / 64.0f, 1.0f / 64.0f, 33.0f / 64.0f, 9.0f / 64.0f, 41.0f / 64.0f,
    49.0f / 64.0f, 17.0f / 64.0f, 57.0f / 64.0f, 25.0f / 64.0f, 51.0f / 64.0f, 19.0f / 64.0f, 59.0f / 64.0f, 27.0f / 64.0f,
    15.0f / 64.0f, 47.0f / 64.0f, 7.0f / 64.0f, 39.0f / 64.0f, 13.0f / 64.0f, 45.0f / 64.0f, 5.0f / 64.0f, 37.0f / 64.0f,
    63.0f / 64.0f, 31.0f / 64.0f, 55.0f / 64.0f, 23.0f / 64.0f, 61.0f / 64.0f, 29.0f / 64.0f, 53.0f / 64.0f, 21.0f / 64.0f};

// Initialize BitNet dithering system
void bitnet_dithering_init()
{
    if (g_dithering_initialized)
        return;

    // Set default configuration
    g_dithering_config.enable_dithering = true;
    g_dithering_config.dithering_strength = 0.1f;
    g_dithering_config.bayer_matrix_size = 4;
    g_dithering_config.adaptive_strength = true;
    g_dithering_config.resolution_enhancement = true;

    // Initialize metrics
    g_dithering_metrics.inference_speed_ratio = 1.0f;
    g_dithering_metrics.quality_improvement_ratio = 0.0f;
    g_dithering_metrics.memory_overhead = 0.0f;
    g_dithering_metrics.perplexity_improvement = 0.0f;

    g_dithering_initialized = true;
}

// Cleanup BitNet dithering system
void bitnet_dithering_cleanup()
{
    g_dithering_initialized = false;
}

// Calculate content complexity for adaptive dithering
float bitnet_calculate_content_complexity(const float *weights, int size)
{
    if (size <= 0)
        return 0.0f;

    // Calculate variance and entropy
    float mean = 0.0f;
    float variance = 0.0f;

    // Calculate mean - use simple loop for compatibility
    for (int i = 0; i < size; ++i)
    {
        mean += weights[i];
    }
    mean /= size;

    // Calculate variance
    for (int i = 0; i < size; ++i)
    {
        float diff = weights[i] - mean;
        variance += diff * diff;
    }

    variance /= size;

    // Calculate entropy approximation
    const int bins = 32;
    int histogram[bins] = {0};
    float min_val = weights[0];
    float max_val = weights[0];

    for (int i = 1; i < size; ++i)
    {
        min_val = std::min(min_val, weights[i]);
        max_val = std::max(max_val, weights[i]);
    }

    float range = max_val - min_val;
    if (range > 0.0f)
    {
        for (int i = 0; i < size; ++i)
        {
            int bin = static_cast<int>((weights[i] - min_val) / range * (bins - 1));
            bin = std::max(0, std::min(bins - 1, bin));
            histogram[bin]++;
        }
    }

    float entropy = 0.0f;
    for (int i = 0; i < bins; ++i)
    {
        if (histogram[i] > 0)
        {
            float p = static_cast<float>(histogram[i]) / size;
            entropy -= p * std::log2(p);
        }
    }

    // Combine variance and entropy for complexity measure
    return variance * 0.6f + entropy * 0.4f;
}

// Determine if dithering should be applied based on content
bool bitnet_should_apply_dithering(const float *weights, int size)
{
    if (!g_dithering_initialized || !g_dithering_config.enable_dithering)
    {
        return false;
    }

    // Calculate content complexity
    float complexity = bitnet_calculate_content_complexity(weights, size);

    // Apply dithering for moderate to high complexity content
    return complexity > 0.02f;
}

// Calculate adaptive dithering strength based on content
float bitnet_calculate_adaptive_strength(const float *weights, int size)
{
    if (!g_dithering_config.adaptive_strength)
    {
        return g_dithering_config.dithering_strength;
    }

    float complexity = bitnet_calculate_content_complexity(weights, size);

    // Map complexity to adaptive strength
    // Low complexity: reduce strength to avoid over-dithering
    // High complexity: increase strength for better quality
    float base_strength = g_dithering_config.dithering_strength;
    float adaptive_factor = 1.0f + (complexity - 0.1f) * 2.0f;

    // Clamp adaptive factor to reasonable bounds
    adaptive_factor = std::max(0.5f, std::min(2.0f, adaptive_factor));

    return base_strength * adaptive_factor;
}

// Apply ordered dithering using Bayer matrix
void bitnet_ordered_dither(float *data, int size, const float *bayer_matrix, int matrix_size, float strength)
{
    int matrix_area = matrix_size * matrix_size;

    for (int i = 0; i < size; ++i)
    {
        // Calculate Bayer matrix index with wrapping
        int row = (i / matrix_size) % matrix_size;
        int col = i % matrix_size;
        int bayer_idx = row * matrix_size + col;

        // Apply ordered dithering
        float noise = (bayer_matrix[bayer_idx] - 0.5f) * strength;
        data[i] += noise;
    }
}

// Apply ordered dithering for resolution enhancement
void bitnet_apply_ordered_dithering(float *weights, int size, int layer_idx, const BitNetDitheringConfig *config)
{
    if (!config || !config->enable_dithering)
        return;

    // Use adaptive strength if enabled
    float strength = config->adaptive_strength ? bitnet_calculate_adaptive_strength(weights, size) : config->dithering_strength;

    // Choose Bayer matrix based on configuration
    const float *bayer_matrix;
    int matrix_size;

    if (config->bayer_matrix_size == 8)
    {
        bayer_matrix = BAYER_8X8;
        matrix_size = 8;
    }
    else
    {
        bayer_matrix = BAYER_4X4;
        matrix_size = 4;
    }

    // Apply ordered dithering
    bitnet_ordered_dither(weights, size, bayer_matrix, matrix_size, strength);
}

// Apply resolution enhancement dithering
void bitnet_apply_resolution_dithering(float *weights, int size, int layer_idx, float scale)
{
    if (!g_dithering_initialized || !g_dithering_config.resolution_enhancement)
        return;

    // Create a configuration for resolution enhancement
    BitNetDitheringConfig config = g_dithering_config;
    config.dithering_strength = scale;
    config.bayer_matrix_size = 8; // Use 8x8 for higher resolution

    bitnet_apply_ordered_dithering(weights, size, layer_idx, &config);
}

// Enhanced resolution dithering for inference quality improvement
void bitnet_enhance_resolution_dithering(float *activations, int size, int sequence_length, int hidden_size)
{
    if (!g_dithering_initialized || !g_dithering_config.resolution_enhancement)
        return;

    // Apply different dithering strategies based on the layer type
    // For attention layers, use finer dithering
    // For feed-forward layers, use coarser dithering

    int elements_per_token = hidden_size;
    int total_tokens = size / elements_per_token;

    if (total_tokens == sequence_length)
    {
        // This looks like an attention layer - use fine dithering
        BitNetDitheringConfig config = g_dithering_config;
        config.dithering_strength = 0.05f; // Fine dithering
        config.bayer_matrix_size = 8;

        for (int i = 0; i < size; i += elements_per_token)
        {
            bitnet_apply_ordered_dithering(&activations[i], elements_per_token, 0, &config);
        }
    }
    else
    {
        // Use standard dithering for other layers
        bitnet_apply_ordered_dithering(activations, size, 0, &g_dithering_config);
    }
}

// Set dithering configuration
void bitnet_set_dithering_config(const BitNetDitheringConfig *config)
{
    if (config)
    {
        g_dithering_config = *config;
    }
}

// Get current dithering configuration
BitNetDitheringConfig bitnet_get_dithering_config()
{
    return g_dithering_config;
}

// Get dithering performance metrics
BitNetDitheringMetrics bitnet_get_dithering_metrics()
{
    return g_dithering_metrics;
}