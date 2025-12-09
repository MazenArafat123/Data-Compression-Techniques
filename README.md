# Data Compression Algorithms Library

A comprehensive Python library implementing various lossless and lossy compression techniques for text and image data. This project provides implementations of classical compression algorithms along with quantization methods for image processing.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Algorithms Implemented](#algorithms-implemented)
- [Usage Examples](#usage-examples)
- [Compression Metrics](#compression-metrics)
- [Project Structure](#project-structure)
- [Requirements](#requirements)


## Overview

This library provides implementations of both lossless and lossy compression algorithms, designed for educational purposes and practical applications. The project includes comprehensive metrics calculation to evaluate compression performance across different techniques.

## Features

- Multiple lossless compression algorithms (RLE, Huffman, LZW, Golomb)
- Lossy compression through quantization (Uniform and Non-uniform)
- Comprehensive metrics calculation (entropy, compression ratio, MSE, efficiency)
- File I/O support for all compression methods
- Support for both grayscale and RGB images
- Support for text in multiple languages including Unicode characters

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/compression-algorithms.git
cd compression-algorithms

# Install required dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy
scipy
Pillow
```

## Algorithms Implemented

### Lossless Compression

#### 1. Run-Length Encoding (RLE)

RLE is a simple compression algorithm that replaces sequences of repeated characters with a count and the character itself.

**How it works:**
- Scans through the data sequentially
- Counts consecutive occurrences of the same symbol
- Encodes each run as `count|character`
- Particularly effective for data with long runs of repeated values

**Best use cases:**
- Images with large uniform areas
- Simple graphics and icons
- Data with repetitive patterns

**Example:**
```
Input:  "AAABBBCCC"
Output: "3|A;3|B;3|C"
```

#### 2. Huffman Coding

Huffman coding is an optimal prefix-free encoding scheme that assigns variable-length codes to characters based on their frequency.

**How it works:**
- Builds a frequency table of all characters
- Constructs a binary tree where frequent characters have shorter paths
- Assigns binary codes based on tree traversal
- More frequent characters receive shorter codes

**Best use cases:**
- Text compression
- Data with non-uniform symbol distribution
- File compression utilities

**Key properties:**
- Optimal for symbol-by-symbol encoding
- No code is a prefix of another (prefix-free property)
- Achieves compression ratio close to entropy limit

#### 3. Lempel-Ziv-Welch (LZW)

LZW is a dictionary-based compression algorithm that builds a dictionary of patterns encountered in the data.

**How it works:**
- Starts with a dictionary of single characters
- Reads input and looks for longest matching pattern
- Outputs code for matched pattern
- Adds new pattern to dictionary
- Dictionary grows dynamically during compression

**Best use cases:**
- Text files
- GIF image format
- Data with repeating patterns or phrases

**Key properties:**
- Adapts to input data characteristics
- Does not require transmitting the dictionary
- Effective for files with repeated substrings

#### 4. Golomb Coding

Golomb coding is optimal for encoding non-negative integers following geometric distributions.

**How it works:**
- Divides input value n by parameter m to get quotient q and remainder r
- Encodes quotient in unary (q ones followed by zero)
- Encodes remainder in truncated binary
- Uses differential encoding for text (encodes differences between consecutive characters)

**Best use cases:**
- Data following geometric or exponential distributions
- Residual coding in image/video compression
- Delta encoding applications

**Parameters:**
- m: Golomb parameter (affects encoding efficiency)
- Optimal m depends on data distribution

### Lossy Compression

#### 5. Uniform Quantization

Uniform quantization divides the value range into equal-sized intervals.

**How it works:**
- Divides the range [0, 255] into equal bins of size `step`
- Maps each pixel value to its bin index
- Reconstructs using the midpoint of each bin
- Can be applied per-channel for RGB images

**Parameters:**
- step: Size of quantization interval (larger = more compression, lower quality)

**Best use cases:**
- Images with uniform intensity distribution
- Fast compression when simplicity is preferred
- Baseline for comparing other quantization methods

**Trade-offs:**
- Simple and fast computation
- May not be optimal for images with non-uniform distributions
- Higher MSE compared to non-uniform methods for same bit rate

#### 6. Non-uniform Quantization

Non-uniform quantization adapts bin sizes to the data distribution using clustering.

**How it works:**
- Uses Lloyd-Max algorithm (k-means variant)
- Iteratively finds optimal quantization levels (centroids)
- Starts with splitting centroids until desired number of levels reached
- Assigns pixels to nearest centroid
- Updates centroids based on assigned pixels
- Converges to locally optimal solution

**Parameters:**
- bits: Number of bits per pixel (determines number of levels: 2^bits)
- epsilon: Perturbation factor for centroid splitting
- max_iter: Maximum iterations for convergence

**Best use cases:**
- Images with non-uniform intensity distributions
- Applications requiring better quality at same bit rate
- Scenarios where computational cost is acceptable

**Trade-offs:**
- Better quality than uniform quantization
- Higher computational complexity
- Requires storing quantization table

## Usage Examples

### Text Compression with RLE

```python
from compression import RLECompression

# Initialize compressor
rle = RLECompression()

# Compress a file
compressed_path = rle.compress_file("input.txt", "output.rle")

# Decompress
decompressed_path = rle.decompress_file("output.rle", "restored.txt")
```

### Text Compression with Huffman

```python
from compression import HuffmanCompression

# Initialize compressor
huffman = HuffmanCompression()

# Compress a file
compressed_path = huffman.compress_file("input.txt", "output.huf")

# Decompress
decompressed_path = huffman.decompress_file("output.huf", "restored.txt")
```

### Text Compression with LZW

```python
from compression import LZWCompression

# Initialize compressor
lzw = LZWCompression()

# Compress a file
compressed_path = lzw.compress_file("input.txt", "output.lzw")

# Decompress
decompressed_path = lzw.decompress_file("output.lzw", "restored.txt")
```

### Text Compression with Golomb

```python
from compression import GolombCompression

# Initialize with Golomb parameter
golomb = GolombCompression(m=4)

# Compress a file
compressed_path = golomb.compress_file("input.txt", "output.golomb")

# Decompress
decompressed_path = golomb.decompress_file("output.golomb", "restored.txt")
```

### Image Quantization (Uniform)

```python
from compression import Quantization
import numpy as np
from PIL import Image

# Initialize quantizer
quant = Quantization()

# Load image
image = np.array(Image.open("input.jpg"))

# Quantize with step size 32
quantized, table = quant.uniform_quantize_rgb(image, step=32)

# Dequantize to reconstruct
reconstructed = quant.uniform_dequantize_rgb(table)

# Save result
Image.fromarray(reconstructed).save("output.jpg")
```

### Image Quantization (Non-uniform)

```python
from compression import Quantization
import numpy as np
from PIL import Image

# Initialize quantizer
quant = Quantization()

# Load image
image = np.array(Image.open("input.jpg"))

# Quantize with 4 bits per channel
quantized, tables = quant.nonuniform_quantize_rgb(image, bits=4)

# Dequantize to reconstruct
reconstructed = quant.nonuniform_dequantize_rgb(tables)

# Save result
Image.fromarray(reconstructed).save("output.jpg")
```

## Compression Metrics

The library includes comprehensive metrics for evaluating compression performance:

### Lossless Compression Metrics

```python
from compression import CompressionMetrics

metrics = CompressionMetrics()

# Calculate entropy
entropy = metrics.calculate_entropy(data)

# Calculate average code length
avg_length = metrics.calculate_avg_code_length(codes, data)

# Calculate compression ratio
ratio = metrics.calculate_compression_ratio(original_size, compressed_size)

# Calculate efficiency
efficiency = metrics.calculate_compression_efficiency(entropy, avg_length)

# Get all text metrics
all_metrics = metrics.get_text_metrics(text, codes)
metrics.print_metrics(all_metrics, "Huffman")
```

### Metrics Explained

- **Entropy**: Theoretical minimum average code length (in bits per symbol)
  - Formula: H = -Σ(p(x) * log₂(p(x)))
  - Lower entropy indicates more predictable data

- **Average Code Length**: Actual average bits used per symbol
  - Formula: L = Σ(p(x) * length(code(x)))
  - Should approach entropy for optimal compression

- **Compression Ratio**: Original size divided by compressed size
  - Ratio > 1 indicates compression
  - Ratio < 1 indicates expansion

- **Efficiency**: Ratio of entropy to average code length
  - Values close to 1.0 indicate near-optimal compression
  - Formula: Efficiency = H / L

### Lossy Compression Metrics

```python
# Calculate MSE
mse = metrics.calculate_mse(original_image, reconstructed_image)

# Get all image metrics
image_metrics = metrics.get_image_metrics(original_image, reconstructed_image)
metrics.print_metrics(image_metrics, "Quantization")
```

### Image Quality Metrics

- **Mean Squared Error (MSE)**: Average squared difference between original and compressed
  - Formula: MSE = (1/N) * Σ(original - compressed)²
  - Lower values indicate better quality
  - MSE = 0 means perfect reconstruction

## Project Structure

```
compression-algorithms/
│
├── compression.py          # Main implementation file
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── examples/
│   ├── text_compression.py
│   └── image_compression.py
│
└── tests/
    ├── test_rle.py
    ├── test_huffman.py
    ├── test_lzw.py
    ├── test_golomb.py
    └── test_quantization.py
```

## Algorithm Comparison

| Algorithm | Type | Best For | Compression Ratio | Speed |
|-----------|------|----------|-------------------|-------|
| RLE | Lossless | Repeated patterns | Variable | Very Fast |
| Huffman | Lossless | Variable frequency data | Good | Fast |
| LZW | Lossless | Repeated substrings | Very Good | Moderate |
| Golomb | Lossless | Geometric distributions | Good | Fast |
| Uniform Quantization | Lossy | Fast image compression | High | Very Fast |
| Non-uniform Quantization | Lossy | Quality-focused compression | High | Moderate |

## Performance Considerations

- **RLE**: O(n) time complexity, best for data with long runs
- **Huffman**: O(n log n) build time, O(n) encode/decode time
- **LZW**: O(n) time complexity, memory usage grows with dictionary
- **Golomb**: O(n) time complexity, optimal for specific distributions
- **Quantization**: O(n) for uniform, O(n * k * iterations) for non-uniform

## Future Enhancements

- Arithmetic coding implementation
- JPEG-style DCT-based compression
- Adaptive Golomb coding with automatic parameter selection
- Parallel processing support for large files
- Additional image formats support
- Compression benchmarking suite


---

**Note**: This implementation is designed for educational purposes and may not be optimized for production use in all scenarios.
