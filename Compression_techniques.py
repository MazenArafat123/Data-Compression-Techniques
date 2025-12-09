import math
import numpy as np
from collections import Counter
from typing import Union, Dict, List,Tuple
import os
from pathlib import Path
import heapq
import json
from scipy.spatial.distance import cdist
from PIL import Image



class CompressionMetrics:
    """
    A class for calculating compression metrics for various compression techniques.
    Supports both lossless (RLE, Huffman, Golomb) and lossy (Quantizers) methods.
    """
    
    def __init__(self):
        """Initialize the metrics calculator"""
        pass
    
    def calculate_entropy(self, data: Union[str, List, np.ndarray]) -> float:
        """
        Calculate Shannon entropy: H = -sum(pi * log2(pi))
        
        Args:
            data: Input data (text string, list of values, or numpy array)
            
        Returns:
            Entropy value in bits
        """
        # Convert numpy array to list if needed
        if isinstance(data, np.ndarray):
            data = data.flatten().tolist()
        
        # Handle string or list
        if isinstance(data, str):
            freq = Counter(data)
        else:
            freq = Counter(data)
        
        total = len(data) if len(data) > 0 else 1
        
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def calculate_avg_code_length(self, codes: Dict, data: Union[str, List]) -> float:
        """
        Calculate average code length: Lav = sum(pi * Li)
        For lossless compression techniques (Huffman, RLE, Golomb)
        
        Args:
            codes: Dictionary mapping symbols to their codes
                   For RLE: {symbol: (symbol, count)}
                   For Huffman: {symbol: 'binary_code'}
                   For Golomb: {value: 'encoded_bits'}
            data: Original data
            
        Returns:
            Average code length in bits per symbol
        """
        if not data:
            return 0.0
        
        total_length = 0
        for symbol in data:
            code = codes.get(symbol, '')
            # Handle different code formats
            if isinstance(code, tuple):  # RLE format
                # Assume fixed bits for symbol + count representation
                total_length += len(str(code))
            elif isinstance(code, str):  # Binary string
                total_length += len(code)
            else:
                total_length += 0
        
        return total_length / len(data) if len(data) > 0 else 0.0
    
    def calculate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """
        Calculate compression ratio: CR = original_size / compressed_size
        
        Args:
            original_size: Size before compression (in bits or bytes)
            compressed_size: Size after compression (in bits or bytes)
            
        Returns:
            Compression ratio (>1 means compression, <1 means expansion)
        """
        if original_size == 0:
            return 0.0
        return (original_size / compressed_size) if compressed_size > 0 else float('inf')
    
    def calculate_mse(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """
        Calculate Mean Squared Error: MSE = (1/N) * sum((original - compressed)^2)
        For lossy compression techniques (Quantizers)
        
        Args:
            original: Original image/data as numpy array
            compressed: Compressed/reconstructed image/data as numpy array
            
        Returns:
            MSE value (lower is better, 0 means identical)
        """
        # Ensure both arrays have the same shape
        if original.shape != compressed.shape:
            raise ValueError(f"Shape mismatch: original {original.shape} vs compressed {compressed.shape}")
        
        # Convert to float to avoid overflow
        original = original.astype(np.float64)
        compressed = compressed.astype(np.float64)
        
        # Calculate MSE
        mse = np.mean((original - compressed) ** 2)
        
        return mse
    
    
    def calculate_compression_efficiency(self, entropy: float, avg_code_length: float) -> float:
        """
        Calculate compression efficiency: Efficiency = Entropy / Lav
        Perfect efficiency = 1.0 (theoretical limit)
        
        Args:
            entropy: Shannon entropy of the data
            avg_code_length: Average code length achieved
            
        Returns:
            Efficiency ratio (1.0 is optimal)
        """
        if avg_code_length == 0:
            return 0.0
        return avg_code_length/entropy  
    
    def get_text_metrics(self, text: str, codes: Dict) -> Dict:
        """
        Get all metrics for text compression (RLE, Huffman, Golomb)
        
        Args:
            text: Original text
            codes: Encoding dictionary
            
        Returns:
            Dictionary with all metrics
        """
        entropy = self.calculate_entropy(text)
        avg_length = self.calculate_avg_code_length(codes, text)
        
        # Calculate sizes
        original_size = len(text) * 8  # assuming 8 bits per character
        compressed_size = avg_length * len(text)
        ratio = self.calculate_compression_ratio(original_size, compressed_size)
        efficiency = self.calculate_compression_efficiency(entropy, avg_length)
        
        return {
            'entropy': entropy,
            'avg_code_length': avg_length,
            'original_size_bits': original_size,
            'compressed_size_bits': compressed_size,
            'compression_ratio': ratio,
            'efficiency': efficiency
        }
    
    def get_image_metrics(self, original_img: np.ndarray, compressed_img: np.ndarray, 
                         compressed_size_bits: int = None) -> Dict:
        """
        Get all metrics for image compression (Quantizers)
        
        Args:
            original_img: Original image as numpy array
            compressed_img: Compressed/reconstructed image
            compressed_size_bits: Size of compressed data in bits (optional)
            
        Returns:
            Dictionary with all metrics
        """
        mse = self.calculate_mse(original_img, compressed_img)
        
        metrics = {
            'mse': mse,
        }
        
        # Add compression ratio if size provided
        if compressed_size_bits is not None:
            original_size = original_img.size * 8  # assuming 8 bits per pixel
            ratio = self.calculate_compression_ratio(original_size, compressed_size_bits)
            metrics['compression_ratio'] = ratio
            metrics['original_size_bits'] = original_size
            metrics['compressed_size_bits'] = compressed_size_bits
        
        return metrics
    
    def print_metrics(self, metrics: Dict, technique: str = "Compression"):
        """
        Pretty print metrics
        
        Args:
            metrics: Dictionary of metrics
            technique: Name of compression technique
        """
        print(f"\n{'='*50}")
        print(f"{technique} Metrics")
        print(f"{'='*50}")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if value == float('inf'):
                    print(f"{key.replace('_', ' ').title()}: ∞")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        print(f"{'='*50}\n")


class RLECompression:
    """Run-Length Encoding compression class with file I/O support."""
    
    def __init__(self):
        self.original_text = ""
        self.encoded_text = ""
    
    @staticmethod
    def encode(data: str) -> str:
        """
        Encode data using Run-Length Encoding with delimiter
        Format: count|character (e.g., "3|A" means "AAA")
        """
        if not data:
            return ""
        
        encoded = []
        count = 1
        
        for i in range(1, len(data)):
            if data[i] == data[i - 1]:
                count += 1
            else:
                # Use delimiter to separate count from character
                encoded.append(f"{count}|{data[i - 1]}")
                count = 1
        
        # Add the last run
        encoded.append(f"{count}|{data[-1]}")
        
        return ";".join(encoded)
    
    @staticmethod
    def decode(encoded: str) -> str:
        """
        Decode RLE encoded data with delimiter
        Format: count|character separated by semicolons
        """
        if not encoded:
            return ""
        
        decoded = ""
        
        # Split by semicolon to get each run
        runs = encoded.split(";")
        
        for run in runs:
            if "|" in run:
                parts = run.split("|", 1)  # Split only on first |
                if len(parts) == 2:
                    count_str, char = parts
                    try:
                        count = int(count_str)
                        decoded += char * count
                    except ValueError:
                        # Handle malformed data
                        continue
        
        return decoded
    
    def compress_file(self, input_path: str, output_path: str = None) -> str:
        """
        Compress a text file using RLE
        
        Args:
            input_path: Path to input text file
            output_path: Path to output file (optional, auto-generates .rle)
            
        Returns:
            Path to compressed file
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            self.original_text = f.read()
        
        self.encoded_text = self.encode(self.original_text)
        
        if output_path is None:
            output_path = input_path.stem + ".rle"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.encoded_text)
        
        return str(output_path)
    
    def decompress_file(self, input_path: str, output_path: str = None) -> str:
        """
        Decompress an RLE compressed file
        
        Args:
            input_path: Path to .rle file
            output_path: Path to output file (optional)
            
        Returns:
            Path to decompressed file
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            self.encoded_text = f.read()
        
        self.original_text = self.decode(self.encoded_text)
        
        if output_path is None:
            output_path = input_path.stem + "_decompressed.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.original_text)
        
        return str(output_path)


class HuffmanCompression:
    """Huffman Encoding compression class with file I/O support."""
    
    def __init__(self):
        self.original_text = ""
        self.encoded_bits = ""
        self.codes = {}
    
    @staticmethod
    def build_probability(text: str) -> Dict[str, float]:
        """Build probability distribution for characters"""
        freq = Counter(text)
        total = len(text)
        return {char: freq[char] / total for char in freq}
    
    @staticmethod
    def build_codes(text: str) -> Dict[str, str]:
        """Build Huffman codes from text"""
        prob = HuffmanCompression.build_probability(text)
        heap = [[weight, [char, ""]] for char, weight in prob.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            smallest = heapq.heappop(heap)
            secsmallest = heapq.heappop(heap)
            
            for pair in smallest[1:]:
                pair[1] = "0" + pair[1]
            for pair in secsmallest[1:]:
                pair[1] = "1" + pair[1]
            
            heapq.heappush(heap, [smallest[0] + secsmallest[0]] + smallest[1:] + secsmallest[1:])
        
        return dict(heap[0][1:])
    
    @staticmethod
    def encode(text: str, codes: Dict[str, str]) -> str:
        """Encode text using Huffman codes"""
        return "".join(codes[ch] for ch in text)
    
    @staticmethod
    def decode(encoded_bits: str, codes: Dict[str, str]) -> str:
        """Decode Huffman encoded bits"""
        reverse_codes = {v: k for k, v in codes.items()}
        decoded = ""
        buffer = ""
        
        for bit in encoded_bits:
            buffer += bit
            if buffer in reverse_codes:
                decoded += reverse_codes[buffer]
                buffer = ""
        
        return decoded
    
    def compress_file(self, input_path: str, output_path: str = None) -> str:
        """
        Compress a text file using Huffman encoding
        
        Args:
            input_path: Path to input text file
            output_path: Path to output file (optional, auto-generates .huf)
            
        Returns:
            Path to compressed file
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            self.original_text = f.read()
        
        # Build codes and encode
        self.codes = self.build_codes(self.original_text)
        self.encoded_bits = self.encode(self.original_text, self.codes)
        
        if output_path is None:
            output_path = input_path.stem + ".huf"
        
        # Save compressed file with codes
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write codes as JSON
            f.write(json.dumps(self.codes) + '\n')
            # Write separator
            f.write('---DATA---\n')
            # Write encoded bits
            f.write(self.encoded_bits)
        
        return str(output_path)
    
    def decompress_file(self, input_path: str, output_path: str = None) -> str:
        """
        Decompress a Huffman compressed file
        
        Args:
            input_path: Path to .huf file
            output_path: Path to output file (optional)
            
        Returns:
            Path to decompressed file
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read().split('---DATA---\n')
            
            if len(content) != 2:
                raise ValueError("Invalid Huffman file format")
            
            # Load codes
            self.codes = json.loads(content[0].strip())
            # Load encoded bits
            self.encoded_bits = content[1]
        
        # Decode
        self.original_text = self.decode(self.encoded_bits, self.codes)
        
        if output_path is None:
            output_path = input_path.stem + "_decompressed.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.original_text)
        
        return str(output_path)
    


class LZWCompression:
    """LZW (Lempel-Ziv-Welch) compression class with file I/O support."""
    
    def __init__(self):
        self.original_text = ""
        self.encoded_codes = []
    
    @staticmethod
    def encode(text: str) -> List[int]:
        """Encode text using LZW algorithm"""
        dictionary = {chr(i): i for i in range(256)}
        
        next_code = 256
        current_c = ""
        result = []
        
        for next_n in text:
            combined = current_c + next_n
            if combined in dictionary:
                current_c = combined
            else:
                result.append(dictionary[current_c])
                dictionary[combined] = next_code
                next_code += 1
                current_c = next_n
        
        if current_c:
            result.append(dictionary[current_c])
        
        return result
    
    @staticmethod
    def decode(codes: List[int]) -> str:
        """Decode LZW encoded codes"""
        if not codes:
            return ""
        
        rev_dict = {i: chr(i) for i in range(256)}
        
        next_code = 256
        first_code = codes[0]
        current_string = rev_dict[first_code]
        result = current_string
        
        for code in codes[1:]:
            if code in rev_dict:
                entry = rev_dict[code]
            else:
                entry = current_string + current_string[0]
            
            result += entry
            rev_dict[next_code] = current_string + entry[0]
            next_code += 1
            current_string = entry
        
        return result
    
    def compress_file(self, input_path: str, output_path: str = None) -> str:
        """
        Compress a text file using LZW encoding
        
        Args:
            input_path: Path to input text file
            output_path: Path to output file (optional, auto-generates .lzw)
            
        Returns:
            Path to compressed file
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            self.original_text = f.read()
        
        # Encode
        self.encoded_codes = self.encode(self.original_text)
        
        if output_path is None:
            output_path = input_path.stem + ".lzw"
        
        # Save compressed file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoded_codes))
        
        return str(output_path)
    
    def decompress_file(self, input_path: str, output_path: str = None) -> str:
        """
        Decompress an LZW compressed file
        
        Args:
            input_path: Path to .lzw file
            output_path: Path to output file (optional)
            
        Returns:
            Path to decompressed file
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            self.encoded_codes = json.loads(f.read())
        
        # Decode
        self.original_text = self.decode(self.encoded_codes)
        
        if output_path is None:
            output_path = input_path.stem + "_decompressed.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.original_text)
        
        return str(output_path)
    



class GolombCompression:
    """Golomb Coding compression class for text"""
    
    def __init__(self, m: int = 4):
        """
        Initialize Golomb compression
        
        Args:
            m: Golomb parameter (default=4, optimal for geometric distributions)
        """
        self.m = m
        self.original_data = None
        self.encoded_bits = ""
        self.data_type = None  # 'text', 'image', 'integers'
    
    @staticmethod
    def encode_value(n: int, m: int) -> str:
        """Encode a single value using Golomb coding"""
        q = n // m
        r = n % m
        
        quotient_code = "1" * q + "0"
        
        if (m & (m - 1)) == 0:
            k = int(math.log2(m))
            remainder_code = format(r, f"0{k}b")
        else:
            b = math.ceil(math.log2(m))
            T = 2**b - m
            
            if r < T:
                remainder_code = format(r, f"0{b-1}b")
            else:
                remainder_code = format(r + T, f"0{b}b")
        
        return quotient_code + remainder_code
    
    @staticmethod
    def decode_value(code: str, m: int) -> Tuple[int, int]:
        """
        Decode a single value from Golomb code
        
        Returns:
            Tuple of (decoded_value, bits_consumed)
        """
        i = 0
        n = len(code)
        q = 0
        
        while i < n and code[i] == "1":
            q += 1
            i += 1
        
        if i < n and code[i] == "0":
            i += 1
        
        if (m & (m - 1)) == 0:
            k = int(math.log2(m))
            r = int(code[i:i+k], 2) if i+k <= n else 0
            consumed = i + k
        else:
            b = math.ceil(math.log2(m))
            T = 2**b - m
            
            r_val = int(code[i:i+(b-1)], 2) if i+(b-1) <= n else 0
            
            if r_val < T:
                r = r_val
                consumed = i + (b - 1)
            else:
                r_extended = int(code[i:i+b], 2) if i+b <= n else 0
                r = r_extended - T
                consumed = i + b
        
        return q * m + r, consumed
    
    def encode_text(self, text: str) -> str:
        """
        Encode text using differential encoding + Golomb
        
        Args:
            text: Input text (English, Arabic, or mixed)
            
        Returns:
            Encoded bit string
        """
        if not text:
            return ""
        
        # Convert to Unicode code points
        values = [ord(c) for c in text]
        
        # Store first value directly
        encoded = format(values[0], '032b')  # 32 bits for first character
        
        # Encode differences
        for i in range(1, len(values)):
            diff = values[i] - values[i-1]
            # Map negative differences to positive
            mapped = 2 * diff if diff >= 0 else -2 * diff - 1
            encoded += self.encode_value(mapped, self.m)
        
        return encoded
    
    def decode_text(self, encoded: str, length: int) -> str:
        """
        Decode Golomb encoded text
        
        Args:
            encoded: Encoded bit string
            length: Original text length
            
        Returns:
            Decoded text
        """
        if not encoded or length == 0:
            return ""
        
        # Decode first character
        first_val = int(encoded[:32], 2)
        values = [first_val]
        
        # Decode differences
        pos = 32
        for _ in range(length - 1):
            if pos >= len(encoded):
                break
            mapped, consumed = self.decode_value(encoded[pos:], self.m)
            pos += consumed
            
            # Unmap difference
            if mapped % 2 == 0:
                diff = mapped // 2
            else:
                diff = -(mapped + 1) // 2
            
            values.append(values[-1] + diff)
        
        # Convert back to characters
        return ''.join(chr(v) for v in values)
    
    
    def compress_file(self, input_path: str, output_path: str = None, 
                     data_type: str = 'text') -> str:
        """
        Compress a file using Golomb encoding
        
        Args:
            input_path: Path to input file
            output_path: Path to output file (optional, auto-generates .golomb)
            data_type: Type of data ('text', 'image', 'integers')
            
        Returns:
            Path to compressed file
        """
        input_path = Path(input_path)
        self.data_type = data_type
        
        if data_type == 'text':
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.original_data = text
            self.encoded_bits = self.encode_text(text)
            metadata = {'type': 'text', 'length': len(text), 'm': self.m}
            
       
        if output_path is None:
            output_path = input_path.stem + ".golomb"
        
        # Save compressed file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(metadata) + '\n')
            f.write('---DATA---\n')
            f.write(self.encoded_bits)
        
        return str(output_path)
    
    def decompress_file(self, input_path: str, output_path: str = None) -> str:
        """
        Decompress a Golomb compressed file
        
        Args:
            input_path: Path to .golomb file
            output_path: Path to output file (optional)
            
        Returns:
            Path to decompressed file
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read().split('---DATA---\n')
            
            if len(content) != 2:
                raise ValueError("Invalid Golomb file format")
            
            metadata = json.loads(content[0].strip())
            self.encoded_bits = content[1]
            self.m = metadata['m']
            self.data_type = metadata['type']
        
        if self.data_type == 'text':
            self.original_data = self.decode_text(self.encoded_bits, metadata['length'])
            if output_path is None:
                output_path = input_path.stem + "_decompressed.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.original_data)
                
        
        return str(output_path)
    


class Quantization:
    def __init__(self):
        self.original = None
        self.quantized = None
        self.reconstructed = None
        self.table = None                    

    # UNIFORM QUANTIZATION (GRAYSCALE)
    def uniform_quantize(self, image: np.ndarray, step: int):
        self.original = image.astype(np.float32)
        self.quantized = np.floor(self.original / step)

        num_levels = int(np.ceil(256 / step))
        self.table = {}

        for q in range(num_levels):
            low = q * step
            high = min((q + 1) * step - 1, 255)
            recon = low + step / 2
            self.table[q] = {"range": (low, high), "reconstruction": recon}

        return self.quantized, self.table

    def uniform_dequantize(self):
        if self.quantized is None:
            raise ValueError("Quantized image not available.")

        step = list(self.table.values())[1]["range"][0] - list(self.table.values())[0]["range"][0]
        rec = self.quantized * step + step / 2
        self.reconstructed = np.clip(rec, 0, 255).astype(np.uint8)
        return self.reconstructed

    # RGB UNIFORM
    def uniform_dequantize_rgb(self, tables):
        reconstructed_channels = []

        for i in range(3):
            table = tables[i]
            quantized_channel = self.quantized[:, :, i] if self.quantized.ndim == 3 else self.quantized
            # compute step exactly like in uniform_dequantize()
            step = list(table.values())[1]["range"][0] - list(table.values())[0]["range"][0]
            rec = quantized_channel * step + step / 2
            reconstructed_channels.append(np.clip(rec, 0, 255).astype(np.uint8))

        return np.stack(reconstructed_channels, axis=2)

    def uniform_quantize_rgb(self, image: np.ndarray, step: int):
        quantized_channels = []
        tables = []

        for c in range(3):
            q, t = self.uniform_quantize(image[:, :, c], step)
            quantized_channels.append(q)
            tables.append(t)

        self.quantized = np.stack(quantized_channels, axis=2)  
        return self.quantized, tables

    # NON-UNIFORM QUANTIZATION (GRAY)
    def nonuniform_quantize(self, image: np.ndarray, bits: int, epsilon=0.01, max_iter=50):
        self.original = image.astype(np.float32)
        x = self.original.flatten()

        L = 2 ** bits
        centroids = [np.mean(x)]

        while len(centroids) < L:
            new_centroids = []
            for c in centroids:
                new_centroids.append(c * (1 + epsilon))
                new_centroids.append(c * (1 - epsilon))
            centroids = np.array(new_centroids)

            for _ in range(max_iter):
                distances = np.abs(x[:, None] - centroids[None, :])
                labels = np.argmin(distances, axis=1)

                updated = []
                for i in range(len(centroids)):
                    cluster = x[labels == i]
                    updated.append(np.mean(cluster) if len(cluster) > 0 else centroids[i])
                updated = np.array(updated)

                if np.sum((updated - centroids) ** 2) < 1e-4:
                    break

                centroids = updated

        self.table = {i: {"reconstruction": float(centroids[i])} for i in range(L)}

        distances = np.abs(x[:, None] - centroids[None, :])
        labels = np.argmin(distances, axis=1)

        self.quantized = labels.reshape(self.original.shape)
        return self.quantized, self.table

    def nonuniform_dequantize(self):
        if self.quantized is None:
            raise ValueError("Quantized image not available.")

        rec = np.zeros_like(self.quantized, dtype=np.float32)
        for q, info in self.table.items():
            rec[self.quantized == q] = info["reconstruction"]

        self.reconstructed = np.clip(rec, 0, 255).astype(np.uint8)
        return self.reconstructed

    # FIXED RGB NON-UNIFORM 
    def nonuniform_dequantize_rgb(self, tables):
        reconstructed_channels = []

        for i in range(3):
            table = tables[i]
            quantized_channel = self.quantized[:, :, i]

            rec_channel = np.zeros_like(quantized_channel, dtype=np.float32)
            for q, info in table.items():
                rec_channel[quantized_channel == q] = info["reconstruction"]

            reconstructed_channels.append(np.clip(rec_channel, 0, 255).astype(np.uint8))

        return np.stack(reconstructed_channels, axis=2)

    def nonuniform_quantize_rgb(self, image: np.ndarray, bits: int):
        quantized_channels = []
        tables = []

        for c in range(3):
            q, t = self.nonuniform_quantize(image[:, :, c], bits)
            quantized_channels.append(q)
            tables.append(t)

        self.quantized = np.stack(quantized_channels, axis=2)
        return self.quantized, tables

    # IMAGE LOADING / SAVING 
    def _load_image(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))

    def _save_image(self, image: np.ndarray, path: str):
        Image.fromarray(image.astype(np.uint8)).save(path)