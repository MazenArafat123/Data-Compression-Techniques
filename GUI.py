import streamlit as st
import numpy as np
from PIL import Image
import os
import tempfile
from pathlib import Path
import json

# Import your classes
from Compression_techniques import (
    RLECompression,
    HuffmanCompression,
    LZWCompression,
    GolombCompression,
    Quantization,
    CompressionMetrics
)

# Page configuration
st.set_page_config(
    page_title="Data Compression Tool",
    page_icon="🗜️",
    layout="wide"
)



# Title
st.markdown('Data Compression Project', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Compression Options")

# Main radio button for compression type
compression_category = st.sidebar.radio(
    "Select Compression Category",
    ["Lossless Compression", "Lossy Compression"]
)

st.sidebar.markdown("---")

# ==========================
# LOSSLESS COMPRESSION
# ==========================
if compression_category == "Lossless Compression":
    st.header("📝 Lossless Compression")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Select Method",
        ["RLE", "Huffman", "LZW", "Golomb"]
    )
    
    # Golomb parameter if Golomb is selected
    if method == "Golomb":
        st.sidebar.markdown("### Golomb Parameters")
        m_parameter = st.sidebar.slider("m (Golomb Parameter)", min_value=2, max_value=32, value=4, step=1)
        data_type = st.sidebar.selectbox("Data Type", ["Text"])
    
    st.subheader(f"{method} Compression")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Text File", type=['txt'])

    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_input_path = tmp_file.name
        
        # Display original content
        if method != "Golomb" or (method == "Golomb" and data_type == "Text"):
            original_text = uploaded_file.getvalue().decode('utf-8')
            st.text_area("Original Text", original_text, height=200)
        
        # Create columns for compress/decompress
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Compression")
            compress_filename = st.text_input(
                "Compressed Filename", 
                value=f"compressed_{Path(uploaded_file.name).stem}",
                key="compress_name"
            )
            
            if st.button("Compress", key="compress_btn"):
                try:
                    # Initialize metrics
                    metrics = CompressionMetrics()
                    
                    # Compress based on method
                    if method == "RLE":
                        compressor = RLECompression()
                        compressed_path = compressor.compress_file(tmp_input_path, compress_filename + ".rle")
                        
                        # Calculate metrics
                        text = original_text
                        codes = {ch: compressor.encode(ch) for ch in set(text)}
                        
                    elif method == "Huffman":
                        compressor = HuffmanCompression()
                        compressed_path = compressor.compress_file(tmp_input_path, compress_filename + ".huf")
                        
                        # Calculate metrics
                        text = original_text
                        codes = compressor.codes
                        
                    elif method == "LZW":
                        compressor = LZWCompression()
                        compressed_path = compressor.compress_file(tmp_input_path, compress_filename + ".lzw")
                        
                        # Calculate metrics
                        text = original_text
                        codes = {str(i): str(code) for i, code in enumerate(compressor.encoded_codes)}
                        
                    elif method == "Golomb":
                        compressor = GolombCompression(m=m_parameter)
                        if data_type == "Text":
                            compressed_path = compressor.compress_file(tmp_input_path, compress_filename + ".golomb", data_type='text')
                            text = original_text
                        
                    st.success(f"Compression successful!")
                    if method != "Golomb" or (method == "Golomb" and data_type == "Text"):
                        st.markdown("### 📊 Compression Metrics")
                        
                        # Calculate entropy
                        entropy = metrics.calculate_entropy(text)
                        
                        # Calculate average code length
                        avg_length = metrics.calculate_avg_code_length(codes, text)
                        
                        # Get file sizes
                        original_size = os.path.getsize(tmp_input_path)
                        compressed_size = os.path.getsize(compressed_path)
                        
                        # Calculate compression ratio
                        compression_ratio = metrics.calculate_compression_ratio(original_size, compressed_size)
                        
                        # Calculate efficiency
                        efficiency = metrics.calculate_compression_efficiency(entropy, avg_length)
                        
                        # Display metrics in columns
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            st.metric(
                                "Entropy (H)", 
                                f"{entropy:.4f}",
                            )
                        
                        with col_m2:
                            st.metric(
                                "Avg Code Length (Lav)", 
                                f"{avg_length:.4f}",
                            )
                        
                        with col_m3:
                            st.metric(
                                "Compression Ratio", 
                                f"{compression_ratio:.2f}",
                                delta=f"{((compression_ratio - 1) * 100):.1f}%" if compression_ratio != 0 else "0%",
                                delta_color="normal" if compression_ratio > 1 else "inverse",
                                help="Original size / Compressed size (>1 is good)"
                            )
                        
                        with col_m4:
                            st.metric(
                                "Efficiency", 
                                f"{efficiency:.2%}",
                            )
                        
                        # Size information
                        space_saved = original_size - compressed_size
                        space_saved_percent = (space_saved / original_size * 100) if original_size > 0 else 0
                        
                        if compression_ratio > 1:
                            st.success(f"SUCCESS Compressed: {original_size:,} bytes → {compressed_size:,} bytes | Space saved: {space_saved:,} bytes ({space_saved_percent:.1f}%)")
                        elif compression_ratio < 1:
                            st.error(f"WARRNING Expansion: {original_size:,} bytes → {compressed_size:,} bytes | File grew by {-space_saved:,} bytes ({-space_saved_percent:.1f}%)")
                        else:
                            st.info(f"No change: {original_size:,} bytes → {compressed_size:,} bytes")
                    
                    # Download button
                    with open(compressed_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Compressed File",
                            data=f.read(),
                            file_name=Path(compressed_path).name,
                            mime="application/octet-stream"
                        )
                    
                except Exception as e:
                    st.error(f"Compression failed: {str(e)}")
        
        with col2:
            st.markdown("### Decompression")
            
            # Upload compressed file
            compressed_file = st.file_uploader("Upload Compressed File", type=['rle', 'huf', 'lzw', 'golomb'], key="decompress_upload")
            
            if compressed_file is not None:
                decompress_filename = st.text_input(
                    "Decompressed Filename",
                    value=f"decompressed_{Path(compressed_file.name).stem}",
                    key="decompress_name"
                )
                
                if st.button("Decompress", key="decompress_btn"):
                    try:
                        # Save compressed file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(compressed_file.name).suffix) as tmp_comp:
                            tmp_comp.write(compressed_file.getvalue())
                            tmp_compressed_path = tmp_comp.name
                        
                        # Decompress based on method
                        if method == "RLE":
                            compressor = RLECompression()
                            decompressed_path = compressor.decompress_file(tmp_compressed_path, decompress_filename + ".txt")
                            
                        elif method == "Huffman":
                            compressor = HuffmanCompression()
                            decompressed_path = compressor.decompress_file(tmp_compressed_path, decompress_filename + ".txt")
                            
                        elif method == "LZW":
                            compressor = LZWCompression()
                            decompressed_path = compressor.decompress_file(tmp_compressed_path, decompress_filename + ".txt")
                            
                        elif method == "Golomb":
                            compressor = GolombCompression(m=m_parameter)
                            decompressed_path = compressor.decompress_file(tmp_compressed_path, decompress_filename)
                        
                        st.success(f"✅ Decompression successful!")
                        
                        # Show decompressed content for text
                        if Path(decompressed_path).suffix == '.txt':
                            with open(decompressed_path, 'r', encoding='utf-8') as f:
                                decompressed_text = f.read()
                            st.text_area("Decompressed Text", decompressed_text, height=200)
                        
                        # Download button
                        with open(decompressed_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Decompressed File",
                                data=f.read(),
                                file_name=Path(decompressed_path).name,
                                mime="application/octet-stream"
                            )
                        
                    except Exception as e:
                        st.error(f"ERROR Decompression failed: {str(e)}")
# ==========================
# LOSSY COMPRESSION
# ==========================
else:
    st.header("Lossy Compression (Image Quantization)")

    quant_method = st.sidebar.selectbox(
        "Select Quantization Method",
        ["Uniform Quantizer", "Non-Uniform Quantizer"]
    )

    st.sidebar.markdown("### Parameters")

    if quant_method == "Uniform Quantizer":
        step = st.sidebar.slider("Step Size (Δ)", 1, 128, 16, 1)
        num_bits = int(np.ceil(np.log2(256 / step)))
        st.sidebar.info(f"Levels = {int(256/step)}   |   Bits = {num_bits}")

    else:
        bits = st.sidebar.slider("Bits (2^bits levels)", 1, 32, 3, 1)
        st.sidebar.info(f"Levels = {2**bits}")

    st.subheader(quant_method)

    uploaded_image = st.file_uploader("Upload Color Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_image:
        # Load RGB
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)

        st.markdown("### Original Image")
        st.image(image, use_container_width=True)

        if st.button("Quantize Image", type="primary"):
            try:
                quantizer = Quantization()
                metrics = CompressionMetrics()

                if quant_method == "Uniform Quantizer":
                    quantized, tables = quantizer.uniform_quantize_rgb(img_array, step)      # ← fixed: tables
                    reconstructed = quantizer.uniform_dequantize_rgb(tables)                 # ← fixed: tables
                    used_bits = num_bits

                else:  # Non-Uniform Quantizer
                    quantized, tables = quantizer.nonuniform_quantize_rgb(img_array, bits)   # ← fixed: tables
                    reconstructed = quantizer.nonuniform_dequantize_rgb(tables)              # ← fixed: tables
                    used_bits = bits

                st.markdown("---")
                st.markdown("### Comparison")
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption="Original", use_container_width=True)

                with col2:
                    st.image(reconstructed, caption="Quantized", use_container_width=True)

                # Metrics
                st.markdown("---")
                st.markdown("Quality Metrics")

                mse = metrics.calculate_mse(img_array, reconstructed)

                original_bits = img_array.size * 8
                compressed_bits = img_array.size * used_bits
                ratio = metrics.calculate_compression_ratio(original_bits, compressed_bits)

                colA, colB, colC = st.columns(3)
                colA.metric("MSE", f"{mse:.2f}")
                colC.metric("Compression Ratio", f"{ratio:.2f}:1")

                # Download
                st.markdown("---")
                st.markdown("### Download Results")

                # Quantized image
                out_img = Image.fromarray(reconstructed.astype(np.uint8))
                buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                out_img.save(buf.name)

                st.download_button(
                    "Download Quantized Image",
                    data=open(buf.name, 'rb').read(),
                    file_name=f"quantized_{uploaded_image.name}",
                    mime="image/png"
                )

                # Save compressed data (.npz)
                comp = tempfile.NamedTemporaryFile(delete=False, suffix='.npz')

                np.savez_compressed(
                    comp.name,
                    quantized=quantized,
                    table_R=json.dumps(tables[0]),    # ← fixed: tables instead of table
                    table_G=json.dumps(tables[1]),
                    table_B=json.dumps(tables[2])
                )

                st.download_button(
                    "Download Compressed Data (.npz)",
                    data=open(comp.name, 'rb').read(),
                    file_name=f"compressed_{Path(uploaded_image.name).stem}.npz",
                    mime="application/octet-stream"
                )

            except Exception as e:
                st.error(f"Quantization failed: {str(e)}")