import io
import time
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

st.set_page_config(page_title="Watermark Lab", page_icon="🛡️", layout="wide")

MAGIC = b"WMK1"

def image_to_rgb_array(image):
    return np.array(image.convert("RGB"), dtype=np.uint8)

def bytes_to_bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) % 8:
        bits = bits[:len(bits) - (len(bits) % 8)]
    return np.packbits(bits).tobytes()

def embed_lsb(image, message):

    rgb = image_to_rgb_array(image)
    payload = MAGIC + len(message.encode("utf-8")).to_bytes(4, "big") + message.encode("utf-8")
    bits = bytes_to_bits(payload)
    flat = rgb.reshape(-1)
    if len(bits) > len(flat):
        max_bytes = max(0, (len(flat) // 8) - 8)
        raise ValueError(
            f"Message is too large for this image. Approximate maximum message size: {max_bytes} bytes."
        )
    flat[:len(bits)] = (flat[:len(bits)] & 0xFE) | bits
    result = flat.reshape(rgb.shape)
    return Image.fromarray(result, mode="RGB"), len(bits)

def extract_lsb(image):
    rgb = image_to_rgb_array(image)
    flat = rgb.reshape(-1)
    header_bits = flat[:64] & 1
    header = bits_to_bytes(header_bits)
    if len(header) < 8 or header[:4] != MAGIC:
        raise ValueError("No valid LSB watermark header found in this image.")
    message_len = int.from_bytes(header[4:8], "big")
    total_bits = (8 + message_len) * 8
    if total_bits > len(flat):
        raise ValueError("Watermark header is invalid or the image has been damaged.")
    payload = bits_to_bytes(flat[:total_bits] & 1)
    try:
        return payload[8:8 + message_len].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("The watermark may be damaged or corrupted.") from exc

def sha256_pixels(image):
    arr = image_to_rgb_array(image)
    return hashlib.sha256(arr.tobytes()).hexdigest()

def psnr(original, processed):
    a = image_to_rgb_array(original).astype(np.float64)
    b = image_to_rgb_array(processed).astype(np.float64)
    if a.shape != b.shape:
        return float("nan")
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def image_bytes(image, fmt="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()

def show_image_pair(left, right, left_label, right_label):
    c1, c2 = st.columns(2)
    with c1:
        st.image(left, caption=left_label, use_container_width=True)
    with c2:
        st.image(right, caption=right_label, use_container_width=True)

st.title("🛡️ Digital Watermarking & Tamper Detection")

with st.sidebar:
    st.header("Project information")
    st.write("Least Significant Bit (LSB) watermark embedding")
    st.write("SHA-256 fingerprint of image pixels")

tab_embed, tab_extract, tab_tamper = st.tabs([
    "1 · Embed", "2 · Extract", "3 · Tamper Check"
])

with tab_embed:
    st.subheader("Embed a text watermark")
    uploaded = st.file_uploader("Upload an image ", type=["png", "jpg", "jpeg"], key="embed_upload")
    message = st.text_input("Watermark text", placeholder="enter watermark  ")
    if uploaded:
        try:
            original = Image.open(uploaded).convert("RGB")
            st.image(original, caption=f"Original image • {original.width} × {original.height}", use_container_width=True)
            capacity = max(0, (original.width * original.height * 3 // 8) - 8)
            st.caption(f"Approximate maximum watermark capacity: {capacity:,} bytes")
            if st.button("Embed watermark", type="primary", disabled=not message):
                start = time.perf_counter()
                watermarked, bit_count = embed_lsb(original, message)
                elapsed_ms = (time.perf_counter() - start) * 1000
                st.session_state["watermarked_image"] = watermarked
                st.session_state["watermarked_reference_hash"] = sha256_pixels(watermarked)
                st.session_state["watermark_message"] = message
                st.success("Watermark embedded successfully.")

                st.download_button(
                    "Download watermarked PNG",
                    data=image_bytes(watermarked, "PNG"),
                    file_name="watermarked_image.png",
                    mime="image/png"
                )
                st.info("Keep this downloaded PNG as the trusted reference for the Tamper Check tab.")
        except Exception as exc:
            st.error(f"Could not process image: {exc}")
    else:
        st.info("Upload an image to begin. PNG is recommended for preserving embedded LSB data.")

with tab_extract:
    st.subheader("Extract a watermark")
    uploaded_extract = st.file_uploader("Upload a watermarked PNG", type=["png", "jpg", "jpeg"], key="extract_upload")
    if uploaded_extract and st.button("Extract watermark", type="primary"):
        try:
            watermarked = Image.open(uploaded_extract).convert("RGB")
            start = time.perf_counter()
            recovered = extract_lsb(watermarked)
            elapsed_ms = (time.perf_counter() - start) * 1000
            st.success("Watermark header detected.")
            st.text_area("Recovered watermark", value=recovered, height=100)
        except Exception as exc:
            st.error(str(exc))

with tab_tamper:
    st.subheader("Check whether an image has changed")
    st.write("Upload a trusted watermarked PNG and the image you want to verify.")
    ref_file = st.file_uploader("Trusted reference image", type=["png", "jpg", "jpeg"], key="ref_upload")
    suspect_file = st.file_uploader("Image to verify", type=["png", "jpg", "jpeg"], key="suspect_upload")
    if ref_file and suspect_file and st.button("Verify integrity", type="primary"):
        try:
            ref = Image.open(ref_file).convert("RGB")
            suspect = Image.open(suspect_file).convert("RGB")
            show_image_pair(ref, suspect, "Trusted reference", "Image under test")
            if ref.size != suspect.size:
                st.error(f"Image dimensions differ: reference {ref.size}, test {suspect.size}. Integrity check failed.")
            else:
                ref_arr = image_to_rgb_array(ref)
                suspect_arr = image_to_rgb_array(suspect)
                changed = np.any(ref_arr != suspect_arr, axis=2)
                changed_pixels = int(changed.sum())
                total_pixels = int(changed.size)
                ratio = changed_pixels / total_pixels * 100 if total_pixels else 0
                same_hash = sha256_pixels(ref) == sha256_pixels(suspect)
                if same_hash:
                    st.success("Integrity verified: pixel data matches the trusted reference exactly.")
                else:
                    st.error("Modification detected: pixel data differs from the trusted reference.")
                a, b, c = st.columns(3)
                a.metric("Changed pixels", f"{changed_pixels:,}")
                b.metric("Changed-pixel ratio", f"{ratio:.4f}%")
                c.metric("SHA-256 match", "Yes" if same_hash else "No")
                if changed_pixels:
                    diff_img = Image.fromarray((changed.astype(np.uint8) * 255), mode="L")
                    st.image(diff_img, caption="Difference map (white = changed pixel)", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not compare images: {exc}")

