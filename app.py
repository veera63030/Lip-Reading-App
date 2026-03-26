import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tempfile
from pathlib import Path
import subprocess

try:
    import face_recognition
except ImportError:
    st.warning("Installing required packages for mouth detection...")
    os.system("pip install face_recognition dlib -q")
    import face_recognition

# ── Setup TensorFlow precision ───────────────────────────────────────
HAS_GPU = len(tf.config.list_physical_devices('GPU')) > 0
if HAS_GPU:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
else:
    tf.keras.backend.set_floatx("float32")
    tf.keras.mixed_precision.set_global_policy("float32")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Custom layers + loss ─────────────────────────────────────────────
class ReduceMean(layers.Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.reduce_mean(x, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.rate      = rate
        self.att        = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn        = keras.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1   = layers.Dropout(rate)
        self.dropout2   = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn = self.att(inputs, inputs)
        attn = self.dropout1(attn, training=training)
        attn = tf.cast(attn, inputs.dtype)
        out1 = self.layernorm1(inputs + attn)
        out1 = tf.cast(out1, inputs.dtype)
        ffn  = self.ffn(out1)
        ffn  = self.dropout2(ffn, training=training)
        ffn  = tf.cast(ffn, inputs.dtype)
        return self.layernorm2(out1 + ffn)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config


def ctc_loss(y_true, y_pred):
    y_true       = tf.cast(y_true, tf.int32)
    y_pred       = tf.nn.log_softmax(y_pred)
    batch_size   = tf.shape(y_pred)[0]
    time_steps   = tf.shape(y_pred)[1]
    input_length = tf.fill([batch_size], time_steps)
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, -1), tf.int32), axis=1)
    y_true_clean = tf.maximum(y_true, 0)
    loss = tf.nn.ctc_loss(
        labels=y_true_clean, logits=y_pred,
        label_length=label_length, logit_length=input_length,
        logits_time_major=False, blank_index=-1)
    return tf.reduce_mean(loss)


# ── Mouth ROI Cropping ───────────────────────────────────────────────
MOUTH_H = 46
MOUTH_W = 140
FPS = 25


def extract_mouth_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if len(frames) == 0:
        return None

    cropped_frames = []
    successful_crops = 0
    frame_count = len(frames)

    for gray_frame in frames:
        try:
            rgb = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            faces = face_recognition.face_landmarks(rgb, model="small")
            if not faces:
                raise ValueError("No face detected")
            lm = faces[0]
            left_eye_x  = int(np.mean([p[0] for p in lm["left_eye"]]))
            right_eye_x = int(np.mean([p[0] for p in lm["right_eye"]]))
            eye_distance = right_eye_x - left_eye_x
            nose_x = int(np.mean([p[0] for p in lm["nose_tip"]]))
            nose_y = int(np.mean([p[1] for p in lm["nose_tip"]]))
            crop_size = int(eye_distance * 1.35)
            half = crop_size // 2
            y1 = nose_y + int(eye_distance * 0.1)
            y2 = y1 + crop_size
            x1 = nose_x - half
            x2 = nose_x + half
            h, w = gray_frame.shape
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            mouth = gray_frame[y1:y2, x1:x2]
            if mouth.size == 0:
                raise ValueError("Empty crop")
            successful_crops += 1
        except Exception:
            h, w = gray_frame.shape
            mouth = gray_frame[h//4:3*h//4, w//4:3*w//4]

        mouth = cv2.resize(mouth, (MOUTH_W, MOUTH_H))
        cropped_frames.append(mouth)

    return np.array(cropped_frames), successful_crops, frame_count


def convert_video_to_mp4(input_path, output_path):
    """Convert video to MP4 using ffmpeg (with audio) or OpenCV fallback (no audio)."""
    # Try ffmpeg first — preserves audio track
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-vcodec", "libx264", "-acodec", "aac",
             "-strict", "experimental", output_path],
            capture_output=True, timeout=60
        )
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
    except Exception:
        pass

    # Fallback: OpenCV (video only, no audio)
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_count += 1
        cap.release()
        writer.release()
        return frame_count > 0
    except Exception:
        return False


def save_frames_as_video(frames, output_path):
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'XVID'),
        FPS, (MOUTH_W, MOUTH_H), isColor=False
    )
    for f in frames:
        f_uint8 = (f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        writer.write(f_uint8)
    writer.release()


# ── Vocab setup ──────────────────────────────────────────────────────
chars = "abcdefghijklmnopqrstuvwxyz "
char_to_num = tf.keras.layers.StringLookup(vocabulary=list(chars), mask_token=None)
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


# ── Load model (cached) ──────────────────────────────────────────────
@st.cache_resource
def load_lip_reading_model():
    WEIGHTS_PATH = "best_model_beam.keras"
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"Model file '{WEIGHTS_PATH}' not found.")
        st.stop()

    CUSTOM_OBJECTS = {'TransformerEncoder': TransformerEncoder, 'ReduceMean': ReduceMean, 'ctc_loss': ctc_loss}
    import keras
    import json
    model = None

    try:
        model = keras.models.load_model(WEIGHTS_PATH, custom_objects=CUSTOM_OBJECTS, compile=False, safe_mode=False)
        st.info("✅ Model loaded successfully (direct)")
    except Exception as e:
        st.warning(f"Direct load failed: {str(e)[:80]}...")

    if model is None:
        try:
            import h5py
            with h5py.File(WEIGHTS_PATH, 'r') as f:
                if 'model_config' in f.attrs:
                    config_json = f.attrs['model_config']
                    if isinstance(config_json, bytes):
                        config_json = config_json.decode('utf-8')
                    config = json.loads(config_json)
                    def clean_config(cfg):
                        if isinstance(cfg, dict):
                            cfg.pop('quantization_config', None)
                            if 'dtype' in cfg and isinstance(cfg['dtype'], dict):
                                cfg['dtype'] = {'class_name': 'DTypePolicy', 'config': {'name': 'float32'}}
                            for v in cfg.values():
                                clean_config(v)
                        elif isinstance(cfg, list):
                            for item in cfg:
                                clean_config(item)
                    clean_config(config)
                    model = keras.models.model_from_json(json.dumps(config), custom_objects=CUSTOM_OBJECTS)
                    st.info("✅ Model loaded successfully (weights-only)")
        except Exception as e2:
            st.warning(f"Weights recovery failed: {str(e2)[:80]}...")

    if model is None:
        try:
            model = tf.keras.models.load_model(WEIGHTS_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
            st.info("✅ Model loaded successfully (TensorFlow)")
        except Exception as e3:
            st.error(f"All loading attempts failed: {str(e3)[:300]}")
            st.stop()

    if not HAS_GPU and model is not None:
        st.info("🔄 Converting model to CPU-compatible float32...")
        try:
            model_config = model.get_config()
            def convert_dtype_recursive(obj):
                if isinstance(obj, dict):
                    if 'dtype' in obj:
                        obj['dtype'] = 'float32'
                    if 'config' in obj and isinstance(obj['config'], dict):
                        if 'dtype' in obj['config']:
                            obj['config']['dtype'] = 'float32'
                    for value in obj.values():
                        convert_dtype_recursive(value)
                elif isinstance(obj, list):
                    for item in obj:
                        convert_dtype_recursive(item)
            convert_dtype_recursive(model_config)
            weights = model.get_weights()
            model = keras.Model.from_config(model_config, custom_objects=CUSTOM_OBJECTS)
            model.set_weights(weights)
            st.info("✅ Model converted to float32 for CPU")
        except Exception as e:
            st.warning(f"Could not fully convert model: {str(e)[:100]}")

    return model


# ── Inference helpers ────────────────────────────────────────────────
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(75):
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(frame, (140, 46)))
    cap.release()
    while len(frames) < 75:
        frames.append(np.zeros((46, 140), dtype=np.float32))
    video = np.array(frames, dtype=np.float32) / 255.0
    video = video[..., np.newaxis]
    return np.expand_dims(video, axis=0)


def ids_to_text(row):
    idx = row[row != -1]
    chars = num_to_char(tf.cast(idx, tf.int64))
    return tf.strings.reduce_join(chars).numpy().decode('utf-8')


def predict(model, video_path, use_beam=True, beam_width=10, top_k=1):
    video    = load_video(video_path)
    video_tf = tf.constant(video, dtype=tf.float32)
    with tf.device('/CPU:0') if not HAS_GPU else tf.device('/GPU:0'):
        y_pred = model(video_tf, training=False)
        y_pred = tf.cast(y_pred, tf.float32).numpy()
    input_len = tf.fill([1], y_pred.shape[1])
    if use_beam:
        decoded_sparse, _ = tf.nn.ctc_beam_search_decoder(
            tf.transpose(y_pred, [1, 0, 2]), input_len, beam_width=beam_width, top_paths=top_k)
        predictions = []
        for i in range(top_k):
            decoded_dense = tf.sparse.to_dense(decoded_sparse[i], default_value=-1)
            predictions.append(ids_to_text(decoded_dense.numpy()[0]))
        return predictions if top_k > 1 else predictions[0]
    else:
        decoded_sparse, _ = tf.nn.ctc_greedy_decoder(
            tf.transpose(y_pred, [1, 0, 2]), input_len, blank_index=-1)
        decoded_dense = tf.sparse.to_dense(decoded_sparse[0], default_value=-1)
        return ids_to_text(decoded_dense.numpy()[0])


# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Lip Reading Prediction", page_icon="👄", layout="centered")

try:
    model = load_lip_reading_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False

# ── Header ───────────────────────────────────────────────────────────
st.markdown('<h1 style="text-align: center;">👄 Lip Reading Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d;">Academic Project Demo • Video-based Speech Recognition</p>', unsafe_allow_html=True)
st.divider()
st.markdown("**Upload a short video or GIF of a person speaking.**  \nThe system analyzes lip movements and predicts the spoken text.")

# ── File Upload ───────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📤 Upload a video or GIF", type=["mp4", "mov", "avi", "gif", "mpg", "mpeg"])

# ── Preview ───────────────────────────────────────────────────────────
if uploaded_file is not None:
    st.subheader("🎞️ Media Preview")

    if uploaded_file.type.startswith("video"):
        file_ext = uploaded_file.name.split('.')[-1].lower()

        if file_ext in ['avi', 'mpg', 'mpeg']:
            st.info(f"📝 Converting {file_ext.upper()} to MP4 for preview...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_original:
                tmp_original.write(uploaded_file.getbuffer())
                original_path = tmp_original.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_mp4:
                converted_path = tmp_mp4.name

            if convert_video_to_mp4(original_path, converted_path):
                with open(converted_path, 'rb') as f:
                    video_bytes = f.read()
                st.video(video_bytes, format="video/mp4")
                st.caption(f"✅ Converted from {file_ext.upper()} to MP4 for preview")
            else:
                st.warning(f"⚠️ Could not convert {file_ext.upper()} file for preview. Proceeding with prediction anyway.")

            for p in [original_path, converted_path]:
                try:
                    os.unlink(p)
                except:
                    pass
        else:
            st.video(uploaded_file)
            st.caption("🔊 Audio preview enabled (if audio track exists)")
    else:
        st.image(uploaded_file)
        st.caption("ℹ️ GIF files do not contain audio")

    st.divider()

    # ── Prediction ────────────────────────────────────────────────────
    if st.button("🔍 Predict Lip Reading"):
        if not model_loaded:
            st.error("Model failed to load. Cannot run prediction.")
        else:
            try:
                with st.spinner("Extracting mouth region..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name

                    result = extract_mouth_frames(tmp_path)
                    if result is None:
                        st.error("Failed to detect face in video. Please ensure the video shows a clear face.")
                        os.unlink(tmp_path)
                    else:
                        mouth_frames, successful_crops, total_frames = result

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as cropped_file:
                            cropped_path = cropped_file.name
                        save_frames_as_video(mouth_frames, cropped_path)

                        st.subheader("🎯 Mouth Region Extraction")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Frames Processed", total_frames)
                        with col2:
                            st.metric("Successful Crops", successful_crops)

                        st.subheader("📹 Cropped Mouth Video")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as cropped_mp4:
                            cropped_mp4_path = cropped_mp4.name

                        if convert_video_to_mp4(cropped_path, cropped_mp4_path):
                            with open(cropped_mp4_path, 'rb') as f:
                                cropped_video_bytes = f.read()
                            st.video(cropped_video_bytes, format="video/mp4")
                            st.caption("Extracted mouth region (140x46 pixels) - Input to lip reading model")
                            try:
                                os.unlink(cropped_mp4_path)
                            except:
                                pass
                        else:
                            st.warning("⚠️ Could not convert cropped video for preview, but proceeding with inference...")

                        st.divider()

                with st.spinner("Running lip reading inference..."):
                    greedy_pred = predict(model, cropped_path, use_beam=False)
                    beam_preds  = predict(model, cropped_path, use_beam=True, beam_width=10, top_k=5)

                st.success("✅ Prediction Complete")
                st.subheader("📝 Predicted Text")

                # Single-line HTML — avoids Streamlit markdown misparse of multi-line style blocks
                st.markdown(
                    f'<div style="padding:1rem;background-color:#e9ecef;border-left:6px solid #0d6efd;border-radius:6px;font-size:1.1rem;color:#212529;font-weight:600;margin-bottom:1rem;"><strong>Greedy Decoding:</strong><br>{greedy_pred}</div>',
                    unsafe_allow_html=True
                )

                st.markdown("<strong>Beam Search (Top 5 - Ranked):</strong>", unsafe_allow_html=True)
                for idx, pred in enumerate(beam_preds, 1):
                    color        = "#d1ecf1" if idx == 1 else "#f8f9fa"
                    border_color = "#0c5460" if idx == 1 else "#6c757d"
                    st.markdown(
                        f'<div style="padding:0.75rem;background-color:{color};border-left:4px solid {border_color};border-radius:4px;font-size:1rem;color:#212529;margin-bottom:0.5rem;"><strong>#{idx}:</strong> {pred}</div>',
                        unsafe_allow_html=True
                    )

                st.caption("⚙️ Output generated using CNN-Transformer lip-reading model")

                for p in ['tmp_path', 'cropped_path']:
                    try:
                        path = locals().get(p)
                        if path and os.path.exists(path):
                            os.unlink(path)
                    except:
                        pass

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please ensure the video format is compatible (.avi, .mp4, .mov, .mpg)")
                for p in ['tmp_path', 'cropped_path']:
                    try:
                        path = locals().get(p)
                        if path and os.path.exists(path):
                            os.unlink(path)
                    except:
                        pass

# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.caption("👨‍🎓 Academic Project • Lip Reading System Prototype")