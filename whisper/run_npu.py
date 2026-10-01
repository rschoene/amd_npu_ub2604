#!/usr/bin/env python3
"""
run_npu.py — Transcribe audio with Whisper on the AMD NPU (or CPU).

This uses the Ryzen AI SDK's onnxruntime build, which includes the
VitisAIExecutionProvider (VitisEP). The VitisEP partitions the Whisper
encoder/decoder ONNX graphs and compiles the NPU-supported subgraphs to AIE
at runtime (first run takes ~15 min; subsequent runs load from cache).

It runs the pre-quantized Whisper ONNX models published by AMD on HuggingFace
(amd/whisper-*-onnx-npu). No manual AIE compilation step is needed — the
VitisEP handles that internally.

Prerequisites:
  1. ./setup_npu.sh install          (driver, XRT, memlock)
  2. ./setup_whisper.sh rai          (Miniforge + RAI SDK with VitisEP)

Usage (from the RAI conda env):
  tools/miniforge3/bin/conda run -n ryzen-ai python whisper/run_npu.py \
      --model-type whisper-small --device npu --input audio.wav

  # CPU fallback (same models, no NPU):
  ... --device cpu
"""

import argparse
import os
import sys
import time

SAMPLE_RATE = 16000

# NPU-optimized ONNX models published by AMD on HuggingFace.
# (whisper-base / whisper-tiny have no NPU ONNX in this map — pass
#  --encoder/--decoder explicitly if you have them.)
HF_MODEL_MAP = {
    "whisper-small": "amd/whisper-small-onnx-npu",
    "whisper-medium": "amd/whisper-medium-onnx-npu",
    "whisper-large-v3-turbo": "amd/whisper-large-turbo-onnx-npu",
}


def download_whisper_onnx(model_type: str):
    """Download the NPU-optimized Whisper ONNX encoder/decoder from HF."""
    from huggingface_hub import snapshot_download

    repo_id = HF_MODEL_MAP.get(model_type)
    if repo_id is None:
        raise ValueError(
            f"No auto-download for '{model_type}'. "
            f"Supported: {', '.join(HF_MODEL_MAP)}. "
            "Pass --encoder/--decoder for other sizes."
        )
    local_dir = snapshot_download(repo_id=repo_id)
    encoder = os.path.join(local_dir, "encoder_model.onnx")
    decoder = os.path.join(local_dir, "decoder_model.onnx")
    if not (os.path.exists(encoder) and os.path.exists(decoder)):
        raise FileNotFoundError(f"encoder/decoder ONNX not found in {local_dir}")
    return encoder, decoder


def build_providers(device: str, cache_dir: str, model_key: str,
                    encoder_cfg: str = None, decoder_cfg: str = None):
    """Build onnxruntime provider lists for the encoder and decoder."""
    import onnxruntime as ort

    if device == "cpu":
        return ["CPUExecutionProvider"], ["CPUExecutionProvider"]

    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        print("WARNING: VitisAIExecutionProvider not available. "
              "Is the RAI SDK installed? Falling back to CPU.", file=sys.stderr)
        return ["CPUExecutionProvider"], ["CPUExecutionProvider"]

    def vai_opts(cfg):
        opts = {"cache_dir": cache_dir,
                "cache_key": f"whisper_{model_key}",
                "enable_cache_file_io_in_mem": "0"}
        if cfg:
            opts["config_file"] = cfg
        return [("VitisAIExecutionProvider", opts)]

    return vai_opts(encoder_cfg), vai_opts(decoder_cfg)


class WhisperNPU:
    def __init__(self, encoder_path, decoder_path, model_type,
                 encoder_providers, decoder_providers, language=None):
        import numpy as np
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        self.np = np
        print(f"Loading encoder: {encoder_path}")
        self.encoder = ort.InferenceSession(encoder_path, providers=encoder_providers)
        print(f"Loading decoder: {decoder_path}")
        self.decoder = ort.InferenceSession(decoder_path, providers=decoder_providers)

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{model_type}")
        self.tokenizer = WhisperTokenizer.from_pretrained(f"openai/{model_type}")
        self.eos_token = self.tokenizer.eos_token_id
        self.max_length = min(448, self.decoder.get_inputs()[0].shape[1])
        if not isinstance(self.max_length, int):
            raise ValueError("Invalid/dynamic decoder input shape")

        if language:
            self.tokenizer.set_prefix_tokens(language=language, task="transcribe")
            self.initial_tokens = list(self.tokenizer.prefix_tokens)
        else:
            self.initial_tokens = [self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]

    def _preprocess(self, audio):
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        return inputs["input_features"]

    def _encode(self, input_features):
        name = self.encoder.get_inputs()[0].name
        return self.encoder.run(None, {name: input_features})[0]

    def _decode(self, encoder_out):
        np = self.np
        tokens = list(self.initial_tokens)
        decoder_inputs = self.decoder.get_inputs()
        input_ids_name = decoder_inputs[0].name
        encoder_out_name = decoder_inputs[1].name
        # Disambiguate by dtype if the input order is not guaranteed.
        if decoder_inputs[0].type != "tensor(int64)":
            input_ids_name, encoder_out_name = encoder_out_name, input_ids_name

        for _ in range(len(tokens), self.max_length):
            decoder_input = np.full((1, self.max_length), self.eos_token, dtype=np.int64)
            decoder_input[0, :len(tokens)] = tokens
            outputs = self.decoder.run(None, {
                input_ids_name: decoder_input,
                encoder_out_name: encoder_out,
            })
            logits = outputs[0]
            next_token = int(np.argmax(logits[0, len(tokens) - 1]))
            if next_token == self.eos_token:
                break
            tokens.append(next_token)
        return tokens

    def transcribe(self, audio, chunk_length_s=30):
        np = self.np
        chunk_size = SAMPLE_RATE * chunk_length_s
        overlap = SAMPLE_RATE * 1
        parts = []
        t0 = time.time()
        for start in range(0, len(audio), chunk_size - overlap):
            end = min(start + chunk_size, len(audio))
            feats = self._preprocess(audio[start:end])
            enc = self._encode(feats)
            tokens = self._decode(enc)
            text = self.tokenizer.decode(tokens[len(self.initial_tokens):],
                                         skip_special_tokens=True).strip()
            parts.append(text)
        rtf = (time.time() - t0) / (len(audio) / SAMPLE_RATE)
        return " ".join(parts), rtf


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe audio with Whisper on the AMD NPU")
    ap.add_argument("--input", required=True, help="Path to a .wav file (16 kHz mono)")
    ap.add_argument("--model-type", default="whisper-small",
                    choices=["whisper-tiny", "whisper-base", "whisper-small",
                             "whisper-medium", "whisper-large-v3-turbo"])
    ap.add_argument("--device", choices=["cpu", "npu"], default="npu")
    ap.add_argument("--encoder", help="Path to encoder ONNX (else auto-download)")
    ap.add_argument("--decoder", help="Path to decoder ONNX (else auto-download)")
    ap.add_argument("--encoder-config", help="VitisEP config JSON for the encoder (BF16)")
    ap.add_argument("--decoder-config", help="VitisEP config JSON for the decoder (BF16)")
    ap.add_argument("--language", help="Force a language code (e.g. 'en')")
    ap.add_argument("--cache-dir", default=None,
                    help="VitisEP cache dir (default: artifacts/whisper/cache)")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: audio file not found: {args.input}", file=sys.stderr)
        return 1

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_dir = args.cache_dir or os.path.join(repo_root, "artifacts", "whisper", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    if args.encoder and args.decoder:
        encoder_path, decoder_path = args.encoder, args.decoder
    else:
        print(f"Downloading NPU ONNX models for {args.model_type} from HuggingFace ...")
        encoder_path, decoder_path = download_whisper_onnx(args.model_type)

    model_key = args.model_type.replace("whisper-", "")
    enc_prov, dec_prov = build_providers(args.device, cache_dir, model_key,
                                         args.encoder_config, args.decoder_config)
    print(f"Encoder providers: {enc_prov}")
    print(f"Decoder providers: {dec_prov}")

    model = WhisperNPU(encoder_path, decoder_path, args.model_type,
                       enc_prov, dec_prov, language=args.language)

    import torchaudio
    waveform, sr = torchaudio.load(args.input)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
    audio = waveform.squeeze(0).numpy()

    print(f"Transcribing {args.input} on {args.device.upper()} ...")
    text, rtf = model.transcribe(audio)
    print(f"\nRTF: {rtf:.2f}")
    print("\n--- Transcript ---")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
