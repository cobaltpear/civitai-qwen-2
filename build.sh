#!/bin/sh
# Qwen Image 2 Universal (any Qwen 2 / Qwen 2.1 model from Civitai or Hugging Face) — everything the worker image needs. The Dockerfile in your copy downloads this file
# from the original repo on every build, so fixes made there reach you with a rebuild.
set -eu
# civitai.env in your copy (made by Cobalt Pear's «Add your token» link) wins over the two Dockerfile lines
if [ -f /setup/civitai.env ]; then
  for k in CIVITAI_TOKEN CIVITAI_MODEL HF_TOKEN; do
    v=$(sed -n "s/^[[:space:]]*$k[[:space:]]*=//p" /setup/civitai.env | tr -d '\r"'"'"' ' | head -n 1)
    [ -z "$v" ] || eval "$k=\$v"
  done
fi
export CIVITAI_TOKEN CIVITAI_MODEL HF_TOKEN
case "${CIVITAI_MODEL:-}" in ""|PASTE_*) echo "ERROR: put the model in civitai.env (or at the top of the Dockerfile) — its Civitai page link or version number, or its Hugging Face link" >&2; exit 1;; esac

# Find the one file the link means, check it is a Qwen Image 2 model, then download it and check its SHA-256.
# Civitai: its API names the file. Hugging Face: the repo's single-file model (a .safetensors at its top, not the diffusers folders).
# The model is always saved as diffusion_models/model.safetensors — the pack's workflow loads that name.
python3 - <<'PY'
import hashlib, json, os, re, struct, sys, urllib.request
token, link, hf_token = os.environ.get("CIVITAI_TOKEN", ""), os.environ["CIVITAI_MODEL"].strip(), os.environ.get("HF_TOKEN", "")
def fail(msg): sys.exit("ERROR: " + msg)
def download(url, dest, sha, name, headers={}):
    h = hashlib.sha256()
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cobaltpear", **headers}), timeout=600) as r, open(dest, "wb") as out:
        while chunk := r.read(1 << 22): h.update(chunk); out.write(chunk)
    if sha and h.hexdigest() != sha: fail(f"{name} does not match its SHA-256 — try the build again")
    print("SHA-256 OK" if sha else "No SHA-256 listed for this file — not checked")
dest = "/comfyui/models/diffusion_models/model.safetensors"
# Qwen Image 2 (32 blocks, one shared modulation) — the older Qwen Image has no top-level modulation.N weights
def is_qwen2(keys): return any(re.search(r"(^|\.)transformer_blocks\.", k) for k in keys) and any(re.search(r"(^|\.)modulation\.\d+\.", k) for k in keys)
hf = None if "civitai.com" in link or re.fullmatch(r"\d+", link) else \
    re.fullmatch(r"(?:https?://huggingface\.co/)?([\w.-]+/[\w.-]+)(?:/(?:blob|resolve|tree)/[^/]+(?:/(.+?))?)?/?(?:\?.*)?", link)
if hf:
    repo, want = hf.group(1), hf.group(2)
    auth = {"Authorization": "Bearer " + hf_token} if hf_token and not hf_token.startswith("PASTE_") else {}
    def get(url, **kw):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cobaltpear", **auth, **kw}), timeout=60)
        except urllib.error.HTTPError as e:
            fail(f"Hugging Face answered {e.code} for {repo}" + (" — it is gated: accept its license on its page and put HF_TOKEN=<a Read token> in civitai.env" if e.code in (401, 403) else ""))
    info = json.load(get(f"https://huggingface.co/api/models/{repo}?blobs=true"))
    files = [f for f in info.get("siblings", []) if f["rfilename"].endswith(".safetensors") and (f["rfilename"] == want if want else "/" not in f["rfilename"])]
    files = [f for f in files if f.get("size", 0) > 3e+09 or want]  # a whole model, not a VAE or a LoRA
    if not files: fail(f"{repo} has no single-file checkpoint (.safetensors) at its top — it only has diffusers folders, which this worker can't load")
    f = min(files, key=lambda f: f.get("size", 0))  # fp16 rather than fp32
    url = f"https://huggingface.co/{repo}/resolve/main/{f['rfilename']}"
    head = get(url, Range="bytes=0-7").read(); n = struct.unpack("<Q", head)[0]
    keys = list(json.loads(get(url, Range=f"bytes=8-{7 + n}").read()))
    if not is_qwen2(keys): fail(f"{f['rfilename']} is not a Qwen Image 2 model — this worker takes Qwen 2 / Qwen 2.1 models only")
    print(f"Hugging Face: {repo} — {f['rfilename']}")
    print(f"Downloading {f['rfilename']} ({f.get('size', 0) / 1e9:.1f} GB)")
    download(url, dest, (f.get("lfs") or {}).get("sha256", ""), f["rfilename"], auth)
    json.dump({"repo": repo, "name": repo.split("/")[-1], "file": f["rfilename"], "base": "Qwen 2.1"}, open("/comfyui/models/diffusion_models/model.json", "w"))
    sys.exit(0)
if not token or token.startswith("PASTE_"): fail("put your Civitai token in civitai.env (or at the top of the Dockerfile)")
# a model file ComfyUI loads as is: .safetensors, not NF4 / NVFP4 / W4A8 / W6A8 / MXFP8 / INT4 (those need other kernels). INT8 only when nothing plainer is there.
QUANT = r"nvfp4|nf4|mxfp8|w[46]a8|int4"
def usable(v):
    ok = [f for f in v.get("files", []) if f.get("type") in ("Model", "Diffusion Model") and f.get("metadata", {}).get("format") == "SafeTensor"
          and f.get("metadata", {}).get("fp") not in ("nf4", "nvfp4", "mxfp8", "int4") and not re.search(QUANT, f["name"], re.I)]
    return [f for f in ok if f.get("metadata", {}).get("fp") != "int8" and not re.search(r"int8|convrot", f["name"], re.I)] or ok
def signed(f): return f["downloadUrl"] + ("&" if "?" in f["downloadUrl"] else "?") + "token=" + token
def opens(f):  # Early Access files answer 401/403 until you pay Buzz on Civitai
    try: urllib.request.urlopen(urllib.request.Request(signed(f), headers={"Range": "bytes=0-7", "User-Agent": "cobaltpear"}), timeout=120).read(); return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403): return False
        raise
smallest = lambda files: min(files, key=lambda f: f.get("sizeKB", 0))  # the smallest full model file: faster to build
def get(path):
    req = urllib.request.Request("https://civitai.com/api/v1/" + path, headers={"Authorization": "Bearer " + token, "User-Agent": "cobaltpear"})
    try: return json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e: fail(f"could not ask Civitai about {link!r} ({e})")
m = re.search(r"modelVersionId=(\d+)", link) or re.fullmatch(r"(\d+)", link)
if m: version = get(f"model-versions/{m.group(1)}")
else:
    m = re.search(r"/models/(\d+)", link) or fail(f"CIVITAI_MODEL {link!r} is not a Civitai model link")
    vs = get(f"models/{m.group(1)}")["modelVersions"]
    mine = [v for v in vs if v.get("baseModel") in ("Qwen 2", "Qwen 2.1")] or fail("no version of this model is Qwen Image 2 (base model Qwen 2 or Qwen 2.1 on Civitai)")
    version = next((v for v in mine if usable(v) and opens(smallest(usable(v)))), None) or fail("no Qwen Image 2 version of this model has a file your token can download (Early Access versions need Buzz on Civitai)")   # the newest one
    version = get(f"model-versions/{version['id']}")
base, kind = version.get("baseModel", ""), version.get("model", {}).get("type", "")
print(f"Civitai: {version.get('model', {}).get('name')} — {version.get('name')} (version {version['id']}, {base}, {kind})")
if kind != "Checkpoint": fail(f"this is a {kind}, not a Checkpoint")
if base not in ("Qwen 2", "Qwen 2.1"): fail(f"this is a {base} model — this worker takes Qwen Image 2 models (base model Qwen 2 or Qwen 2.1 on Civitai)")
files = usable(version)
if not files: fail("no .safetensors model file in this version that plain ComfyUI can load (GGUF, NF4, NVFP4, W4A8, W6A8, MXFP8 and INT4 files are not taken) — pick another version")
f = smallest(files)
if not opens(f): fail("Civitai won't let your token download this version — it may be in Early Access (pay with Buzz on its page) — pick another version")
print(f"Downloading {f['name']} ({f.get('sizeKB', 0) / 1e6:.1f} GB)")
download(signed(f), dest, f.get("hashes", {}).get("SHA256", "").lower(), f["name"])
json.dump({"version": version["id"], "name": version.get("model", {}).get("name"), "file": f["name"], "base": base}, open("/comfyui/models/diffusion_models/model.json", "w"))
def local_keys(p):
    with open(p, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; return list(json.loads(fh.read(n)))
keys = local_keys(dest)
if not is_qwen2(keys): fail(f"{f['name']} is not a Qwen Image 2 model inside, though Civitai lists it as {base} — pick another version")
PY

# the text encoder and VAE every Qwen Image 2 model uses, pinned by SHA-256
file() {
  wget -q -O "$1" "$2"
  echo "$3  $1" | sha256sum -c - || { echo "ERROR: $1 changed upstream — wait for an update of the original repo, then rebuild" >&2; exit 1; }
}

file /comfyui/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors \
  "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/aacf31a6d48e707f352ecba714cee0680da3479b/text_encoders/qwen3vl_8b_int8_convrot.safetensors" \
  8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f

file /comfyui/models/vae/qwen_image_2.1_vae_bf16.safetensors \
  "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/aacf31a6d48e707f352ecba714cee0680da3479b/vae/qwen_image_2.1_vae_bf16.safetensors" \
  bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9

# Qwen Image 2 needs a newer ComfyUI than the base image has
cd /comfyui
git fetch --depth 1 https://github.com/Comfy-Org/ComfyUI.git tag v0.43.1
git checkout -f v0.43.1
uv pip install --python /opt/venv/bin/python -r requirements.txt "transformers>=4.50.3,<5" "huggingface-hub<1.0"
timeout 300 /opt/venv/bin/python main.py --quick-test-for-ci --cpu
