# %%
# Cell 3: Download and load QUASR device database
import gzip
import json
from io import BytesIO
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import time
import numpy as np
from simsopt.field import Current
from simsopt.geo import SurfaceRZFourier
from simsopt._core import load
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import shutil

# %%
SIMSOPT_DIR = 'quasr_simsopt_files'
LOG_CSV = "flat_surf_quasr_log.csv"
data_dir = Path(SIMSOPT_DIR)
output_dir = Path('flat_surface_coil_tfrecords')
os.makedirs(SIMSOPT_DIR, exist_ok=True)

MAX_COILS = 6
FEATURES_PER_COIL = 100
NUM_THREAD_WORKERS = 48
NUM_PROCESS_WORKERS = 1
CHUNK_SIZE = 10000
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds
MAX_NFP = 5
MAX_COEFS = 441

# %%
# url = "https://quasr.flatironinstitute.org/database.json.gz"
# print('Downloading device database...')
# r = requests.get(url)
# r.raise_for_status()

# with gzip.open(BytesIO(r.content), 'rt', encoding='utf-8') as f:
#     data = json.load(f)

# df = pd.DataFrame(**data)
# print(f"Loaded {len(df)} devices.")

# df.to_hdf('QUASR_Stellarators.h5', key = 'full_dataset')
df = pd.read_hdf('QUASR_Stellarators.h5', key = 'full_dataset')

# %%
# Cell 4: Apply filters to select matching devices
filtered = df[
    (df["Nfourier_coil"] == 16) &
    (df['qs_error'] >= -4) &
    # (df["max_elongation"] <= 10) &
    # (df["aspect_ratio"] >= 4) & (df["aspect_ratio"] <= 10) &
    (df["nc_per_hp"] >= 1) & (df["nc_per_hp"] <= 6) &
    (df["nfp"] >= 1) & (df["nfp"] <= 5)
].copy()

print(f"{len(filtered)} devices match your criteria.")

# %%
def simsopt_url(device_id):
    pid = device_id.zfill(7)
    return f"https://quasr.flatironinstitute.org/simsopt_serials/{pid[:4]}/serial{pid}.json"

# %%
# Cell 6: Robust download with retries
def download_with_retries(url: str, path: str) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                return True
            else:
                print(f"{url} returned status {r.status_code} (attempt {attempt})")
        except Exception as e:
            print(f"Error on {url} (attempt {attempt}): {e}")
        time.sleep(RETRY_DELAY)
    return False

# %%
# Cell 7: Prepare log and list of device IDs to download
if os.path.exists(LOG_CSV):
    log_df = pd.read_csv(LOG_CSV, dtype=str)
else:
    log_df = pd.DataFrame(columns=["ID", 'simsopt_url', "status"])

processed = set(log_df["ID"])
device_ids = [str(d) for d in filtered["ID"] if str(d) not in processed] #this is where you change which df you want the device ids from
chunks = [device_ids[i:i+CHUNK_SIZE] for i in range(0, len(device_ids), CHUNK_SIZE)]
print(f"{len(device_ids)} devices to download in {len(chunks)} chunks.")

# %%
def process_device(dev_id):
        pid = dev_id.zfill(7)
        # vmec_path = os.path.join(VMEC_DIR, f"input.{pid}")
        simsopt_path = os.path.join(SIMSOPT_DIR, f"input_{pid}.json")
        # vmec_ok = os.path.exists(vmec_path) or download_with_retries(vmec_url(dev_id), vmec_path)
        simsopt_ok = os.path.exists(simsopt_path) or download_with_retries(simsopt_url(dev_id), simsopt_path)
        status = "success" if simsopt_ok else 'failed'
        return {
            "ID": dev_id,
            # "vmec_url": vmec_url(dev_id),
            'simsopt_url': simsopt_url(dev_id),
            "status": status
        }

# %%
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_data(id: str, coils: np.ndarray, coil_mask: np.ndarray, surface: np.ndarray, surface_mask: np.ndarray):
    feature = {
        'ID': _bytes_feature(id.encode('utf-8')),
        'coil_data': _float_feature(coils),
        'coil_mask': _int_feature(coil_mask),
        'surface_data': _float_feature(surface),
        'surface_mask': _int_feature(surface_mask)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# %%
def _get_c0(curve, key):
    # Robustly fetch a single scalar coefficient like 'x:c(0)' from a simsopt curve.
    # Will raise if not present (which is good — it flags non-Fourier curves early).
    if key=='x':
        k=0
    elif key=='y':
        k=33
    elif key=='z':
        k=66
    val = curve.x[k]
    # Make sure we return a python float / scalar
    return float(val.item())

def _coil_centroid_and_phi_fourier(coil):
    """
    Use the zeroth cosine Fourier coefficients (c(0)) of x,y,z as the centroid.
    This is exact for CurveXYZFourier and avoids any sampling.
    """
    curve = getattr(coil, "curve", coil)
    x0 = _get_c0(curve, 'x')
    y0 = _get_c0(curve, 'y')
    z0 = _get_c0(curve, 'z')
    phi = (np.arctan2(y0, x0) + 2*np.pi) % (2*np.pi)
    return x0, y0, z0, phi

def _coil_current_scalar(coil):
    # Use same access you used earlier for consistency. Adjust if needed.
    # Fallbacks (commented) are typical simsopt APIs if your object differs.
    try:
        return float(coil.current.current_to_scale.current)
    except Exception:
        try:
            return float(coil.current.get_value()/(795774.7154594767))
        except Exception:
            return float(coil.current)

# %%
def process_coils(file_path):
    try:
        id = str(file_path)[-12:-5]
        surfaces, coils = load(str(file_path))
        s = surfaces[-1]

        # --- compute per-coil stats ---
        nfp = int(s.nfp)
        sector_width = 2*np.pi / nfp
        num_coils = len(coils) // (s.nfp * 2)
        num_coils = min(num_coils, MAX_COILS)
        base_indices = list(range(num_coils))

        # --- compute quick stats via Fourier c(0) ---
        stats = []
        for idx in base_indices:
            coil = coils[idx]
            x0, y0, z0, phi = _coil_centroid_and_phi_fourier(coil)
            # reduced angle in [0, sector_width)
            k = int(np.floor(phi / sector_width)) % nfp
            phi_red = phi - k * sector_width
            I = _coil_current_scalar(coil)
            stats.append({
                "idx": idx,
                "phi": phi,
                "phi_red": phi_red,
                "z0": z0,
                "I": I
            })

        # --- canonical sort (no sampling, sector-invariant via phi_red) ---
        # Primary: phi within a sector; then prefer upper coils (z0 high), then current sign/mag.
        stats.sort(key=lambda st: (st["phi_red"], -st["z0"], -np.sign(st["I"]), -abs(st["I"])))

        # --- how many to keep? ---
        # If the file contains all toroidal copies, base_indices typically already de-duplicates them.
        # Keep up to MAX_COILS in that canonical order.
        num_coils = min(len(stats), MAX_COILS)

        coil_array = np.zeros((MAX_COILS, FEATURES_PER_COIL), dtype=np.float32)
        # order_indices = []  # if you want to track permutation

        for dst_i in range(num_coils):
            src_idx = stats[dst_i]["idx"]
            # pack features: [current, last 99 Fourier params] (as in your original code)
            params = coils[src_idx].x[-99:]  # adjust if your layout changes
            curr = np.array(_coil_current_scalar(coils[src_idx]), dtype=np.float32)
            coil_array[dst_i] = np.concatenate([curr[None], params.astype(np.float32)], axis=0)
            # order_indices.append(src_idx)

        coil_mask = np.array([1]*num_coils + [0]*(MAX_COILS - num_coils), dtype=np.int64)

        # Optional: return order_indices if you want to log them or store in TFRecord
        return id, coil_array, coil_mask  # , order_indices

    except Exception as e:
        print(f"Failed on {file_path}: {e}")
        return None

# %%
def process_surface(file_path):
    """
    Convert s.x and metadata to [N_modes, 5] array:
    columns = [m_norm, n_norm, is_cos, is_R, coeff_value]
    """
    try:
        # id = str(file_path)[-12:-5]
        surfaces, coils = load(str(file_path))
        outer_surface = surfaces[-1]
        s = outer_surface.to_RZFourier() #the bottleneck in speed

        x = s.x  # the coeff vector
        nfp = s.nfp

        num_coefs = len(x)
        if num_coefs > MAX_COEFS:
            print(num_coefs)
        
        # m = s.m  # mode numbers, shape (N_modes,)
        # n = s.n
        
        # num_modes = (len(m)+1)//2
        # # Normalize mode indices
        # max_m = np.max(np.abs(m)) or 1
        # max_n = np.max(np.abs(n)) or 1

        # # Type flags
        # is_cos = np.concatenate([
        #     np.ones(num_modes, dtype=bool),  # R_cos
        #     np.zeros(num_modes-1, dtype=bool)  # Z_sin
        # ])

        surface_set = np.zeros((MAX_COEFS+1,), dtype=np.float32)
        surface_mask = np.zeros((MAX_COEFS,), dtype=np.int64)
        nfp_norm = float(nfp) / MAX_NFP 

        # for i in range(min(num_coefs, MAX_COEFS)):
        #     surface_set[i] = [
        #         m[i] / max_m,
        #         n[i] / max_n,
        #         float(is_cos[i]),
        #         x[i]
        #     ]
        #     surface_mask[i] = 1
        surface_set[:num_coefs] = x
        surface_set[-1] = nfp_norm
            
        return surface_set, surface_mask #serialize_surface(id, surface_set, surface_mask)

    except Exception as e:
        print(f"Failed on {file_path}: {e}")
        return None

# %%
def process_file(file_path):
    id, coil_array, coil_mask = process_coils(file_path)

    try: 
        surface_set, surface_mask = process_surface(file_path)
        return serialize_data(id, coil_array, coil_mask, surface_set, surface_mask)
    except Exception as e:
        print(f'{file_path} failure handled.')
        return None

# %%
def write_tfrecord_chunk(serialized_examples, output_path):
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for ex in serialized_examples:
            if ex:
                writer.write(ex)

def datasets_to_tfrecords(directory: Path, output_dir: Path, idx,
                               chunk_size=CHUNK_SIZE, num_workers=NUM_PROCESS_WORKERS):
    files = list(directory.glob("*.json"))
    total_files = len(files)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        serialized_examples = list(tqdm(
            executor.map(process_file, files),
            total=total_files,
            desc=f"Chunk {idx:03d}"
        ))

    serialized_examples = [ex for ex in serialized_examples if ex is not None]

    output_path = output_dir / f"surface_coil_chunk_{idx:03d}.tfrecord"

    write_tfrecord_chunk(serialized_examples, output_path)
    print(f"✅ Saved {len(serialized_examples)} surface coil pair samples to {output_dir}")


# %%
for idx, chunk in enumerate(chunks, start=15):
    print(f"\n=== Chunk {idx}/{len(chunks)}: {len(chunk)} devices ===")
    results = []

    os.makedirs(SIMSOPT_DIR, exist_ok=True)

    with ThreadPoolExecutor(max_workers=NUM_THREAD_WORKERS) as executor:
        futures = {executor.submit(process_device, dev): dev for dev in chunk}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Chunk {idx}"):
            results.append(fut.result())
    
    datasets_to_tfrecords(directory=data_dir, output_dir=output_dir, idx=idx)

    log_df = pd.concat([log_df, pd.DataFrame(results)], ignore_index=True)
    log_df.to_csv(LOG_CSV, index=False)
    success = sum(r["status"] == "success" for r in results)
    print(f"Chunk {idx} completed: {success}/{len(results)} successful.")

    try:
        shutil.rmtree(SIMSOPT_DIR)
    except OSError as e:
        print(f'Error deleting directory: {e}')


