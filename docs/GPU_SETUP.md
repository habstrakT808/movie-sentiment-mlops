# GPU Setup untuk Docker Container

## Masalah: Container Menggunakan CPU

Dari monitoring (Acer NitroSense), terlihat:
- **CPU**: 92°C, 65% load (sangat tinggi) ❌
- **GPU**: 41°C, 24% load (rendah) ❌

Ini menunjukkan training berjalan di **CPU**, bukan GPU.

## Penyebab

1. **GPU support di-comment** di `docker-compose.yml`
2. **PyTorch CPU-only** yang terinstall (`torch==2.5.1` tanpa CUDA)
3. **NVIDIA Container Toolkit** belum dikonfigurasi

## Solusi: Enable GPU Support

### Prerequisites

1. **NVIDIA GPU** dengan driver terinstall
2. **NVIDIA Container Toolkit** untuk Docker
3. **WSL2** dengan CUDA support (jika di Windows)

### Step 1: Install NVIDIA Container Toolkit (Windows + WSL2)

**Di WSL2 Ubuntu:**

```bash
# Update package list
sudo apt-get update

# Install prerequisites
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add NVIDIA GPG key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add NVIDIA repository
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker
```

**Atau di Windows (Docker Desktop):**

1. Install **NVIDIA GPU drivers** terbaru
2. Install **Docker Desktop** dengan WSL2 backend
3. Enable **WSL2 integration** di Docker Desktop settings
4. GPU support akan otomatis tersedia jika driver sudah terinstall

### Step 2: Update PyTorch untuk CUDA

**Edit `requirements.txt`:**

```txt
# Deep Learning
torch==2.5.1+cu121  # CUDA 12.1 version
# Atau install via pip: pip install torch --index-url https://download.pytorch.org/whl/cu121
transformers==4.35.2
```

**Atau install PyTorch dengan CUDA setelah container running:**

```bash
docker exec -it movie-sentiment-api pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Uncomment GPU Config di docker-compose.yml

**File sudah di-update** - GPU support sudah di-uncomment.

### Step 4: Rebuild Container dengan GPU Support

```bash
cd docker
docker-compose down
docker-compose build --no-cache sentiment-api
docker-compose up -d sentiment-api
```

### Step 5: Verify GPU Access

```bash
# Check CUDA availability
docker exec movie-sentiment-api python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Check GPU info
docker exec movie-sentiment-api nvidia-smi
```

**Expected output:**
```
CUDA available: True
Device count: 1
```

## Troubleshooting

### Issue: `nvidia-smi: command not found`

**Penyebab:** NVIDIA Container Toolkit belum terinstall atau Docker tidak detect GPU

**Solusi:**
1. Pastikan NVIDIA drivers terinstall di Windows
2. Restart Docker Desktop
3. Check WSL2 integration enabled

### Issue: `CUDA available: False`

**Penyebab:** PyTorch CPU-only version terinstall

**Solusi:**
1. Install PyTorch dengan CUDA:
   ```bash
   docker exec movie-sentiment-api pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
2. Restart container

### Issue: `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

**Penyebab:** NVIDIA Container Toolkit tidak terinstall atau tidak running

**Solusi:**
1. Install NVIDIA Container Toolkit (lihat Step 1)
2. Restart Docker daemon
3. Verify: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

## Performance Comparison

### CPU (Current)
- Training time: **~60-90 menit** (3 epochs)
- CPU usage: **65-100%**
- Temperature: **92°C** (sangat tinggi)

### GPU (After Setup)
- Training time: **~15-30 menit** (3 epochs) ⚡
- GPU usage: **80-100%**
- CPU usage: **10-20%**
- Temperature: **Normal** (GPU lebih efisien)

## Catatan Penting

1. **Windows + WSL2**: GPU support memerlukan setup khusus
2. **Docker Desktop**: Harus menggunakan WSL2 backend (bukan Hyper-V)
3. **NVIDIA Drivers**: Harus terinstall di Windows, bukan di WSL2
4. **CUDA Version**: Pastikan CUDA version match dengan PyTorch

## Quick Check Commands

```bash
# Check GPU di host
nvidia-smi

# Check GPU di container
docker exec movie-sentiment-api python -c "import torch; print(torch.cuda.is_available())"

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Alternative: Train di Host (Bukan Container)

Jika setup GPU di Docker terlalu kompleks, bisa train langsung di host:

```bash
# Aktifkan venv
source venv/Scripts/activate

# Train langsung (akan pakai GPU jika tersedia)
python src/models/train_transformer.py
```

Ini akan lebih mudah karena tidak perlu setup Docker GPU support.
