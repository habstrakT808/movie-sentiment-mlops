# Docker Disk Usage - Mengapa Folder WSL Besar?

## Masalah: Folder Docker WSL Memakan 165 GB

Folder `C:\Users\acern\AppData\Local\Docker\wsl` yang memakan **165 GB** adalah **normal** untuk Docker Desktop di Windows, tetapi bisa dibersihkan.

## Penyebab Ukuran Besar

### 1. **Docker Images** (Paling Besar)
- Base images (Ubuntu, Python, dll) bisa 1-5 GB per image
- Multiple versions dari image yang sama
- Dangling images (images tanpa tag)

### 2. **Build Cache**
- Layer-layer dari Docker build yang di-cache
- Bisa menumpuk setelah banyak build

### 3. **Containers**
- Containers yang sudah di-stop tapi tidak dihapus
- Logs dari containers

### 4. **Volumes**
- Data yang disimpan di Docker volumes
- Prometheus data (bisa besar jika retention lama)
- Grafana data

### 5. **WSL2 Virtual Disk**
- Docker Desktop menggunakan WSL2
- Virtual disk bisa grow tapi tidak shrink otomatis

## Solusi: Cleanup Docker

### Cara 1: Safe Cleanup Script (RECOMMENDED - Hanya Hapus yang Tidak Terpakai)

Script ini **AMAN** karena hanya menghapus:
- ✅ Dangling images (images tanpa tag)
- ✅ Build cache (layer-layer build yang tidak terpakai)
- ✅ Stopped containers (yang bukan dari project ini)
- ❌ **TIDAK** menghapus images/containers yang sedang digunakan
- ❌ **TIDAK** menghapus volumes yang digunakan aplikasi

**Windows PowerShell:**
```powershell
cd D:\Project\movie-sentiment-mlops
.\scripts\docker_safe_cleanup.ps1
```

**Linux/Mac/Git Bash:**
```bash
cd D:/Project/movie-sentiment-mlops
bash scripts/docker_safe_cleanup.sh
```

**Fitur:**
- ✅ Dry run mode (lihat apa yang akan dihapus tanpa menghapus)
- ✅ Melindungi containers yang sedang running
- ✅ Melindungi volumes project (Prometheus, Grafana)
- ✅ Pilihan cleanup yang aman

### Cara 2: Manual Cleanup Script (Lebih Agresif)

**Windows PowerShell:**
```powershell
cd D:\Project\movie-sentiment-mlops
.\scripts\docker_cleanup.ps1
```

**Linux/Mac/Git Bash:**
```bash
cd D:/Project/movie-sentiment-mlops
bash scripts/docker_cleanup.sh
```

### Cara 3: Manual Cleanup (Command Line)

#### 1. Check Disk Usage
```bash
docker system df
```

#### 2. Remove Unused Images
```bash
# Remove dangling images
docker image prune -f

# Remove all unused images (⚠️ careful!)
docker image prune -a -f
```

#### 3. Remove Stopped Containers
```bash
docker container prune -f
```

#### 4. Remove Unused Volumes
```bash
# ⚠️ WARNING: This may delete data!
docker volume prune -f
```

#### 5. Remove Build Cache
```bash
docker builder prune -f
```

#### 6. Remove Everything Unused (Nuclear Option)
```bash
# ⚠️ DANGEROUS: Removes ALL unused resources
docker system prune -a --volumes -f
```

### Cara 3: Docker Desktop Settings

1. Buka **Docker Desktop**
2. Klik **Settings** (⚙️)
3. Pilih **Resources** → **Advanced**
4. Klik **Clean / Purge data** untuk membersihkan

### Cara 4: Compact WSL2 Virtual Disk

Jika setelah cleanup masih besar, compact virtual disk:

```powershell
# 1. Stop Docker Desktop
# 2. Stop WSL
wsl --shutdown

# 3. Compact disk (run as Administrator)
wsl --export docker-desktop-data docker-desktop-data.tar
wsl --unregister docker-desktop-data
wsl --import docker-desktop-data C:\Users\acern\AppData\Local\Docker\wsl\data docker-desktop-data.tar --version 2
del docker-desktop-data.tar
```

**⚠️ WARNING:** Proses ini bisa memakan waktu lama dan berisiko. Backup dulu!

## Untuk Project Ini

### Volumes yang Digunakan

Dari `docker-compose.yml`, project ini menggunakan:

1. **prometheus_data** - Data Prometheus (bisa besar jika retention lama)
2. **grafana_data** - Data Grafana (relatif kecil)

### Cleanup Prometheus Data

Jika Prometheus data terlalu besar:

```bash
# Stop containers
docker-compose down

# Remove Prometheus volume
docker volume rm docker_prometheus_data

# Start again (akan create volume baru)
docker-compose up -d
```

Atau ubah retention time di `docker-compose.yml`:
```yaml
prometheus:
  command:
    - '--storage.tsdb.retention.time=200h'  # Kurangi dari 200h ke 24h
```

## Best Practices

### 1. Regular Cleanup
- Jalankan cleanup setiap 1-2 minggu
- Monitor disk usage dengan `docker system df`

### 2. Limit Build Cache
```bash
# Set build cache size limit
docker builder prune --filter "until=24h" -f
```

### 3. Remove Old Images
```bash
# Remove images older than 7 days
docker image prune -a --filter "until=168h" -f
```

### 4. Monitor Volume Sizes
```bash
# Check volume sizes
docker system df -v
```

## Expected Sizes

Setelah cleanup, ukuran normal:
- **Base images:** ~5-10 GB (Python, Ubuntu, dll)
- **Project images:** ~2-5 GB per image
- **Volumes:** ~1-5 GB (tergantung retention)
- **Total normal:** ~15-30 GB

Jika masih >50 GB setelah cleanup, ada yang tidak normal.

## Troubleshooting

### Issue: Cleanup tidak mengurangi ukuran

**Penyebab:** WSL2 virtual disk tidak shrink otomatis

**Solusi:** Compact virtual disk (lihat Cara 4 di atas)

### Issue: Tidak bisa delete volume

**Penyebab:** Container masih menggunakan volume

**Solusi:**
```bash
# Stop all containers
docker-compose down

# Remove volume
docker volume rm <volume_name>
```

### Issue: Disk masih penuh setelah cleanup

**Penyebab:**
- Images yang masih digunakan
- Volumes dengan data besar
- Build cache yang besar

**Solusi:**
1. Check dengan `docker system df -v`
2. Identify largest components
3. Remove secara selective

## Quick Commands Reference

```bash
# Check disk usage
docker system df

# Detailed breakdown
docker system df -v

# Remove dangling images
docker image prune -f

# Remove stopped containers
docker container prune -f

# Remove unused volumes
docker volume prune -f

# Remove build cache
docker builder prune -f

# Remove everything unused
docker system prune -a --volumes -f

# List all images
docker images

# List all containers
docker ps -a

# List all volumes
docker volume ls
```

## References

- [Docker System Prune Documentation](https://docs.docker.com/engine/reference/commandline/system_prune/)
- [Docker Disk Usage Guide](https://docs.docker.com/config/pruning/)
- [WSL2 Disk Management](https://learn.microsoft.com/en-us/windows/wsl/disk-space)
