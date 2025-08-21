# ASR Testing & Analysis UI - Mac M1 Oppsett

Dette er oppsettinstruksjoner spesifikt for Mac M1/M2/M3 (Apple Silicon).

## Forutsetninger

- Mac med Apple Silicon (M1/M2/M3)
- Docker Desktop for Mac med Apple Silicon-støtte
- Minst 8 GB RAM (16 GB anbefales)

## Installasjon

### 1. Docker Desktop Oppsett

Sørg for at du har Docker Desktop installert med Apple Silicon-støtte:

```bash
# Sjekk at Docker kjører på ARM64
docker version --format '{{.Server.Arch}}'
# Skal vise "arm64"
```

### 2. Klon og Oppsett

```bash
git clone <repository-url>
cd ASR_ABTEST
```

### 3. Bygg og start containere

```bash
# Bygg images optimalisert for Mac M1
docker-compose up --build

# Eller kjør i bakgrunnen
docker-compose up --build -d
```

### 4. Tilgang til tjenestene

- **ASR Server**: http://localhost:8000
- **UI**: http://localhost:7860

## Mac M1 Optimaliseringer

### PyTorch MPS Støtte

Dette oppsettet inkluderer:
- PyTorch 2.0+ med Metal Performance Shaders (MPS) støtte
- ARM64-optimaliserte base-images
- Automatisk fallback til CPU hvis MPS ikke er tilgjengelig

### Ytelsestips

1.  **Minnehåndtering**: Mac M1 bruker unified memory. Lukk andre tunge apper mens du kjører ASR-systemet.
2.  **Docker Ressurser**: Øk Docker sin minneallokering i Docker Desktop-innstillingene:
    - Gå til Docker Desktop > Settings > Resources
    - Sett minne til minst 6 GB (8 GB anbefales)
3.  **Termisk Håndtering**: Ved intensive arbeidsbelastninger kan Mac M1 strupe ytelsen. Sørg for god ventilasjon.

## Feilsøking

### Vanlige Problemer

1.  **"No module named 'pandas'"**
    ```bash
    # Hvis du får denne feilen, installer avhengigheter manuelt:
    docker-compose exec asr-server pip install -r requirements-mac.txt
    ```
2.  **Treg Ytelse**
    ```bash
    # Sjekk om MPS er tilgjengelig:
    docker-compose exec asr-server python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
    ```
3.  **Minnefeil**
    ```bash
    # Restart containere med mindre batch size
    docker-compose down
    docker-compose up --build
    ```

### Feilsøkingskommandoer

```bash
# Se logger
docker-compose logs asr-server
docker-compose logs ui-app

# Shell-tilgang til container
docker-compose exec asr-server bash
docker-compose exec ui-app bash

# Sjekk systemressurser
docker stats
```

## Utvikling

### Live Reload

Koden er montert som et volum, så endringer reflekteres umiddelbart uten en rebuild.

### Rebuild etter endringer i avhengigheter

```bash
docker-compose down
docker-compose up --build
```

## Ytelsessammenligning

På Mac M1 kan du forvente:
- **CPU-only**: Grunnleggende ytelse
- **Med MPS**: 2-3x hastighetsøkning for PyTorch-operasjoner
- **Minne**: Lavere minnebruk takket være unified memory

## Alternativt Oppsett (uten Docker)

Hvis du foretrekker å kjøre direkte på macOS:

```bash
# Opprett et conda-miljø
conda create -n asr python=3.10
conda activate asr

# Installer avhengigheter
pip install -r requirements-mac.txt

# Installer prosjektet
pip install -e .

# Start serveren
serve-asr
```

## Lenker

- [PyTorch Mac M1 Documentation](https://pytorch.org/blog/pytorch-m1-support/)
- [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/)
- [Apple Silicon Performance Guide](https://developer.apple.com/documentation/accelerate)