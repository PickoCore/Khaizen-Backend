from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import zipfile
import io
import os
import shutil
from pathlib import Path
import tempfile
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minecraft Texture Pack Optimizer API")

# CORS - Allow Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain Vercel Anda di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary directory untuk processing
TEMP_DIR = Path(tempfile.gettempdir()) / "texture_optimizer"
TEMP_DIR.mkdir(exist_ok=True)


def optimize_png(image_path: Path, quality: int = 85, max_size: Optional[int] = None) -> None:
    """Optimize PNG file dengan compression dan optional resize"""
    try:
        with Image.open(image_path) as img:
            # Convert RGBA jika perlu
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGBA', img.size, (255, 255, 255, 0))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if len(img.split()) > 3 else None)
                img = background
            
            # Resize jika diminta
            if max_size and max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save dengan optimization
            img.save(
                image_path,
                "PNG",
                optimize=True,
                quality=quality,
                compress_level=9
            )
            logger.info(f"Optimized: {image_path.name}")
    except Exception as e:
        logger.error(f"Error optimizing {image_path}: {str(e)}")


def process_texture_pack(zip_path: Path, output_path: Path, quality: int, max_size: Optional[int]) -> dict:
    """Process texture pack ZIP file"""
    stats = {
        "total_files": 0,
        "optimized_files": 0,
        "original_size": 0,
        "optimized_size": 0,
        "errors": []
    }
    
    extract_dir = TEMP_DIR / f"extract_{zip_path.stem}"
    extract_dir.mkdir(exist_ok=True)
    
    try:
        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Get original size
        stats["original_size"] = sum(
            f.stat().st_size for f in extract_dir.rglob('*') if f.is_file()
        )
        
        # Process all PNG files
        png_files = list(extract_dir.rglob('*.png'))
        stats["total_files"] = len(png_files)
        
        for png_file in png_files:
            try:
                optimize_png(png_file, quality, max_size)
                stats["optimized_files"] += 1
            except Exception as e:
                stats["errors"].append(f"{png_file.name}: {str(e)}")
        
        # Create optimized ZIP
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for file_path in extract_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(extract_dir)
                    zip_out.write(file_path, arcname)
        
        # Get optimized size
        stats["optimized_size"] = output_path.stat().st_size
        
    finally:
        # Cleanup extract directory
        shutil.rmtree(extract_dir, ignore_errors=True)
    
    return stats


@app.get("/")
async def root():
    return {
        "message": "Minecraft Texture Pack Optimizer API",
        "version": "1.0.0",
        "endpoints": {
            "/optimize": "POST - Upload texture pack ZIP for optimization",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "texture-optimizer"}


@app.post("/optimize")
async def optimize_texture_pack(
    file: UploadFile = File(...),
    quality: int = 85,
    max_size: Optional[int] = None
):
    """
    Optimize Minecraft texture pack
    
    - **file**: ZIP file texture pack
    - **quality**: Compression quality (1-100, default 85)
    - **max_size**: Maximum texture dimension in pixels (optional)
    """
    
    # Validate file
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    # Validate quality
    if not 1 <= quality <= 100:
        raise HTTPException(status_code=400, detail="Quality must be between 1 and 100")
    
    # Create unique filenames
    import uuid
    unique_id = str(uuid.uuid4())
    input_path = TEMP_DIR / f"input_{unique_id}.zip"
    output_path = TEMP_DIR / f"optimized_{unique_id}.zip"
    
    try:
        # Save uploaded file
        content = await file.read()
        input_path.write_bytes(content)
        
        logger.info(f"Processing texture pack: {file.filename}")
        
        # Process texture pack
        stats = process_texture_pack(input_path, output_path, quality, max_size)
        
        # Calculate compression ratio
        if stats["original_size"] > 0:
            compression_ratio = (1 - stats["optimized_size"] / stats["original_size"]) * 100
        else:
            compression_ratio = 0
        
        stats["compression_ratio"] = round(compression_ratio, 2)
        stats["output_filename"] = f"optimized_{file.filename}"
        
        logger.info(f"Optimization complete: {compression_ratio:.2f}% reduction")
        
        # Return file with stats in headers
        return FileResponse(
            output_path,
            media_type="application/zip",
            filename=stats["output_filename"],
            headers={
                "X-Original-Size": str(stats["original_size"]),
                "X-Optimized-Size": str(stats["optimized_size"]),
                "X-Compression-Ratio": str(stats["compression_ratio"]),
                "X-Total-Files": str(stats["total_files"]),
                "X-Optimized-Files": str(stats["optimized_files"])
            },
            background=cleanup_files(input_path, output_path)
        )
        
    except Exception as e:
        # Cleanup on error
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        logger.error(f"Error processing texture pack: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing texture pack: {str(e)}")


def cleanup_files(*files):
    """Background task untuk cleanup files"""
    from starlette.background import BackgroundTask
    
    def cleanup():
        for file in files:
            try:
                if file.exists():
                    file.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up {file}: {str(e)}")
    
    return BackgroundTask(cleanup)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
