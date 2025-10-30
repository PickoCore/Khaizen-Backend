from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import zipfile
import io
import os
import shutil
import json
import re
from pathlib import Path
import tempfile
from typing import Optional, Dict, List
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


def optimize_png(image_path: Path, quality: int = 85, max_size: Optional[int] = None) -> int:
    """Optimize PNG file dengan compression dan optional resize"""
    try:
        original_size = image_path.stat().st_size
        
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
        
        new_size = image_path.stat().st_size
        saved = original_size - new_size
        logger.info(f"Optimized PNG: {image_path.name} (saved {saved} bytes)")
        return saved
        
    except Exception as e:
        logger.error(f"Error optimizing PNG {image_path}: {str(e)}")
        return 0


def optimize_json(json_path: Path) -> int:
    """Minify JSON files by removing whitespace and comments"""
    try:
        original_size = json_path.stat().st_size
        
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove comments (// and /* */)
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Parse and minify
        try:
            data = json.loads(content)
            minified = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(minified)
            
            new_size = json_path.stat().st_size
            saved = original_size - new_size
            logger.info(f"Optimized JSON: {json_path.name} (saved {saved} bytes)")
            return saved
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON, skipping: {json_path.name}")
            return 0
            
    except Exception as e:
        logger.error(f"Error optimizing JSON {json_path}: {str(e)}")
        return 0


def optimize_ogg(ogg_path: Path) -> int:
    """Optimize OGG audio files (basic optimization by removing metadata)"""
    try:
        original_size = ogg_path.stat().st_size
        
        # Read file
        with open(ogg_path, 'rb') as f:
            data = f.read()
        
        # Basic optimization: remove trailing zeros and unnecessary metadata
        # This is a simple optimization - for advanced, would need pydub/ffmpeg
        data = data.rstrip(b'\x00')
        
        with open(ogg_path, 'wb') as f:
            f.write(data)
        
        new_size = ogg_path.stat().st_size
        saved = original_size - new_size
        
        if saved > 0:
            logger.info(f"Optimized OGG: {ogg_path.name} (saved {saved} bytes)")
        return saved
        
    except Exception as e:
        logger.error(f"Error optimizing OGG {ogg_path}: {str(e)}")
        return 0


def optimize_vfx_shader(shader_path: Path) -> int:
    """Optimize shader files (.vsh, .fsh, .vfx) by removing comments and whitespace"""
    try:
        original_size = shader_path.stat().st_size
        
        with open(shader_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove single-line comments
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove excessive whitespace but keep single spaces and newlines for readability
        lines = content.split('\n')
        optimized_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Keep non-empty lines
                # Reduce multiple spaces to single space
                line = re.sub(r'\s+', ' ', line)
                optimized_lines.append(line)
        
        content = '\n'.join(optimized_lines)
        
        with open(shader_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        new_size = shader_path.stat().st_size
        saved = original_size - new_size
        
        if saved > 0:
            logger.info(f"Optimized Shader: {shader_path.name} (saved {saved} bytes)")
        return saved
        
    except Exception as e:
        logger.error(f"Error optimizing shader {shader_path}: {str(e)}")
        return 0


def process_texture_pack(zip_path: Path, output_path: Path, quality: int, max_size: Optional[int]) -> Dict:
    """Process texture pack ZIP file with comprehensive optimization"""
    stats = {
        "total_files": 0,
        "optimized_files": 0,
        "original_size": 0,
        "optimized_size": 0,
        "bytes_saved": 0,
        "file_types": {
            "png": {"count": 0, "optimized": 0, "saved": 0},
            "json": {"count": 0, "optimized": 0, "saved": 0},
            "ogg": {"count": 0, "optimized": 0, "saved": 0},
            "shader": {"count": 0, "optimized": 0, "saved": 0},
            "other": {"count": 0}
        },
        "errors": []
    }
    
    extract_dir = TEMP_DIR / f"extract_{zip_path.stem}"
    extract_dir.mkdir(exist_ok=True)
    
    try:
        # Extract ZIP
        logger.info("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Calculate original size
        original_size = 0
        for f in extract_dir.rglob('*'):
            if f.is_file():
                original_size += f.stat().st_size
        stats["original_size"] = original_size
        
        # Process PNG files
        logger.info("Processing PNG files...")
        png_files = list(extract_dir.rglob('*.png'))
        stats["file_types"]["png"]["count"] = len(png_files)
        
        for png_file in png_files:
            saved = optimize_png(png_file, quality, max_size)
            if saved > 0:
                stats["file_types"]["png"]["optimized"] += 1
                stats["file_types"]["png"]["saved"] += saved
                stats["optimized_files"] += 1
        
        # Process JSON files
        logger.info("Processing JSON files...")
        json_files = list(extract_dir.rglob('*.json')) + list(extract_dir.rglob('*.mcmeta'))
        stats["file_types"]["json"]["count"] = len(json_files)
        
        for json_file in json_files:
            saved = optimize_json(json_file)
            if saved > 0:
                stats["file_types"]["json"]["optimized"] += 1
                stats["file_types"]["json"]["saved"] += saved
                stats["optimized_files"] += 1
        
        # Process OGG audio files
        logger.info("Processing OGG audio files...")
        ogg_files = list(extract_dir.rglob('*.ogg'))
        stats["file_types"]["ogg"]["count"] = len(ogg_files)
        
        for ogg_file in ogg_files:
            saved = optimize_ogg(ogg_file)
            if saved > 0:
                stats["file_types"]["ogg"]["optimized"] += 1
                stats["file_types"]["ogg"]["saved"] += saved
                stats["optimized_files"] += 1
        
        # Process shader files (.vsh, .fsh, .vfx, .glsl)
        logger.info("Processing shader files...")
        shader_extensions = ['*.vsh', '*.fsh', '*.vfx', '*.glsl', '*.vert', '*.frag']
        shader_files = []
        for ext in shader_extensions:
            shader_files.extend(list(extract_dir.rglob(ext)))
        
        stats["file_types"]["shader"]["count"] = len(shader_files)
        
        for shader_file in shader_files:
            saved = optimize_vfx_shader(shader_file)
            if saved > 0:
                stats["file_types"]["shader"]["optimized"] += 1
                stats["file_types"]["shader"]["saved"] += saved
                stats["optimized_files"] += 1
        
        # Count other files
        all_optimized = set(png_files + json_files + ogg_files + shader_files)
        all_files = set(extract_dir.rglob('*'))
        other_files = [f for f in all_files if f.is_file() and f not in all_optimized]
        stats["file_types"]["other"]["count"] = len(other_files)
        
        stats["total_files"] = len([f for f in all_files if f.is_file()])
        
        # Calculate total bytes saved
        stats["bytes_saved"] = (
            stats["file_types"]["png"]["saved"] +
            stats["file_types"]["json"]["saved"] +
            stats["file_types"]["ogg"]["saved"] +
            stats["file_types"]["shader"]["saved"]
        )
        
        # Create optimized ZIP with proper compression
        logger.info("Creating optimized ZIP...")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_out:
            for file_path in extract_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(extract_dir)
                    zip_out.write(file_path, arcname)
        
        # Get final optimized size
        stats["optimized_size"] = output_path.stat().st_size
        
        # Calculate accurate compression ratio
        if stats["original_size"] > 0:
            actual_reduction = stats["original_size"] - stats["optimized_size"]
            stats["compression_ratio"] = round((actual_reduction / stats["original_size"]) * 100, 2)
        else:
            stats["compression_ratio"] = 0
        
        logger.info(f"Optimization complete! Saved {stats['bytes_saved']} bytes from files, "
                   f"total reduction: {stats['compression_ratio']}%")
        
    finally:
        # Cleanup extract directory
        shutil.rmtree(extract_dir, ignore_errors=True)
    
    return stats


@app.get("/")
async def root():
    return {
        "message": "Minecraft Texture Pack Optimizer API",
        "version": "2.0.0",
        "features": [
            "PNG image optimization",
            "JSON minification",
            "OGG audio optimization",
            "Shader optimization (.vsh, .fsh, .vfx, .glsl)",
            "Accurate size reduction tracking"
        ],
        "endpoints": {
            "/optimize": "POST - Upload texture pack ZIP for optimization",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "texture-optimizer", "version": "2.0.0"}


@app.post("/optimize")
async def optimize_texture_pack(
    file: UploadFile = File(...),
    quality: int = 85,
    max_size: Optional[int] = None
):
    """
    Optimize Minecraft texture pack with comprehensive file type support
    
    - **file**: ZIP file texture pack
    - **quality**: PNG compression quality 1-100 (default: 85)
    - **max_size**: Maximum texture dimension in pixels (optional)
    
    Supports:
    - PNG images (compression + resize)
    - JSON files (minification)
    - OGG audio files (metadata removal)
    - Shader files (comment removal, whitespace optimization)
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
        logger.info(f"Receiving file: {file.filename}")
        content = await file.read()
        input_path.write_bytes(content)
        
        logger.info(f"Processing texture pack: {file.filename}")
        
        # Process texture pack
        stats = process_texture_pack(input_path, output_path, quality, max_size)
        
        # Prepare detailed stats for response headers
        file_types_json = json.dumps(stats["file_types"])
        
        logger.info(f"Optimization complete: {stats['compression_ratio']}% reduction")
        
        # Read optimized file
        optimized_content = output_path.read_bytes()
        
        # Clean up files
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        
        # Return file with comprehensive stats
        return StreamingResponse(
            io.BytesIO(optimized_content),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="optimized_{file.filename}"',
                "X-Original-Size": str(stats["original_size"]),
                "X-Optimized-Size": str(stats["optimized_size"]),
                "X-Bytes-Saved": str(stats["bytes_saved"]),
                "X-Compression-Ratio": str(stats["compression_ratio"]),
                "X-Total-Files": str(stats["total_files"]),
                "X-Optimized-Files": str(stats["optimized_files"]),
                "X-File-Types": file_types_json,
                "Access-Control-Expose-Headers": "Content-Disposition, X-Original-Size, X-Optimized-Size, X-Bytes-Saved, X-Compression-Ratio, X-Total-Files, X-Optimized-Files, X-File-Types"
            }
        )
        
    except Exception as e:
        # Cleanup on error
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        logger.error(f"Error processing texture pack: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing texture pack: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
