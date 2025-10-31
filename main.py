from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import zipfile
import rarfile
import py7zr
import io
import os
import shutil
import json
import re
from pathlib import Path
import tempfile
from typing import Optional, Dict, List
import logging
import asyncio

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

# Supported archive formats
SUPPORTED_FORMATS = ['.zip', '.rar', '.7z', '.tar', '.gz']


def cleanup_temp_files(*files):
    """Cleanup temporary files"""
    for file_path in files:
        try:
            if file_path and file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error cleaning up {file_path}: {e}")


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """Extract berbagai format archive dengan better error handling"""
    try:
        ext = archive_path.suffix.lower()
        logger.info(f"Extracting {ext} archive...")
        
        if ext == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif ext == '.rar':
            with rarfile.RarFile(archive_path, 'r') as rar_ref:
                rar_ref.extractall(extract_dir)
        elif ext == '.7z':
            with py7zr.SevenZipFile(archive_path, 'r') as seven_ref:
                seven_ref.extractall(extract_dir)
        else:
            logger.error(f"Unsupported archive format: {ext}")
            return False
        
        logger.info(f"Extracted {ext} successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting archive: {str(e)}")
        return False


def optimize_png(image_path: Path, quality: int = 85, max_size: Optional[int] = None) -> int:
    """Optimize PNG with better memory management"""
    try:
        original_size = image_path.stat().st_size
        
        with Image.open(image_path) as img:
            # Convert RGBA if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGBA', img.size, (255, 255, 255, 0))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if len(img.split()) > 3 else None)
                img = background
            
            # Resize if requested
            if max_size and max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save with optimization
            img.save(
                image_path,
                "PNG",
                optimize=True,
                quality=quality,
                compress_level=9
            )
        
        new_size = image_path.stat().st_size
        saved = original_size - new_size
        return max(saved, 0)
        
    except Exception as e:
        logger.error(f"Error optimizing PNG {image_path.name}: {str(e)}")
        return 0


def optimize_json(json_path: Path) -> int:
    """Minify JSON files"""
    try:
        original_size = json_path.stat().st_size
        
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove comments
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
            return max(saved, 0)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON, skipping: {json_path.name}")
            return 0
            
    except Exception as e:
        logger.error(f"Error optimizing JSON {json_path.name}: {str(e)}")
        return 0


def optimize_ogg(ogg_path: Path) -> int:
    """Optimize OGG audio files"""
    try:
        original_size = ogg_path.stat().st_size
        
        with open(ogg_path, 'rb') as f:
            data = f.read()
        
        # Remove trailing zeros
        data = data.rstrip(b'\x00')
        
        with open(ogg_path, 'wb') as f:
            f.write(data)
        
        new_size = ogg_path.stat().st_size
        saved = original_size - new_size
        return max(saved, 0)
        
    except Exception as e:
        logger.error(f"Error optimizing OGG {ogg_path.name}: {str(e)}")
        return 0


def optimize_shader(shader_path: Path) -> int:
    """Optimize shader files"""
    try:
        original_size = shader_path.stat().st_size
        
        with open(shader_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove comments
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Optimize whitespace
        lines = content.split('\n')
        optimized_lines = []
        for line in lines:
            line = line.strip()
            if line:
                line = re.sub(r'\s+', ' ', line)
                optimized_lines.append(line)
        
        content = '\n'.join(optimized_lines)
        
        with open(shader_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        new_size = shader_path.stat().st_size
        saved = original_size - new_size
        return max(saved, 0)
        
    except Exception as e:
        logger.error(f"Error optimizing shader {shader_path.name}: {str(e)}")
        return 0


def process_texture_pack(archive_path: Path, output_path: Path, quality: int, max_size: Optional[int]) -> Dict:
    """Process texture pack with improved performance"""
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
    
    extract_dir = TEMP_DIR / f"extract_{archive_path.stem}"
    extract_dir.mkdir(exist_ok=True)
    
    try:
        # Extract archive
        if not extract_archive(archive_path, extract_dir):
            raise Exception(f"Failed to extract {archive_path.suffix} file")
        
        # Calculate original size
        logger.info("Calculating original size...")
        original_size = sum(f.stat().st_size for f in extract_dir.rglob('*') if f.is_file())
        stats["original_size"] = original_size
        
        # Process PNG files
        logger.info("Processing PNG files...")
        png_files = list(extract_dir.rglob('*.png'))
        stats["file_types"]["png"]["count"] = len(png_files)
        
        for i, png_file in enumerate(png_files):
            if i % 50 == 0:
                logger.info(f"PNG: {i}/{len(png_files)}")
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
        
        # Process OGG files
        logger.info("Processing OGG files...")
        ogg_files = list(extract_dir.rglob('*.ogg'))
        stats["file_types"]["ogg"]["count"] = len(ogg_files)
        
        for ogg_file in ogg_files:
            saved = optimize_ogg(ogg_file)
            if saved > 0:
                stats["file_types"]["ogg"]["optimized"] += 1
                stats["file_types"]["ogg"]["saved"] += saved
                stats["optimized_files"] += 1
        
        # Process shader files
        logger.info("Processing shader files...")
        shader_extensions = ['*.vsh', '*.fsh', '*.vfx', '*.glsl', '*.vert', '*.frag']
        shader_files = []
        for ext in shader_extensions:
            shader_files.extend(list(extract_dir.rglob(ext)))
        
        stats["file_types"]["shader"]["count"] = len(shader_files)
        
        for shader_file in shader_files:
            saved = optimize_shader(shader_file)
            if saved > 0:
                stats["file_types"]["shader"]["optimized"] += 1
                stats["file_types"]["shader"]["saved"] += saved
                stats["optimized_files"] += 1
        
        # Count other files
        all_optimized = set(png_files + json_files + ogg_files + shader_files)
        all_files = [f for f in extract_dir.rglob('*') if f.is_file()]
        other_files = [f for f in all_files if f not in all_optimized]
        stats["file_types"]["other"]["count"] = len(other_files)
        stats["total_files"] = len(all_files)
        
        # Calculate bytes saved from optimization
        stats["bytes_saved"] = sum([
            stats["file_types"]["png"]["saved"],
            stats["file_types"]["json"]["saved"],
            stats["file_types"]["ogg"]["saved"],
            stats["file_types"]["shader"]["saved"]
        ])
        
        # Create optimized ZIP
        logger.info("Creating optimized ZIP...")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_out:
            for i, file_path in enumerate(all_files):
                if i % 100 == 0:
                    logger.info(f"Zipping: {i}/{len(all_files)}")
                arcname = file_path.relative_to(extract_dir)
                zip_out.write(file_path, arcname)
        
        # Get final size
        stats["optimized_size"] = output_path.stat().st_size
        
        # Calculate compression ratio
        if stats["original_size"] > 0:
            actual_reduction = stats["original_size"] - stats["optimized_size"]
            stats["compression_ratio"] = round((actual_reduction / stats["original_size"]) * 100, 2)
            stats["actual_bytes_saved"] = actual_reduction
        else:
            stats["compression_ratio"] = 0
            stats["actual_bytes_saved"] = 0
        
        logger.info(f"✓ Optimization complete! {stats['compression_ratio']}% reduction")
        
    except Exception as e:
        logger.error(f"Error in process_texture_pack: {str(e)}")
        raise
    finally:
        # Cleanup
        cleanup_temp_files(extract_dir)
    
    return stats


@app.get("/")
async def root():
    return {
        "message": "Minecraft Texture Pack Optimizer API",
        "version": "2.1.0",
        "status": "operational",
        "features": [
            "PNG optimization (compression + resize)",
            "JSON minification",
            "OGG audio optimization",
            "Shader optimization (.vsh, .fsh, .vfx, .glsl)",
            "Multi-format support (ZIP, RAR, 7Z)",
            "Fixed download for large files",
            "Better performance for 20MB+ files"
        ],
        "supported_formats": SUPPORTED_FORMATS
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "texture-optimizer",
        "version": "2.1.0",
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK)
    }


@app.post("/optimize")
async def optimize_texture_pack(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    quality: int = 85,
    max_size: Optional[int] = None
):
    """
    Optimize Minecraft texture pack (v2.1 - Improved for large files)
    
    Supports:
    - Archive formats: ZIP, RAR, 7Z
    - PNG images (compression + resize)
    - JSON files (minification)
    - OGG audio (metadata removal)
    - Shader files (code optimization)
    """
    
    # Validate file format
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Validate quality
    if not 1 <= quality <= 100:
        raise HTTPException(status_code=400, detail="Quality must be between 1-100")
    
    # Create unique filenames
    import uuid
    unique_id = str(uuid.uuid4())
    input_path = TEMP_DIR / f"input_{unique_id}{file_ext}"
    output_path = TEMP_DIR / f"optimized_{unique_id}.zip"
    
    try:
        # Save uploaded file
        logger.info(f"Receiving: {file.filename} ({file_ext})")
        content = await file.read()
        input_path.write_bytes(content)
        logger.info(f"Saved input file: {len(content)} bytes")
        
        # Process
        logger.info("Starting optimization...")
        stats = process_texture_pack(input_path, output_path, quality, max_size)
        
        # Verify output exists
        if not output_path.exists():
            raise Exception("Output file was not created")
        
        output_size = output_path.stat().st_size
        logger.info(f"Output file created: {output_size} bytes")
        
        # Prepare stats header
        file_types_json = json.dumps(stats["file_types"])
        
        # Prepare filename
        base_name = Path(file.filename).stem
        output_filename = f"optimized_{base_name}.zip"
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(cleanup_temp_files, input_path, output_path)
        
        # Return file using FileResponse (better for large files)
        return FileResponse(
            path=output_path,
            media_type="application/zip",
            filename=output_filename,
            headers={
                "X-Original-Size": str(stats["original_size"]),
                "X-Optimized-Size": str(stats["optimized_size"]),
                "X-Bytes-Saved": str(stats["bytes_saved"]),
                "X-Actual-Bytes-Saved": str(stats.get("actual_bytes_saved", 0)),
                "X-Compression-Ratio": str(stats["compression_ratio"]),
                "X-Total-Files": str(stats["total_files"]),
                "X-Optimized-Files": str(stats["optimized_files"]),
                "X-File-Types": file_types_json,
                "Access-Control-Expose-Headers": "X-Original-Size, X-Optimized-Size, X-Bytes-Saved, X-Actual-Bytes-Saved, X-Compression-Ratio, X-Total-Files, X-Optimized-Files, X-File-Types",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        # Cleanup on error
        cleanup_temp_files(input_path, output_path)
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
