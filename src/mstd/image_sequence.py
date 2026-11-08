"""Image-folder to VisualStimulus movie DAT, with an explicit frame-order audit."""
from io import BytesIO
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import struct

from .calibration import STIM_SIGNATURE, file_sha256, inspect_stimulus

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def natural_key(name):
    """Frame2 precedes frame10; case/leading-zero ties have a stable ordering."""
    return tuple(int(part) if part.isdigit() else part.casefold()
                 for part in re.split(r'(\d+)', name))


def convert_images(input_dir, output, *, mode='gray', frame_ms=50., geometry='native',
                   size=None, pattern='*', max_frames=None, progress_every=20):
    """Write a uint8, interleaved RGB or grayscale movie without changing inputs.

    geometry=native: preserve dimensions; reject inconsistent frames.
    geometry=resize: exact requested size, bilinear interpolation.
    geometry=pad: center on black; no scaling/cropping, requested size >= inputs.
    The DAT stores no timing; its companion JSON records frame_ms.
    """
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError('image conversion needs Pillow; install it in the existing environment: '
                           'python -m pip install Pillow') from error
    if mode not in ('gray', 'rgb') or geometry not in ('native', 'resize', 'pad'):
        raise ValueError('invalid image mode or geometry')
    if (isinstance(frame_ms, bool) or not isinstance(frame_ms, (int, float))
            or not isfinite(frame_ms) or frame_ms <= 0):
        raise ValueError('positive finite frame_ms required')
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError('positive integer max_frames required')
    if type(progress_every) is not int or progress_every < 0:
        raise ValueError('nonnegative progress_every required')
    if geometry == 'native' and size is not None:
        raise ValueError('--size requires --geometry resize or --geometry pad')
    if geometry != 'native' and (size is None or len(size) != 2
                                or any(type(v) is not int or v < 19 for v in size)):
        raise ValueError('resize/pad requires --size WIDTH HEIGHT, each >=19')
    source = Path(input_dir).resolve()
    if not source.is_dir():
        raise ValueError(f'image folder does not exist: {source}')
    paths = [p for p in source.glob(pattern) if p.is_file() and p.suffix.casefold() in IMAGE_SUFFIXES]
    paths.sort(key=lambda p: (natural_key(p.relative_to(source).as_posix()), p.as_posix()))
    available = len(paths)
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError('no matching JPG/PNG/BMP/TIFF image files')
    if len(paths) > 2**31-1:
        raise ValueError('too many frames for the DAT int32 header')
    # Header preflight avoids producing a partially valid-looking mixed-size movie.
    originals = []
    for path in paths:
        with Image.open(path) as image:
            if getattr(image, 'n_frames', 1) != 1:
                raise ValueError(f'one image per file required: {path.name}')
            if image.mode not in ('1', 'L', 'LA', 'P', 'RGB', 'RGBA'):
                raise ValueError(f'unsupported pixel mode {image.mode}: {path.name}; supply 8-bit images')
            originals.append(image.size)
    target = tuple(size) if size is not None else originals[0]
    if min(target) < 19 or max(target) > 2**31-1:
        raise ValueError('model-compatible width/height must be integers >=19 within int32')
    if geometry == 'native' and any(v != target for v in originals):
        raise ValueError('image sizes differ; use an explicit resize or pad policy')
    if geometry == 'pad' and any(w > target[0] or h > target[1] for w, h in originals):
        raise ValueError('pad target is smaller than an image; cropping is not performed')
    output = Path(output)
    metadata = output.with_suffix(output.suffix+'.json')
    if output.exists() or metadata.exists():
        raise FileExistsError(f'output DAT/JSON already exists: {output}')
    channels = 1 if mode == 'gray' else 3
    output.parent.mkdir(parents=True, exist_ok=True)
    created_dat = created_metadata = False
    entries = []
    try:
        with output.open('xb') as stream:
            created_dat = True
            stream.write(struct.pack('<ifiBiii', STIM_SIGNATURE, 1., 5, channels, *target, len(paths)))
            for index, (path, original_size) in enumerate(zip(paths, originals)):
                raw = path.read_bytes()
                with Image.open(BytesIO(raw)) as image:
                    if image.size != original_size or getattr(image, 'n_frames', 1) != 1:
                        raise ValueError(f'image changed during conversion: {path.name}')
                    if image.mode in ('RGBA', 'LA') or image.mode == 'P' and 'transparency' in image.info:
                        rgba = image.convert('RGBA')
                        background = Image.new('RGBA', rgba.size, (0, 0, 0, 255))
                        background.alpha_composite(rgba)
                        image = background.convert('RGB')
                    image = image.convert('L' if channels == 1 else 'RGB')
                    left = top = 0
                    if geometry == 'resize':
                        image = image.resize(target, Image.Resampling.BILINEAR)
                    elif geometry == 'pad':
                        left, top = (target[0]-image.width)//2, (target[1]-image.height)//2
                        padded = Image.new(image.mode, target, 0)
                        padded.paste(image, (left, top))
                        image = padded
                    payload = image.tobytes()
                if len(payload) != target[0]*target[1]*channels:
                    raise ValueError(f'invalid converted frame byte count: {path.name}')
                stream.write(payload)
                entries.append(dict(input_index=index, display_frame=index+1,
                                    source=path.relative_to(source).as_posix(),
                                    source_sha256=hashlib.sha256(raw).hexdigest(),
                                    source_width=original_size[0], source_height=original_size[1],
                                    pad_left=left, pad_top=top, timestamp_s=index*frame_ms/1000))
                if progress_every and (index == 0 or (index+1) % progress_every == 0 or index+1 == len(paths)):
                    print(f'[images-to-dat] {index+1}/{len(paths)} {path.name}', flush=True)
        info = inspect_stimulus(output)
        result = dict(schema='image-sequence-dat-v1', input_dir=str(source), stimulus=output.name,
                      stimulus_sha256=file_sha256(output), **info,
                      pixel_mode=mode, dtype='uint8', payload_layout='TYX' if channels == 1 else 'TYXC_RGB_interleaved',
                      frame_ms=frame_ms, fps=1000/frame_ms, duration_ms=len(paths)*frame_ms,
                      available_images=available, sorting='natural filename order; numeric tokens ascending',
                      geometry=geometry, resampling='bilinear' if geometry == 'resize' else None,
                      gray_conversion='Pillow convert L' if mode == 'gray' else None,
                      alpha_policy='composite on black', exif_orientation_policy='stored pixels; no rotation',
                      coordinates='x-right_y-down', frame_order=entries,
                      note='DAT contains no frame timing; set model frame_ms to this JSON value')
        with metadata.open('x', encoding='utf-8') as stream:
            created_metadata = True
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        return result
    except BaseException:
        # Remove only files created by this attempt, never existing inputs/outputs.
        if created_metadata:
            metadata.unlink(missing_ok=True)
        if created_dat:
            output.unlink(missing_ok=True)
        raise
