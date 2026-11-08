"""Create a source-only server bundle with hashes; exclude data and environments."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    folders=['src','configs','examples','tests','docs','reference_v1','matlab','srcipts','native','tools']
    suffixes={'.py','.md','.json','.html','.h','.hpp','.cpp','.cu','.m','.sh','.yml','.yaml','.toml'}
    excluded={'__pycache__','.git','.venv','outputs','bw_grating_output','build','dist','migrations','cache','exports','input','experiments'}
    files=[root/name for name in ('AGENTS.md','README.md','pyproject.toml','THIRD_PARTY_NOTICES.md','run_model.py')]
    for name in folders:
        for path in (root/name).rglob('*'):
            relative=path.relative_to(root)
            if any(part in excluded for part in relative.parts) or path.is_symlink():continue
            if path.is_file() and (path.suffix.lower() in suffixes or path.name in ('LICENSE','CMakeLists.txt')):
                if not path.resolve().is_relative_to(root):raise ValueError('source escapes project')
                files.append(path)
    manifest={'schema':'mstd-v1-upload-v1','root':'mstd-v1-v050','files':{}}
    with tarfile.open(output,'x:gz') as archive:
        for path in sorted(set(files)):
            relative=path.relative_to(root).as_posix()
            data=path.read_bytes()
            info=tarfile.TarInfo('mstd-v1-v050/'+relative)
            info.size=len(data)
            info.mode=0o644
            archive.addfile(info,io.BytesIO(data))
            manifest['files'][relative]=hashlib.sha256(data).hexdigest()
        data=json.dumps(manifest,indent=2).encode('utf-8')
        info=tarfile.TarInfo('mstd-v1-v050/UPLOAD_MANIFEST.json')
        info.size=len(data);info.mode=0o644
        archive.addfile(info,io.BytesIO(data))
    print(json.dumps({'output':str(output),'files':len(files),'bytes':output.stat().st_size,
                      'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
