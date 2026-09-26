"""Resume official ZIP extraction per verified range; preserve existing files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/physics_poc'))
from asset_zip import RemoteFile

p=argparse.ArgumentParser()
p.add_argument('--archive',required=True)
p.add_argument('--size',type=int,required=True)
p.add_argument('--prefix',action='append',required=True)
p.add_argument('--destination',type=Path,required=True)
p.add_argument('--cache',type=Path,required=True)
a=p.parse_args()
url=f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/3dc3b798668feb99ac61cc9086d84cbcc3d79186/{a.archive}'
a.cache.mkdir(parents=True,exist_ok=True)
remote=RemoteFile(url,a.size)
with zipfile.ZipFile(remote) as z:
    missing=[]
    for item in z.infolist():
        if not any(item.filename.startswith(prefix) for prefix in a.prefix) or item.is_dir():continue
        assert not item.filename.startswith('/') and '..' not in PurePosixPath(item.filename).parts
        dest=a.destination/item.filename
        if dest.exists():
            data=dest.read_bytes()
            assert len(data)==item.file_size and zlib.crc32(data)==item.CRC, str(dest)
        else:missing.append(item)
    print(json.dumps(dict(missing_files=len(missing),missing_bytes=sum(x.file_size for x in missing))),flush=True)
    runs=[]
    for item in sorted(missing,key=lambda x:x.header_offset):
        low=item.header_offset
        high=min(a.size,low+30+len(item.filename.encode())+len(item.extra)+item.compress_size+256)
        if runs and low-runs[-1][1]<65536 and high-runs[-1][0]<32*1024**2:
            runs[-1][1]=high;runs[-1][2].append(item)
        else:runs.append([low,high,[item]])
    def fetch(run):
        low,high,items=run
        path=a.cache/f'{low}_{high}.range'
        if path.exists() and path.stat().st_size==high-low:return path
        for attempt in range(4):
            try:
                reader=RemoteFile(url,a.size);reader.seek(low)
                data=reader.read(high-low)
                temporary=path.with_suffix('.partial');temporary.write_bytes(data);temporary.replace(path)
                return path
            except Exception:
                if attempt==3:raise
                time.sleep(2*(attempt+1))
    with ThreadPoolExecutor(max_workers=3) as executor:
        for run,path in zip(runs,executor.map(fetch,runs)):
            low,high,items=run
            remote.cache=[(low,path.read_bytes())]
            for item in items:
                data=z.read(item)  # zipfile verifies member CRC
                dest=a.destination/item.filename;dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:f.write(data)
            remote.cache=[]
            path.unlink()  # Only this run's temporary verified range cache, never existing assets.
            print('EXTRACTED',low,high,len(items),flush=True)
print('ASSET_EXTRACTION_COMPLETE',flush=True)
