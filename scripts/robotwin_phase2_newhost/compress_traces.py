"""Lossless storage conversion for completed, owned traces; verify before replacement."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

root=Path(__file__).resolve().parents[2]/'results/robotwin_phase2_newhost'
records=[]
zstd=shutil.which('zstd')
assert zstd
version=subprocess.check_output([zstd,'--version'],text=True).strip()
for folder in sorted((root/'episodes').iterdir()):
    summary_path=folder/'summary.json'
    if not folder.is_dir() or not summary_path.exists():continue
    summary=json.loads(summary_path.read_text())
    if summary['status']!='completed':continue
    source=folder/'physical_trace.jsonl.gz'
    if not source.exists():continue
    target=folder/'physical_trace.jsonl.zst'
    temporary=target.with_suffix('.zst.partial')
    assert not target.exists() and not temporary.exists(), 'Inspect interrupted compression before retry'
    before=hashlib.sha256();after=hashlib.sha256();count=0
    with subprocess.Popen(['gzip','-dc',str(source)],stdout=subprocess.PIPE) as decoder:
        with subprocess.Popen([zstd,'-q','-3','-o',str(temporary)],stdin=subprocess.PIPE) as encoder:
            for block in iter(lambda:decoder.stdout.read(1024**2),b''):
                before.update(block);count+=len(block);encoder.stdin.write(block)
            encoder.stdin.close()
            assert encoder.wait()==0
        assert decoder.wait()==0, 'Original gzip decode/CRC failed; original retained'
    with subprocess.Popen([zstd,'-q','-dc',str(temporary)],stdout=subprocess.PIPE) as decoder:
        for block in iter(lambda:decoder.stdout.read(1024**2),b''):after.update(block)
        assert decoder.wait()==0
    assert before.digest()==after.digest(), 'Lossless round-trip failed; original retained'
    temporary.rename(target)
    record=dict(episode=folder.name,uncompressed_sha256=before.hexdigest(),uncompressed_bytes=count,
                original_gzip_bytes=source.stat().st_size,zstd_bytes=target.stat().st_size,
                codec='zstandard',level=3,version=version,verified_lossless=True,unix=time.time())
    (folder/'trace_storage.json').write_text(json.dumps(record,indent=2))
    summary['physical_trace_file']=target.name
    summary_path.write_text(json.dumps(summary,indent=2))
    source.unlink()  # Only this run's completed trace, after complete decode/hash verification.
    records.append(record)
    print(json.dumps(record),flush=True)
print(json.dumps(dict(converted=len(records),saved_bytes=sum(r['original_gzip_bytes']-r['zstd_bytes'] for r in records))),flush=True)
