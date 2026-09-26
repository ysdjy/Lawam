"""Archive only this run's finalized traces, retaining verified workspace links."""
import hashlib
import json
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[2]/'results/robotwin_phase2_newhost'
archive=Path('/media/zbh/4E4A-BDAC/LaWAM_robotwin_phase2_newhost_20260925')
assert archive.parent.is_mount()
archive.mkdir(exist_ok=True)
records=[]
manifest=root/'evidence/external_storage.jsonl'
sources=list(root.glob('episodes/*/physical_trace.jsonl.zst'))+list(root.glob('attempts/*/physical_trace.jsonl.gz'))
for source in sorted(sources,key=lambda p:p.stat().st_size,reverse=True):
    if source.is_symlink():continue
    if shutil.disk_usage(root).free>16*1024**3:break
    status=json.loads((source.parent/'summary.json').read_text())['status']
    assert status=='completed' or (source.parent.parent.name=='attempts' and status=='system_error')
    target=archive/source.relative_to(root)
    assert not target.exists()
    assert shutil.disk_usage(archive).free>source.stat().st_size+10*1024**3
    target.parent.mkdir(parents=True,exist_ok=True)
    before=hashlib.sha256();after=hashlib.sha256()
    with source.open('rb') as reader,target.open('xb') as writer:
        for block in iter(lambda:reader.read(1024**2),b''):before.update(block);writer.write(block)
    with target.open('rb') as reader:
        for block in iter(lambda:reader.read(1024**2),b''):after.update(block)
    assert before.digest()==after.digest()
    record=dict(workspace=str(source),archive=str(target),bytes=target.stat().st_size,sha256=before.hexdigest(),verified=True,
                episode_status=status,verification='byte-identical copy; does not imply incomplete-attempt trace is finalized')
    with manifest.open('a') as f:f.write(json.dumps(record)+'\n')
    link=source.with_suffix('.archive_link');link.symlink_to(target);link.replace(source)
    records.append(record)
    print(json.dumps(record),flush=True)
print(json.dumps(dict(archived=len(records),root_free_bytes=shutil.disk_usage(root).free)),flush=True)
