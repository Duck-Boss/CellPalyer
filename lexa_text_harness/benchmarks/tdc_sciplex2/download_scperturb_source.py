"""Bounded resumable retrieval of the scPerturb authors' source file.

This is NOT claimed byte-identical to the inaccessible TDC Dataverse copy.
Only a complete file matching the published MD5 is promoted to .h5ad.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,math,time,urllib.request

ROOT=Path(__file__).resolve().parent
NAME='SrivatsanTrapnell2020_sciplex2.h5ad'
URL='https://zenodo.org/records/10044268/files/'+NAME+'?download=1'
SIZE=145178504
MD5='12a48293a166a859f994551653a2ad8c'
CHUNK=1048576

def dump(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)

def run(seconds=180,workers=8):
    data=ROOT/'data';parts=data/'range_parts';parts.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();deadline=start+seconds
    old=data/(NAME+'.partial')
    if old.exists():
        with old.open('rb') as f:
            for i in range(old.stat().st_size//CHUNK):
                b=f.read(CHUNK);p=parts/f'{i:04d}.bin'
                if not p.exists():p.write_bytes(b)
    n=math.ceil(SIZE/CHUNK)
    def partsize(i):return min(CHUNK,SIZE-i*CHUNK)
    def valid(i):
        p=parts/f'{i:04d}.bin';return p.exists() and p.stat().st_size==partsize(i)
    failures=[]
    def fetch(i):
        if valid(i):return {'index':i,'status':'cached'}
        lo=i*CHUNK;hi=min(SIZE,(i+1)*CHUNK)-1
        for attempt in range(2):
            if time.monotonic()>=deadline:return {'index':i,'status':'budget_expired'}
            try:
                req=urllib.request.Request(URL,headers={'Range':f'bytes={lo}-{hi}'})
                with urllib.request.urlopen(req,timeout=15) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{SIZE}':
                        raise ValueError(f'Range response mismatch: status={response.status}, '
                                         f'actual={response.headers.get("Content-Range")}, '
                                         f'expected=bytes {lo}-{hi}/{SIZE}')
                    b=bytearray()
                    while len(b)<partsize(i):
                        if time.monotonic()>=deadline:return {'index':i,'status':'budget_expired'}
                        item=response.read(min(65536,partsize(i)-len(b)))
                        if not item:raise ValueError('Short response')
                        b.extend(item)
                p=parts/f'{i:04d}.bin';tmp=p.with_suffix('.tmp');tmp.write_bytes(b);tmp.replace(p)
                return {'index':i,'status':'downloaded'}
            except Exception as e:
                error=type(e).__name__;error_detail=str(e)[:500]
        return {'index':i,'status':'failed','error_type':error,'error_detail':error_detail}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch,i) for i in range(n) if not valid(i)]
        for f in concurrent.futures.as_completed(futures):
            r=f.result()
            if r['status'] not in ['cached','downloaded']:failures.append(r)
    missing=[i for i in range(n) if not valid(i)]
    status={'status':'incomplete' if missing else 'downloaded_unverified','source_url':URL,
        'record_url':'https://zenodo.org/records/10044268','official_size_bytes':SIZE,'official_md5':MD5,
        'complete_parts':n-len(missing),'total_parts':n,'complete_bytes':sum(partsize(i) for i in range(n) if valid(i)),
        'missing_parts':missing,'failures':failures,'workers':workers,'wall_budget_seconds':seconds,
        'wall_seconds':time.monotonic()-start,
        'tdc_dataverse_file_id':9845394,'tdc_dataverse_byte_equivalence':'unverified: HTTP403',
        'is_leaderboard_ready':False}
    if not missing:
        target=data/NAME;tmp=data/(NAME+'.assembled')
        with tmp.open('wb') as out:
            for i in range(n):out.write((parts/f'{i:04d}.bin').read_bytes())
        actual=hashlib.md5(tmp.read_bytes()).hexdigest()
        if actual!=MD5:raise ValueError('Published checksum mismatch; assembled file preserved')
        tmp.replace(target)
        status.update(status='complete_source_file_md5_verified',sha256=hashlib.sha256(target.read_bytes()).hexdigest(),file=str(target))
    dump(data/'source_download_status.json',status)
    print(json.dumps({k:v for k,v in status.items() if k not in ['failures','missing_parts']},indent=2))
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=180);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();run(a.seconds,a.workers)
