"""Decode review clips and register gate-relative kinematics against the arena rim."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, signal

from video_gate_response import BASE, DEST, FPS


def ring_patches(gray, arena):
    small = gray[::2,::2].astype(np.float32)
    dx = ndimage.sobel(small, axis=1)
    dy = ndimage.sobel(small, axis=0)
    xx, yy, xy = [ndimage.uniform_filter(m,7) for m in (dx*dx, dy*dy, dx*dy)]
    corner = (xx*yy-xy**2)/(xx+yy+1)
    y,x = np.ogrid[:small.shape[0],:small.shape[1]]
    rx, ry = 2*x-arena[0], 2*y-arena[1]
    radius = np.hypot(rx,ry)
    annulus = (radius>arena[2]+123) & (radius<arena[2]+163)
    sector = ((np.arctan2(ry,rx)+np.pi)*32/(2*np.pi)).astype(int)%32
    refs, patches = [], []
    for k in range(32):
        idx = np.argmax(np.where(annulus & (sector==k), corner, -1))
        row,col = np.unravel_index(idx,small.shape)
        x0,y0 = 2*col,2*row
        ref = gray[y0-12:y0+13,x0-12:x0+13].astype(float)
        ref -= ref.mean()
        refs.append((x0,y0))
        patches.append(ref / max(np.linalg.norm(ref),1e-8))
    return np.array(refs),patches


def find_patches(gray,refs,patches):
    measured,valid = [],[]
    for (x,y),patch in zip(refs,patches):
        tile = gray[y-27:y+28,x-27:x+28].astype(float)
        # Sliding-window normalization of the correlation with the reference patch.
        side = patch.shape[0]
        def window_sum(arr):
            s = np.pad(arr.cumsum(0).cumsum(1),((1,0),(1,0)))
            return s[side:,side:] + s[:-side,:-side] - s[side:,:-side] - s[:-side,side:]
        norm = np.sqrt(np.maximum(window_sum(tile**2)-window_sum(tile)**2/side**2,1e-5))
        corr = signal.correlate(tile,patch,mode='valid',method='fft') / norm
        iy,ix = np.unravel_index(corr.argmax(),corr.shape)
        peak = corr[iy,ix]
        ok = 0 < ix < 30 and 0 < iy < 30 and peak > .78
        def sub(a,b):
            return np.clip(.5*(a-b)/(a-2*peak+b-1e-15),-.5,.5)
        fx = sub(corr[iy,ix-1],corr[iy,ix+1]) if ok else 0.
        fy = sub(corr[iy-1,ix],corr[iy+1,ix]) if ok else 0.
        measured.append((x+ix-15+fx,y+iy-15+fy))
        valid.append(ok)
    return np.array(measured),np.array(valid)


def affine(obs,ref,valid):
    x = np.column_stack([obs,np.ones(len(obs))])
    for _ in range(3):
        if valid.sum()<8:
            raise ValueError('Too few arena landmarks for registration')
        matrix = np.linalg.lstsq(x[valid],ref[valid],rcond=None)[0]
        err = np.linalg.norm(x@matrix-ref,axis=1)
        valid = valid & (err < max(.75,3*np.median(err[valid])))
    matrix = np.linalg.lstsq(x[valid],ref[valid],rcond=None)[0]
    return matrix, np.linalg.norm(x@matrix-ref,axis=1)


def extract_recording(rows, paths, ffmpeg, cache):
    tag = rows.recording.iloc[0]
    n,pwm,diam = tag.split('_')
    videos = [p for root in paths for p in root.glob('*.MP4')
              if f'{n[1:]}_bots_PWM_{pwm[3:]}_' in p.name and f'D_{diam[1:]}cm' in p.name]
    if len(videos)!=1:
        raise ValueError(f'Expected one MP4 for {tag}, found {videos}')
    with np.load(BASE/f'{tag}.npz') as src:
        tracks, ids = src['raw_pos'],src['ids']
    arena = json.loads((BASE.parent/'N30_PWM10_D41/training.json').read_text())['arenas'][tag[-2:]]
    output = []
    for r in rows.itertuples():
        name = cache/f'{r.event:03}.npz'
        lo,hi = r.gate_frame-5,max(r.gate_frame+2,r.release+1)
        idx = [np.flatnonzero(ids==rid).item() for rid in (r.i,r.j)]
        xy = tracks[lo:hi+1,idx]
        rel = xy[:,1]-xy[:,0]
        if not name.exists():
            cmd = [ffmpeg,'-v','error','-threads','3','-ss',str(lo/FPS),'-i',str(videos[0]),
                   '-map','0:v:0','-frames:v',str(len(xy)),'-fps_mode','passthrough','-threads','1',
                   '-pix_fmt','rgb24','-f','rawvideo','pipe:1']
            crops,registered,errors = [],[],[]
            with subprocess.Popen(cmd,stdout=subprocess.PIPE) as proc:
                for k,pair in enumerate(xy):
                    buf = proc.stdout.read(3840*2160*3)
                    if len(buf)!=3840*2160*3:
                        raise ValueError(f'Short video read at {tag}, frame {lo+k}')
                    im = Image.frombytes('RGB',(3840,2160),buf)
                    if k<=7:
                        gray = np.asarray(im.convert('L'))
                        if k==0:
                            refs,patches = ring_patches(gray,arena)
                        obs,valid = find_patches(gray,refs,patches)
                        check = np.arange(len(obs))%4==0
                        _,err = affine(obs,refs,valid&~check)
                        matrix,_ = affine(obs,refs,valid)
                        registered.append(rel[k]@matrix[:2])
                        errors.append(np.median(err[check&valid]))
                    u = rel[k]/np.linalg.norm(rel[k])
                    v = np.array([-u[1],u[0]])
                    origin = pair.mean(0)-60*u-40*v
                    crop = im.transform((120,80),Image.Transform.AFFINE,
                        (u[0],v[0],origin[0],u[1],v[1],origin[1]),Image.Resampling.BILINEAR)
                    crops.append(np.asarray(crop))
                if proc.wait()!=0:
                    raise RuntimeError(f'FFmpeg failed for {tag}')
            np.savez_compressed(name,crops=crops,registered=registered,errors=errors,
                                key=[r.i,r.j,r.gate_frame],recording=tag)
        with np.load(name) as src:
            assert src['recording'].item()==tag and list(src['key'])==[r.i,r.j,r.gate_frame]
            reg,errs = src['registered'],src['errors']
        n0 = reg[5]/np.linalg.norm(reg[5])
        tangent = np.array([-n0[1],n0[0]])
        s = reg@n0
        before = (s[4]-s[2])*FPS/2
        output.append(dict(event=r.event,closing=-before,side=(reg[4]-reg[2])@tangent*FPS/2,
            lead_change=before-(s[2]-s[0])*FPS/2,change_40=(s[6]-s[5])*FPS-before,
            change_80=(s[7]-s[5])*FPS/2-before,registration_residual=float(np.median(errs))))
    return output


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir',type=Path,action='append',required=True)
    parser.add_argument('--ffmpeg',default='ffmpeg')
    parser.add_argument('--cache',required=True,type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True)
    groups = pd.read_csv(DEST/'selection.csv').groupby('recording')
    with ThreadPoolExecutor(3) as pool:
        tasks = [pool.submit(extract_recording,rows,args.video_dir,args.ffmpeg,args.cache)
                 for _,rows in groups]
        values = [v for task in tasks for v in task.result()]
    pd.DataFrame(values).to_csv(DEST/'kinematics.csv',index=False)
