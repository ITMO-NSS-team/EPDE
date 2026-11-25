"""Measure independent body rims and marker rotation in sampled reviewed encounters."""
import argparse
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from shared_interaction_data import BASE, DEST


def measure(im,xy,other,diameter):
    rays = np.exp(1j*np.arange(96)*2*np.pi/96)
    p = complex(*xy)
    # The outside of an eligible radial edge must be clear of neighboring bodies.
    outside = p+(diameter/2+7)*rays
    clear = (abs(outside[:,None]-other[None,:])>diameter/2+3).all(1)
    radius = np.arange(round(diameter/2)-13,round(diameter/2)+14)
    coords = p+rays[:,None]*radius
    def sample(q):
        return map_coordinates(im,[q.imag,q.real],order=1,mode='nearest',output=float)
    contrast = sample(coords-3*rays[:,None])-sample(coords+3*rays[:,None])
    best = contrast.argmax(1)
    usable = clear&(contrast[np.arange(len(rays)),best]>12)&(best>0)&(best<len(radius)-1)
    points = coords[np.arange(len(rays)),best][usable]-p
    if len(points)<24 or abs(np.mean(rays[usable]))>.65:
        return None
    fit = least_squares(lambda b:abs(points-complex(*b[:2]))-b[2],
                        [0,0,diameter/2],loss='soft_l1',f_scale=1.4)
    if np.median(abs(fit.fun))>2.5 or abs(complex(*fit.x[:2]))>14:
        return None
    # Fourth angular harmonic of the square black marker, excluding its inner code.
    y,x = np.mgrid[-85:86,-85:86]
    region=x+1j*y
    sel=(abs(region)>42)&(abs(region)<82)
    q=region[sel]
    weight=np.maximum(125-sample(p+q),0)
    m=np.sum(weight*(q/abs(q))**4)/max(weight.sum(),1)
    return (*fit.x,float(np.median(abs(fit.fun))),len(points),np.angle(m)/4,abs(m))


def main(roots,ffmpeg):
    sel=pd.read_csv(BASE/'video_gate_response/selection.csv').merge(
        pd.read_csv(BASE/'video_gate_response/optical_labels.csv'),on=['event','recording','gate_frame','release'])
    out=[]
    for tag,g in sel.groupby('recording'):
        with np.load(BASE/f'{tag}.npz') as z:
            pos,phi,ids,d=z['raw_pos'],z['phi'],z['ids'],float(z['cap_px'])
        n,pwm,arena=tag.split('_')
        video=next(v for root in roots for v in root.glob('*.MP4') if
                   f'{n[1:]}_bots_PWM_{pwm[3:]}_' in v.name and f'D_{arena[1:]}cm' in v.name)
        for _,items in g.groupby('optical_label'):
            for row in items.sort_values('event').iloc[[len(items)//4,3*len(items)//4]].itertuples():
                start=row.gate_frame-2
                cmd=[str(ffmpeg),'-loglevel','error','-threads','2','-ss',str(start/25),'-i',str(video),
                     '-map','0:v:0','-frames:v','5','-fps_mode','passthrough','-threads','1','-pix_fmt','gray','-f','rawvideo','pipe:1']
                buf=subprocess.run(cmd,check=True,stdout=subprocess.PIPE).stdout
                frames=np.frombuffer(buf,np.uint8).reshape(5,2160,3840)
                for k,im in enumerate(frames):
                    t=start+k
                    for rid in (row.i,row.j):
                        ix=int(np.flatnonzero(ids==rid)[0])
                        other=np.delete(pos[t,:,0]+1j*pos[t,:,1],ix)
                        obs=measure(im,pos[t,ix],other,d)
                        if obs is not None:
                            out.append((tag,row.event,rid,t,phi[t,ix],*obs))
        print('Rims measured',tag,flush=True)
    columns=['recording','event','id','frame','phi','dx','dy','radius','rim_residual','rays',
             'marker_angle_mod_pi_2','marker_strength']
    pd.DataFrame(out,columns=columns).to_csv(DEST/'rim_centers.csv',index=False)


if __name__=='__main__':
    scratch = Path(tempfile.gettempdir())
    p=argparse.ArgumentParser()
    p.add_argument('--video-dir',type=Path,action='append')
    p.add_argument('--ffmpeg',type=Path,default=scratch/'robot_video_pilot_N30/ffmpeg.exe')
    a=p.parse_args()
    main(a.video_dir or [Path('D:/Downloads'),scratch/'robot_video_holdout'],a.ffmpeg)
