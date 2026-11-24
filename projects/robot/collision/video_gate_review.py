"""Build optical review sheets without displaying track distances or velocity outcomes."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from video_gate_response import DEST


def review_pages(cache, out):
    rows = pd.read_csv(DEST/'selection.csv')
    pages = []
    font = ImageFont.load_default(size=14)
    for page, first in enumerate(range(0,len(rows),10),1):
        batch = rows.iloc[first:first+10]
        sheet = Image.new('RGB',(960,48+86*len(batch)),'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8,8),'Optical review    |    Frame offsets from the common gate, in 40 ms steps',
                  fill='black',font=font)
        ready = True
        for line, item in enumerate(batch.itertuples()):
            path = cache/f'{item.event:03}.npz'
            if not path.exists():
                ready = False
                continue
            with np.load(path) as data:
                crops = data['crops']
            y = 38+line*86
            draw.text((5,y+18),f'{item.event:03}',fill='black',font=font)
            for k, crop in enumerate(crops[3:]):
                x=48+k*70
                patch=Image.fromarray(crop).crop((28,7,92,73))
                sheet.paste(patch,(x,y))
                draw.text((x+16,y+65),f'{k-2:+d}',fill='black',font=font)
        if ready:
            sheet.save(cache/f'review_{page:02}.png')
            pages.append(sheet)
    if len(pages)==(len(rows)+9)//10:
        pages[0].save(out,save_all=True,append_images=pages[1:],format='PDF',resolution=100,
                      quality=96,title='Robot encounters aligned to the common approach gate')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',required=True,type=Path)
    args=parser.parse_args()
    review_pages(args.cache,DEST/'frame_review.pdf')
