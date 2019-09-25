'''
Code for extracting frames from videos at a rate of 25fps in RGB diff methods

Need to extract 26fps RGB, using Numpy to calculate the difference.

Usage:
    python extract_frames_RGBdiff.py video_dir frame_dir

    video_dir ==> path of video files
    frame_dir ==> path of extracted RGBdiff jpg frames
'''

import sys, os, pdb, cv2, subprocess, struct
import numpy as np
from tqdm import tqdm
import torch
import torchvision as tv
from PIL import Image

I2T = tv.transforms.ToTensor()
T2I = tv.transforms.ToPILImage()

def cal_diff(frame1_pth, frame2_pth):
    fm1 = Image.open(frame1_pth)
    fm2 = Image.open(frame2_pth)
    fm1_t = I2T(fm1)
    fm2_t = I2T(fm2)
    fm_diff_t = fm2_t - fm1_t
    fm_diff = T2I(fm_diff_t)
    return fm_diff

def cal_diff_all(frames_file, tmp_outdir, outdir):
    for i in range(len(frames_file)-1):
        f1 = tmp_outdir + frames_file[i]
        f2 = tmp_outdir + frames_file[i+1]
        fm_diff = cal_diff(f1, f2)
        fm_diff.save(outdir + ('"%05d".jpg') % i, 'jpeg')
    

def extract_RGBdiff(vid_dir, frame_dir, start, end, redo=False):
    class_list = sorted(os.listdir(vid_dir))[start:end]
    
    print("Classes =", class_list)
    
    for ic, cls in enumerate(class_list):
        vlist = sorted(os.listdir(vid_dir + cls))
        print("")
        print(ic+1, len(class_list), cls, len(vlist))
        print("")
        for v in tqdm(vlist):
            outdir = os.path.join(frame_dir, cls, v[:-4])
            tmp_outdir = os.path.join(frame_dir, cls, v[:-4], 'tmp')
            
            # Checking if frames already extracted
            if os.path.isfile(os.path.join(outdir, 'done')) and not redo: continue
            try:
                os.system('mkdir -p "%s"'%(tmp_outdir))
                # check if horizontal or vertical scaling factor
                o = subprocess.check_output('ffprobe -v error -show_entries stream=width, height -of default=noprint_wrappers=1 "%s"' %(os.path.join(vid_dir, cls, v)), shell=True).decode('utf-8')
                lines = os.splitlines()
                width = int(lines[0].split('=')[1])
                height = int(lines[0].split('=')[1])
                resize_str = '-1:256' if width>height else '256:-1'

                # extract frames
                os.system('ffmpeg -i "%s" -r 26 -q:v 2 -vf "scale=%s" "%s" > /dev/null 2>&1'%(os.path.join(vid_dir, cls, v), resize_str, os.path.join(tmp_outdir, '%05d.jpg')))
                frames_file = [ fname for fname in os.listdir(tmp_outdir) if fname.endswith('.jpg') and len(fname)==9 ]
                tmp_nframes = len(frames_file)
                if tmp_nframes==0: raise Exception
                cal_diff_all(frames_file, tmp_outdir, outdir)

                nframes = len([ fname for fname in os.listdir(outdir) if fname.endswith('jpg') and len(fname)==9 ])
                if tmp_nframes==0: raise Exception

                os.system('rm -rf "%s"'%(tmpdir))
                os.system('touch "%s"'%(os.path.join(outdir, 'done') ))
            except:
                print("ERROR", cls, v)

if __name__ == '__main__':
    vid_dir = sys.argv[1]
    frame_dir = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    extract_RGBdiff(vid_dir, frame_dir, start, end, redo=True)
