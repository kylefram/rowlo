import torch
import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.cluster import KMeans
from video_process_v1 import KEYPOINT_INDICES,OUTPUT_CSV_PATH,DEF_BOAT

def main():
    df=pd.read_csv(OUTPUT_CSV_PATH, index_col=[0,1,2])
    #attempts to rank each rower by median x position of nose keypoint
    df=repair_ids(df)
    df = assign_seats(df, seat_encoding=DEF_BOAT)

    df.sort_index(inplace=True)
    df.to_csv(OUTPUT_CSV_PATH.split('.csv')[0]+'_repaired.csv')

def repair_ids(df):
    #get and rank nose positions
    nose_frame = df.xs("Nose", level="keypoint")
    frame_pos = nose_frame.groupby("frame")['x'].rank(method='first')
    mp=frame_pos.groupby('id').agg(lambda x: pd.Series.mode(x)[0]).to_dict()
    # Convert values to integers (round to nearest)
    mp = {k: int(round(v)) for k, v in mp.items()}

    df = df.reset_index()
    df['id'] = df['id'].map(mp)
    df = df.set_index(['frame', 'id', 'keypoint'])
    
    return df

def assign_seats(df,seat_encoding):
    #if an eight is facing left, "{1:'s',2:'p',3:'s',4:'p',5:'s',6:'p',7:'s',8:'p',9:'c'}"
    df['seat_label'] = df.index.get_level_values('id').map(lambda x: seat_encoding[x] if x in seat_encoding else 'unknown')
    return df

def crop_boat_area(frame, boat_box):
    
    return 

if __name__ == "__main__":
    main()