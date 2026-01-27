import torch
import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

DEVICE="mps"
#VIDEO_PATH = "../input_videos/lifting.mov"
VIDEO_PATH = "input_videos/avis6.mov"
DETECTION_MODEL_PATH = "models/yolo26x.pt"
POSE_MODEL_PATH = "models/yolo26x-pose.pt"
TRACK_MODEL="models/botsort.yaml"

DEF_BOAT = {1:'s',2:'p',3:'s',4:'p',5:'s',6:'p',7:'s',8:'p',9:'c'}

OUTPUT_VIDEO_PATH = "output_videos/"
OUTPUT_DATA_PATH = "output_data/"

def parse_args():
    p = argparse.ArgumentParser(description="Joint angle detection from video file")
    p.add_argument("--device", type=str, default=DEVICE, required=False, help="Device to run the model on (e.g., 'cpu', 'cuda', 'mps')")
    p.add_argument("--video", type=str, default=VIDEO_PATH, required=False, help="Path to input video file")
    p.add_argument("--def_boat", type=str, default=DEF_BOAT, required=False, help="s=starboard, p=port, c=coxswain")
    p.add_argument("--detection_model", type=str, default=DETECTION_MODEL_PATH, required=False, help="Path to yolov8 pose weights (.pt)")
    p.add_argument("--pose_model", type=str, default=POSE_MODEL_PATH, required=False, help="Path to yolov8 pose weights (-pose.pt)")
    p.add_argument("--tracker", type=str, default=TRACK_MODEL, required=False, help="Path to tracker config file")
    p.add_argument("--output_video", type=str, default=OUTPUT_VIDEO_PATH, required=False, help="Path to save the output video file")
    p.add_argument("--max_frames", type=int, default=60, required=False, help="Maximum number of frames to process from the video")

    return p.parse_args()

def compute_box(x1,y1,x2,y2,padding,width,height):
    x1 = max(0, int(x1 - padding * (x2 - x1)))
    y1 = max(0, int(y1 - padding * (y2 - y1)))
    x2 = min(width, int(x2 + padding * (x2 - x1)))
    y2 = min(height, int(y2 + padding * (y2 - y1)))
    return x1, y1, x2, y2

def main(args):
    detection = YOLO(args.detection_model)

    cap = cv2.VideoCapture(args.video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video properties: {width}x{height} at {fps} FPS")   

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_name = args.video.split("/")[-1].split(".")[0]
    output_video = args.output_video + f"{input_name}_annotated.mp4"
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    #pass 1: track boat
    boat_df = boat_track(cap,args,width,height)
    #pass 2: track rowers within boat box
    rower_df = rower_track(cap,args,boat_df,width,height)

    #first pass: identifty boat box

def skeleton_identify(cap,args,):
    pass


def rower_track(cap,args,boat_df,w,h):
    cap = cv2.VideoCapture(args.video)

    #track humans within boat box
    rower_data = []
    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #certain frames may not have been repaired
        if i in boat_df.index:
            x1,y1,x2,y2 = map(int, boat_df.loc[i,['x1','y1','x2','y2']])
        else:
            print(f"Frame {i} missing boat data, using full frame for rower detection")
            x1,y1,x2,y2 = None,None,None,None

        #detect rowers
        detection_results_rower,row = detect_rower(i, frame[y1:y2, x1:x2], args, x1,y1)

        if len(detection_results_rower.boxes) > 0:
            
            for box in detection_results_rower.boxes:
                #we want to find the center of the rower box and move it
                frame = draw_rower_box(frame, box, w, h,x1,y1)
                rower_data.extend(row)
                #cv2.imshow("Rower Frame", frame)

        i += 1
        if i >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save rower data to DataFrame and CSV
    df = pd.DataFrame(rower_data)
    df = postprocess_box_data(df,i-1)
    input_video_name = args.video.split("/")[-1].split(".")[0]
    output_path = OUTPUT_DATA_PATH + input_video_name + "_rower_output.csv"
    df.to_csv(output_path, index=True)
    print(f"Rower detection data saved to {output_path}")

    return df

def draw_rower_box(frame, box, fr_width, fr_height,bx1, by1, ver_pad=0.2, hor_pad=0.2):
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    x1 += bx1
    y1 += by1
    x2 += bx1
    y2 += by1
    bx_width = x2 - x1
    bx_height = y2 - y1
    x1 = max(0, int(x1 - hor_pad * bx_width))
    x2 = min(fr_width, int(x2 + hor_pad * bx_width))
    #adjust y1 (top) upwards (recall images increase downwards)
    y1 = max(0, int(y1 - ver_pad * bx_height))
    y2 = min(fr_height, int(y2 + ver_pad * bx_height))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def detect_rower(i, frame, args, bx1,by1):
    rower_data = []
    detection = YOLO(args.detection_model)
    results = detection.track(frame, persist=True,
                               classes=[0], 
                               device=args.device, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = box.conf[0].cpu().numpy()
        cls = box.cls[0].cpu().numpy()
        id = box.id[0].item() if box.id is not None else -1
        rower_data.append({
            'frame': i,
            'id': id,
            'x1': x1 + bx1,
            'y1': y1 + by1,
            'x2': x2 + bx1,
            'y2': y2 + by1,
            'confidence': conf,
            'class': cls
        })

    return results, rower_data

def boat_track(cap,args,w,h):
    # Initialize list to collect boat detection data
    boat_data = []

    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        #detect boats
        detection_results_boat,row = detect_boat(i, frame, args)

        if len(detection_results_boat.boxes) > 0:
            #we want to find the center of the boat box and move it
            frame = draw_boat_box(frame, detection_results_boat.boxes[0], w, h)
            boat_data.extend(row)
            #cv2.imshow("Boat Frame", frame)
        else:
            #often, no boat is detected. TODO: cache last boat position and interpolate movement 
            continue
        
        i += 1
        if i >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save boat data to DataFrame and CSV
    df = pd.DataFrame(boat_data)
    df = postprocess_box_data(df,i-1)
    input_video_name = args.video.split("/")[-1].split(".")[0]
    output_path = OUTPUT_DATA_PATH + input_video_name + "_boat_output.csv"
    df.to_csv(output_path, index=True)
    print(f"Boat detection data saved to {output_path}")

    return df

def postprocess_box_data(df,n):
    #only take the most common id. TODO: improve to persist id across frames
    df = correct_ids_simple(df)
    df = df.drop_duplicates(subset='frame')
    df.set_index('frame', inplace=True)

    interpolated_dfs = []
    full_index = pd.RangeIndex(start=0, stop=n, step=1)
    for id_val, group in df.groupby('id'):
        group = group.reindex(full_index)
        group[['x1', 'y1', 'x2', 'y2']] = group[['x1', 'y1', 'x2', 'y2']].interpolate(method='index', limit_direction='both')
        group['id'] = id_val  # Restore id column
        interpolated_dfs.append(group.reset_index())
    
    df = pd.concat(interpolated_dfs, ignore_index=True)

    return df

def correct_ids_simple(df, dist_thresh=50, max_gap=5):
    df = df.sort_values(['frame', 'id']).copy()
    track_centers = {}  # id -> (last_frame, center_x, center_y)
    
    for idx, row in df.iterrows():
        frame, id_val = row['frame'], row['id']
        cx = (row['x1'] + row['x2']) / 2
        cy = (row['y1'] + row['y2']) / 2
        
        # Find closest previous track within gap
        min_dist = float('inf')
        best_id = None
        for tid, (last_frame, px, py) in track_centers.items():
            if frame - last_frame <= max_gap:
                dist = ((cx - px)**2 + (cy - py)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
        
        if best_id and min_dist < dist_thresh:
            df.at[idx, 'id'] = best_id
            track_centers[best_id] = (frame, cx, cy)
        else:
            track_centers[id_val] = (frame, cx, cy)
    
    return df

def detect_boat(i,frame,args):
    boat_data = []
    #we need to track boats with special thresholds because YOLO often misses boats
    #edited in .yaml tracker
    detection = YOLO(args.detection_model)
    results = detection.track(frame, persist=True,
                               classes=[8], 
                               device=args.device, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = box.conf[0].cpu().numpy()
        cls = box.cls[0].cpu().numpy()
        id = box.id[0].item() if box.id is not None else -1
        boat_data.append({
            'frame': i,
            'id': id,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'confidence': conf,
            'class': cls
        })

    return results, boat_data

# generally, rowers might sit within 3x of the boat box, but we just want a bit wider for viewing
def draw_boat_box(frame, box, fr_width, fr_height, ver_pad=3.0, hor_pad=0.4):
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    bx_width = x2 - x1
    bx_height = y2 - y1
    x1 = max(0, int(x1 - hor_pad * bx_width))
    x2 = min(fr_width, int(x2 + hor_pad * bx_width))
    #adjust y1 (top) upwards (recall images increase downwards)
    y1 = max(0, int(y1 - ver_pad * bx_height))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame

if __name__ == "__main__":
    args = parse_args()
    main(args)