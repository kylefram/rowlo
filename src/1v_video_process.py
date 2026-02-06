import torch
import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import scipy.stats

DEVICE="mps"
#VIDEO_PATH = "../input_videos/lifting.mov"
VIDEO_PATH = "input_videos/avis5.mov"
DETECTION_MODEL_PATH = "models/yolo26x.pt"
POSE_MODEL_PATH = "models/yolo26x-pose.pt"
#TRACK_MODEL="bytetrack.yaml"
TRACK_MODEL="models/botsort.yaml"


OUTPUT_VIDEO_PATH = "output_videos/"
OUTPUT_CSV_DIR = "output_data/"

PADDING=.3
#https://docs.ultralytics.com/tasks/pose/
# 1=Nose
# 2=Left Eye
# 3=Right Eye
# 4=Left Ear
# 5=Right Ear
# 6=Left Shoulder
# 7=Right Shoulder
# 8=Left Elbow
# 9=Right Elbow
# 10=Left Wrist
# 11=Right Wrist
# 12=Left Hip
# 13=Right Hip
# 14=Left Knee
# 15=Right Knee
# 16=Left Ankle
# 17=Right Ankle
KEYPOINT_INDICES = [1,6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
KEYPOINT_DICT = {1:"Nose", 6: "L_Shoulder", 7: "R_Shoulder", 8: "L_Elbow", 9: "R_Elbow", 10: "L_Wrist", 11: "R_Wrist", 12: "L_Hip", 13: "R_Hip", 14: "L_Knee", 15: "R_Knee", 16: "L_Ankle", 17: "R_Ankle"}
#useful for reference
NOSE=[1]
SKELETON_UPPER_BODY = [(6,7), (6, 8), (8, 10), (7, 9), (9, 11)]
SKELETON_TRUNK = [(6,12), (7,13), (12,13), (12,14), (13,15)]
#default boat configuration
DEF_BOAT = {1:'s',2:'p',3:'s',4:'p',5:'s',6:'p',7:'s',8:'p',9:'c'}

def parse_args():
    p = argparse.ArgumentParser(description="Joint angle detection from video file")
    p.add_argument("--device", type=str, default=DEVICE, required=False, help="Device to run the model on (e.g., 'cpu', 'cuda', 'mps')")
    p.add_argument("--video", type=str, default=VIDEO_PATH, required=False, help="Path to input video file")
    p.add_argument("--def_boat", type=str, default=DEF_BOAT, required=False, help="s=starboard, p=port, c=coxswain")
    p.add_argument("--detection_model", type=str, default=DETECTION_MODEL_PATH, required=False, help="Path to yolov8 pose weights (.pt)")
    p.add_argument("--pose_model", type=str, default=POSE_MODEL_PATH, required=False, help="Path to yolov8 pose weights (-pose.pt)")
    p.add_argument("--tracker", type=str, default=TRACK_MODEL, required=False, help="Path to tracker config file")
    p.add_argument("--output_video", type=str, default=OUTPUT_VIDEO_PATH, required=False, help="Path to save the output video file")
    p.add_argument("--output_csv_dir", type=str, default=OUTPUT_CSV_DIR, required=False, help="Path to save the output CSV file")
    p.add_argument("--max_frames", type=int, default=60, required=False, help="Maximum number of frames to process from the video")

    return p.parse_args()

def compute_box(x1,y1,x2,y2,padding,width,height):
    x1 = max(0, int(x1 - padding * (x2 - x1)))
    y1 = max(0, int(y1 - padding * (y2 - y1)))
    x2 = min(width, int(x2 + padding * (x2 - x1)))
    y2 = min(height, int(y2 + padding * (y2 - y1)))
    return x1, y1, x2, y2

def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video properties: {width}x{height} at {fps} FPS")   

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_name = args.video.split("/")[-1].split(".")[0]
    output_video = args.output_video + f"{input_name}_annotated.mp4"
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    #first pass: track boat and rowers
    data = process_video(args,cap,width,height)
    df = pd.DataFrame(data)
    df = df_postprocess(df)

    df,out = second_pass(args,cap,out,df)

    output_csv = args.output_csv_dir + f"{input_name.split('.')[0].split('/')[-1]}_data.csv"
    df.to_csv(output_csv)

    #second pass, crop video and then draw skeletons and charts below 
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    #save video 
    print(f"Annotated video saved to {output_video}")
    return df

#dynamic crop, drawing skeletons, and adding charts
def second_pass(args,cap,out,df,alp=0.7):

    bounds = df.groupby('frame')[['x1', 'y1', 'x2', 'y2']].agg(
        {'x1': 'min', 'y1': 'min', 'x2': 'max', 'y2': 'max'}
    ).reindex(range(args.max_frames)).interpolate(method='linear')

    # Calculate the center (cx, cy) of the bounding box
    bounds['cx'] = (bounds['x1'] + bounds['x2']) / 2
    bounds['cy'] = (bounds['y1'] + bounds['y2']) / 2

    # Apply Exponential Weighted Moving Average (EWM) for smooth camera motion
    # Lower alpha = smoother camera, higher alpha = more twitchy
    smooth_ops = bounds[['cx', 'cy']].ewm(alpha=alp).mean()
    bounds['smooth_cx'] = smooth_ops['cx']
    bounds['smooth_cy'] = smooth_ops['cy']
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to start
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-calculate center of the screen
    screen_cx, screen_cy = w // 2, h // 2

    for i in tqdm(range(args.max_frames), desc="Stabilizing & Annotating"):
        ret, frame = cap.read()
        if not ret: 
            break

        # If we have tracking data for this frame
        if i in bounds.index:
            # Calculate the shift needed: Target (Screen Center) - Source (Rower Center)
            src_cx = bounds.loc[i, 'smooth_cx']
            src_cy = bounds.loc[i, 'smooth_cy']
            
            dx = screen_cx - src_cx
            dy = screen_cy - src_cy

            # Create Affine Translation Matrix
            M = np.float32([[1, 0, dx], [0, 1, dy]])

            # Apply shift (stabilization). borderValue=(0,0,0) creates the black mask.
            stabilized_frame = cv2.warpAffine(frame, M, (w, h), borderValue=(0, 0, 0))

            # Filter data for current frame
            frame_data = df[df['frame'] == i]
            
            # Draw skeletons on the stabilized frame
            # We must pass dx, dy so we can shift the keypoints to match the video
            draw_skeleton_from_df(stabilized_frame, frame_data, dx, dy)
            
            out.write(stabilized_frame)
        else:
            # Write original frame if no tracking data exists
            out.write(frame)

    return df, out  

def draw_skeleton_from_df(frame, frame_df, dx, dy):
    """
    Draws skeletons from the DataFrame. 
    Applies dx/dy offset to match the stabilized video.
    """
    # Pivot so we can access anatomy by ID: index=id, columns=keypoint
    # We only care about x, y, confidence
    pivoted = frame_df.pivot(index='id', columns='keypoint', values=['x', 'y', 'confidence'])

    for person_id, row in pivoted.iterrows():
        # Helper to get (x,y) tuple shifted by dx, dy
        def get_pt(name):
            try:
                x = int(row['x'][name] + dx)
                y = int(row['y'][name] + dy)
                conf = row['confidence'][name]
                return (x, y), conf
            except KeyError:
                return None, 0.0

        # Draw Lines (Limbs)
        # Using SKELETON_UPPER_BODY and SKELETON_TRUNK global definitions
        # Note: Your globals use Indices (6,7), but DF uses Names ("L_Shoulder")
        # We need to map indices back to names for the DF lookup
        
        all_connections = SKELETON_UPPER_BODY + SKELETON_TRUNK
        colors = [(0, 255, 0)] * len(SKELETON_UPPER_BODY) + [(0, 0, 255)] * len(SKELETON_TRUNK)
        
        for connection, color in zip(all_connections, colors):
            k1_idx, k2_idx = connection
            k1_name = KEYPOINT_DICT.get(k1_idx)
            k2_name = KEYPOINT_DICT.get(k2_idx)
            
            if k1_name and k2_name:
                pt1, conf1 = get_pt(k1_name)
                pt2, conf2 = get_pt(k2_name)
                
                if conf1 > 0.5 and conf2 > 0.5: # Threshold
                    cv2.line(frame, pt1, pt2, color, 2)

        # Draw Dots (Joints)
        for k_idx, k_name in KEYPOINT_DICT.items():
            pt, conf = get_pt(k_name)
            if conf > 0.5:
                cv2.circle(frame, pt, 3, (255, 0, 0), -1)

#go through the video once, track rowers and boats, save data to dataframe
def process_video(args,cap,w,h):
    detection = YOLO(args.detection_model)
    pose  = YOLO(args.pose_model)
    data=[]
    for i in tqdm(range(args.max_frames),desc="Processing frames:"):
        ret, frame = cap.read()
        if not ret:
            continue

        if not cap.isOpened():
            break

        #white canvas
        #canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

        detection_results = detection.track(frame, persist=True, tracker=args.tracker, classes=[0],device=args.device,verbose=False)[0]
        boxes = detection_results.boxes
        ids = detection_results.boxes.id.cpu().numpy()

        for id, box in zip(ids, boxes):
            if box.cls[0] == 0:  #person         
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = compute_box(x1, y1, x2, y2, PADDING, w, h)

                person_crop = frame[y1:y2, x1:x2]
                pose_results = pose(person_crop, device=args.device,verbose=False)[0]
        
                #draw_skeleton(frame[y1:y2, x1:x2], pose_results.keypoints.data.cpu().numpy())
                #draw_skeleton(canvas[y1:y2, x1:x2], pose_results.keypoints.data.cpu().numpy())
                data.extend(database_append_rower(i, id, x1, x2, y1, y2, pose_results.keypoints.data.cpu().numpy()))
            elif box.cls[0] == 8:  # boat
                print("Boat detected")
                # Handle boat detection if needed
                data.append(database_append_boat(i, id, x1, y1, x2, y2))
    return data
        

def df_postprocess(df,seat_encoding=DEF_BOAT):
    #group by frame and keypoint to compute distances
    dfg=df.groupby(['frame','keypoint']).apply(distance_func,include_groups=False).reset_index()
    #assign seats
    seat_positions = {}
    for _, row in dfg.iterrows():
        ordered_ids = row['ordered_ids']
        for pos, id_val in enumerate(ordered_ids):
            seat_positions.setdefault(id_val, []).append(pos)
    
    # Compute seat_map using mode of positions
    seat_map = {}
    for id_val, locs in seat_positions.items():
        if locs:
            mode_pos = scipy.stats.mode(locs, keepdims=False).mode
            seat_map[id_val] = int(mode_pos) + 1  # Seat numbers start at 1
        else:
            seat_map[id_val] = -1
    
    df['seat_label']=df['id'].map(lambda x: seat_map.get(x,-1))
    df['seat_side']=df['seat_label'].map(lambda x: seat_encoding.get(x,'unknown'))
    return df



def distance_func(x):
    # Sort keypoints from right to left (assuming higher x is right)
    x = x.sort_values('x', ascending=False).reset_index(drop=True)
    ordered_ids = x['id'].tolist()
    distances = []
    for i in range(len(x) - 1):
        # Compute signed L1 distance between consecutive keypoints
        dx = x.iloc[i]['x'] - x.iloc[i + 1]['x']
        distances.append(dx)
    d = {}
    d['ordered_ids'] = ordered_ids
    #might want these later! but for now not necessary
    #d['distances'] = distances
    #d['x'] = x['x']
    #d['y'] = x['y']

    return pd.Series(d)

def draw_skeleton(frame,keypoints):
    if keypoints is None or len(keypoints) == 0:
        print ("No keypoints detected.")
        return
    
    for connection in SKELETON_UPPER_BODY:
        part_a = connection[0]-1
        part_b = connection[1]-1
        if keypoints[0,part_a][2] > 0.2 and keypoints[0,part_b][2] > 0.2:
            x1, y1 = int(keypoints[0,part_a][0]), int(keypoints[0,part_a][1])
            x2, y2 = int(keypoints[0,part_b][0]), int(keypoints[0,part_b][1])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    for connection in SKELETON_TRUNK:
        part_a = connection[0]-1
        part_b = connection[1]-1
        if keypoints[0,part_a][2] > 0.2 and keypoints[0,part_b][2] > 0.2:
            x1, y1 = int(keypoints[0,part_a][0]), int(keypoints[0,part_a][1])
            x2, y2 = int(keypoints[0,part_b][0]), int(keypoints[0,part_b][1])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    for dot in KEYPOINT_DICT.keys():
        idx = dot - 1
        if keypoints[0,idx][2] > 0.2:
            x, y = int(keypoints[0,idx][0]), int(keypoints[0,idx][1])
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

def database_append_rower(frame_num, rower_id,x1,x2,y1,y2, keypoints):
    keypoints_data = []
    if keypoints is None or len(keypoints) == 0:
        print ("No keypoints detected.")
        return []
    
    for k,keypoint in enumerate(keypoints[0]):
        if k not in KEYPOINT_DICT.keys():
            continue

        keypoint_name = KEYPOINT_DICT.get(k)
        x, y, conf = keypoint
        yield {
            "frame": frame_num,
            "id": int(rower_id),
            "keypoint": keypoint_name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "x": x1+x,
            "y": y1+y,
            "confidence": conf,
            "interpolated": False
        }

def database_append_boat(frame_num, boat_id, x1, y1, x2, y2):
    return {
        "frame": frame_num,
        "id": int(boat_id),
        "keypoint": "boat_box",
        "x": x1,
        "y": y1,
        "x_rel": x2,
        "y_rel": y2
    }

if __name__ == "__main__":
    main()