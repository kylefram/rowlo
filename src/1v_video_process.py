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
# Shift all keys down by 1 to match YOLO 0-based indexing
KEYPOINT_DICT = {
    0: "Nose",
    5: "L_Shoulder",
    6: "R_Shoulder", 
    7: "L_Elbow", 
    8: "R_Elbow", 
    9: "L_Wrist", 
    10: "R_Wrist", 
    11: "L_Hip", 
    12: "R_Hip", 
    13: "L_Knee", 
    14: "R_Knee", 
    15: "L_Ankle", 
    16: "R_Ankle"
}

# UPDATE YOUR SKELETON LISTS
# These also need to shift down by 1 to match the new dictionary keys
SKELETON_UPPER_BODY = [(5,6), (5,7), (7,9), (6,8), (8,10)]
SKELETON_TRUNK = [(5,11), (6,12), (11,12), (11,13), (12,14)]
#default boat configuration

def parse_args():
    p = argparse.ArgumentParser(description="Joint angle detection from video file")
    p.add_argument("--device", type=str, default=DEVICE, required=False, help="Device to run the model on (e.g., 'cpu', 'cuda', 'mps')")
    p.add_argument("--video", type=str, default=VIDEO_PATH, required=False, help="Path to input video file")
    p.add_argument("--def_boat", type=str, default="spspspspc", required=False, help="s=starboard, p=port, c=coxswain")
    p.add_argument("--detection_model", type=str, default=DETECTION_MODEL_PATH, required=False, help="Path to yolov8 pose weights (.pt)")
    p.add_argument("--pose_model", type=str, default=POSE_MODEL_PATH, required=False, help="Path to yolov8 pose weights (-pose.pt)")
    p.add_argument("--tracker", type=str, default=TRACK_MODEL, required=False, help="Path to tracker config file")
    p.add_argument("--output_video", type=str, default=OUTPUT_VIDEO_PATH, required=False, help="Path to save the output video file")
    p.add_argument("--output_csv_dir", type=str, default=OUTPUT_CSV_DIR, required=False, help="Path to save the output CSV file")
    p.add_argument("--max_frames", type=int, default=60, required=False, help="Maximum number of frames to process from the video")
    p.add_argument("--alpha", type=float, default=0.3, required=False, help="Smoothing factor for camera stabilization (0-1, lower is smoother)")

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

    #first pass: track boat and rowers
    data = process_video(args,cap,width,height)
    df = pd.DataFrame(data)
    df = df_postprocess(df,args.def_boat)

    df,out = second_pass(args,cap,df,alpha=args.alpha)
    
    input_name = args.video.split("/")[-1].split(".")[0]
    output_csv = args.output_csv_dir + f"{input_name.split('.')[0].split('/')[-1]}_data.csv"
    df.to_csv(output_csv)

    #second pass, crop video and then draw skeletons and charts below 
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    #save video 
    return df

def second_pass(args, cap, df, alpha=0.05): # Alpha lowered to 0.05 for smoother cam
    print("Computing stable crop window...")

    # --- 1. Compute Stable Center (Centroid) ---
    # Instead of (min+max)/2, we take the mean of ALL points.
    # If one person flickers, the mean of the other ~150 points barely moves.
    frame_centroids = df.groupby('frame')[['x', 'y']].mean()
    
    # Reindex to ensure we have an entry for every frame (fill missing with previous known spot)
    # This handles cases where the boat completely disappears (rare, but good for safety)
    full_idx = range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_centroids = frame_centroids.reindex(full_idx).interpolate(method='linear').ffill().bfill()

    # Apply Heavy Smoothing
    smooth_cx = frame_centroids['x'].ewm(alpha=alpha).mean()
    smooth_cy = frame_centroids['y'].ewm(alpha=alpha).mean()

    # --- 2. Determine Static Crop Dimensions ---
    # We need a box big enough to hold the boat relative to our new Centroid.
    # For every frame, we check: How far is the furthest point from the Centroid?
    
    # Join the smoothed center back to the main DF
    df_merged = df.merge(smooth_cx.rename('cx'), on='frame').merge(smooth_cy.rename('cy'), on='frame')
    
    # Calculate distance from center to edges
    df_merged['dist_x_left']  = df_merged['cx'] - df_merged['x'] # Positive if point is left of center
    df_merged['dist_x_right'] = df_merged['x'] - df_merged['cx'] # Positive if point is right of center
    df_merged['dist_y_top']   = df_merged['cy'] - df_merged['y']
    df_merged['dist_y_bot']   = df_merged['y'] - df_merged['cy']
    
    # Find the maximum necessary reach from the center
    # We use quantile(0.99) instead of max() to ignore extreme single-pixel outliers/glitches
    margin = 150
    max_reach_left  = df_merged['dist_x_left'].quantile(0.995) + margin
    max_reach_right = df_merged['dist_x_right'].quantile(0.995) + margin
    max_reach_up    = df_merged['dist_y_top'].quantile(0.995) + margin
    max_reach_down  = df_merged['dist_y_bot'].quantile(0.995) + margin
    
    # Total Width/Height needed
    req_w = int(max_reach_left + max_reach_right)
    req_h = int(max_reach_up + max_reach_down)
    
    # Clamp to original video size
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    CROP_W = min(req_w, orig_w)
    CROP_H = min(req_h, orig_h)
    
    # Canvas Settings
    SKELETON_H = 350 # Fixed height for skeleton strip
    CANVAS_W = CROP_W
    CANVAS_H = CROP_H + SKELETON_H
    
    print(f"Canvas: {CANVAS_W}x{CANVAS_H} (Crop: {CROP_W}x{CROP_H})")

    # Re-init Writer
    output_filename = args.output_video + f"{args.video.split('/')[-1].split('.')[0]}_final.mp4"
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (CANVAS_W, CANVAS_H))

    # --- 3. Render Loop ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Pre-calculate the "Window Center" (Where the centroid should sit in the output)
    # If the boat is asymmetric (more length behind centroid), we shift the window center slightly
    # to ensure the crop fits perfectly around the centroid.
    win_cx = int(max_reach_left) 
    win_cy = int(max_reach_up)

    df_indexed = df.set_index('frame')

    for i in tqdm(range(args.max_frames), desc="Rendering Stabilized"):
        ret, frame = cap.read()
        if not ret: break

        master_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # Retrieve smoothed center for this frame
        curr_cx = smooth_cx.iloc[i] if i < len(smooth_cx) else smooth_cx.iloc[-1]
        curr_cy = smooth_cy.iloc[i] if i < len(smooth_cy) else smooth_cy.iloc[-1]
        
        # Calculate Shift: Move (curr_cx, curr_cy) to (win_cx, win_cy)
        dx = int(win_cx - curr_cx)
        dy = int(win_cy - curr_cy)

        # --- TOP: Video ---
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        video_crop = cv2.warpAffine(frame, M, (CROP_W, CROP_H), borderValue=(0,0,0))
        master_canvas[0:CROP_H, 0:CROP_W] = video_crop
        
        # --- BOTTOM: Skeletons ---
        try:
            frame_data = df_indexed.loc[i]
            if isinstance(frame_data, pd.Series): 
                frame_data = frame_data.to_frame().T
            
            # The skeleton strip is centered horizontally same as video
            # Vertically, we center it in the SKELETON_H strip
            # We use the same 'dx' to keep horizontal alignment with the video
            
            # We want the 'centroid' of the skeletons to be in the middle of the bottom strip
            strip_center_y = CROP_H + (SKELETON_H // 2)
            
            # Use the same 'dy' logic but targeted at the strip center
            # Actually, reusing dy + offset keeps it "locked" to the video movement,
            # which is usually preferred so they move in sync.
            draw_skeletons(master_canvas, frame_data, dx, dy,True)
            draw_skeletons(master_canvas, frame_data, dx, dy + CROP_H)
            
        except KeyError:
            pass 

        out.write(master_canvas)

    return df, out

def draw_skeletons(canvas, frame_df, dx, dy, include_labels=False):
    pivoted = frame_df.pivot(index='id', columns='keypoint', values=['x', 'y', 'confidence'])
    
    # Get seat labels (they are the same for all keypoints of an ID, so just take first)
    # We iterate unique IDs in the frame
    unique_ids = frame_df['id'].unique()
    
    for person_id in unique_ids:
        # Extract row for this person
        if person_id not in pivoted.index: continue
        row = pivoted.loc[person_id]
        
        # Get Seat Label
        seat_label = frame_df[frame_df['id'] == person_id]['seat_label'].iloc[0]

        def get_pt(name):
            try:
                x = int(row['x'][name] + dx)
                y = int(row['y'][name] + dy)
                return (x, y)
            except:
                return None

        # Draw Limbs
        all_connections = SKELETON_UPPER_BODY + SKELETON_TRUNK
        colors = [(0, 255, 0)] * len(SKELETON_UPPER_BODY) + [(0, 0, 255)] * len(SKELETON_TRUNK)

        for connection, color in zip(all_connections, colors):
            k1_name = KEYPOINT_DICT.get(connection[0]) # KEYPOINT_DICT must be 0-based now
            k2_name = KEYPOINT_DICT.get(connection[1])
            
            if k1_name and k2_name:
                pt1 = get_pt(k1_name)
                pt2 = get_pt(k2_name)
                if pt1 and pt2:
                    cv2.line(canvas, pt1, pt2, color, 2)
        
        # Draw Label (Above Head)
        nose_pt = get_pt("Nose")
        if nose_pt and include_labels:
            label_pos = (nose_pt[0] - 10, nose_pt[1] - 20)
            cv2.putText(canvas, str(seat_label), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

        detection_results = detection.track(frame, persist=True, tracker=args.tracker, classes=[0],device=args.device,verbose=False)[0]
        boxes = detection_results.boxes
        ids = detection_results.boxes.id.cpu().numpy()

        for id, box in zip(ids, boxes):
            if box.cls[0] == 0:  #person         
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = compute_box(x1, y1, x2, y2, PADDING, w, h)

                person_crop = frame[y1:y2, x1:x2]
                pose_results = pose(person_crop, device=args.device,verbose=False)[0]
        
                data.extend(database_append_rower(i, id, x1, x2, y1, y2, pose_results.keypoints.data.cpu().numpy()))
            elif box.cls[0] == 8:  # boat
                print("Boat detected")
                # Handle boat detection if needed
                data.append(database_append_boat(i, id, x1, y1, x2, y2))
    return data
        
def df_postprocess(df, boat_config="spspspspc"):
    """
    1. Interpolates missing tracking data.
    2. Ranks actors Left-to-Right (Ascending X) to match boat_config order.
    3. Maps ranks to Seat Names (Bow, 2, 3... Stroke, Cox).
    """
    # --- 1. Vectorized Interpolation ---
    # Create a dense grid of Frame x ID x Keypoint to handle missing data
    # This ensures every person has a row for every frame, even if YOLO missed them
    full_idx = pd.MultiIndex.from_product(
        [range(df.frame.min(), df.frame.max() + 1), df.id.unique(), df.keypoint.unique()],
        names=['frame', 'id', 'keypoint']
    )
    
    # Reindex
    df = df.set_index(['frame', 'id', 'keypoint']).reindex(full_idx)
    
    # CHANGE: Add limit to interpolation
    # limit=30 means: If a rower is missing for >30 frames (1 sec), DON'T connect the dots.
    # This prevents lines streaking across the screen when IDs swap.
    df[['x', 'y']] = df[['x', 'y']].interpolate(method='linear', limit=30, limit_direction='both')
    
    df = df.reset_index()
    
    # Filter out rows that are STILL NaN after interpolation (meaning the gap was too big)
    # This cleans up the "ghost" tracks that shouldn't exist.
    df = df.dropna(subset=['x', 'y'])
    
    # Fill static columns (like confidence) that got NaN'd by reindex
    df['confidence'] = df['confidence'].interpolate(linear=True, limit_direction='both').fillna(0)

    # --- 2. Determine Seat Order (Left -> Right) ---
    # Calculate the average X position of each person in each frame
    # Rank them: 1 = Leftmost (Low X), N = Rightmost (High X)
    frame_ranks = df.groupby(['frame', 'id'])['x'].mean().groupby('frame').rank(method='first', ascending=True)
    
    # Find the most frequent rank for each ID (Mode) to assign a permanent seat
    # (Subtract 1 to make it 0-indexed for string lookup)
    id_seat_indices = frame_ranks.reset_index().groupby('id')['x'].agg(lambda x: x.mode()[0] - 1).astype(int)

    # --- 3. Map Indices to Names/Sides ---
    def get_seat_info(idx):
        if idx < 0 or idx >= len(boat_config): 
            return "Unknown", "?"
        
        side = boat_config[idx]
        
        # Naming Logic based on boat length
        is_last = (idx == len(boat_config) - 1)
        is_second_last = (idx == len(boat_config) - 2)
        last_is_cox = (boat_config[-1] == 'c')

        if idx == 0: 
            name = "Bow"
        elif is_last and side == 'c': 
            name = "Cox"
        elif is_last: 
            name = "Stroke"
        elif is_second_last and last_is_cox: 
            name = "Stroke"
        else: 
            name = str(idx + 1)
            
        return name, side

    # Create mapping dictionaries
    seat_info_map = {uid: get_seat_info(idx) for uid, idx in id_seat_indices.items()}
    
    # Apply to DataFrame
    df['seat_label'] = df['id'].map(lambda x: seat_info_map.get(x, ("?", "?"))[0])
    df['seat_side']  = df['id'].map(lambda x: seat_info_map.get(x, ("?", "?"))[1])
    
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