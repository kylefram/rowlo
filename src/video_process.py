import torch
import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import scipy.stats

DEVICE="mps"
VIDEO_PATH = "input_videos/avis4.mov"
DETECTION_MODEL_PATH = "models/yolo26x.pt"
POSE_MODEL_PATH = "models/yolo26x-pose.pt"
TRACK_MODEL="models/custom_botsort.yaml" 

OUTPUT_VIDEO_PATH = "output_videos/"
OUTPUT_CSV_DIR = "output_data/"
PADDING=.3

KEYPOINT_DICT = {
    0: "Nose", 5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow", 
    9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip", 13: "L_Knee", 
    14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle"
}
SKELETON_UPPER_BODY = [(5,6), (5,7), (7,9), (6,8), (8,10)]
SKELETON_TRUNK = [(5,11), (6,12), (11,12), (11,13), (12,14)]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument("--video", type=str, default=VIDEO_PATH)
    p.add_argument("--def_boat", type=str, default="spspspspc")
    p.add_argument("--detection_model", type=str, default=DETECTION_MODEL_PATH)
    p.add_argument("--pose_model", type=str, default=POSE_MODEL_PATH)
    p.add_argument("--tracker", type=str, default=TRACK_MODEL)
    p.add_argument("--output_video", type=str, default=OUTPUT_VIDEO_PATH)
    p.add_argument("--output_csv_dir", type=str, default=OUTPUT_CSV_DIR)
    p.add_argument("--max_frames", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--trim_start", type=int, default=0)
    p.add_argument("--trim_end", type=int, default=0)
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
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 1. Track & Process
    data = process_video(args, cap, w, h)
    df = pd.DataFrame(data)
    df = df_postprocess(df, args.def_boat)

    # 2. Render
    df, out = second_pass(args, cap, df, alpha=args.alpha)
    
    name = args.video.split("/")[-1].split(".")[0]
    df.to_csv(args.output_csv_dir + f"{name}_data.csv")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return df

def second_pass(args, cap, df, alpha=0.05):
    print("Computing stable crop...")

    # --- 1. Smoothing (Full Data) ---
    # Center of mass calculation
    centroids = df.groupby('frame')[['x', 'y']].mean()
    centroids = centroids.reindex(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))).interpolate().ffill().bfill()
    
    # Exponential smoothing
    scx = centroids['x'].ewm(alpha=alpha).mean()
    scy = centroids['y'].ewm(alpha=alpha).mean()

    # --- 2. Crop Calculation (Trimmed Data) ---
    start, limit = args.trim_start, min(args.max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    end = limit - args.trim_end
    
    if start >= end: return df, cv2.VideoWriter()

    # Determine max reach relative to smoothed center
    merged = df.merge(scx.rename('cx'), on='frame').merge(scy.rename('cy'), on='frame')
    viewable = merged[(merged['frame'] >= start) & (merged['frame'] < end)]

    margin = 150
    rl = (viewable['cx'] - viewable['x']).quantile(0.95) + margin
    rr = (viewable['x'] - viewable['cx']).quantile(0.95) + margin
    rt = (viewable['cy'] - viewable['y']).quantile(0.95) + margin
    rb = (viewable['y'] - viewable['cy']).quantile(0.95) + margin
    
    # Canvas Dims
    orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
    CW, CH = min(int(rl + rr), orig_w), min(int(rt + rb), orig_h)
    SH = 350 # Strip height
    FH = CH + SH
    
    print(f"Canvas: {CW}x{FH}")

    # --- 3. Render ---
    out = cv2.VideoWriter(args.output_video + f"{args.video.split('/')[-1].split('.')[0]}_final.mp4", 
                          cv2.VideoWriter_fourcc(*'mp4v'), 30, (CW, FH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    win_cx, win_cy = int(rl), int(rt)
    df_idx = df.set_index('frame')

    for i in tqdm(range(start, end), desc="Rendering"):
        ret, frame = cap.read()
        if not ret: break

        canvas = np.zeros((FH, CW, 3), dtype=np.uint8)
        dx, dy = int(win_cx - scx.iloc[i]), int(win_cy - scy.iloc[i])

        # Video
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas[0:CH, :] = cv2.warpAffine(frame, M, (CW, CH))
        
        # Skeletons
        try:
            row = df_idx.loc[i]
            if isinstance(row, pd.Series): row = row.to_frame().T
            draw_skeletons(canvas, row, dx, dy, True)
            draw_skeletons(canvas, row, dx, dy + CH, False)
        except KeyError: pass 

        out.write(canvas)
    return df, out

def draw_skeletons(canvas, df, dx, dy, labels=False):
    pivoted = df.pivot(index='id', columns='keypoint', values=['x', 'y'])
    
    for pid in df['id'].unique():
        if pid not in pivoted.index: continue
        row = pivoted.loc[pid]
        seat = df[df['id'] == pid]['seat_label'].iloc[0]
        
        def pt(n):
            try: return (int(row['x'][n] + dx), int(row['y'][n] + dy))
            except: return None

        # Limbs
        conns = SKELETON_UPPER_BODY + SKELETON_TRUNK
        cols = [(0,255,0)] * len(SKELETON_UPPER_BODY) + [(0,0,255)] * len(SKELETON_TRUNK)
        for (k1, k2), c in zip(conns, cols):
            p1, p2 = pt(KEYPOINT_DICT[k1]), pt(KEYPOINT_DICT[k2])
            if p1 and p2: cv2.line(canvas, p1, p2, c, 2)
        
        # Label
        nose = pt("Nose")
        if nose and labels:
            cv2.putText(canvas, str(seat), (nose[0]-10, nose[1]-20), 0, 0.7, (255,255,255), 2)

def process_video(args, cap, w, h):
    det, pose = YOLO(args.detection_model), YOLO(args.pose_model)
    data = []
    
    for i in tqdm(range(args.max_frames), desc="Tracking"):
        ret, frame = cap.read()
        if not ret: break

        # Track
        res = det.track(frame, persist=True, tracker=args.tracker, classes=[0], device=args.device,
                        verbose=False, imgsz=args.imgsz, conf=args.conf, iou=args.iou)[0]
        if res.boxes.id is None: continue

        # Collect crops, then run pose on all of them in one batched call
        crops, metas = [], []
        for id, box in zip(res.boxes.id.cpu().numpy(), res.boxes):
            x1, y1, x2, y2 = compute_box(*box.xyxy[0], PADDING, w, h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            crops.append(crop)
            metas.append((id, x1, y1))

        if not crops: continue
        for result, (id, x1, y1) in zip(pose(crops, device=args.device, verbose=False), metas):
            kpts = result.keypoints.data.cpu().numpy()
            data.extend(append_rower(i, id, x1, y1, kpts))
    return data

def stitch_tracks(df, max_gap=450):
    """Merge track fragments belonging to the same rower.

    The boat is rigid, so each person's x-position relative to the crew centroid
    is nearly constant. If one track ends and another starts later at the same
    relative position (e.g. the cox drifting out of frame and back), they are
    the same person.
    """
    pos = df.groupby(['frame', 'id'])[['x', 'y']].mean().reset_index()
    cent = pos.groupby('frame')[['x', 'y']].transform('mean')
    pos['rel_x'] = pos['x'] - cent['x']

    span = pos.groupby('frame')['rel_x'].agg(lambda s: s.max() - s.min()).median()
    tol = span * 0.06  # roughly half a seat spacing in an 8+

    info = pos.groupby('id').agg(start=('frame', 'min'), end=('frame', 'max'),
                                 relx=('rel_x', 'median')).sort_values('start')

    canonical = {}  # fragment id -> surviving id
    open_tracks = []  # (end_frame, relx, root_id) of tracks that may continue
    for tid, row in info.iterrows():
        merged = False
        for k, (end, relx, root) in enumerate(open_tracks):
            if end < row['start'] and row['start'] - end <= max_gap and abs(row['relx'] - relx) < tol:
                canonical[tid] = root
                open_tracks[k] = (row['end'], row['relx'], root)
                merged = True
                break
        if not merged:
            canonical[tid] = tid
            open_tracks.append((row['end'], row['relx'], tid))

    df['id'] = df['id'].map(canonical)
    return df
        
def df_postprocess(df, boat_config):
    # 1. Clean Glitches (Spread > 3x Median)
    stats = df.groupby(['frame','id']).agg(min_x=('x','min'), max_x=('x','max'), min_y=('y','min'), max_y=('y','max'))
    stats['diag'] = np.sqrt((stats.max_x - stats.min_x)**2 + (stats.max_y - stats.min_y)**2)
    limit = stats['diag'].median() * 3.0
    bad_ids = stats[stats['diag'] > limit].index
    df = df.set_index(['frame','id']).drop(bad_ids, errors='ignore').reset_index()

    # 2. Merge track fragments, then keep the N best tracks for an N-seat boat
    df = stitch_tracks(df)
    n_seats = len(boat_config)
    presence = df.groupby('id')['frame'].nunique().sort_values(ascending=False)
    keep = presence.index[:n_seats]
    df = df[df['id'].isin(keep)]

    # 3. Interpolate
    idx = pd.MultiIndex.from_product([range(df.frame.min(), df.frame.max()+1), df.id.unique(), df.keypoint.unique()], names=['frame','id','keypoint'])
    df = df.set_index(['frame','id','keypoint']).reindex(idx)
    df[['x','y']] = df[['x','y']].interpolate(limit=30, limit_direction='both')
    df = df.reset_index().dropna(subset=['x'])

    # 4. GLOBAL SEAT ASSIGNMENT (Fixes Flickering)
    # Rank each ID by its median x-position RELATIVE to the per-frame crew
    # centroid. Raw x breaks when the camera pans or a track only exists for
    # part of the video; relative x is constant because the boat is rigid.
    pos = df.groupby(['frame','id'])['x'].mean().reset_index()
    pos['rel_x'] = pos['x'] - pos.groupby('frame')['x'].transform('mean')
    id_rel_x = pos.groupby('id')['rel_x'].median()

    # Sort IDs by their relative position (Left to Right)
    sorted_ids = id_rel_x.sort_values().index.tolist()

    # Assign seats based on this global sort order
    def get_name(rank):
        if rank < 0 or rank >= len(boat_config): return "?"
        if rank == 0: return "Bow"
        if rank == len(boat_config)-1: return "Cox" if boat_config[-1]=='c' else "Stroke"
        return str(rank+1)

    # Map ID -> Rank -> Name
    id_to_seat = {uid: get_name(i) for i, uid in enumerate(sorted_ids)}

    df['seat_label'] = df['id'].map(id_to_seat)
    return df

def append_rower(f, pid, x1, y1, kpts):
    if kpts is None or len(kpts)==0: return []
    for k, (x, y, conf) in enumerate(kpts[0]):
        if k in KEYPOINT_DICT:
            yield {"frame":f, "id":int(pid), "keypoint":KEYPOINT_DICT[k], "x":x1+x, "y":y1+y, "confidence":conf}

if __name__ == "__main__":
    main()