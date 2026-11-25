"""
export_to_unity.py
Export dữ liệu chấm điểm động tác nhảy sang JSON cho Unity MVP (hoặc stream trực tiếp qua UDP).
Đề tài NCKH: Dance Movement Evaluation.
"""

import os
import sys
import json
import time
import argparse

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from catboost import CatBoostClassifier

# Danh sách 13 khớp MediaPipe chuẩn của đề tài
JOINTS = [
    'nose',
    'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 
    'left_knee', 'right_knee', 
    'left_ankle', 'right_ankle'
]

# Các cặp xương nối 13 khớp (để Unity LineRenderer vẽ Skeleton)
BONES = [
    ['nose', 'left_shoulder'],
    ['nose', 'right_shoulder'],
    ['left_shoulder', 'right_shoulder'],
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['left_shoulder', 'left_hip'],
    ['right_shoulder', 'right_hip'],
    ['left_hip', 'right_hip'],
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle']
]

PAIR_IDS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 15]  # Loại video 6 (đứng yên)
TEST_VIDEO_IDS = [5, 9]
FRAME_SCORE_THRESHOLD = 7.1
FPS = 30.0

def is_dancing(df_student, variance_threshold=0.015):
    key_joints = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']
    df_key = df_student[df_student['Joint_Name'].isin(key_joints)]
    std = df_key.groupby('Joint_Name')[['X', 'Y']].std().mean().mean()
    return std >= variance_threshold, std

def preprocess_data(df):
    df = df.sort_values(by=['Frame', 'Joint_Name']).reset_index(drop=True)
    mask = df['Visibility'] < 0.5
    df.loc[mask, ['X', 'Y', 'Z']] = np.nan
    df[['X', 'Y', 'Z']] = df.groupby('Joint_Name')[['X', 'Y', 'Z']].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    df.fillna(0, inplace=True)
    
    normalized_frames = []
    for frame_id, frame_data in df.groupby('Frame'):
        l_hip = frame_data[frame_data['Joint_Name'] == 'left_hip']
        r_hip = frame_data[frame_data['Joint_Name'] == 'right_hip']
        l_sh = frame_data[frame_data['Joint_Name'] == 'left_shoulder']
        r_sh = frame_data[frame_data['Joint_Name'] == 'right_shoulder']
        
        if not l_hip.empty and not r_hip.empty and not l_sh.empty and not r_sh.empty:
            root_x = (l_hip['X'].values[0] + r_hip['X'].values[0]) / 2.0
            root_y = (l_hip['Y'].values[0] + r_hip['Y'].values[0]) / 2.0
            sh_x = (l_sh['X'].values[0] + r_sh['X'].values[0]) / 2.0
            sh_y = (l_sh['Y'].values[0] + r_sh['Y'].values[0]) / 2.0
            torso = np.hypot(sh_x - root_x, sh_y - root_y)
            torso = max(torso, 1e-5)
        else:
            root_x, root_y, torso = 0.0, 0.0, 1.0
            
        frame_data = frame_data.copy()
        frame_data['X_norm'] = (frame_data['X'] - root_x) / torso
        frame_data['Y_norm'] = (frame_data['Y'] - root_y) / torso
        normalized_frames.append(frame_data)
        
    return pd.concat(normalized_frames, ignore_index=True)

def extract_features(df_s, df_t):
    def get_vecs(df):
        wide = df.pivot(index='Frame', columns='Joint_Name', values=['X_norm', 'Y_norm']).fillna(0)
        return wide.values, wide.index.tolist()

    vs, fs = get_vecs(df_s)
    vt, ft = get_vecs(df_t)
    
    _, path = fastdtw(vs, vt, dist=euclidean)
    
    # 1-to-1 mapping: mỗi khung học viên ghép với đúng 1 khung giáo viên gần nhất
    best_match = {}
    for i, j in path:
        d = np.linalg.norm(vs[i] - vt[j])
        if i not in best_match or d < best_match[i][0]:
            best_match[i] = (d, j)
            
    aligned = [{'Frame_s': fs[i], 'Frame_t': ft[best_match[i][1]]} for i in range(len(fs))]
    df_path = pd.DataFrame(aligned)
    
    df_s_aln = pd.merge(df_path, df_s, left_on='Frame_s', right_on='Frame').drop(columns=['Frame'])
    df_t_aln = pd.merge(df_path, df_t, left_on='Frame_t', right_on='Frame').drop(columns=['Frame'])
    
    df_s_aln = df_s_aln.rename(columns={'X_norm': 'X_s', 'Y_norm': 'Y_s', 'X': 'raw_X_s', 'Y': 'raw_Y_s'})
    df_t_aln = df_t_aln.rename(columns={'X_norm': 'X_t', 'Y_norm': 'Y_t', 'X': 'raw_X_t', 'Y': 'raw_Y_t'})
    
    label_col = 'Label' if 'Label' in df_s_aln.columns else None
    cols_s = ['Frame_s', 'Frame_t', 'Joint_Name', 'X_s', 'Y_s', 'raw_X_s', 'raw_Y_s'] + ([label_col] if label_col else [])
    
    merged = pd.merge(df_s_aln[cols_s],
                      df_t_aln[['Frame_s', 'Frame_t', 'Joint_Name', 'X_t', 'Y_t', 'raw_X_t', 'raw_Y_t']],
                      on=['Frame_s', 'Frame_t', 'Joint_Name'])
                      
    merged['Joint_Error'] = np.hypot(merged['X_s'] - merged['X_t'], merged['Y_s'] - merged['Y_t'])
    dot = (merged['X_s'] * merged['X_t']) + (merged['Y_s'] * merged['Y_t'])
    norm_s = np.hypot(merged['X_s'], merged['Y_s'])
    norm_t = np.hypot(merged['X_t'], merged['Y_t'])
    merged['Joint_Cosine'] = dot / (norm_s * norm_t + 1e-9)
    
    merged['Vel_X_s'] = merged.groupby('Joint_Name')['X_s'].diff().fillna(0)
    merged['Vel_Y_s'] = merged.groupby('Joint_Name')['Y_s'].diff().fillna(0)
    merged['Vel_X_t'] = merged.groupby('Joint_Name')['X_t'].diff().fillna(0)
    merged['Vel_Y_t'] = merged.groupby('Joint_Name')['Y_t'].diff().fillna(0)
    merged['Velocity_Error'] = np.hypot(merged['Vel_X_s'] - merged['Vel_X_t'], merged['Vel_Y_s'] - merged['Vel_Y_t'])
    
    if label_col:
        merged.rename(columns={label_col: 'Ground_Truth_Label'}, inplace=True)
        
    return merged

def build_dataset_and_train():
    print("=== [1/3] Đang nạp và trích xuất dữ liệu huấn luyện ===")
    train_dfs = []
    for vid in PAIR_IDS:
        s_file = f'student_dance_{vid}.csv'
        t_file = f'teacher_dance_{vid}.csv'
        if not os.path.exists(s_file) or not os.path.exists(t_file):
            continue
            
        df_s = pd.read_csv(s_file)
        df_t = pd.read_csv(t_file)
        
        is_mov, _ = is_dancing(df_s)
        if not is_mov:
            continue
            
        feat_df = extract_features(preprocess_data(df_s), preprocess_data(df_t))
        feat_df['Video_ID'] = vid
        if vid not in TEST_VIDEO_IDS:
            train_dfs.append(feat_df)
            
    train_data = pd.concat(train_dfs, ignore_index=True)
    features = ['Joint_Error', 'Joint_Cosine', 'Velocity_Error']
    X_train = train_data[features]
    y_train = train_data['Ground_Truth_Label']
    
    print(f"-> Tập TRAIN: {len(X_train):,} mẫu khớp. Đang huấn luyện CatBoost...")
    model = CatBoostClassifier(iterations=150, depth=6, learning_rate=0.08, random_seed=42, verbose=0)
    model.fit(X_train, y_train)
    print("-> Huấn luyện CatBoost hoàn tất!")
    return model

def generate_unity_json(target_video_id=5, output_file='dance_demo_data.json'):
    model = build_dataset_and_train()
    
    print(f"\n=== [2/3] Đang xử lý Video kiểm thử: {target_video_id} ===")
    s_file = f'student_dance_{target_video_id}.csv'
    t_file = f'teacher_dance_{target_video_id}.csv'
    
    df_s = pd.read_csv(s_file)
    df_t = pd.read_csv(t_file)
    
    feat_df = extract_features(preprocess_data(df_s), preprocess_data(df_t))
    features = ['Joint_Error', 'Joint_Cosine', 'Velocity_Error']
    feat_df['Predicted_Pass'] = model.predict(feat_df[features]).astype(int)
    
    frames_list = []
    total_frames = feat_df['Frame_s'].nunique()
    cumulative_pass_count = 0
    
    for frame_idx, frame_df in feat_df.groupby('Frame_s'):
        frame_score = float(frame_df['Predicted_Pass'].sum())
        is_frame_pass = bool(frame_score >= FRAME_SCORE_THRESHOLD)
        if is_frame_pass:
            cumulative_pass_count += 1
            
        frame_teacher_idx = int(frame_df['Frame_t'].iloc[0])
        timestamp = round(float(frame_idx) / FPS, 3)
        
        # Format tọa độ cho Unity
        # MediaPipe: X từ 0 (trái) đến 1 (phải), Y từ 0 (trên) đến 1 (dưới)
        # Trong Unity: Y hướng lên trên, nên unity_y = 1.0 - raw_y
        student_joints = {}
        teacher_joints = {}
        
        for _, row in frame_df.iterrows():
            j_name = row['Joint_Name']
            student_joints[j_name] = {
                'x': round(float(row['raw_X_s']), 4),
                'y': round(1.0 - float(row['raw_Y_s']), 4), # Lật trục Y cho Unity
                'is_pass': int(row['Predicted_Pass']),
                'joint_error': round(float(row['Joint_Error']), 4)
            }
            teacher_joints[j_name] = {
                'x': round(float(row['raw_X_t']), 4),
                'y': round(1.0 - float(row['raw_Y_t']), 4)  # Lật trục Y cho Unity
            }
            
        frames_list.append({
            'frame_index': int(frame_idx),
            'teacher_frame_index': frame_teacher_idx,
            'timestamp': timestamp,
            'frame_score': frame_score,
            'max_score': 13.0,
            'is_pass': is_frame_pass,
            'cumulative_accuracy': round(cumulative_pass_count / (len(frames_list) + 1) * 100, 1),
            'student_joints': student_joints,
            'teacher_joints': teacher_joints
        })
        
    export_payload = {
        'project': 'Dance Movement Evaluation MVP',
        'video_id': target_video_id,
        'fps': FPS,
        'total_frames': len(frames_list),
        'score_threshold': FRAME_SCORE_THRESHOLD,
        'bones': BONES,
        'frames': frames_list
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)
        
    print(f"\n=== [3/3] Xuất file thành công: {output_file} ===")
    print(f"- Tổng số frames: {len(frames_list)}")
    print(f"- Điểm tích lũy toàn bài: {export_payload['frames'][-1]['cumulative_accuracy']}%")
    print("Dữ liệu đã sẵn sàng để đưa vào Unity!")
    return output_file

def stream_udp(json_file='dance_demo_data.json', host='127.0.0.1', port=5065):
    """Phát lại (replay) dữ liệu qua socket UDP theo đúng FPS."""
    import socket
    if not os.path.exists(json_file):
        print(f"Lỗi: Không tìm thấy {json_file}. Hãy chạy sinh file trước.")
        return
        
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (host, port)
    fps = data.get('fps', 30.0)
    delay = 1.0 / fps
    frames = data.get('frames', [])
    
    print(f"[*] Bắt đầu phát UDP tới {host}:{port} ({len(frames)} frames @ {fps} FPS)...")
    print("Nhấn Ctrl+C để dừng.")
    
    try:
        while True:
            for fr in frames:
                t0 = time.time()
                payload = json.dumps(fr).encode('utf-8')
                sock.sendto(payload, dest)
                elapsed = time.time() - t0
                time.sleep(max(0, delay - elapsed))
            print("[*] Hết bài nhảy. Tự động lặp lại từ đầu...")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[*] Đã dừng phát UDP.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export dance evaluation data to Unity JSON or stream UDP")
    parser.add_argument('--video', type=int, default=5, help="Target video ID (default: 5)")
    parser.add_argument('--out', type=str, default='dance_demo_data.json', help="Output JSON path")
    parser.add_argument('--stream', action='store_true', help="Phát lại qua UDP cho Unity")
    parser.add_argument('--port', type=int, default=5065, help="UDP port (default: 5065)")
    args = parser.parse_args()
    
    if args.stream:
        stream_udp(args.out, port=args.port)
    else:
        generate_unity_json(args.video, args.out)
