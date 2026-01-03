import os, json

# 1. Directories
INPUT_DIR  = './data/pose'
OUTPUT_DIR = './data/pose_coco17'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving COCO-17 JSONs into: {OUTPUT_DIR}")

# 2. Define Mediapipe→COCO17 index mapping
#    COCO keypoint order: [nose, left_eye, right_eye, left_ear, right_ear,
#                          left_shoulder, right_shoulder, left_elbow, right_elbow,
#                          left_wrist, right_wrist, left_hip, right_hip,
#                          left_knee, right_knee, left_ankle, right_ankle]
MP_TO_COCO = [
    0,   # nose
    2,   # left_eye
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28   # right_ankle
]

err_list = []
for fname in os.listdir(INPUT_DIR):
    if not fname.endswith('.json'):
        continue

    in_path = os.path.join(INPUT_DIR, fname)
    try:
      with open(in_path, 'r') as f:
        data = json.load(f)
    except:
      err_list.append(in_path)
      continue

    # assume your 33-point list is under "transformed_points"
    mp_pts = data.get('transformed_points')
    if mp_pts is None or len(mp_pts) != 33:
        print(f"  ❗ skipping {fname} (no 33-point data)")
        continue

    # build COCO-17 flattened [x,y,v] array
    coco_kpts = []
    for mp_idx in MP_TO_COCO:
        x, y = mp_pts[mp_idx]
        # mark all detected as visible (v=2); you can adjust logic if needed
        v = 2
        coco_kpts.append([x, y])

    # prepare output JSON
    out = {
        # carry over image reference if you like
        'original_tof_frame': data.get('original_tof_frame'),
        # standard COCO keypoints field:
        'keypoints': coco_kpts
    }

    # write it
    out_path = os.path.join(OUTPUT_DIR, fname)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"  ✔ wrote COCO17 for {fname}")

print("✅ All done!")
print(err_list)

