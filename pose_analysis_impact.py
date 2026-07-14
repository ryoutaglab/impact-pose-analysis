import sys
import os
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# キーポイントインデックス定数（YOLOv8 17点モデル）
LEFT_EAR       = 3
RIGHT_EAR      = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16

# 解析する最大人数
MAX_PERSONS = 2

# スケルトン接続定義（0-indexed）
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11],  [6, 12],  [5, 6],   [5, 7],   [6, 8],
    [7, 9],   [8, 10],  [1, 2],   [0, 1],   [0, 2],
    [1, 3],   [2, 4],
]

# 人物ごとの描画色（BGR）
PERSON_COLORS = [
    (0, 255, 255),   # 黄
    (0, 165, 255),   # オレンジ
    (255, 0, 255),   # マゼンタ
]

# 信頼度しきい値
CONF_THRESH = 0.5

# 計算不能時の表示色（グレー）
GRAY = (128, 128, 128)


def resolve_source(argv):
    if len(argv) > 1:
        return int(argv[1]) if argv[1].isdigit() else argv[1]
    return 'sample.mp4'


def resolve_max_persons(argv):
    if len(argv) > 2:
        try:
            return max(1, int(argv[2]))
        except ValueError:
            pass
    return MAX_PERSONS


def make_output_path(source):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    if isinstance(source, int):
        return f'output_{timestamp}.mp4'
    base, _ = os.path.splitext(source)
    return f'{base}_{timestamp}.mp4'


def draw_person(frame, kp, conf, color):
    for i, (x, y) in enumerate(kp):
        if x > 0 and y > 0 and conf[i] > CONF_THRESH:
            cv2.circle(frame, (int(x), int(y)), 4, color, cv2.FILLED)

    for a, b in SKELETON:
        if a >= len(kp) or b >= len(kp):
            continue
        x1, y1 = kp[a]
        x2, y2 = kp[b]
        if (x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0
                and conf[a] > CONF_THRESH and conf[b] > CONF_THRESH):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


def draw_midline(frame, kp, conf):
    if len(kp) <= RIGHT_HIP:
        return
    pts = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if any(conf[i] <= CONF_THRESH or kp[i][0] == 0 for i in pts):
        return

    mid_sh  = (int((kp[LEFT_SHOULDER][0] + kp[RIGHT_SHOULDER][0]) / 2),
               int((kp[LEFT_SHOULDER][1] + kp[RIGHT_SHOULDER][1]) / 2))
    mid_hip = (int((kp[LEFT_HIP][0] + kp[RIGHT_HIP][0]) / 2),
               int((kp[LEFT_HIP][1] + kp[RIGHT_HIP][1]) / 2))

    cv2.line(frame, mid_sh, mid_hip, (0, 255, 255), 4)
    cv2.circle(frame, mid_sh,  5, (0, 0, 255), cv2.FILLED)
    cv2.circle(frame, mid_hip, 5, (0, 0, 255), cv2.FILLED)

    # 肩中点から耳への線（両耳→中点、片耳→その耳、両方なし→スキップ）
    if len(kp) > RIGHT_EAR:
        l_ok = conf[LEFT_EAR]  > CONF_THRESH and kp[LEFT_EAR][0]  > 0
        r_ok = conf[RIGHT_EAR] > CONF_THRESH and kp[RIGHT_EAR][0] > 0
        if l_ok and r_ok:
            head_pt = (int((kp[LEFT_EAR][0] + kp[RIGHT_EAR][0]) / 2),
                       int((kp[LEFT_EAR][1] + kp[RIGHT_EAR][1]) / 2))
        elif l_ok:
            head_pt = (int(kp[LEFT_EAR][0]),  int(kp[LEFT_EAR][1]))
        elif r_ok:
            head_pt = (int(kp[RIGHT_EAR][0]), int(kp[RIGHT_EAR][1]))
        else:
            head_pt = None
        if head_pt is not None:
            cv2.line(frame, mid_sh, head_pt, (0, 255, 255), 4)
            cv2.circle(frame, head_pt, 5, (0, 0, 255), cv2.FILLED)


def select_top_persons(r, max_n):
    """面積の大きい上位 max_n 人のインデックスを返す"""
    if r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = r.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return np.argsort(areas)[::-1][:max_n].tolist()


def update_hip_angular_velocity(hip_rotation_angle, fps, rank, prev_angles, angle_histories):
    """腰の回転角速度（°/秒）をHip Rot Angle（腰幅ベースの回転角度）の時間微分から算出し、
    直近10フレームの移動平均で平滑化する。腰幅比率は正面/真横の区別のみで左右の向きを
    持たないため、速度は常に非負の「回転の速さ」を表す。"""
    if hip_rotation_angle is None:
        return None

    prev_angle = prev_angles[rank]
    prev_angles[rank] = hip_rotation_angle
    if prev_angle is None:
        return None

    angular_velocity = abs(hip_rotation_angle - prev_angle) * fps

    angle_histories[rank].append(angular_velocity)
    return sum(angle_histories[rank]) / len(angle_histories[rank])


def compute_hip_width(kp, conf):
    """左右股関節のx座標の差（腰幅）"""
    if len(kp) <= RIGHT_HIP:
        return None
    if conf[LEFT_HIP] <= CONF_THRESH or conf[RIGHT_HIP] <= CONF_THRESH:
        return None
    if kp[LEFT_HIP][0] == 0 or kp[RIGHT_HIP][0] == 0:
        return None
    return abs(kp[RIGHT_HIP][0] - kp[LEFT_HIP][0])


def compute_hip_rotation_angle(hip_width, max_hip_width):
    """腰幅を最大腰幅で正規化した水平回転角度の近似値（0°=真横、90°=正面）"""
    if hip_width is None or max_hip_width <= 0:
        return None
    rotation_ratio = hip_width / max_hip_width
    return np.degrees(np.arccos(np.clip(rotation_ratio, 0, 1)))


def compute_body_axis_angle(kp, conf):
    """体軸（肩中点→腰中点）の傾き角度（度）。前傾+ / 後傾-"""
    if len(kp) <= RIGHT_HIP:
        return None
    pts = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if any(conf[i] <= CONF_THRESH or kp[i][0] == 0 for i in pts):
        return None

    mid_sh  = ((kp[LEFT_SHOULDER][0] + kp[RIGHT_SHOULDER][0]) / 2,
               (kp[LEFT_SHOULDER][1] + kp[RIGHT_SHOULDER][1]) / 2)
    mid_hip = ((kp[LEFT_HIP][0] + kp[RIGHT_HIP][0]) / 2,
               (kp[LEFT_HIP][1] + kp[RIGHT_HIP][1]) / 2)

    dx = mid_sh[0] - mid_hip[0]
    dy = mid_sh[1] - mid_hip[1]
    return np.degrees(np.arctan2(dx, -dy))


def compute_balance(kp, conf):
    """左右重心バランス（%）。腰中点と両足首のx座標から算出"""
    if len(kp) <= RIGHT_ANKLE:
        return None
    pts = [LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]
    if any(conf[i] <= CONF_THRESH or kp[i][0] == 0 for i in pts):
        return None

    mid_hip_x = (kp[LEFT_HIP][0] + kp[RIGHT_HIP][0]) / 2
    l_ankle_x = kp[LEFT_ANKLE][0]
    r_ankle_x = kp[RIGHT_ANKLE][0]

    total = abs(mid_hip_x - l_ankle_x) + abs(mid_hip_x - r_ankle_x)
    if total == 0:
        return None

    left_pct  = abs(mid_hip_x - r_ankle_x) / total * 100
    right_pct = 100 - left_pct
    return left_pct, right_pct


def color_for_angular_velocity(angular_velocity):
    av = abs(angular_velocity)
    if av >= 100:
        return (0, 255, 0)
    if av >= 50:
        return (0, 255, 255)
    return (0, 0, 255)


def color_for_hip_rotation_angle(angle):
    if angle <= 30:
        return (0, 255, 0)
    if angle <= 60:
        return (0, 255, 255)
    return (0, 0, 255)


def color_for_body_axis(angle):
    a = abs(angle)
    if a <= 5:
        return (0, 255, 0)
    if a <= 15:
        return (0, 255, 255)
    return (0, 0, 255)


def color_for_balance(left_pct):
    if 40 <= left_pct <= 60:
        return (0, 255, 0)
    if 30 <= left_pct < 40 or 60 < left_pct <= 70:
        return (0, 255, 255)
    return (0, 0, 255)


def draw_metrics(frame, rank, width, angular_velocity, max_angular_velocity,
                  hip_rotation_angle, max_hip_rotation_angle, body_axis, balance):
    """腰回転角速度・腰回転角度・体軸傾き・重心バランスを画面に数値表示する（英語表記、cv2.putTextは日本語非対応のため）"""
    label = f'P{rank + 1}'
    x = 20 if rank == 0 else width - 300
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2

    if angular_velocity is None:
        text, color = f'{label} Hip Rot: ---', GRAY
    else:
        text  = (f'{label} Hip Rot: {angular_velocity:.1f} deg/s '
                  f'(Max {max_angular_velocity:.1f})')
        color = color_for_angular_velocity(angular_velocity)
    cv2.putText(frame, text, (x, 40), font, scale, color, thickness)

    if hip_rotation_angle is None:
        text, color = f'{label} Hip Rot Angle: ---', GRAY
    else:
        text  = (f'{label} Hip Rot Angle: {hip_rotation_angle:.1f} deg '
                  f'(Max {max_hip_rotation_angle:.1f} deg)')
        color = color_for_hip_rotation_angle(hip_rotation_angle)
    cv2.putText(frame, text, (x, 80), font, scale, color, thickness)

    if body_axis is None:
        text, color = f'{label} Axis: ---', GRAY
    else:
        text  = f'{label} Axis: {body_axis:+.1f} deg'
        color = color_for_body_axis(body_axis)
    cv2.putText(frame, text, (x, 120), font, scale, color, thickness)

    if balance is None:
        text, color = f'{label} Balance: ---', GRAY
    else:
        left_pct, right_pct = balance
        text  = f'{label} Balance: L{left_pct:.0f}% R{right_pct:.0f}%'
        color = color_for_balance(left_pct)
    cv2.putText(frame, text, (x, 160), font, scale, color, thickness)


def main():
    source      = resolve_source(sys.argv)
    max_persons = resolve_max_persons(sys.argv)
    out_path    = make_output_path(source)

    model = YOLO('yolo26x-pose.pt')
    cap   = cv2.VideoCapture(source)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f'解析開始: {source}  →  出力: {out_path}')
    print("終了するには 'q' を押してください。")

    frame_count   = 0
    display_frame = None

    angle_histories = [
        deque(maxlen=10),
        deque(maxlen=10),
    ]
    prev_angles = [None, None]
    max_angular_velocities  = [0.0, 0.0]
    max_hip_widths          = [0.0, 0.0]
    max_hip_rotation_angles = [0.0, 0.0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 2 == 1 or display_frame is None:
            results = model(frame, stream=True, verbose=False, imgsz=640)

            canvas = frame.copy()

            for r in results:
                if r.keypoints is None:
                    continue

                top_idx  = select_top_persons(r, max_persons)
                kp_all   = r.keypoints.xy.cpu().numpy()
                conf_all = (r.keypoints.conf.cpu().numpy()
                            if r.keypoints.conf is not None
                            else np.ones((len(kp_all), 17)))

                for rank, idx in enumerate(top_idx):
                    if idx >= len(kp_all):
                        continue
                    color = PERSON_COLORS[rank % len(PERSON_COLORS)]
                    kp    = kp_all[idx]
                    conf  = conf_all[idx]
                    draw_person(canvas, kp, conf, color)
                    draw_midline(canvas, kp, conf)

                    if rank < len(prev_angles):
                        body_axis = compute_body_axis_angle(kp, conf)
                        balance   = compute_balance(kp, conf)

                        hip_width = compute_hip_width(kp, conf)
                        if hip_width is not None:
                            max_hip_widths[rank] = max(max_hip_widths[rank], hip_width)
                        hip_rotation_angle = compute_hip_rotation_angle(
                            hip_width, max_hip_widths[rank])
                        if hip_rotation_angle is not None:
                            max_hip_rotation_angles[rank] = max(
                                max_hip_rotation_angles[rank], hip_rotation_angle)

                        angular_velocity = update_hip_angular_velocity(
                            hip_rotation_angle, fps, rank, prev_angles, angle_histories)
                        if angular_velocity is not None:
                            max_angular_velocities[rank] = max(
                                max_angular_velocities[rank], angular_velocity)

                        draw_metrics(canvas, rank, width,
                                     angular_velocity, max_angular_velocities[rank],
                                     hip_rotation_angle, max_hip_rotation_angles[rank],
                                     body_axis, balance)

            display_frame = canvas

        out_frame = display_frame if display_frame is not None else frame
        writer.write(out_frame)
        cv2.imshow('MMA Pose Analysis', out_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f'完了: {out_path}')


if __name__ == '__main__':
    main()
