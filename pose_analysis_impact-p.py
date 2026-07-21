import os
import argparse
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

# 負荷軽減のため、姿勢推定はNフレームに1回だけ実行する（間引き）
FRAME_SKIP = 2

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='腰回転・体軸・重心バランスを解析する姿勢解析ツール')
    parser.add_argument('source', nargs='?', default='sample.mp4',
                         help='動画ファイルパス、またはWebカメラ番号（省略時: sample.mp4）')
    parser.add_argument('--persons', '-p', type=int, default=MAX_PERSONS,
                         help=f'解析する人数（省略時: {MAX_PERSONS}）')
    args = parser.parse_args()
    args.source  = int(args.source) if str(args.source).isdigit() else args.source
    args.persons = max(1, args.persons)
    return args


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
    移動平均（ウィンドウ = angle_historiesのmaxlen）で平滑化する。腰幅比率は正面/真横の
    区別のみで左右の向きを持たないため、速度は常に非負の「回転の速さ」を表す。
    ウィンドウは短め（例:3フレーム）にする必要がある。長くしすぎると、パンチのような
    一瞬(ウィンドウ時間より短い時間)で完了する速い回転ほど、前後の静止フレームと
    平均されて大きく過小評価されてしまう。移動平均バッファが埋まりきるまではNoneを
    返す。ここで弾かないと、サンプル数1〜数個のほぼ生値に近い平均が、平均化による
    保護なしにそのままMaxへ焼き付いてしまう。"""
    if hip_rotation_angle is None:
        return None

    prev_angle = prev_angles[rank]
    prev_angles[rank] = hip_rotation_angle
    if prev_angle is None:
        return None

    angular_velocity = abs(hip_rotation_angle - prev_angle) * fps

    history = angle_histories[rank]
    history.append(angular_velocity)
    if len(history) < history.maxlen:
        return None
    return sum(history) / len(history)


def compute_hip_width(kp, conf):
    """左右股関節のx座標の差（腰幅）"""
    if len(kp) <= RIGHT_HIP:
        return None
    if conf[LEFT_HIP] <= CONF_THRESH or conf[RIGHT_HIP] <= CONF_THRESH:
        return None
    if kp[LEFT_HIP][0] == 0 or kp[RIGHT_HIP][0] == 0:
        return None
    return abs(kp[RIGHT_HIP][0] - kp[LEFT_HIP][0])


def smooth_hip_width(hip_width, rank, hip_width_histories):
    """腰幅を移動平均（ウィンドウ = hip_width_historiesのmaxlen）で平滑化する。
    バッファが埋まりきるまで（トラッキング開始直後の数フレーム）はNoneを
    返す。ここで弾かないと、平均サンプル数が少ないままの不安定な値が
    後段の計算に混入してしまう。

    呼び出し側で2種類の用途に使い分ける：
    - 長いウィンドウ（例:20フレーム）でmax_hip_width（正面向きの基準値）を
      算出：arccosはratio=1付近で微分値が発散するため、基準値自体は
      キーポイント検出のジッターに左右されない安定した値にする必要がある。
    - 短いウィンドウ（例:5フレーム）で現在の腰幅（角度計算の分子）を算出：
      基準値と同じ長さで平滑化すると、パンチ等の素早い回転運動まで
      鈍ってしまうため、応答性を優先して短めにする。"""
    if hip_width is None:
        return None
    hip_width_histories[rank].append(hip_width)
    history = hip_width_histories[rank]
    if len(history) < history.maxlen:
        return None
    return sum(history) / len(history)


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
    args        = parse_args()
    source      = args.source
    max_persons = args.persons
    out_path    = make_output_path(source)

    model = YOLO('yolo26x-pose.pt')
    cap   = cv2.VideoCapture(source)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 姿勢推定はFRAME_SKIPフレームに1回しか実行しないため、角速度計算に
    # 使う実効fpsも同じ比率で下げる（動画本来のfpsのままだと経過時間を
    # 半分に見積もり、角速度が実際の2倍に水増しされる）
    analysis_fps = fps / FRAME_SKIP

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f'解析開始: {source}  →  出力: {out_path}')
    print("終了するには 'q' を押してください。")

    frame_count   = 0
    display_frame = None

    # 角速度の移動平均：短いウィンドウで一瞬の速い動き(パンチ等)のピークを
    # 薄めすぎないようにする（長いと素早い回転ほど過小評価されてしまう）
    angle_histories = [deque(maxlen=3) for _ in range(max_persons)]
    # max_hip_width（正面向きの基準値）算出用：安定性重視の長いウィンドウ
    hip_width_histories_slow = [deque(maxlen=20) for _ in range(max_persons)]
    # 現在の腰幅（角度計算の分子）算出用：応答性重視の短いウィンドウ
    hip_width_histories_fast = [deque(maxlen=5) for _ in range(max_persons)]
    prev_angles             = [None] * max_persons
    max_angular_velocities  = [0.0] * max_persons
    max_hip_widths          = [0.0] * max_persons
    max_hip_rotation_angles = [0.0] * max_persons

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP == 1 or display_frame is None:
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
                        hip_width_slow = smooth_hip_width(
                            hip_width, rank, hip_width_histories_slow)
                        hip_width_fast = smooth_hip_width(
                            hip_width, rank, hip_width_histories_fast)
                        if hip_width_slow is not None:
                            max_hip_widths[rank] = max(
                                max_hip_widths[rank], hip_width_slow)
                        hip_rotation_angle = compute_hip_rotation_angle(
                            hip_width_fast, max_hip_widths[rank])
                        if hip_rotation_angle is not None:
                            max_hip_rotation_angles[rank] = max(
                                max_hip_rotation_angles[rank], hip_rotation_angle)

                        angular_velocity = update_hip_angular_velocity(
                            hip_rotation_angle, analysis_fps, rank, prev_angles, angle_histories)
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
