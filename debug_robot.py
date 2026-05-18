#!/usr/bin/env python3
"""FrankaVC 진단 스크립트 - 문제 원인 파악용"""
import time
import numpy as np
import requests
from robot_infra.envs.franka_vc_env import FrankaVC
from modules.config import Config

CONFIG_PATH = '/home/rosaic/manip_catkin_ws/src/franka-controller/robot_infra/configs/robot_params.yaml'

config_robot = Config(CONFIG_PATH).get_config()
control = FrankaVC(config_robot=config_robot, hz=30, start_gripper=0)

# ── 1. 설정값 확인 ───────────────────────────────────
print("\n=== 1. 설정값 ===")
print(f"서버 URL  : {control.url}")
print(f"pose_limit: {control.pose_limit}")

# ── 2. 현재 포즈 확인 ───────────────────────────────
print("\n=== 2. 현재 포즈 ===")
raw = control.get_ee_pose()
print(f"get_ee_pose() 전체: {raw}")
current_pose = np.array(raw['pose'])
print(f"배열 길이: {len(current_pose)}")
print(f"XYZ      : {current_pose[:3]}")
print(f"Quat     : {current_pose[3:]}")

# ── 3. ensure_within_limits 클리핑 확인 ─────────────
print("\n=== 3. 클리핑 테스트 (+1cm X이동) ===")
target = current_pose.copy()
target[0] += 0.01
print(f"이동 목표 XYZ: {target[:3]}")
clipped = control.ensure_within_limits(target)
print(f"클리핑 후 XYZ: {clipped[:3]}")
if not np.allclose(target[:3], clipped[:3], atol=1e-6):
    print(f"⚠️  클리핑 발생! 차이: {clipped[:3] - target[:3]}")
else:
    print("✅ 클리핑 없음 - 목표값 그대로 전달됨")

# ── 4. HTTP /pose 응답 직접 확인 ─────────────────────
print("\n=== 4. HTTP /pose 요청 테스트 ===")
arr = np.array(clipped).astype(np.float32)
data = {"arr": arr.tolist()}
print(f"전송 데이터: {data['arr']}")
try:
    response = requests.post(control.url + 'pose', json=data, timeout=5)
    print(f"HTTP 상태코드: {response.status_code}")
    print(f"HTTP 응답    : {response.text[:300]}")
except Exception as e:
    print(f"❌ 요청 실패: {e}")

# ── 5. 이동 후 실제 포즈 변화 확인 ──────────────────
print("\n=== 5. 이동 후 포즈 확인 ===")
time.sleep(1.0)
after = control.get_ee_pose()
after_pose = np.array(after['pose'])
print(f"이동 후 XYZ : {after_pose[:3]}")
diff = after_pose[:3] - current_pose[:3]
print(f"실제 이동량 : {diff}")
if np.allclose(diff, 0, atol=1e-4):
    print("❌ 로봇이 움직이지 않음")
else:
    print("✅ 로봇 이동 확인됨")
