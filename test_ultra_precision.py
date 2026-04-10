#!/usr/bin/env python3
"""
Ultra Precision Mode 테스트
- 최대 stiffness (6000)로 설정
- Position controller처럼 빠른 추종 테스트
"""
import requests
import time

SERVER_URL = "http://localhost:5000/"

print("[TEST] Switching to ultra precision mode...")
response = requests.post(SERVER_URL + "ultra_precision_mode")
print(f"[TEST] Response: {response.text}")

print("[TEST] Waiting 2 seconds for parameters to apply...")
time.sleep(2)

print("[TEST] Getting current state...")
state = requests.post(SERVER_URL + "getstate").json()
print(f"[TEST] Current pose: {state['pose'][:3]}")

print("[TEST] Ultra precision mode test complete!")
print("[TEST] Now you can run VLA with fast tracking")
