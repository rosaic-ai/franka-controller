import socket
import json

class FrankaClient:
    def __init__(self, host='127.0.0.1', port=4999):
        self.host = host
        self.port = port

    def send_command(self, target_pose, gripper_command=None):
        """
        target_pose: [x, y, z, qx, qy, qz, qw]
        gripper_command: 'open' or 'close' (optional)
        """
        data = {
            "target_pose": target_pose
        }
        if gripper_command:
            data["gripper_command"] = gripper_command

        message = json.dumps(data) + '\n'
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.sendall(message.encode('utf-8'))
                print(f"Sent: {gripper_command} at {target_pose[:3]}")
        except Exception as e:
            print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    # 사용 예시
    client = FrankaClient(host='172.27.190.125', port=4999)
    
    # 그리퍼 닫기 (현재 위치가 [0.45, 0, 0.2]라고 가정할 때)
    current_pose = [0.45, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    client.send_command(current_pose, gripper_command='close')
