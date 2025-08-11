"""
구미 로봇 제어 서버

- 클라이언트로부터 target_pose 데이터를 받아서 처리
- 포트 5000에서 대기
"""

import socket
import json
import numpy as np

def start_server(host='0.0.0.0', port=5000):
    # IPv4, TCP 소켓 생성
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 소켓 옵션 설정 (포트 재사용 가능하도록)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # 서버 소켓 바인딩
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"서버가 {host}:{port}에서 대기 중입니다.")
        
        while True:
            # 클라이언트 연결 대기
            client_socket, addr = server_socket.accept()
            print(f"{addr}에서 연결됨")
            
            # 수신 버퍼
            buffer = ""
            
            try:
                while True:
                    # 데이터 수신
                    data = client_socket.recv(1024).decode('utf-8')
                    if not data:
                        break
                    
                    # 버퍼에 데이터 추가
                    buffer += data
                    
                    # 개행문자로 분리된 메시지 처리
                    while '\n' in buffer:
                        message, buffer = buffer.split('\n', 1)
                        try:
                            # JSON 디코딩
                            received_data = json.loads(message)
                            target_pose = np.array(received_data['target_pose'])
                            timestamp = received_data['timestamp']
                            
                            # 데이터 처리 및 출력
                            print(f"수신 시각: {timestamp}")
                            print(f"Target pose: {target_pose}")
                            
                            # TODO: 여기에 로봇 제어 코드 추가
                            
                        except json.JSONDecodeError as e:
                            print(f"JSON 디코딩 오류: {e}")
                            continue
                        
            except ConnectionResetError:
                print("클라이언트와의 연결이 끊어졌습니다.")
            finally:
                client_socket.close()
                print(f"{addr} 연결 종료")
                
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server_socket.close()

if __name__ == '__main__':
    start_server()