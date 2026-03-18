import socket
import os

SOCK_FILE = "/tmp/nyx.sock"

if os.path.exists(SOCK_FILE):
	os.remove(SOCK_FILE)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCK_FILE)
server.listen(5)

print(f"Listening for interceptor messages on {SOCK_FILE}..")
print("__"*30)

try:
	while True:
		conn,_=server.accept()
		data=conn.recv(1024)
		if data:
			print(f" [SCHEDULER RECEIVED] {data.decode('utf-8')}")
		conn.close()
except KeyboardInterrupt:
	print("Scheduler Turning OFF")
	os.remove(SOCK_FILE)
