# TCP Server Code running on Jetson Nano
# Uses Built in socket library
# Server creates a new thread to handle each connection
# Listens on all available ipv4 addresses using port 8080 on Jetson Nano
# The code automatically runs with the main.py



import socket
import threading

# Server Configuration
HEADER = 64
FORMAT = 'utf-8'
DISCONNECT_MSG = "!DISCONNECT"
PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
if SERVER.startswith('127.0.'):
    SERVER = '0.0.0.0'
ADDR = (SERVER, PORT)
print(ADDR)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

# Buffers to hold the messages to be sent and received
msgrx = 'buff'
sndtx = 'buff'
client_connection = None

# Sends message through the channel to the active connection
def send_message():
    if client_connection:
        message = sndtx.encode(FORMAT)
        message_length = len(message)
        send_length = str(message_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length))
        client_connection.send(send_length)
        client_connection.send(message)

# For each different conn handle runs on a different thread
def handle_client(conn,addr):
    global client_connection
    print(f"NewConnection {addr} connected.")
    client_connection = conn
    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MSG:
                connected = False
            print(f"[{addr}] {msg}")
            global msgrx
            msgrx = msg
    conn.close()

# Main Server Function
# For each connection request it calls the function handle_client
def start_server():
    server.listen()
    print(f"Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread1 = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        thread1.start()
        print(f"activeconnections: {threading.activeCount() - 1}")
