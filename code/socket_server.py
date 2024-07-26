import socket
import threading

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

msgrx = 'buff'
sndtx = 'buff'
client_connection = None

def send_message():
    if client_connection:  # Check if there is a client connection
        message = sndtx.encode(FORMAT)
        message_length = len(message)
        send_length = str(message_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length))
        client_connection.send(send_length)
        client_connection.send(message)

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

def start_server():
    server.listen()
    print(f"Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread1 = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        thread1.start()
        print(f"activeconnections: {threading.activeCount() - 1}")
