
# This is the code for mobile application
# To run from the pc, install flet first https://flet.dev/docs/getting-started/
# Following the documentation use, flet run app.py
# ------------------------------------------------------------------------
# The app is written fully in python to natively support TCP Client code
# Rest of the app is all visualization elements written with the help of flet library
# Functions at the top of the page handle the TCP Client process, rest is visuals




from flet import *
import socket
import asyncio

# Socket Parameters
HEADER = 64
PORT = 8080
FORMAT = 'utf-8'
DISCONNECT_MSG = "!DISCONNECT"
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


async def main(page: Page):

# Functions
# ************

    async def onConnect(e):
        ADDR = (IPtextbar.value, PORT)
        try:
            client.connect(ADDR)
        except:
            chatFeed.controls.append(Text(f"[ERROR]: Connection Failed",weight=FontWeight.BOLD,color=colors.ORANGE))
        else:
            chatFeed.controls.append(Text(f"[Connected]: {ADDR}",weight=FontWeight.BOLD,color=colors.GREEN))
            asyncio.create_task(receive_messages())
        IPtextbar.value = ""
        page.update()

    async def receive_messages():
        while True:
            try:
                msg_length = await asyncio.get_event_loop().run_in_executor(None, client.recv, HEADER)
                msg_length = msg_length.decode(FORMAT)
                if msg_length:
                    msg_length = int(msg_length)
                    msg = await asyncio.get_event_loop().run_in_executor(None, client.recv, msg_length)
                    msg = msg.decode(FORMAT)
                    chatFeed.controls.append(Text(f"[Server]: {msg}", weight=FontWeight.BOLD, color=colors.BLUE))
                    page.update()
            except Exception as e:
                chatFeed.controls.append(Text(f"[ERROR]: {str(e)}", weight=FontWeight.BOLD, color=colors.ORANGE))
                page.update()
                break


    async def onDisconnect(e):
        msg = "!DISCONNECT"
        message = msg.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length))
        client.send(send_length)
        client.send(message)
        client.close()
        chatFeed.controls.append(Text(f"[DISCONNECTED]",weight=FontWeight.BOLD,color=colors.RED))
        messagetextfield.value = ""
        page.update()

    async def send_click(e):
        msg = messagetextfield.value
        message = msg.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length))
        client.send(send_length)
        client.send(message)
        chatFeed.controls.append(Text(f"[Client]: {msg}",weight=FontWeight.BOLD,color=colors.BLACK))
        messagetextfield.value = ""
        page.update()

# Display Layout Design
# *****************************

# variables
    IPtextbar = TextField(label="Enter IP Adress: ", border="underline", filled=True,adaptive = False)
# IP Text Box 
    IPtextField = Container(
        width= page.width,
        height=150,
        bgcolor= colors.GREEN_50,
        padding=padding.only(
            left=20,right=20,
            bottom=30,top=30,
        ),
        content=IPtextbar
    )

# ************

# variables
    chatFeed = Column(auto_scroll=True,scroll=ScrollMode.AUTO)
    messagetextfield = TextField(border="underline", filled=True,border_color=colors.WHITE70,border_width=2,adaptive = False)
# Chat Window
    chatWindow = Column(
        alignment='spaceBetween',
        controls=[
            Container(
                height = 350,
                width = page.width,
                border=border.all(width=4,color=colors.BLACK45),
                bgcolor = colors.WHITE,
                padding=padding.only(
                left=10,right=10,
                bottom=20,top=20
                ),
                content=chatFeed
            ),
            Row(
                alignment='spaceBetween',
            controls=[
                Container(
                    width=250,
                    content=messagetextfield,
                    padding=padding.only(
                    left=10,right=0,
                    bottom=10,top=20
                    )
                ),
                Container(
                padding=padding.only(right=10),
                content = FilledButton(
                    adaptive = False,
                    style=ButtonStyle(color=colors.WHITE,bgcolor=colors.BROWN_300),
                    text = "Send",
                    on_click=send_click
                )
                )
            ]
            )
        ]
    )

# ************

# Button Layout
    buttonArea = Container(
        width=page.width,
        height=100,
        alignment= alignment.bottom_center,
        bgcolor=colors.BROWN_300,
        padding=padding.only(
            left=10,right=10,
            bottom=20,top=20
        ),
        content=Row(
            alignment='spaceBetween',
            controls=[
                Container(
                content=FilledButton(
                    adaptive = False,
                    text = "Connect",
                    on_click=onConnect,
                    style=ButtonStyle(
                    shape=RoundedRectangleBorder(radius=30),
                    bgcolor={MaterialState.DEFAULT: colors.BLACK45},
                    overlay_color=colors.TRANSPARENT,
                    color={
                        MaterialState.DEFAULT: colors.WHITE,
                        MaterialState.FOCUSED: colors.WHITE
                    },
                    side={
                        MaterialState.HOVERED: BorderSide(3, colors.WHITE)
                    }
                    )
                )
                ),   
                Container(
                content=FilledButton(
                    adaptive = False,
                    text = "Disconnect",
                    on_click=onDisconnect,
                    style=ButtonStyle(
                    shape=RoundedRectangleBorder(radius=30),
                    bgcolor={MaterialState.DEFAULT: colors.YELLOW},
                    overlay_color=colors.TRANSPARENT,
                    color={
                        MaterialState.DEFAULT: colors.BLACK,
                        MaterialState.FOCUSED: colors.WHITE
                    },
                    side={
                        MaterialState.HOVERED: BorderSide(3, colors.BLACK)
                    }
                    )
                )
                )
            ]
        )
    )

# ************

# Main Page Container
    container = Container(
        width=page.width,
        height=page.height,
        bgcolor=colors.GREEN_50,
        content=Column(
            alignment='start',
            spacing= 10,
            controls=[
             IPtextField,
             chatWindow,
             buttonArea,
            ]
        )
    )
    page.adaptive = True
    page.bgcolor = colors.GREEN_50
    page.padding = 0
    page.add(container)

app(target=main, view=None, port=5050)

