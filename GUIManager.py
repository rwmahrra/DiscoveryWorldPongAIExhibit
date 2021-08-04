import PySimpleGUI as sg

# import Queue
import threading
import time
# layout = [[sg.Button(f'{row}, {col}') for col in range(4)] for row in range(4)]

# event, values = sg.Window('List Comprehensions', layout, no_titlebar=True, alpha_channel=0.7).read(close=True)
import os
def openFile():
    fileName = 'string' #listbox_1.get(ACTIVE)
    os.system("start " + fileName)
# *****


def long_function_thread(window):
    time.sleep(10)
    window.write_event_value('-THREAD DONE-', '')

def long_function():
    threading.Thread(target=long_function_thread, args=(window,), daemon=True).start()


sg.theme('DarkAmber')   #
# All the stuff inside your window.
layout = [  [sg.Text('Some text on Row 1')],
            [sg.Text('Enter something on Row 2'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Close')],
            [sg.Button('mqtt server'), sg.Button('game'), sg.Button('visualization'), sg.Button('Emulate3D')] ]

# Create the Window
window = sg.Window('Pong Controller', layout, no_titlebar=True, alpha_channel=0.8)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Close': # if user closes window or clicks cancel
        # close down everything
        break
    elif event == 'mqtt server':
        print('starting up mqtt server')
    elif event == 'game':
        print('starting up game driver')
    elif event == 'visualization':
        print('starting up visualization')
    elif event == 'Emulate3D':
        print('starting up Emulate3D')
        openFile()
    elif event == 'Ok':
        print('You entered ', values[0])
        long_function()
        print('Long function has returned from starting')
    elif event == '-THREAD DONE-':
        print('Your long operation completed')
    else:
        print(event, values)

window.close()