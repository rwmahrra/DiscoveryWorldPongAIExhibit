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

def long_function_thread(window):
    time.sleep(3)
    window.write_event_value('-THREAD DONE-', '')

def long_function():
    threading.Thread(target=long_function_thread, args=(window,), daemon=True).start()

mqttActive = False


sg.theme('DarkAmber')   #
mqttButton = sg.Button('mqtt server',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
gameButton = sg.Button('game',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
visualizationButton = sg.Button('visualization',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
Emulate3DButton = sg.Button('Emulate3D',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
# All the stuff inside your window.
layout = [  [sg.Text('Some text on Row 1')],
            [sg.Text('Enter something on Row 2'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Close')],
            [mqttButton, gameButton, visualizationButton, Emulate3DButton] ]

# Create the Window
window = sg.Window('Pong Controller', layout, no_titlebar=True, alpha_channel=0.9)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Close': # if user closes window or clicks cancel
        # close down everything
        break
    elif event == 'mqtt server':
        if mqttActive:
            print('shuting down mqtt server')
            mqttActive = False
            mqttButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))

        else:
            mqttActive = True
            print('starting up mqtt server')
            #mqttButton.button_text = "ehh"
            #mqttButton.ButtonColor = sg.theme_background_color()            
            mqttButton.update(button_color=(sg.theme_background_color() +' on '+ sg.theme_element_text_color()))

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