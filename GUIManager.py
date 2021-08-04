import PySimpleGUI as sg

# import Queue
import threading
import time
from queue import Queue
# layout = [[sg.Button(f'{row}, {col}') for col in range(4)] for row in range(4)]

# event, values = sg.Window('List Comprehensions', layout, no_titlebar=True, alpha_channel=0.7).read(close=True)
import os
import exhibit.game.game_driver as gd


killObject = "endThreads"
q = Queue()

def openMqttShell():
    fileLoc = "cd C:\\'Program Files'\\Mosquitto\\" #cd C:\'Program Files'\Mosquitto\
    commandM = "mosquitto -v -c ./mosquitto.conf"
    #"C:\Users\lawood\OneDrive - Rockwell Automation, Inc\Desktop\windowsPongScriptMosquitto.bat"
    os.system(fileLoc)
    os.system(commandM)
    #os.system("start " + fileName)
def closeMosquittoShell():
    os.system(signal.SIGINT)


def long_function_thread(window):
    time.sleep(3)
    window.write_event_value('-THREAD DONE-', '')

def long_function():
    y.start()
    #time.sleep(1)

mqttActive = False
gameActive = False

sg.theme('DarkAmber')   #
mqttButton = sg.Button('mqtt server',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
gameButton = sg.Button('game',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
visualizationButton = sg.Button('visualization',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
Emulate3DButton = sg.Button('Emulate3D',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
# All the stuff inside your window.

def startGameDriver():
    threading.Thread(target=gd.main, args=(q,), name='gameThread', daemon=True).start()

layout = [  [sg.Text('Some text on Row 1')],
            [sg.Text('Enter something on Row 2'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Close')],
            [mqttButton, gameButton, visualizationButton, Emulate3DButton] ]

# Create the Window
window = sg.Window('Pong Controller', layout, no_titlebar=True, alpha_channel=0.9)
# Event Loop to process "events" and get the "values" of the inputs
#z = threading.Thread(target=gd.main, args=(q,), name='gameThread', daemon=True)
y = threading.Thread(target=long_function_thread, args=(window,), name='testThread', daemon=True)

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
            openMqttShell()
            #mqttButton.button_text = "ehh"
            #mqttButton.ButtonColor = sg.theme_background_color()            
            mqttButton.update(button_color=(sg.theme_background_color() +' on '+ sg.theme_element_text_color()))

    elif event == 'game':
        if gameActive:
            gameActive = False
            print('shutting down game driver')
            q.put("endThreads")
            #z.join()
            gameButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))

        else:
            gameActive = True
            print('starting up game driver')
            import exhibit.game.game_driver as gd
            q.put("don't endThreads")
            startGameDriver()
            gameButton.update(button_color=(sg.theme_background_color() +' on '+ sg.theme_element_text_color()))

    elif event == 'visualization':
        print('starting up visualization')
    elif event == 'Emulate3D':
        print('starting up Emulate3D')
        
    elif event == 'Ok':
        print('You entered ', values[0])
        long_function()
        print('Long function has returned from starting')
    elif event == '-THREAD DONE-':
        print('Your long operation completed')
    else:
        print(event, values)
q.put("endThreads")
# if z.is_alive:
#     z.join()
#y.join()
window.close()