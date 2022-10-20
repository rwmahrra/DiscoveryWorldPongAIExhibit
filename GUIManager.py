import PySimpleGUI as sg

# import Queue
import threading
import time
from queue import Queue
# layout = [[sg.Button(f'{row}, {col}') for col in range(4)] for row in range(4)]

# event, values = sg.Window('List Comprehensions', layout, no_titlebar=True, alpha_channel=0.7).read(close=True)
import os
from exhibit import motion
import exhibit.game
from exhibit.game import game_driver
# import exhibit.motion
# from exhibit.motion import motion_driver
import exhibit.ai
from exhibit.ai import ai_driver
import exhibit.visualization
from exhibit.visualization import visualization_driver as vd
import webbrowser
import importlib


mqttActive = False
gameActive = False
# motionActive = False
aiActive = False
visualizationActive = False
# emulate3DActive = False
killObject = "endThreads"
q_game = Queue()
q_game.put('noneActive')
# q_motion = Queue()
# q_motion.put('noneActive')
q_ai = Queue()
q_ai.put('noneActive')

def openMqttShell():
    print("Opening MQTT Shell:")
    # Probably done this way to open separate shell - otherwise overrides current GUI
    os.system('start exhibit\\windowsScripts\\startMosquittoPong.bat')
def closeMosquitto():
    os.system("taskkill /f /im mosquitto.exe")

# We are ignoring Emulate3D for the initial deployment of this demo
# def openEmulate3DShell():
#     # fileLoc = 'cd C:\\Users\\"DW Pong"\\Downloads\\DiscoveryWorldPongAIExhibit-master\\DiscoveryWorldPongAIExhibit-master\\' 
#     # commandM = 'C:\\"Program Files"\\Mosquitto\\mosquitto -v -c ./mosquitto.conf'
#     #"C:\Users\lawood\OneDrive - Rockwell Automation, Inc\Desktop\windowsPongScriptMosquitto.bat"
#     os.system('start C:\\Users\\"DW Pong"\\Downloads\\DiscoveryWorldPongAIExhibit-master\\Pong_with_level.demo3d')
# def closeEmulate3D():
#     # print('not yet implemented')
#     os.system("taskkill/im Demo3D2020x86.exe")

def long_function_thread(window):
    time.sleep(3)
    window.write_event_value('-THREAD DONE-', '')

def long_function(new_value):
    max_score = new_value
    print(f'max_score is now {max_score}...')
    #time.sleep(1)


sg.theme('DarkGrey')   #
mqttButton = sg.Button('mqtt server',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
gameButton = sg.Button('game',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
visualizationButton = sg.Button('visualization',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
# motionButton = sg.Button('motion',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
aiButton = sg.Button('AI - only if single pc',button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
emptyString = '  '
updateWhenStr = 'Change will take effect after restarting game'
varText = sg.Text(emptyString, size=(48,1))
# All the stuff inside window.

def startGameDriver():
    varText.update(value=emptyString)
    importlib.reload(game_driver)
    threading.Thread(target=game_driver.main, args=(q_game,), name='gameThread', daemon=True).start()
    time.sleep(0.5)

# def startMotionDriver():
#     varText.update(value=emptyString)
#     importlib.reload(motion_driver)
#     threading.Thread(target=motion_driver.main,args=(q_motion,), name='motionThread', daemon=True).start()

def startAIDriver():
    threading.Thread(target=ai_driver.main, args=(q_ai,), name='ai_thread', daemon=True).start()

def startVisualizationDriver():
    threading.Thread(target=vd.main, args=('',), name='visualizationThread', daemon=True).start()

def openVisualizationBrowser():
    webbrowser.open("http://localhost:8000/")
    #os.system('start C:\\Users\\"DW Pong"\\Downloads\\DiscoveryWorldPongAIExhibit-master\\DiscoveryWorldPongAIExhibit-master\\visualizer\\index.html')

layout = [  [sg.Text('Pong Placeholder Text')], 
            [mqttButton, gameButton, visualizationButton, aiButton], #Emulate3DButton left out
            [sg.Text('Change Points per Level:'), sg.InputText(size=(10, 1)), sg.Button('Accept')],
            [varText, sg.Button('Close')] ]

# Create the Window
window = sg.Window('Pong Controller', layout, no_titlebar=False, alpha_channel=0.9, keep_on_top=False)
# Event Loop to process "events" and get the "values" of the inputs
#z = threading.Thread(target=game_driver.main, args=(q,), name='gameThread', daemon=True)
#y = threading.Thread(target=long_function_thread, args=(window,), name='testThread', daemon=True)

while True:

    event, values = window.read(timeout=250)

    if not q_game.empty():
        tempQ2 = q_game.get()
        q_game.put(tempQ2)
        if tempQ2 == 'noneActive':
            gameActive = False
            gameButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))

    # if not q_motion.empty():
    #     tempQ2 = q_motion.get()
    #     q_motion.put(tempQ2)
    #     if tempQ2 == 'noneActive':
    #         motionActive = False
    #         motionButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))

    if not q_ai.empty():
        tempQ2 = q_ai.get()
        q_ai.put(tempQ2)
        if tempQ2 == 'noneActive':
            aiActive = False
            aiButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))

    if event == sg.WIN_CLOSED or event == 'Close': # if user closes window or clicks cancel
        # close down everything
        break
    elif event == 'mqtt server':
        if mqttActive: #mqttActive:
            print('shuting down mqtt server')
            closeMosquitto()
            mqttActive = False
            mqttButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))

        else: # if mqttActive == False:
            mqttActive = True
            print('starting up mqtt server')
            openMqttShell()
            #mqttButton.ButtonColor = sg.theme_background_color()            
            mqttButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_button_color()[1]))

    elif event == 'game':
        if gameActive:
            
            if q_game.empty():
                print('shutting down game driver')
                q_game.put("endThreads")
                #z.join()                
                
                # gameActive = False
                # gameButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
            else:
                print('no game thread active')

        else:
            
            if not q_game.empty():
                tempQ = q_game.get()
                if tempQ == 'noneActive':
                    while not q_game.empty(): # clear the queue
                        q_game.get()
                    print('starting up game driver')
                    gameActive = True
                    startGameDriver()
                    gameButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_button_color()[1]))
                else:
                    print('old game thread is not exited yet')

    elif event == 'visualization':
        if not visualizationActive:
            startVisualizationDriver()
            openVisualizationBrowser()
            visualizationActive = True
            visualizationButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_button_color()[1]))
            print('starting up visualization')

    # elif event == 'Emulate3D':
    #     if emulate3DActive:
    #         closeEmulate3D()
    #         emulate3DActive = False
    #         Emulate3DButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))

    #     else:
    #         print('starting up Emulate3D')
    #         openEmulate3DShell()                    
    #         Emulate3DButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_button_color()[1]))
    #         emulate3DActive = True

    elif event == 'Accept':
        print('Changing points per level to ', values[0])
        max_score += (int(values[0]) - max_score)
        print(f'max_score is now {max_score}...')
        #long_function(values[0])
        print('Restart game_driver for new max score to take effect.')
        varText.update(value=updateWhenStr)

    # elif event == 'motion':
    #     if motionActive:
            
    #         if q_motion.empty():
    #             print('shutting down motion driver')
    #             q_motion.put("endThreads")
    #             #z.join()                
                
    #             # gameActive = False
    #             # gameButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
    #         else:
    #             print('no motion thread active')

    #     else:
            
    #         if not q_motion.empty():
    #             tempQ = q_motion.get()
    #             if tempQ == 'noneActive':
    #                 while not q_motion.empty(): # clear the queue
    #                     q_motion.get()
    #                 print('starting up motion driver')
    #                 motionActive = True
    #                 startMotionDriver()
    #                 motionButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_button_color()[1]))
    #             else:
    #                 print('old ai thread is not exited yet')
    
    elif event == 'AI - only if single pc':
        if aiActive:
            
            if q_ai.empty():
                print('shutting down ai driver')
                q_ai.put("endThreads")
                #z.join()                
                
                # gameActive = False
                # gameButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_background_color()))
            else:
                print('no ai thread active')

        else:
            
            if not q_ai.empty():
                tempQ = q_ai.get()
                if tempQ == 'noneActive':
                    while not q_ai.empty(): # clear the queue
                        q_ai.get()
                    print('starting up game driver')
                    aiActive = True
                    startAIDriver()
                    aiButton.update(button_color=(sg.theme_element_text_color() +' on '+ sg.theme_button_color()[1]))
                else:
                    print('old ai thread is not exited yet')

    elif event == '-THREAD DONE-':
        print('Your long operation completed')
    #else:
    #    print(event, values)
q_game.put("endThreads")
q_ai.put("endThreads")
# closeEmulate3D()
# if z.is_alive:
#     z.join()
#y.join()
window.close()