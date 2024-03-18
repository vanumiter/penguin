from microbit import *

work = 25*60 # 25 minutes converted to seconds
rest = 5*60
UPDATE_INTERVAL = (work / 5) * 1000 # time required to pass for timer to dim a column of lights
REST_INTERVAL = (rest / 5) * 1000
i = 4
n = 0 
distraction_count = 0
button_b_pressed = False # flag prevents distraction count from increasing rapidly while holding down button

while True:

    # work interval
    if button_a.is_pressed():
        for led in range(1, 4):
            display.show(led)
            sleep(1000)
        display.scroll("Go!", delay=50)

        for y in range(1, 4):
            for x in range(5):
                display.set_pixel(x, y, 9)
        
        start_time = running_time()
        next_update = running_time() + UPDATE_INTERVAL # time required to pass for timer to dim a column of lights

        while running_time() - start_time < work * 1000: # running_time and start_time is subtracted to account for time taken to start pomodoro timer
            if running_time() > next_update and i > -1:
                display.set_pixel(i, 1, 1)
                display.set_pixel(i, 2, 1)
                display.set_pixel(i, 3, 1)
                next_update = running_time() + UPDATE_INTERVAL
                i -= 1

            if pin_logo.is_touched():
                uart.write('exit\n')
                display.show(Image.HEART)
                sleep(2000)
                reset()

            if button_b.is_pressed() and not button_b_pressed:
                distraction_count += 1
                button_b_pressed = True 
            elif not button_b.is_pressed():
                button_b_pressed = False  # resets the flag when the button is released
                
            uart.write(str(microphone.sound_level()) + '\n')
            sleep(50)

        uart.write("Distraction count: " + str(distraction_count) + '\n')
        set_volume(200)
        audio.play(Sound.TWINKLE, wait=False)
        display.show(Image.YES)
        sleep(5000)
        reset()

        # break interval
    if button_b.is_pressed():
        display.show(Image.ASLEEP)
        sleep(2000)
        for y in range(1, 4):
            for x in range(5):
                display.set_pixel(x, y, 9)
        start_time = running_time()
        next_update = running_time() + REST_INTERVAL

        while running_time() - start_time < rest * 1000:
            if running_time() > next_update and i > -1:
                display.set_pixel(i, 1, 1)
                display.set_pixel(i, 2, 1)
                display.set_pixel(i, 3, 1)
                next_update = running_time() + REST_INTERVAL
                i -= 1

            if pin_logo.is_touched():
                display.show(Image.HEART)
                sleep(2000)
                reset()

        set_volume(200)
        audio.play(Sound.TWINKLE, wait=False)
        display.show(Image.YES)
        sleep(5000)
        reset()