from teleop_inference import TeleopInference, CONFIG_FILE_DICT

if __name__ == '__main__':
    print "Starting task 4 - after learning."

    #print 'Here is an example of what you will try to do.'
    #TeleopInference("config/task4_example_inference_config.yaml")

    print 'First attempt: your interaction will not be recorded.'
    TeleopInference(CONFIG_FILE_DICT[4]['e'])
    print 'Now, your interaction will be recorded.'
    while True:
        TeleopInference(CONFIG_FILE_DICT[4]['e'])
        resp = raw_input('If you would like to use this recording, please fill out the survey, then press Y.\nPress ENTER if you would like to record again.\n')
        if resp == "y" or resp == "Y":
            break
