class FrameData:
    def __init__(self, frame_number, frame, ear):
        self.__frame_number = frame_number
        self.__frame = frame
        self.__ear = ear
        # self.__blink_data

    @property
    def frame_number():
        return self.__frame_number

    @property
    def frame():
        return self.__frame
    
    @property
    def EAR():
        return self.__ear

    # @property
    # def blink_data():
    #     return self.__blink_data
