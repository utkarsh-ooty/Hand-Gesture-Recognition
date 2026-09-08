from enum import Enum

class ApplicationState(Enum):
    STARTUP = "STARTUP"
    WAITING_FOR_HAND = "WAITING_FOR_HAND"
    MAIN_MENU = "MAIN_MENU"
    DATASET_MENU = "DATASET_MENU"
    DATASET_COLLECTING = "DATASET_COLLECTING"
    TRAINING_CONFIRMATION = "TRAINING_CONFIRMATION"
    TRAINING = "TRAINING"
    MODEL_READY = "MODEL_READY"
    RECOGNITION = "RECOGNITION"
    HELP = "HELP"
    SETTINGS = "SETTINGS"
    ERROR = "ERROR"

class StateManager:
    def __init__(self):
        self.state = ApplicationState.STARTUP
        
    def get_state(self):
        return self.state
        
    def set_state(self, new_state: ApplicationState):
        self.state = new_state
