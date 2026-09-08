import time
from application_state import ApplicationState, StateManager
from voice_assistant import VoiceAssistant
from gesture_config import GESTURES

class CommandEngine:
    def __init__(self, state_manager: StateManager, tts: VoiceAssistant, app_callbacks: dict):
        self.state_manager = state_manager
        self.tts = tts
        self.app_callbacks = app_callbacks
        
        # Menu indices
        self.main_menu_items = ["Dataset Collection", "Model Training", "Recognition", "Settings", "Help", "Exit"]
        self.main_menu_index = 0
        
        # Dataset Collection state
        self.dataset_gesture_index = 0
        
        # GUI debug variables
        self.last_spoken_text = ""
        self.last_intent = ""
        
    def say(self, message: str):
        self.last_spoken_text = message
        self.tts.speak(message)

    def process_intent(self, intent: str, payload: str = None):
        """Processes intents (NEXT, PREV, SELECT, BACK, HOME, HELP) across different states."""
        self.last_intent = intent
        state = self.state_manager.get_state()
        print(f"[CommandEngine] Processing Intent: {intent} at State: {state}")
        
        if intent == "HOME":
            if state != ApplicationState.MAIN_MENU:
                self.say("Navigating home.")
                self.state_manager.set_state(ApplicationState.MAIN_MENU)
                self.say(f"You have returned to the main menu. I'll guide you through the options.")
            else:
                self.say(f"You are already at the main menu. You can show one finger to scroll options, or show three fingers to select.")
            return

        if intent == "HELP":
            self._handle_help(state)
            return

        if state == ApplicationState.STARTUP or state == ApplicationState.WAITING_FOR_HAND:
            # First interaction moves us to the main menu
            self.state_manager.set_state(ApplicationState.MAIN_MENU)
            self.main_menu_index = 0
            self.say("I can see your hand. You're now at the main menu. I'll guide you through the available options.")
            return
            
        elif state == ApplicationState.MAIN_MENU:
            if intent == "NEXT":
                self.main_menu_index = (self.main_menu_index + 1) % len(self.main_menu_items)
                self._announce_menu_option()
            elif intent == "PREV":
                self.main_menu_index = (self.main_menu_index - 1) % len(self.main_menu_items)
                self._announce_menu_option()
            elif intent == "SELECT":
                self._execute_main_menu_selection()
            elif intent == "BACK":
                self.say("You are already at the main menu. Show three fingers to select an option, or swipe to browse.")
                
        elif state == ApplicationState.DATASET_MENU:
            if intent == "BACK":
                self.state_manager.set_state(ApplicationState.MAIN_MENU)
                self.say("Main menu.")
            elif intent == "SELECT":
                self.state_manager.set_state(ApplicationState.DATASET_COLLECTING)
                self.dataset_gesture_index = 0
                self.say("Dataset Collection is active.")
                self._start_collecting_gesture()

        elif state == ApplicationState.DATASET_COLLECTING:
            if intent == "BACK":
                self.app_callbacks.get("stop_recording")()
                self.state_manager.set_state(ApplicationState.MAIN_MENU)
                self.say("Recording cancelled. Returned to main menu.")
            # Progression logic inside DATASET_COLLECTING is mainly handled iteratively by the app directly,
            # but user can skip or re-record if we wanted. For now, it auto-advances.
                
        elif state == ApplicationState.TRAINING_CONFIRMATION:
            if intent == "SELECT":
                self.state_manager.set_state(ApplicationState.TRAINING)
                self.say("I'm starting model training. Please wait.")
                success = self.app_callbacks.get("train_model")()
                if success:
                    self.state_manager.set_state(ApplicationState.MODEL_READY)
                    self.say("Training is complete. Your model has been saved successfully.")
                else:
                    self.state_manager.set_state(ApplicationState.MAIN_MENU)
                    self.say("I wasn't able to complete the training. Please check that you have collected enough gesture samples and try again.")
            elif intent == "BACK":
                self.state_manager.set_state(ApplicationState.MAIN_MENU)
                self.say("Training cancelled. Returning to main menu.")

        elif state == ApplicationState.MODEL_READY:
            if intent == "SELECT":
                # Assuming this enters Recognition
                self._execute_main_menu_selection("Recognition")
            elif intent == "BACK":
                self.state_manager.set_state(ApplicationState.MAIN_MENU)
                self.say("Returned to main menu.")

        elif state == ApplicationState.RECOGNITION:
            # During recognition, gestures map directly to intents
            if intent == "NEXT":
                self.say("Moved to the next option.")
            elif intent == "PREV":
                self.say("Moved to the previous option.")
            elif intent == "SELECT":
                self.say("Confirmed.")
            elif intent == "BACK":
                self.app_callbacks.get("stop_recognition")()
                self.state_manager.set_state(ApplicationState.MAIN_MENU)
                self.say("Recognition stopped. Returned to main menu.")

    def _announce_menu_option(self):
        selection = self.main_menu_items[self.main_menu_index]
        if selection == "Dataset Collection":
            self.say("You've selected Dataset Collection. Use the select gesture to open it.")
        elif selection == "Model Training":
            self.say("You've selected Model Training. This option will train the gesture recognition model using your collected data.")
        elif selection == "Recognition":
            self.say("You've selected Gesture Recognition. This will allow you to control the application using recognized hand gestures.")
        elif selection == "Help":
            self.say("You've selected Help. I'll explain how the gestures work.")
        elif selection == "Exit":
            self.say("You've selected Exit. Use the select gesture to close the application.")
        else:
            self.say(f"You've selected {selection}. Use the select gesture to confirm.")

    def _execute_main_menu_selection(self, override=None):
        selection = override if override else self.main_menu_items[self.main_menu_index]
        
        if selection == "Dataset Collection":
            self.state_manager.set_state(ApplicationState.DATASET_MENU)
            self.say("You're now in Dataset Collection. Show three fingers to begin recording, or a closed fist to cancel.")
            
        elif selection == "Model Training":
            self.state_manager.set_state(ApplicationState.TRAINING_CONFIRMATION)
            self.say("I'm opening Model Training. Show three fingers to confirm training or make a fist to cancel.")
            
        elif selection == "Recognition":
            if not self.app_callbacks.get("is_model_trained")():
                self.say("No trained model is available. Please collect the required gesture samples and train the model first.")
                return
            self.state_manager.set_state(ApplicationState.RECOGNITION)
            self.app_callbacks.get("start_recognition")()
            self.say("Gesture recognition is now active. You can control the application using your hand.")
            
        elif selection == "Settings":
            self.say("Settings are currently unavailable in this prototype.")
            
        elif selection == "Help":
            self._handle_help(ApplicationState.MAIN_MENU)
            
        elif selection == "Exit":
            self.say("Exiting Application.")
            self.app_callbacks.get("close_app")()

    def _start_collecting_gesture(self):
        gesture_name = GESTURES[self.dataset_gesture_index]
        pretty = gesture_name.replace('_', ' ').title()
        self.say(f"Please chose the gesture you want to record. {pretty} is queued. Please hold the gesture clearly in front of the camera.")
        self.app_callbacks.get("select_gesture")(gesture_name)
        # The main.py update_loop should detect when it becomes stable and start collecting,
        # OR we could just start recording and naturally collect 50 frames.
        # User requested: "Open Palm detected. Collecting samples... Open Palm samples completed."
        
    def advance_dataset_collection(self):
        """Called by main.py when it successfully finishes collecting 50 samples for the current gesture."""
        if self.state_manager.get_state() != ApplicationState.DATASET_COLLECTING:
            return
            
        gesture_name = GESTURES[self.dataset_gesture_index]
        pretty = gesture_name.replace('_', ' ').title()
        self.say(f"Recording is complete. Your {pretty} gesture samples have been saved.")
        
        self.dataset_gesture_index += 1
        if self.dataset_gesture_index < len(GESTURES):
            self._start_collecting_gesture()
        else:
            self.dataset_gesture_index = 0
            self.state_manager.set_state(ApplicationState.MAIN_MENU)
            self.say("Dataset collection is fully complete. All required gesture samples have been saved. Returning to main menu.")

    def _handle_help(self, state: ApplicationState):
        if state == ApplicationState.MAIN_MENU:
            self.say("Help. One finger moves to the next option. Two fingers move to the previous option. Three fingers selects. Five fingers returns home. A closed fist goes back.")
        elif state == ApplicationState.DATASET_COLLECTING:
            gest = GESTURES[self.dataset_gesture_index].replace('_', ' ').title()
            self.say(f"Help. The current gesture is {gest}. Show a {gest} to collect samples. A closed fist cancels.")
        elif state == ApplicationState.TRAINING_CONFIRMATION:
            self.say("Help. Show three fingers to confirm training. A closed fist cancels.")
        elif state == ApplicationState.RECOGNITION:
            self.say("Help. You are in ML recognition mode. Swipes trigger next and previous. Closed fist goes back.")
        else:
            self.say("Help. One finger for Next. Two fingers for Previous. Three for Select. Fist to go back. Five to go Home.")
