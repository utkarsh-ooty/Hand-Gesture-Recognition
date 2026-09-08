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
        
    def say(self, message: str):
        self.tts.speak(message)

    def process_intent(self, intent: str, payload: str = None):
        """Processes intents (NEXT, PREV, SELECT, BACK, HOME, HELP) across different states."""
        state = self.state_manager.get_state()
        print(f"[CommandEngine] Processing Intent: {intent} at State: {state}")
        
        if intent == "HOME":
            if state != ApplicationState.MAIN_MENU:
                self.say("Returning to main menu.")
                self.state_manager.set_state(ApplicationState.MAIN_MENU)
                self.say(f"Main menu. {self.main_menu_items[self.main_menu_index]} is selected.")
            else:
                self.say(f"You are already at the main menu. {self.main_menu_items[self.main_menu_index]} is selected. Show one finger to move next, or three fingers to select.")
            return

        if intent == "HELP":
            self._handle_help(state)
            return

        if state == ApplicationState.STARTUP or state == ApplicationState.WAITING_FOR_HAND:
            # First interaction moves us to the main menu
            self.state_manager.set_state(ApplicationState.MAIN_MENU)
            self.main_menu_index = 0
            self.say("Main menu. Dataset Collection, Training Model, Recognition, Settings, Help, and Exit are available. Use the configured hand gestures to navigate and select an option.")
            return
            
        elif state == ApplicationState.MAIN_MENU:
            if intent == "NEXT":
                self.main_menu_index = (self.main_menu_index + 1) % len(self.main_menu_items)
                self.say(f"{self.main_menu_items[self.main_menu_index]} selected.")
            elif intent == "PREV":
                self.main_menu_index = (self.main_menu_index - 1) % len(self.main_menu_items)
                self.say(f"{self.main_menu_items[self.main_menu_index]} selected.")
            elif intent == "SELECT":
                self._execute_main_menu_selection()
            elif intent == "BACK":
                self.say("You are already at the main menu.")
                
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
                self.say("Training confirmed. Starting training. Please wait.")
                success = self.app_callbacks.get("train_model")()
                if success:
                    self.state_manager.set_state(ApplicationState.MODEL_READY)
                    self.say("Training completed successfully. Gesture recognition is ready.")
                else:
                    self.state_manager.set_state(ApplicationState.MAIN_MENU)
                    self.say("Training failed. Please check the collected samples in Dataset Collection.")
            elif intent == "BACK":
                self.state_manager.set_state(ApplicationState.MAIN_MENU)
                self.say("Training cancelled.")

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

    def _execute_main_menu_selection(self, override=None):
        selection = override if override else self.main_menu_items[self.main_menu_index]
        
        if selection == "Dataset Collection":
            self.state_manager.set_state(ApplicationState.DATASET_MENU)
            self.say("Dataset Collection opened. Show three fingers to start the sequence, or a closed fist to cancel.")
            
        elif selection == "Model Training":
            self.state_manager.set_state(ApplicationState.TRAINING_CONFIRMATION)
            self.say("Model Training selected. Training will use the collected gesture samples. Show three fingers to confirm or make a fist to cancel.")
            
        elif selection == "Recognition":
            if not self.app_callbacks.get("is_model_trained")():
                self.say("No trained model is available. Please collect the required gesture samples and train the model first.")
                return
            self.state_manager.set_state(ApplicationState.RECOGNITION)
            self.app_callbacks.get("start_recognition")()
            self.say("Gesture Recognition mode is active. Please perform a gesture.")
            
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
        self.say(f"Gesture {self.dataset_gesture_index + 1} of 6: {pretty}. Please show the gesture you want to record.")
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
        self.say(f"{pretty} samples completed.")
        
        self.dataset_gesture_index += 1
        if self.dataset_gesture_index < len(GESTURES):
            self._start_collecting_gesture()
        else:
            self.dataset_gesture_index = 0
            self.state_manager.set_state(ApplicationState.MAIN_MENU)
            self.say("All required gesture samples have been collected. Model Training is ready.")

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
