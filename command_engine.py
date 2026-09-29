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
        print(f"[CommandEngine] Processing Intent: {intent} (payload={payload}) at State: {state}")
        
        # --- Direct Voice Commands have been removed ---
        
        if intent == "HOME":
            if state != ApplicationState.MAIN_MENU:
                self.say("Navigating home.")
                self.state_manager.set_state(ApplicationState.MAIN_MENU)
                self.say("You have returned to the main menu. Show 1 finger for next option, 2 fingers for previous.")
            else:
                self.say("You are already at the main menu.")
            return

        if intent == "HELP":
            self._handle_help(state)
            return

        if intent == "YES":
            intent = "SELECT"
        elif intent == "NO":
            intent = "BACK"

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
                self.say("You are already at the main menu. Use thumbs up to select an option, or 1 or 2 fingers to browse.")
                
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
            self.say("You're now in Dataset Collection. Show a thumbs up to begin recording, or a thumbs down to cancel.")
            
        elif selection == "Model Training":
            self.state_manager.set_state(ApplicationState.TRAINING_CONFIRMATION)
            self.say("I'm opening Model Training. Show a thumbs up to confirm training or thumbs down to cancel.")
            
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

    def get_option_description(self, selection: str = None) -> dict:
        """Returns title, description, and trigger instructions for any menu selection."""
        if selection is None:
            selection = self.main_menu_items[self.main_menu_index]

        desc_map = {
            "Dataset Collection": {
                "title": "📁 Dataset Collection",
                "description": "Records 50 camera sample frames for each hand gesture to build your custom AI dataset.",
                "trigger": "Thumbs Up 🖒 to access"
            },
            "Model Training": {
                "title": "🧠 Model Training",
                "description": "Trains a Machine Learning model (Random Forest) on your collected keypoint samples.",
                "trigger": "Thumbs Up 🖒 to access"
            },
            "Recognition": {
                "title": "✨ Live ML Recognition",
                "description": "Activates live ML hand gesture control.",
                "trigger": "Thumbs Up 🖒 to access"
            },
            "Settings": {
                "title": "⚙️ Settings",
                "description": "Settings are currently disabled in this prototype.",
                "trigger": "Thumbs Up 🖒 to access"
            },
            "Help": {
                "title": "❓ Instructions & Help",
                "description": "Reads out complete hand gesture control instructions.",
                "trigger": "Click to access"
            },
            "Exit": {
                "title": "🚪 Exit Application",
                "description": "Closes camera stream and exits the application cleanly.",
                "trigger": "Thumbs Up 🖒 to exit"
            }
        }
        return desc_map.get(selection, {
            "title": f"📌 {selection}",
            "description": f"Perform actions related to {selection}.",
            "trigger": "Thumbs Up 🖒 to select"
        })

    def _handle_help(self, state: ApplicationState):
        if state == ApplicationState.MAIN_MENU:
            self.say("Help. 1 finger moves to the next option. 2 fingers move to the previous option. Thumbs up selects. Thumbs down goes back.")
        elif state == ApplicationState.DATASET_COLLECTING:
            gest = GESTURES[self.dataset_gesture_index].replace('_', ' ').title()
            self.say(f"Help. The current gesture is {gest}. Show a {gest} to collect samples. Thumbs down cancels.")
        elif state == ApplicationState.TRAINING_CONFIRMATION:
            self.say("Help. Show thumbs up to confirm training. Thumbs down cancels.")
        elif state == ApplicationState.RECOGNITION:
            self.say("Help. You are in ML recognition mode. Thumbs down goes back.")
        else:
            self.say("Help. 1 finger for Next. 2 fingers for Previous. Thumbs up to Select. Thumbs down to go back.")

