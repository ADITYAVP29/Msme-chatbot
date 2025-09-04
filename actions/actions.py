import requests
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Define the URL of our doc-service API
DOC_SERVICE_URL = "http://localhost:8000/ask"

class ActionAnswerFromDocs(Action):
    def name(self) -> Text:
        return "action_answer_from_docs"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        # Get the user's latest message
        user_question = tracker.latest_message.get('text')

        # JSON payload to send to the doc-service
        payload = {"question": user_question}
        
        try:
            # Make the POST request to the doc-service
            response = requests.post(DOC_SERVICE_URL, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Get the answer from the response
            result = response.json()
            answer = result.get("answer", "I couldn't find an answer for that.")
            
            # Send the answer back to the user
            dispatcher.utter_message(text=answer)

        except requests.exceptions.RequestException as e:
            # Handle connection errors or other request issues
            print(f"Error connecting to doc-service: {e}")
            fallback_message = "I'm having trouble connecting to my knowledge base right now. Please try again in a moment."
            dispatcher.utter_message(text=fallback_message)

        return []