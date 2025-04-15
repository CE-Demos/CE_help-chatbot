# CE_help-chatbot
This a small chat bot application which uses Gemini API to answer queries posted by the users.


## To Host it Locally

To run the application on local server, follow the steps:

1. Clone the Git repository to your local filesystem. 
2. Create a Python virtual environment and actiavte it.
3. Install the necessary Python libraries bu using the command `pip install streamlit google-generativeai` .
4. Fetch Gemini API keys from the Cloud console or Google AI Studio
5. Export the API keys as an environment variable for more security. This can be done by editing the .bashrc file in Linux environments by and this line, `export GEMINI_API_KEY="YOUR_API_KEYS"` (replace  `YOUR_API_KEYS` with the value for your API keys), to the .bashrc file.  
6. change into the directory which holds the app.py file and executing the  `streamlit run app.py`  command.

Running the streamlit command automatically opens up a browser tab for you where you can interact with the chatbot interface.
