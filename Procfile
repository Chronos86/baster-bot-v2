# Procfile - Defines processes for Honcho

# Web process: Runs the Streamlit panel, listening on the port provided by Render
web: streamlit run panel/app.py --server.port $PORT --server.address 0.0.0.0

# Bot process: Runs the main trading bot script
bot: python bot/main.py

