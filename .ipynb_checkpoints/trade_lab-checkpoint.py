import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import streamlit as st
from trade import Trade
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN


load_dotenv()
print("loaded")

st.markdown("#Game of Trade")
ticker = st.text_input("Input a ticker:")
print(ticker)
#port = Portfolio(1, [ticker])

