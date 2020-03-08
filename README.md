# Predicting future stock prices using machine learning and historic stock data
A research project to see if it is possible to predict stock prices using machine learning and historic stock data. The method used is currently 500-day historic data and a Gated Recurrent Unit (GRU) neural network.

# Usage
When running the live.py script, it reads 500 equities from the Stockholm Stock Exchange and predicts the change of these stocks. A CSV sheet is outputed to the recommendations folder (has to be created next to live.py). With stock symbols on the left, the following columns shows the percentile change from the latest closing price at different times in the future. 

The first prediction at column 0 shows the predicted closing price for the next market day.
Column 1 shows the predicted closing price 3 market days later. 
Column 2 shows 5 days later.
Column 3: 10 market days ahead. 
Column 4: 20 market days ahead
Column 5: 65 market days ahead. 

live.py should be executed every active market day after closing in order to continually output updated recommendations to the recommendations directory. 

!Disclaimer! Do not blindly follow the recommendations and expect incredible returns. Instead use common sense when trading stock. 