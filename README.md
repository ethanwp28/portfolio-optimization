# portfolio-optimization
Comparison of Portfolio Optimization Methods

Goal: For this project, I wanted to test some of the portfolio optimization methods that we have discussed in class against a portfolio, to see how each portfolio would perform.
To accomplish this, I wanted to first create a program that can pull in stock data that would represent a portfolio. The program would contain the different optimization methods, and since the models rely on historical data, I wanted to use a range of around 5 years. Then once the weights are calculated, I would then pass each portfolio with their respective weights through a date range after the original 5 year period, in order to see how each portfolio would perform based on the historical data it was given.
I did all these calculations in R, even though I originally intended to do this project in python. I found that I was having difficulties with pulling in data, as well as creating my graphs. In the future, I plan to port this project over to python.
The first step was to build a portfolio. I wanted to have a somewhat diversified portfolio, so I chose five different sectors, with two stocks representing each sector for a total of ten stocks. I excluded any index funds at the start for simplicity. For the date range, I chose January 1, 2018 – December 31, 2022.
Portfolio setup:
-	Technology: Microsoft, Apple
-	Healthcare: Pfizer, Novo Nordisk
-	Consumer: Amazon, Tesla
-	Industrials: Lockheed Martin, General Electric
-	Financials: Wells Fargo, BlackRock
As far as the thought process that went into creating the portfolio, I mostly stuck with established companies and companies that I am personally invested in; however, a few of them (Novo Nordisk, Pfizer, BlackRock) I chose due to the world events over the last 5 years. I wanted to see how they performed over this period.
Next I needed to define each of the optimization methods that I wanted to pass my portfolio through. For the purpose of this project, I chose the following five methods for optimization:
-	Monte Carlo simulation
-	CAPM Model simulation
-	Markowitz optimization
-	Black Litterman
-	VaR/C-VaR
After passing the data through each model, I received the optimal weights. Using these weights, I then created a portfolio based on each model. In order to test the performance, I pulled in a fresh set of data after the historical date range that the models used. After testing each model, I then compared several key factors, such as variance, beta, and alpha of each of the portfolios. To put a monetary value on the portfolio, I set an investment amount of $100,000 and applied that to the total returns for each portfolio.

Current issues are with the Black Litterman function. Future updates will involve porting the project to python successfully, adding more optimization methods, and more rigorous testing methods.
