# poverty-prediction

How can we leverage demographic and environmental data (such as the observable characteristics of a house) to determine poverty levels? The motivation for this project lies in aiding efforts in Latin America's model-based approach to identifying citizens/households in need of economic assistance.

This project uses the following data set from Kaggle: 
https://www.kaggle.com/competitions/costa-rican-household-poverty-prediction/data

Although based in the Kaggle problem statement, this project departs from the main goal of this competition in order to answer some of the following questions:

## What is poverty?
How are we defining poverty? Is there a threshold, or continuum? Are we interested in absolute or relative poverty? 

## Who is affected?
What are the demographic patterns of poverty? What are shared characterisitcs between poor households? How do poor families/households compare to wealthier neighbors?

## Where is poverty observed?
Is poverty localized, or randomly distributed throughout the region? Is there clear separation between poor and rich? Is there something that rich neighborhoods have that poor neighborhoods lack? 

## Why are we seeing poverty?
Can the who, the where, and the what help us understand why certain households are poor and others aren’t? What are the underlying, systemic causes of poverty that lead to potential commonalities/patterns/expressions of it? 

Lastly, can we turn this into actionable insight? Is there something we can do to alleviate the poverty? To aid those in poverty? What are these poor households/neighborhoods experiencing that rich ones aren’t (and vice versa)? What is desireable and what is not? 

## Goals
1. Build a poverty predictor:
	a. Regression: predict HOW poor, either the monthly income, or a tiered approach of poverty levels (low, mid, high)
	b. Classification: poor vs not poor
2. Visualization: this will offer insight into the "who?" and the "where?" among other things
3. Impact analysis: Who needs the money most? Network analysis to guage impact in community?
