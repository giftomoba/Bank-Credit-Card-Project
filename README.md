# Bank-Credit-Card-Project (Clustering Technique)

## Background

AllLife Bank wants to focus on its credit card customer base in the next financial year. They have been advised by their marketing research team, that the penetration in the market can be improved. Based on this input, the Marketing team proposes to run personalized campaigns to target new customers as well as upsell to existing customers. Another insight from the market research was that the customers perceive the support services of the back poorly. Based on this, the Operations team wants to upgrade the service delivery model, to ensure that customer queries are resolved faster. Head of Marketing and Head of Delivery both decide to reach out to the Data Science team for help.

## Objective

To identify different segments in the existing customer, based on their spending patterns as well as past interaction with the bank, using clustering algorithms, and provide recommendations to the bank on how to better market to and service these customers.


## Data Description

The data provided is of various customers of a bank and their financial attributes like credit limit, the total number of credit cards the customer has, and different channels through which customers have contacted the bank for any queries (including visiting the bank, online and through a call center).

## Data Dictionary

- Sl_No: Primary key of the records
- Customer Key: Customer identification number
- Average Credit Limit: Average credit limit of each customer for all credit cards
- Total credit cards: Total number of credit cards possessed by the customer
- Total visits bank: Total number of visits that customer made (yearly) personally to the bank
- Total visits online: Total number of visits or online logins made by the customer (yearly)
- Total calls made: Total number of calls made by the customer to the bank or its customer service department (yearly)

## Links to the jupyter notebook file:
https://github.com/giftomoba/Bank-Credit-Card-Project/blob/main/Credit%20Card%20Customer%20Data.ipynb

## Clustering Summary:
- Two groups formed (cluster 0 and cluster 1).
- Cluster 1 contacted the bank mostly online.
- Cluster 1 had the highest Average Credit Limit as well as number of credit cards.
- cluster 0 paid more visits to the bank and also made the most calls to the bank.

## RECOMMENDATIONS:
- Since the customers in cluster 0 contacts the bank the most (via calls and physical visits), they could be the customers having negative ratings for the bank's customer services due to possible delays in resolving their issues. Hence, the bank should focus on making these group happy, perhaps by ensuring their issues are treated with upmost priority.
- Also, by focusing on resolving their disputes faster and making them satisfied, these group (cluster 0) could, in turn, decide to upscale their average credit limit which will also be beneficial to the bank because as an increase in credit limit will also ensure customer retention and more profit for the bank.
- The bank should focus more on ways to encourage customers in cluster 0 to increase their credit limit, so that they can increase their spending.
