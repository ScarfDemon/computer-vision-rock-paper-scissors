# %%
import random

options = ["rock", "paper", "scissors"]

def get_computer_choice():
    #global options
    #computer_choice = random.choice(options)
    return random.choice(["Rock", "Paper", "Scissors"])

print(get_computer_choice())

def get_user_choice():
    user_choice = input("Please input your RPS choice: ").lower()
    return user_choice
# %%
# def roshambo():

#     def valid(user_choice):
#         global options
#         return (user_choice in options) == True #valid=True if user_choice are correct
    
#     global user_choice
#     global computer_choice
    
#     while valid(user_choice) == False:
#         print("Invalid input")
#         user_choice = input("Please input your valid RPS choice: ").lower()
#     else:
#         P1, P2 = user_choice, computer_choice
#         print(f"Game: {P1} vs {P2}")
#         winner = None
#         if P1 == P2:
#             print("Draw!")
#         elif (P1=="rock" and P2=="scissors") or (P1=="paper" and P2=="rock") or (P1=="scissors" and P2=="paper"):
#             winner = "The User" # Player 1
#         else:
#             winner = "The Computer" # Player 2
#         print(f"The winner is: {winner}" if winner!=None else "It's a tie!")