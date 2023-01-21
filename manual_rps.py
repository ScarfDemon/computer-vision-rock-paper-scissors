# %%
import random

def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

print(get_computer_choice())

def get_user_choice():
    return input("Please input your RPS choice: ")

user_choice = get_user_choice().capitalize()
computer_choice = get_computer_choice()

def get_winner(user_choice, computer_choice):

    def valid(user_choice):
        return (user_choice in ['Rock', 'Paper', 'Scissors']) == True #valid=True if user_choice are correct
    
    while valid(user_choice) == False:
        print("Invalid input")
        user_choice = input("Please input your valid RPS choice: ").capitalize()
    
    else:
        P1, P2 = user_choice, computer_choice
        print(f"Game: {P1} vs {P2}")
        if P1 == P2:
            print("It is a tie!")
        elif (P1=="rock" and P2=="scissors") or (P1=="paper" and P2=="rock") or (P1=="scissors" and P2=="paper"):
            print("You win!")
        else:
            print("You lost!")
        