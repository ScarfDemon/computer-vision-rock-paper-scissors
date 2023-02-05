# %%
import random

def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

def get_user_choice():
    return input("Please input your RPS choice: ")

def validate(user_choice):
    while user_choice not in ['Rock', 'Paper', 'Scissors']: #valid=True if user_choice are correct
        print("Invalid input")
        user_choice = input("Please input your valid RPS choice: ").capitalize()
    return user_choice

def get_winner(computer_choice, user_choice):

    user_choice = validate(user_choice)

    P1, P2 = user_choice, computer_choice
    print(f"Game: {P1} vs {P2}")
    if P1 == P2:
        print("It is a tie!")
    elif (P1=="Rock" and P2=="Scissors") or (P1=="Paper" and P2=="Rock") or (P1=="Scissors" and P2=="Paper"):
        print("You won!")
    else:
        print("You lost!")

def play():
    user_choice = get_user_choice().capitalize()
    computer_choice = get_computer_choice()
    get_winner(computer_choice, user_choice)

if __name__ == '__main__':
    play()



