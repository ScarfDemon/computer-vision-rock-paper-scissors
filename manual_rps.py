# %%
import random

def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

def get_user_choice():
    return input("Please input your RPS choice: ")

def validate(user_choice):
    while user_choice not in ["Rock", "Paper", "Scissors"]: #valid=True if user_choice are correct
        print("Invalid input")
        user_choice = input("Please input your valid RPS choice: ").capitalize()
    return user_choice

def get_winner(computer_choice, user_choice):
    user_choice = validate(user_choice)
    user_choice = user_choice.capitalize()
    computer_choice = computer_choice.capitalize()
    #print(f"Game: {user_choice} vs {computer_choice}")
    if user_choice == computer_choice:
        print("It is a tie!")
        winner = None
    elif (user_choice=="Rock" and computer_choice=="Scissors") or (user_choice=="Paper" and computer_choice=="Rock") or (user_choice=="Scissors" and computer_choice=="Paper"):
        print("You won!")
        winner = "User"
    else:
        print("You lost")
        winner = "Computer"
    return winner

def play():
    user_choice = get_user_choice()
    computer_choice = get_computer_choice()
    get_winner(computer_choice, user_choice)

if __name__ == '__main__':
    play()



