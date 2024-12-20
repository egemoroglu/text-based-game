import openai
import sys
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("Error: OpenAI API key is missing. Make sure it's set in your .env file as OPENAI_API_KEY.")
    sys.exit(1)

# Inventory setup
inventory = []
kingdom_power = 50  # Starting power level; range: 0 (weak) to 100 (strong)
rounds_in_event = 0  # Track rounds per event
current_event = None  # Store the current event
used_events = []  # List of events that have already occurred

# Predefined events with potential items
all_events = [
    {"description": "A sudden drought has struck the kingdom, bringing little to no rain for an extended period of time. Crops are withering, threatening the livelihoods of the people.", "item": None},
    {"description": "The kingdom's castle is under attack by a neighboring army. Amidst the chaos, you find a *mystic shield* lying on the battlefield.", "item": "mystic shield"},
    {"description": "The royal treasury is running low on gold. However, a hidden chest of *gold coins* is discovered in a secret passage.", "item": "gold coins"},
    {"description": "A strange plague has started spreading across the kingdom, causing panic. An old healer offers you a *healing potion* for emergencies.", "item": "healing potion"},
    {"description": "The kingdom's main river has been contaminated. You find a rare *purification stone* washed up on the riverbank.", "item": "purification stone"},
    {"description": "A rebellion is brewing among the nobles. In a secret meeting, a spy leaves behind a *sealed letter*.", "item": "sealed letter"}
]

# Function to interact with GPT and generate responses
def generate_dynamic_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful game assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.8
    )
    return response['choices'][0]['message']['content'].strip()

# Inventory functions
def add_to_inventory(item):
    inventory.append(item)
    print(f"You have obtained: {item}")

def remove_from_inventory(item):
    if item in inventory:
        inventory.remove(item)
        print(f"{item} has been removed from your inventory.")
    else:
        print(f"{item} is not in your inventory.")

def show_inventory():
    if inventory:
        print("Current Inventory:", ", ".join(inventory))
    else:
        print("Your inventory is empty.")

# Adjust kingdom power
def adjust_kingdom_power(change):
    global kingdom_power
    kingdom_power += change
    kingdom_power = max(0, min(100, kingdom_power))  # Ensure power stays between 0 and 100
    print(f"Kingdom power is now {kingdom_power}.")

# Generate a new event
def new_event():
    global used_events
    available_events = [event for event in all_events if event not in used_events]

    if not available_events:
        print("\nAll events have occurred. No more events to display.")
        return None  # No events left

    event = random.choice(available_events)  # Pick a random event
    used_events.append(event)  # Mark this event as used

    # Check if the event has an item
    if event.get("item"):
        # Randomize item appearance (e.g., 50% chance of item appearing)
        if random.random() > 0.5:  # 50% probability
            event_with_item = event.copy()
        else:
            event_with_item = event.copy()
            event_with_item["item"] = None  # No item for this instance
            # Update the description dynamically if no item appears
            event_with_item["description"] = event_with_item["description"].split(". Amidst the chaos")[0]
    else:
        # If no item exists in the original event, return as is
        event_with_item = event.copy()

    return event_with_item


# Process player actions
def process_player_action(event_description, action):
    prompt = f"The current event is: '{event_description}' The player decides to '{action}'. Describe the outcome of this action in detail."
    response = generate_dynamic_response(prompt)

    # Adjust kingdom power based on the nature of the action
    if "protect" in action or "help" in action or "reinforce" in action:
        adjust_kingdom_power(+10)  # Positive impact actions
    elif "ignore" in action or "exploit" in action or "tax" in action:
        adjust_kingdom_power(-10)  # Negative impact actions
    
    return response

# Main game logic
def start_game():
    global kingdom_power, rounds_in_event, current_event, used_events
    inventory.clear()  # Reset inventory for a new game
    kingdom_power = 50  # Reset power level for a new game
    rounds_in_event = 0  # Reset rounds
    used_events = []  # Clear used events for a new game

    print("Welcome to the Dynamic Text-based Adventure Game!")
    print(generate_dynamic_response(
        "Generate an opening where the player is a young king and his head advisor is welcoming him."
    ))

    current_event = new_event()  # Pick the first event

    game_over = False

    while not game_over and current_event is not None:
        # Show the current event only at the start or after switching events
        if rounds_in_event == 0:
            print(f"\nEvent: {current_event['description']}")
        
        # Check for item in the event
        if current_event.get("item"):
            print(f"You notice something valuable: {current_event['item']}")
            take_item = input(f"Do you want to take the {current_event['item']}? (yes/no) ").lower()
            if take_item == "yes":
                add_to_inventory(current_event["item"])
                current_event["item"] = None  # Remove the item from the event after it's picked up

        # Increment the rounds counter
        rounds_in_event += 1

        # Show inventory and ask for player action
        show_inventory()
        print(f"Kingdom Power Level: {kingdom_power}")
        action = input("What will you do, my king? ").lower()

        # Handle action logic
        if action.startswith("take"):
            item = action.split("take ", 1)[-1]
            add_to_inventory(item)

        elif action.startswith("use"):
            item = action.split("use ", 1)[-1]
            if item in inventory:
                outcome = process_player_action(current_event['description'], f"use the {item}")
                print("Outcome:", outcome)
                remove_from_inventory(item)
            else:
                print(f"You don't have a {item} to use.")

        elif action == "end game":
            print("You have chosen to end the game.")
            game_over = True

        else:
            # Process a general action
            outcome = process_player_action(current_event['description'], action)
            print("Outcome:", outcome)

        # Check game-ending conditions
        if kingdom_power <= 0:
            print("The kingdom has fallen due to weakened power. Game Over.")
            game_over = True
        elif kingdom_power >= 100:
            print("Congratulations! The kingdom is flourishing with immense power. You have won the game!")
            game_over = True

        # Switch to a new event every 3 rounds
        if rounds_in_event >= 3 and not game_over:
            print("A new event arises in the kingdom!")
            current_event = new_event()  # Fetch a new event
            rounds_in_event = 0  # Reset round counter for the new event

# Replay functionality
def play_again():
    choice = input("Do you want to play again? (yes/no) ").lower()
    if choice == "yes":
        start_game()
    elif choice == "no":
        print("Thanks for playing!")
        sys.exit()
    else:
        print("Invalid choice. Please enter 'yes' or 'no'.")

# Start the game
start_game()
play_again()
