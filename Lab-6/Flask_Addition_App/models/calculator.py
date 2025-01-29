import pickle

# Define the pickle file path
pickle_file = 'models/calculator.pkl'

# Function to load previous computations from the pickle file
def load_computations():
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        # If file doesn't exist or is empty, return an empty list
        return []

# Function to save computations to the pickle file
def save_computations(computations):
    with open(pickle_file, 'wb') as f:
        pickle.dump(computations, f)

# Function to add two numbers and save the result
def add_numbers(a, b):
    # Perform addition
    result = a + b
    
    # Load previous computations
    computations = load_computations()

    # Add the new computation to the list
    computations.append({'operation': f"{a} + {b}", 'result': result})
    
    # Save updated computations back to the pickle file
    save_computations(computations)

    return result