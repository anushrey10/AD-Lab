# generate_calculator_pkl.py
import pickle
from models.calculator import save_computations

# Initialize an empty list for computations
computations = []

# Save an empty list to the pickle file
save_computations(computations)

print("calculator.pkl has been generated successfully.")