class Calculator:
    """A simple calculator class for basic arithmetic operations."""
    
    def add_numbers(self, a, b):
        """Function to add two numbers."""
        return a + b

    def subtract_numbers(self, a, b):
        """Function to subtract two numbers."""
        return a - b

    def multiply_numbers(self, a, b):
        """Function to multiply two numbers."""
        return a * b

    def divide_numbers(self, a, b):
        """Function to divide two numbers."""
        if b == 0:
            return "Error: Division by zero"
        return a / b

# Create an object (instance) of the Calculator class
calc = Calculator()

# Example usage with the object
result_add = calc.add_numbers(5, 3)
print(f"The sum is: {result_add}")

result_sub = calc.subtract_numbers(5, 3)
print(f"The difference is: {result_sub}")

result_mul = calc.multiply_numbers(5, 3)
print(f"The product is: {result_mul}")

result_div = calc.divide_numbers(5, 3)
print(f"The quotient is: {result_div}")