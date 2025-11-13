
# Generate string for Kolmogorov complexity calculation purposes

import string, random, sys

def generate_constant_string(message,size):
    return message*size
    
def generate_random_string(size):
    message = ''
    for i in range(size):
        message += random.choice(string.ascii_letters)
        
    return message
    
def main(argv):
    
    # use:   python generate_string.py rnd 10_000_000 > example.txt
    try:
        size = int(argv[1])
        
        if 'const' in argv[0]:
            print(generate_constant_string('ab',size))      # low complexity
        elif 'rnd' in argv[0]:
            print(generate_random_string(size))             # high complexity
        else:
            print("error! bad parameter.\npermitted parameters are 'const' / 'rnd'")
    except error:
        print("bad parameters.")
    
if __name__ == "__main__":
    main(sys.argv[1:])
