#
# lempel-ziv-v1.py : Lempel-Ziv complexity index with Numba and Numpy
#

import time
import numpy as np
from numba import njit

@njit(nopython=True,nogil=True)
def complexityLempelZiv(s):
	complexity				= 1
	prefix_length			= 1
	length_component		= 1
	max_length_component	= 1
	pointer					= 0
	
	while prefix_length + length_component <= len(s):
		if s[pointer + length_component - 1] == s[prefix_length + length_component - 1]:
			length_component += 1
		else:
			max_length_component = max(length_component, max_length_component)
			pointer += 1
			
			if pointer == prefix_length:
				complexity += 1
				prefix_length += max_length_component
				pointer = 0
				max_length_component = 1
				
			length_component = 1
	
	if length_component != 1:
		complexity += 1
	
	return complexity
	
def main():
	
	print('\n\nLempel-ziv-v1.py : Lempel-Ziv v.1 - complexity index with Numba and Numpy\n')
	print('Menù:')
	print()
	print('Read from [F]ile')
	print()
	print('Read from [K]eyboard')
	print()
	x = input('>')
	
	if x.upper() == 'F':
		file_name = input('What file: ')
		try:
			with open(file_name) as f:
				s = f.readlines()[0]
		except FileNotFoundError:
			print(f'File {file_name} not found.')
			quit()
		
	elif x.upper() == 'K':
		s = input('give me a string: ')
	
	arr = np.frombuffer(s.encode('utf-8'), dtype=np.uint8)
	t0 = time.perf_counter()
	c = complexityLempelZiv(arr)
	t1 = time.perf_counter()
	elapsed = t1 - t0
	
	print(f'\n\nLempel-Ziv complexity index for \n\n{s[:256]}... \n\nis {c}')
	print(f"\nElapsed time: {elapsed:.6f} seconds\n")
    
	
main()
