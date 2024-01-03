from typing import Tuple, List
import utils
from helpers.test_tools import read_text_file,read_word_list

'''
    The DecipherResult is the type defintion for a tuple containing:
    - The deciphered text (string).
    - The shift of the cipher (non-negative integer).
        Assume that the shift is always to the right (in the direction from 'a' to 'b' to 'c' and so on).
        So if you return 1, that means that the text was ciphered by shifting it 1 to the right, and that you deciphered the text by shifting it 1 to the left.
    - The number of words in the deciphered text that are not in the dictionary (non-negative integer).
'''
DechiperResult = Tuple[str, int, int]

def caesar_dechiper(ciphered: str, dictionary: List[str]) -> DechiperResult:
    '''
        This function takes the ciphered text (string)  and the dictionary (a list of strings where each string is a word).
        It should return a DechiperResult (see above for more info) with the deciphered text, the cipher shift, and the number of deciphered words that are not in the dictionary. 
    '''
    
    N = 26 # Number of letters in the alphabet to shift with
    
    words = set(dictionary) # Extract only the unique words
    
    decipheredText = "" # The end result
    shift = 0 # The cipher shift
    wordsNotInDict = float('inf') # Number of deciphered words that are not in the dictionary

    for n in range(N):
        decrypted = ""
        for char in ciphered:
            val = (ord(char) - n - ord('a')) % 26
            decrypted += char if char == ' ' else chr(ord('a') + val)
                
        count = sum(1 for word in decrypted.split() if not word in words)
        
        if count < wordsNotInDict:
            wordsNotInDict, decipheredText, shift = count, decrypted, n

    return decipheredText, shift, wordsNotInDict