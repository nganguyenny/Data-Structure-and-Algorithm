# https://www.codewars.com/users/nganguyen.ny/completed

___
### [Tic-Tac-Toe-like table Generator](https://www.codewars.com/kata/5b817c2a0ce070ace8002be0)
**Description:**
Assuming that you get all the data in one array, you put a space around each value, | as a columns separator and multiple - as rows separator, with something like ["O", "X", " ", " ", "X", " ", "X", "O", " "] you should be returning this structure (inclusive of new lines):

```
 O | X |   
-----------
   | X |   
-----------
 X | O |   
 ```
**My solution:**
```python
def display_board(board, width):
    str = [i.center(3) for i in board]
    lines = f"\n{'-'*(4*width-1)}\n"
    res = []
    for i in range(0, len(str), width):
        res.append("|".join(str[i:i+width]))
    return lines.join(res)
```

___
### [Loose Change](https://www.codewars.com/kata/5571f712ddf00b54420000ee)
**Description:**
Welcome young Jedi! In this Kata you must create a function that takes an amount of US currency in cents, and returns a dictionary/hash which shows the least amount of coins used to make up that amount. The only coin denominations considered in this exercise are: Pennies (1¢), Nickels (5¢), Dimes (10¢) and Quarters (25¢). Therefor the dictionary returned should contain exactly 4 key/value pairs.

**My solution:**
```python
import math

def loose_change(cents):
    cents = math.floor(cents)
    if cents < 0: 
        nickels, pennies, dimes, quarters = 0, 0, 0, 0
    else:
        quarters = cents // 25
        dimes = (cents - quarters*25) // 10
        nickels = (cents - quarters*25 - dimes*10) // 5
        pennies = cents - quarters*25 - dimes*10 - nickels*5
    return {'Nickels': nickels, 'Pennies': pennies, 'Dimes': dimes, 'Quarters': quarters}
```

___
### [Create Phone Number](https://www.codewars.com/kata/525f50e3b73515a6db000b83)
**Description:**
Write a function that accepts an array of 10 integers (between 0 and 9), that returns a string of those numbers in the form of a phone number.


**My solution:**
```python
def create_phone_number(n):
    s = ''.join(str(i) for i in n)
    return f"({s[0:3]}) {s[3:6]}-{s[6:10]}"
```
    
```python
def create_phone_number(n):
    return "({}{}{}) {}{}{}-{}{}{}{}".format(*n)
```

___
### [Sudoku Solution Validator](https://www.codewars.com/kata/63d1bac72de941033dbf87ae)
**Description:**
Write a function that accepts a Sudoku board, and returns true if it is a valid Sudoku solution, or false otherwise. The cells of the input Sudoku board may also contain 0's, which will represent empty cells. Boards containing one or more zeroes are considered to be invalid solutions.

**My solution:**
```python
def valid_solution(grid):
    colums = list(map(list, zip(*grid)))
    cubic_list = []
    num = list(range(1,10))
    for col in range(0, len(grid), 3):
        for row in range(0, len(grid), 3):
            cubic = [grid[i][col:col+3] for i in range(row, row+3)]
            cubic_list.append(cubic[0] + cubic[1] + cubic[2])
    for i in range(9):
        if sorted(grid[i]) != num or sorted(colums[i]) != num or sorted(cubic_list[i]) != num:
            return False
    return True
```
___
### [Stop gninnipS My sdroW!](https://www.codewars.com/kata/5264d2b162488dc400000001)
**Description:**
Write a function that takes in a string of one or more words, and returns the same string, but with all five or more letter words reversed (Just like the name of this Kata). Strings passed in will consist of only letters and spaces. Spaces will be included only when more than one word is present.

Examples:
```
spinWords( "Hey fellow warriors" ) => returns "Hey wollef sroirraw" 
spinWords( "This is a test") => returns "This is a test" 
spinWords( "This is another test" )=> returns "This is rehtona test"
```
**My solution:**
```python
def spin_words(sentence):
    return " ".join([w if len(w) <5 else w[::-1] for w in sentence.split()])
```

___
### [Sum of Digits / Digital Root](https://www.codewars.com/kata/541c8630095125aba6000c00)
**Description:**
Digital root is the recursive sum of all the digits in a number.

Given n, take the sum of the digits of n. If that value has more than one digit, continue reducing in this way until a single-digit number is produced. The input will be a non-negative integer.

Examples
```
    16  -->  1 + 6 = 7
   942  -->  9 + 4 + 2 = 15  -->  1 + 5 = 6
132189  -->  1 + 3 + 2 + 1 + 8 + 9 = 24  -->  2 + 4 = 6
493193  -->  4 + 9 + 3 + 1 + 9 + 3 = 29  -->  2 + 9 = 11  -->  1 + 1 = 2
```
**My solution:**
```python
def digital_root(num):
    if len(str(num)) == 1: return num # If 1 digit, return the digit
    return digital_root(sum([int(n) for n in str(num)])) # Recursion in Python
```

___
### [Potion Class 101](https://www.codewars.com/kata/5981ff1daf72e8747d000091)
**Description:**
This is your first potion class in Hogwarts and professor gave you a homework to figure out what color potion will turn into if he'll mix it with some other potion. All potions have some color that written down as RGB color from [0, 0, 0] to [255, 255, 255]. To make task more complicated teacher will do few mixing and after will ask you for final color. Besides color you also need to figure out what volume will have potion after final mix.

Task
Based on your programming background you managed to figure that after mixing two potions colors will mix as if mix two RGB colors. For example, if you'll mix potion that have color [255, 255, 0] and volume 10 with one that have color [0, 254, 0] and volume 5, you'll get new potion with color [170, 255, 0] and volume 15. So you decided to create a class Potion that will have two properties: color (a list (a tuple in Python) with 3 integers) and volume (a number), and one method mix that will accept another Potion and return a mixed Potion.

Example
```
felix_felicis       =  Potion([255, 255, 255],  7)
shrinking_solution  =  Potion([ 51, 102,  51], 12)
new_potion          =  felix_felicis.mix(shrinking_solution)

new_potion.color   ==  [127, 159, 127]
new_potion.volume  ==  19
```
Note: Use ceiling when calculating the resulting potion's color.
**My solution:**
```python
import math

class Potion:
    def __init__(self, color, volume):
        self.color = color
        self.volume = volume
    
    def mix(self, other):
        new_volume = self.volume + other.volume
        new_color = [math.ceil((self.color[i]*self.volume + other.color[i]*other.volume) / new_volume) for i in range(3)]
        return Potion(tuple(new_color), new_volume)
```

___
### [User class for Banking System]()
**My solution:**

```python
class User(object):
    def __init__(self, name, balance, checking_account):
        self.name = name
        self.balance = balance
        self.checking_account = checking_account
    
    def withdraw(self, money):
        if self.balance >= money:
            self.balance -= money
            return f"{self.name} has {self.balance}."
        else:
            raise ValueError
    
    def check(self, user, money):
        if user.balance < money or user.checking_account is False:
            raise ValueError
        else:
            self.balance += money
            user.balance -= money
            return f"{self.name} has {self.balance} and {user.name} has {user.balance}."
        
    
    def add_cash(self, money):
        self.balance += money
        return f"{self.name} has {self.balance}."
```

___
### [Sleigh Authentication]()
**Description:**
Your task is to implement the authenticate() method of the sleigh, which takes the name of the person, who wants to board the sleigh and a secret password. If, and only if, the name equals "Santa Claus" and the password is "Ho Ho Ho!" (yes, even Santa has a secret password with uppercase and lowercase letters and special characters :D), the return value must be true. Otherwise it should return false.
**My solution:**
```python
class Sleigh(object):
    def authenticate(self, name, password):
        return name == "Santa Claus" and password == "Ho Ho Ho!"
```

___
### [OOP: Object Oriented Piracy](https://www.codewars.com/kata/54fe05c4762e2e3047000add)
**Description:**
Ahoy matey!

You are a leader of a small pirate crew. And you have a plan. With the help of OOP you wish to make a pretty efficient system to identify ships with heavy booty on board!

Unfortunately for you, people weigh a lot these days, so how do you know if a ship is full of gold and not people?
**My solution:**
```python
class Ship:
    def __init__(self, draft, crew):
        self.draft = draft
        self.crew = crew
    
    def is_worth_it(self):
        return (self.draft - self.crew * 1.5) > 20
```

___
### [Arrh, grabscrab!](https://www.codewars.com/kata/52b305bec65ea40fe90007a7)
**Description:**
Write a function that will accept a jumble of letters as well as a dictionary, and output a list of words that the pirate might have meant.

For example:
```
grabscrab( "ortsp", ["sport", "parrot", "ports", "matey"] )
Should return ["sport", "ports"].
```
**My solution:**
```python
def grabscrab(word, possible_words):
    word_chr = [c for c in word]
    result = []
    for possible_word in possible_words:
        possible_word_chr = [c for c in possible_word]
        if sorted(word_chr) == sorted(possible_word_chr):
            result.append(possible_word)
    return result

# refactored code:
def grabscrab(said, possible_words):
    return [ word for word in possible_words if sorted(word) == sorted(said) ]
```

___
### [Find the odd int]()
**Description:**
Given an array of integers, find the one that appears an odd number of times.

There will always be only one integer that appears an odd number of times.

Examples
```
[7] should return 7, because it occurs 1 time (which is odd).
[0] should return 0, because it occurs 1 time (which is odd).
[1,1,2] should return 2, because it occurs 1 time (which is odd).
[0,1,0,1,0] should return 0, because it occurs 3 times (which is odd).
[1,2,2,3,3,3,4,3,3,3,2,2,1] should return 4, because it appears 1 time (which is odd).
```
**My solution:**
```python
def find_it(seq):
    dict = {num: seq.count(num) for num in set(seq) if seq.count(num) % 2 == 1}
    return list(dict)[0]
```

___
### [Duplicate Encoder](https://www.codewars.com/kata/54b42f9314d9229fd6000d9c)
**Description:**
The goal of this exercise is to convert a string to a new string where each character in the new string is "(" if that character appears only once in the original string, or ")" if that character appears more than once in the original string. Ignore capitalization when determining if a character is a duplicate.

Examples
```
"din"      =>  "((("
"recede"   =>  "()()()"
"Success"  =>  ")())())"
"(( @"     =>  "))((" 
```
**My solution:**
```python
def duplicate_encode(word):
    str = ""
    for c in word.lower():
        if word.lower().count(c) == 1:
            str += "("
        else:
            str += ")"
    return str
```

___
### [Counting Duplicates](https://www.codewars.com/kata/54bf1c2cd5b56cc47f0007a1)
**Description:**
Write a function that will return the count of distinct case-insensitive alphabetic characters and numeric digits that occur more than once in the input string. The input string can be assumed to contain only alphabets (both uppercase and lowercase) and numeric digits.

Example
```
"abcde" -> 0 # no characters repeats more than once
"aabbcde" -> 2 # 'a' and 'b'
"aabBcde" -> 2 # 'a' occurs twice and 'b' twice (`b` and `B`)
"indivisibility" -> 1 # 'i' occurs six times
"Indivisibilities" -> 2 # 'i' occurs seven times and 's' occurs twice
"aA11" -> 2 # 'a' and '1'
"ABBA" -> 2 # 'A' and 'B' each occur twice
```
**My solution:**
```python
def duplicate_count(text):
    dict = {chr: text.lower().count(chr) for chr in text.lower()}
    return len([value for value in dict.values() if value > 1])
```

___
### [Multiplication table](https://www.codewars.com/kata/534d2f5b5371ecf8d2000a08)
**Description:**
Your task, is to create N×N multiplication table, of size provided in parameter.

For example, when given size is 3:
```
1 2 3
2 4 6
3 6 9
```
For the given example, the return value should be:
```
[[1,2,3],[2,4,6],[3,6,9]]
```
**My solution:**
```python
def multiplication_table(size):
    result = []
    for x in range(size):
        new_row = [(x+1)*(y+1) for y in range(size)]
        result.append(new_row)
    return result
```

___
### [Password validator](https://www.codewars.com/kata/56a921fa8c5167d8e7000053)
**Description:**
Your job is to create a simple password validation function, as seen on many websites.

The rules for a valid password are as follows:

There needs to be at least 1 uppercase letter.
There needs to be at least 1 lowercase letter.
There needs to be at least 1 number.
The password needs to be at least 8 characters long.
You are permitted to use any methods to validate the password.

**My solution:**
```python
def password(string):
    upper = sum([s.isupper() for s in string])
    lower = sum([s.islower() for s in string])
    number = sum([s.isnumeric() for s in string])
    return upper > 0 and lower > 0 and number > 0 and len(string) >= 8
```

___
### [Mexican Wave](https://www.codewars.com/kata/58f5c63f1e26ecda7e000029)
**Description:**
In this simple Kata your task is to create a function that turns a string into a Mexican Wave. You will be passed a string and you must return that string in an array where an uppercase letter is a person standing up. 
 1.  The input string will always be lower case but maybe empty.
 2.  If the character in the string is whitespace then pass over it as if it was an empty seat

Example
```
wave("hello") => ["Hello", "hEllo", "heLlo", "helLo", "hellO"]
```

**My solution:**
```python
def wave(people):
    result = []
    for i in range(len(people)):
        if people[i] != " ":
            new_wave = people[:i] + people[i].upper() + people[i+1:]
            result.append(new_wave)
    return result
```

___
### [Manhattan Distance](https://www.codewars.com/kata/52998bf8caa22d98b800003a)
**Description:**
Complete the function that accepts two points and returns the Manhattan Distance between the two points.

Example
```
* Input [1, 1], [1, 1] => Output 0
* Input [5, 4], [3, 2] => Output 4
* Input [1, 1], [0, 3] => Output 3
```

**My solution:**
```python
def manhattan_distance(pointA, pointB):
    return abs(pointA[0] - pointB[0]) + abs(pointA[1] - pointB[1])
```

___
### [Return a string's even characters](https://www.codewars.com/kata/566044325f8fddc1c000002c)
**Description:**
Write a function that returns a sequence (index begins with 1) of all the even characters from a string. If the string is smaller than two characters or longer than 100 characters, the function should return "invalid string".

Example
```
"abcdefghijklm" --> ["b", "d", "f", "h", "j", "l"]
"a"             --> "invalid string"
```

**My solution:**
```python
def even_chars(st): 
    if len(st) >= 2 and len(st) <= 100:
        return [chr for i, chr in enumerate(st) if i % 2 == 1]
    else:
        return "invalid string"
```

___
### [Password maker](https://www.codewars.com/kata/5637b03c6be7e01d99000046)
**Description:**
One suggestion to build a satisfactory password is to start with a memorable phrase or sentence and make a password by extracting the first letter of each word.

Even better is to replace some of those letters with numbers (e.g., the letter O can be replaced with the number 0):
- instead of including i or I put the number 1 in the password;
- instead of including o or O put the number 0 in the password;
- instead of including s or S put the number 5 in the password.

**My solution:**
```python
def make_password(phrase):
    password = [word[0] for word in phrase.split(" ")]
    for i in range(len(password)):
        if password[i].lower() == "i":
            password[i] = "1"
        elif password[i].lower() == "o":
            password[i] = "0"
        elif password[i].lower() == "s":
            password[i] = "5"
    return "".join(password)
```

___
### [Sum of two lowest positive integers]()
**Description:**
Create a function that returns the sum of the two lowest positive numbers given an array of minimum 4 positive integers. No floats or non-positive integers will be passed.

For example, when an array is passed like [19, 5, 42, 2, 77], the output should be 7.

[10, 343445353, 3453445, 3453545353453] should return 3453455.

**My solution:**
```python
def sum_two_smallest_numbers(numbers):
    return sum(sorted(numbers)[0:2])
```

___
### [Running out of space](https://www.codewars.com/kata/56576f82ab83ee8268000059)
**Description:**
Kevin is noticing his space run out! Write a function that removes the spaces from the values and returns an array showing the space decreasing. For example, running this function on the array ['i', 'have','no','space'] would produce ['i','ihave','ihaveno','ihavenospace'].

**My solution:**
```python
def spacey(array):
    new_array = []
    base_str = ""
    for i in range(len(array)):
        base_str += array[i]
        new_array.append(base_str)
    return new_array
```

___
### [Testing 1-2-3](https://www.codewars.com/kata/54bf85e3d5b56c7a05000cf9)
**Description:**
Your team is writing a fancy new text editor and you've been tasked with implementing the line numbering.

Write a function which takes a list of strings and returns each line prepended by the correct number.

Example
```
[] --> []
["a", "b", "c"] --> ["1: a", "2: b", "3: c"]
```
**My solution:**
```python
def number(lines):
    return [f"{index + 1}: {line}" for index, line in enumerate(lines)]
```


___
### [The Supermarket Queue](https://www.codewars.com/kata/57b06f90e298a7b53d000a86)
**Description:**
There is a queue for the self-checkout tills at the supermarket. Your task is write a function to calculate the total time required for all the customers to check out!

**My solution:**
```python
def queue_time(customers, n):
    # Create slots for open tills, e.g. n=3 => [0, 0, 0]
    tills = [0]*n
    for i in customers:
        # Add customer to the empty/min till
        tills[0] += i
        # Sort the tills to show the next available till
        tills.sort()
    return max(tills)
```


___
### [Determine if the poker hand is flush](https://www.codewars.com/kata/5acbc3b3481ebb23a400007d)
**Description:**
Determine if the poker hand is flush, meaning if the five cards are of the same suit.

Your function will be passed a list/array of 5 strings, each representing a poker card in the format "5H" (5 of hearts), meaning the value of the card followed by the initial of its suit (Hearts, Spades, Diamonds or Clubs). No jokers included.

Your function should return true if the hand is a flush, false otherwise.

The possible card values are 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A

Examples
```
["AS", "3S", "9S", "KS", "4S"]  ==> true

["AD", "4S", "7H", "KS", "10S"] ==> false
```
**My solution:**
```python
def is_flush(cards):
    # Write your code in here!
    return cards[0][-1] == cards[1][-1] == cards[2][-1] == cards[3][-1] == cards[4][-1]
```


___
### [Letterbox Paint-Squad](https://www.codewars.com/kata/597d75744f4190857a00008d)
**Description:**
Given the start and end letterbox numbers, write a method to return the frequency of all 10 digits painted.

**My solution:**
```python
def paint_letterboxes(start, finish):
    letters = []
    counts = []
    for n in range(int(start), int(finish)+1):
        letters += [i for i in str(n)]
    for n in range(0, 10):
        counts.append(letters.count(str(n)))
    return counts
```


___
### [Split The Bill](https://www.codewars.com/kata/5641275f07335295f10000d0)

**My solution:**
```python
def split_the_bill(x):
    split = (sum(x.values()) / len(x))
    
    for i in x:
        x[i] = round(x[i] - split, 2)
            
    return x
```


___
### [DNA to RNA Conversion]()
**Description:**
Deoxyribonucleic acid, DNA is the primary information storage molecule in biological systems. It is composed of four nucleic acid bases Guanine ('G'), Cytosine ('C'), Adenine ('A'), and Thymine ('T').

Ribonucleic acid, RNA, is the primary messenger molecule in cells. RNA differs slightly from DNA its chemical structure and contains no Thymine. In RNA Thymine is replaced by another nucleic acid Uracil ('U').

Create a function which translates a given DNA string into RNA.

For example:
```
"GCAT"  =>  "GCAU"
The input string can be of arbitrary length - in particular, it may be empty. All input is guaranteed to be valid, i.e. each input string will only ever consist of 'G', 'C', 'A' and/or 'T'.
```
**My solution:**
```python
def dna_to_rna(dna):
    rna = ""
    for i in range(len(dna)):
        if dna[i] == "T":
            rna += "U"
        else:
            rna += dna[i]
    return rna
```
