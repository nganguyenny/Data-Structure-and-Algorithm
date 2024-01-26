# PS1 - Basic Python
### [Problem A - Contingency Planning](https://open.kattis.com/problems/contingencyplanning)
```python
n = int(input())
ans = 0

for i in range(1,n+1):
    ans += perm(n,i)
    
if ans > 10**9:
    print("JUST RUN!!")
else:
    print(ans)
```

### [Problem B - A Furious Cocktail](https://open.kattis.com/problems/cocktail)
```python
N, T = map(int, input().split())
potions = sorted([int(input()) for i in range(N)], reverse=True)

def is_enough_time(N,T,potions):
    for i in range(len(potions)):
        if potions[i] <= (N-i-1)*T:
            return "NO"
    return "YES"

print(is_enough_time(N,T,potions))
```

# PS2 - Sorting Problems
### [Problem A - Charting Progress](https://open.kattis.com/problems/chartingprogress)
```python
import sys

raw_logs = sys.stdin.read().strip().split('\n\n')

for log in raw_logs:
    lines = {i: (len(e), e.count('*')) for i, e in enumerate(log.split())}

    end_dots = 0
    i = 0

    for l in lines:
        n_asterisk = lines[l][1]
        lead_dots = lines[l][0] - n_asterisk - i
        end_dots = lines[l][0] - n_asterisk - lead_dots
        print('.'*lead_dots + '*'*n_asterisk + '.'*end_dots)
        i += n_asterisk

    print()
```
### [Problem B - Massive Card Game](https://open.kattis.com/problems/massivecardgame)
```python
N = int(input())
cards = sorted(map(int, input().split()))

Q = int(input())
ranges = [tuple(map(int, input().split())) for _ in range(Q)]

def find_leftmost_index(cards, num):
    l = 0
    r = len(cards) - 1
    while l <= r:
        m = (l + r) // 2
        if cards[m] < num:
            l = m + 1
        else:
            r = m - 1
    return l

def find_rightmost_index(cards, num):
    l = 0
    r = len(cards) - 1
    while l <= r:
        m = (l + r) // 2
        if cards[m] > num:
            r = m - 1
        else:
            l = m + 1
    return r

for left, right in ranges:
        left_index = find_leftmost_index(cards, left)
        right_index = find_rightmost_index(cards, right)
        print(right_index - left_index + 1)
```

# PS3 - List Problems
### [Problem A - Bracket Matching](https://open.kattis.com/problems/bracketmatching)
```python
n = int(input())
chars = input().strip()

def isMatch(chars):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for c in chars:
        if c in pairs:
            if stack and stack[-1] == pairs[c] :
                stack.pop()
            else:
                return False
        else:
            stack.append(c)
    return stack == []

print("Valid") if isMatch(chars) else print("Invalid")
```

### [Problem B - Conga Line](https://open.kattis.com/problems/congaline)
```python
class Node:
    def __init__(self, value):
        self.name = value
        self.next = None
        self.prev = None
        self.partner = None

class DLL:
    def __init__(self):
        self.head = None
        self.tail = None

    def enqueue(self, node):
        if self.tail == None:
            self.head = self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = self.tail = node
            
    def move_to_back(self, node):
        if node == self.head: self.head = node.next
        if node.next: node.next.prev = node.prev
        if node.prev: node.prev.next = node.next

        node.prev = self.tail
        node.next = None
        self.tail.next = self.tail = node

    def move_behind_partner(self, node):
        if node.partner.next == node: pass

        if node == self.head: self.head = node.next
        if node.next: node.next.prev = node.prev
        if node.prev: node.prev.next = node.next

        node.prev = node.partner
        node.next = node.partner.next
        if node.partner.next: node.partner.next.prev = node
        node.partner.next = node
        if node.partner == self.tail: self.tail = node

import sys

N = sys.stdin.readline().split()[0]
name_list = [sys.stdin.readline().split() for line in range(int(N))]
instructions = sys.stdin.readline()

line = DLL()
for name1, name2 in name_list:
    new1, new2 = Node(name1), Node(name2)
    new1.partner, new2.partner = new2, new1
    line.enqueue(new1)
    line.enqueue(new2)

curr = line.head
for ins in instructions:
    if ins == 'F': curr = curr.prev 
    elif ins == 'B': curr = curr.next
    elif ins == 'R':
        if not curr.next: curr = line.head
        else:
            to_move = curr
            curr = curr.next
            line.move_to_back(to_move)
    elif ins == 'C':
        if not curr.next: curr = line.head
        else:
            to_move = curr
            curr = curr.next
            line.move_behind_partner(to_move)
    elif ins == 'P': print(curr.partner.name)

print()
curr = line.head
while curr:
    print(curr.name)
    curr = curr.next
```

# PS4 - PQ Problems
### [Problem A - Array Smoothening](https://open.kattis.com/problems/arraysmoothening)
```python
from heapq import heapify, heappush, heappop
from collections import Counter

N, K = map(int, input().split())
A = [int(i) for i in input().split()]

max_count_r = [-a for a in Counter(A).values()]
heapify(max_count_r)

while K > 0:
    curr = heappop(max_count_r)
    curr += 1
    K -= 1
    if curr < 0: heappush(max_count_r, curr)
    
print(-max_count_r[0]) if max_count_r else print(0)
```

### [Problem B - Jane Eyre](https://open.kattis.com/problems/janeeyre)
```python
from heapq import heapify, heappop, heappush

n, m, k = map(int, input().split())
unread_books = [("Jane Eyre", k)]
future_books = []

for i in range(n):
    new = input().strip().split('"')
    unread_books.append((new[1], int(new[2])))

for j in range(m):
    new = input().strip().split('"')
    future_books.append((int(new[0]), new[1], int(new[2])))

def time_finish(unread_books, future_books):
    heapify(unread_books)
    heapify(future_books)

    minutes_passed = 0
    while unread_books:
        while future_books and future_books[0][0] <= minutes_passed:
            timestamp, title, pages = heappop(future_books)
            heappush(unread_books, (title, pages))
        
        if future_books and future_books[0][0] == minutes_passed and future_books[0][1] < unread_books[0][0]:
            title, pages = heappop(future_books)
        else:
            title, pages = heappop(unread_books)
        minutes_passed += pages
            
        if title == "Jane Eyre": break

    return minutes_passed

print(time_finish(unread_books, future_books))
```

# PS5 - (Hash) Table Problems
### [Problem A - Variable Names](https://open.kattis.com/problems/variabelnamn)
```python
N = int(input())
c_input = [int(input()) for i in range(N)]

used_names = {}

for c in c_input:
    if c not in used_names:
        used_names[c] = 1
        print(c)
    else:
        new_c = c * used_names[c]
        while new_c in used_names:
            used_names[c] += 1
            new_c = c * used_names[c]
        used_names[new_c] = 1
        print(new_c)
```

### [Problem B - Quickscope](https://open.kattis.com/problems/quickscope)
```python
import sys
from collections import defaultdict

N = int(sys.stdin.readline())
lines = [(sys.stdin.readline().split()) for _ in range(N)]

levels = [{}]
current_vars = defaultdict(list)

def check_type(var, i):
    if var in current_vars:
        all_scopes = current_vars[var]
        for level in reversed(all_scopes):
            if level <= i:
                return levels[level][var]
    return "UNDECLARED"

i = 0
for l in lines:
    if l[0] == "DECLARE":
        if l[1] in levels[-1]:
            print("MULTIPLE DECLARATION")
            break
        else:
            levels[-1][l[1]] = l[2]
            current_vars[l[1]].append(i)
    elif l[0] == "TYPEOF":
        print(check_type(l[1], i))
    elif l[0] == "{":
        levels.append({})
        i += 1
    elif l[0] == "}":
        i -= 1
        last_scope = levels.pop()
        for var in last_scope:
            current_vars[var].pop()
```

# PS6 - Graph DS+Traversal Problems
### [Problem A - Bus Lines](https://open.kattis.com/problems/buslines)
```python
n, m = map(int, input().split())

if (m < n - 1) or (m > 2*n - 3):
    print(-1)
else:
    lines = [(i, i + 1) for i in range(1, n)]
    num_extra_lines = m - (n - 1)
    
    for i in range(1, n+1):
        for j in range(i+2, n+1):
            if num_extra_lines == 0:
                break
            new_line_sum = i + j
            unique = True
            for line in lines:
                if new_line_sum == sum(line):
                    unique = False
                    break            
            if unique:
                lines.append((i, j))
                num_extra_lines -= 1
        if num_extra_lines == 0:
            break
        
    for line in lines:
        print(f"{line[0]} {line[1]}")
```

### [Problem B - Fountain](https://open.kattis.com/problems/fontan)
```python
from collections import deque

N, M = map(int, input().split())
grid = [list(input()) for _ in range(N)]
queue = deque([(r, c) for r in range(N) for c in range(M) if grid[r][c] == 'V'])

while queue:
    r, c = queue.popleft()

    if r + 1 < N:
        if grid[r + 1][c] == '.':
            grid[r + 1][c] = 'V'
            queue.append((r + 1, c))
        if grid[r + 1][c] == '#':
            if c > 0 and grid[r][c - 1] == '.':
                grid[r][c - 1] = 'V'
                queue.append((r, c - 1))
            if c + 1 < M and grid[r][c + 1] == '.':
                grid[r][c + 1] = 'V'
                queue.append((r, c + 1))

for row in grid:
    print(''.join(row))
```

# PS7 - Graph Traversal+SSSP Problems
### [Problem A - Conquest](https://open.kattis.com/problems/conquest)
```python
from collections import defaultdict
import heapq

N, M = map(int, input().split())
graph = defaultdict(list)
for _ in range(M):
    n1, n2 = map(int, input().split())
    graph[n1].append(n2)
    graph[n2].append(n1)
army_sizes = [int(input()) for _ in range(N)]

def conquer_islands(graph, army_sizes):
    visited = set([1])
    unconquered_heap = [(army_sizes[nei - 1], nei) for nei in graph[1]]
    heapq.heapify(unconquered_heap)
    curr_army_size = army_sizes[0]

    while unconquered_heap:
        army_size, island = heapq.heappop(unconquered_heap)
        if island not in visited and army_size < curr_army_size:
            visited.add(island)
            curr_army_size += army_size
            for nei in graph[island]:
                if nei not in visited:
                    heapq.heappush(unconquered_heap, (army_sizes[nei - 1], nei))

    return curr_army_size

max_army_size = conquer_islands(graph, army_sizes)
print(max_army_size)
```

### [Problem B - Escape Wall Maria](https://open.kattis.com/problems/escapewallmaria)
```python
import queue

t, N, M = map(int, input().split())
graph = [list(input()) for _ in range(N)]
ROWS = len(graph)
COLS = len(graph[0])

for r in range(ROWS):
    for c in range(COLS):
        if graph[r][c] == 'S':
            start_pos = (r, c)
            break

def isValidMove(graph, curr, next):
    if next[0] < 0 or next[0] >= ROWS or next[1] < 0 or next[1] >= COLS:
        return False
    next_val = graph[next[0]][next[1]]
    if next_val == '1': return False
    if (next_val == '0' or next_val == 'S') or \
        (next_val == 'U' and next[0] > curr[0]) or \
        (next_val == 'D' and next[0] < curr[0]) or \
        (next_val == 'L' and next[1] > curr[1]) or \
        (next_val == 'R' and next[1] < curr[1]):
        return True
    return False

def bfs(graph, start_pos, timeout):
    visited = set()
    q = queue.Queue()
    q.put((start_pos, 0))
    visited.add(start_pos)

    while not q.empty():
        curr_pos, distance = q.get()
        if distance > timeout: 
            return "NOT POSSIBLE"
        
        if curr_pos[0] == 0 or curr_pos[0] == ROWS - 1 or \
            curr_pos[1] == 0 or curr_pos[1] == COLS - 1:
            return distance
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
            if next_pos not in visited and isValidMove(graph, curr_pos, next_pos):
                visited.add(next_pos)
                q.put((next_pos, distance + 1))
            
    return "NOT POSSIBLE"

print(bfs(graph, start_pos, t))
```

# PS8 - Weighted SSSP Problems
### [Problem A - Bridges](https://open.kattis.com/problems/bryr)
```python
from heapq import heappush, heappop

INF = int(1e9)

n, m = map(int, input().split())
V = n + 1
AL = [[] for u in range(V)]
for _ in range(m):
    a, b, c = map(int, input().split())
    AL[a].append((b,c))
    AL[b].append((a,c))

def sssp(AL, start, end):
    distances = [INF for u in range(V)]
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        distance, vertex = heappop(pq)
        if vertex == end: break
        if distance > distances[vertex]: continue
        for nei, w in AL[vertex]:
            new_distance = distance + w
            if new_distance >= distances[nei]: continue
            distances[nei] = new_distance
            heappush(pq, (new_distance, nei))
        
    return distances[end]

print(sssp(AL, 1, n))
```

### [Problem B - Hopscotch 50](https://open.kattis.com/problems/hopscotch50)
```python
n, k = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(n)]

def hopscotch50(graph, n, k):
    pos = {i: [] for i in range(1, k+1)}
    for i in range(n):
        for j in range(n):
            val = graph[i][j]
            if 1 <= val <= k: 
                pos[val].append((i, j))
    
    for p in pos.values():
        if not p: return -1
            
    inf = int(1e9)
    memo = {i: [inf] * len(pos[i]) for i in range(2, k+1)}
    memo[1] = [0] * len(pos[1])
    
    for num in range(2, k+1):
        for curr_i, curr in enumerate(pos[num]):
            min_d = inf
            for prev_i, prev in enumerate(pos[num-1]):
                dist = abs(prev[0] - curr[0]) + abs(prev[1] - curr[1])
                min_d = min(min_d, memo[num-1][prev_i] + dist)
            memo[num][curr_i] = min_d

    return min(memo[k])

print(hopscotch50(graph, n, k))
```
