---
title: "python数据结构+算法"
date: 2026-03-13
permalink: /posts/Python.md
date: 2026-03-13
location: "Harbin, China"
tags:
  - cool posts
  - category1
  - category2
---

python数据结构+算法

### Python关键字

- break 语句	在语句块执行过程中终止循环，并且跳出整个循环
- continue 语句	在语句块执行过程中终止当前循环，跳出该次循环，执行下一次循环。

### Python常用函数


```python
int()   # 转为整形
x ='1111'
a = int(x, 2) # 二进制转为十进制
print("a:" + str(a))
a = int(x, 16) # 十六进制转为十进制
```


```python
float() # 转为浮点型
```


```python
str()   # 转为字符串
```


```python
hex()   # 转为十六进制
```


```python
ord()   # 将一个字符转换为它的整数值        print(ord('a')) # 97
```


```python
chr()   # 将一个整数值转换为它的字符形式   print(chr(65))  # 'A'
```


```python
oct()   # 转为八进制
```


```python
bin()   # 将数字映射为二进制字符串    print(bin(8)+bin(12^4)) # '0b10000b1000'
```

###  Python位运算符

- ^ 异或
- & 与
- | 或
- ~ 非
- << 左移   运算数的各二进位全部左移若干位，由 << 右边的数字指定了移动的位数，高位丢弃，低位补0
- \>> 右移   把">>"左边的运算数的各二进位全部右移若干位，>> 右边的数字指定了移动的位数


### Python字符串

#### 'a'\~'z'与'A'\~'Z'之间的转换


```python
# 方式一：内置函数
str1 = 'SDA'
str1 = str1.lower()
print(str1)

str2 = 'sda'
str2 = str2.upper()
print(str2)
```

    sda
    SDA



```python
# 方式二：   （a比A的ASCII大32，a为97，A为65）
def to_upper(char):
    if char >= 'a' and char <= 'z':
        return chr(ord(char)-32)
    return char

def to_lower(char):
    if char >= 'A' and char <= 'Z':
        return chr(ord(char)+32)
    return char

str1 = 'S'
str1 = to_lower(str1)
print(str1)

str2 = 'a'
str2 = to_upper(str2)
print(str2)
```

    s
    A


r/R	# 所有的字符串都是直接按照字面的意思来使用，没有转义特殊或不能打印的字符。 原始字符串除在字符串的第一个引号前加上字母"r"（可以大小写）以外，与普通字符串有着几乎完全相同的语法。


```python
print(r'\n') # 输出结果：\n
```

    \n


%	格式字符串


```python
print("My name is %s and weight is %d kg!" % ('Zara', 21))
```

    My name is Zara and weight is 21 kg!


#### 常用函数
str.count(str, beg=0, end=len(str))   # 返回 str 在 str 里面出现的次数，如果 beg 或者 end 指定则返回指定范围内 str 出现的次数
str.find(str, beg=0, end=len(str))    # 检测 str 是否包含在 str 中，如果 beg 和 end 指定范围，则检查是否包含在指定范围内，如果是返回开始的索引值，否则返回-1
str.join(seq)    # 以 str 作为分隔符，将 seq 中所有的元素(的字符串表示)合并为一个新的字符串
str.replace(str1, str2,  num=str.count(str1)) # 把 str 中的 str1 替换成 str2,如果 num 指定，则替换不超过 num 次.
str.split(str="", num=str.count(str)) # 以 str 为分隔符切片 str，如果 num 有指定值，则仅分隔 num+1 个子字符串
### Python元组

---

- 元组用 () 标识。内部元素用逗号隔开。
- 元组不能增加、删除、修改元素，相当于只读列表。

#### 1.创建元组


```python
tup = ('physics', 'chemistry', 1997, 2000)
```

#### 2.增加元素

元组不能二次赋值，相当于只读列表。

#### 3.删除元素

元组中的元素值不允许删除


```python
# 以下删除元组元素操作是非法的
tup = ('physics', 'chemistry', 1997, 2000)
try:
    del tup[0]
except Exception as e:
    print(e)
```

仅可以使用del语句来删除整个元组


```python
tup = ('physics', 'chemistry', 1997, 2000)
del tup
```

#### 4.修改元素

元组中的元素值是不允许修改的


```python
# 以下修改元组元素操作是非法的
tup = ('physics', 'chemistry', 1997, 2000)

try:
    tup[0] = 100
except Exception as e:
    print(e)
```

但可以对元组进行连接组合（易错点）


```python
# 如下实例:
tup1 = (1, 2, 3)
tup2 = (4, 5, 6)
tup3 = tup1 + tup2    # (1, 2, 3, 4, 5, 6)          易错点、
print(tup3)
```

#### 5.查找元素


```python
# 元组可以使用下标索引来访问元组中的值
tup = (1, 2, 3, 4, 5, 6, 7 )
for i in range(len(tup)):
    print(tup[i])

for item in tup:
    print(item)
```

#### 6. 列表转换为元组


```python
lst = [1, 2, 3, 4]
tup = tuple(lst)
print(tup)
```

### Python列表

----------

列表的数据项不需要具有相同的类型


```python
# 获取每个元素的下标和元素值--enumerate
nums = [1, 2, 3, 4, 5, 6, 7]
for index, num in enumerate(nums): 
    print(index, num)
```


```python
# 获取二维数组行数和列数
matrix = [[0, 0, 0, 0], 
          [0, 0, 0, 0], 
          [0, 0, 0, 0], 
          [0, 0, 0, 0], 
          [0, 0, 0, 0]]
row = len(matrix)
col = len(matrix[0])
print(row,col)
```

#### 1.创建列表


```python
#创建一维数组
lst = []
lst2 = [1, 2, 3, 4, 5, 6, 7]
```


```python
# 创建二维数组 
# 方式一：使用循环
rows = 5
cols = 4
array = []
for i in range(rows):
    row = []
    for j in range(cols):
        row.append(0)  # 或者使用其他值初始化
    array.append(row)
print(array)
```


```python
# 创建二维数组 
# 方式二：使用列表推导式
rows = 5
cols = 4
array = [[0 for _ in range(cols)] for _ in range(rows)]
print(array)
```

#### 2.增加元素


```python
# 增加元素--append()
lst = []

lst.append('Google')

print(lst)
```


```python
# 增加元素--insert()
lst = []

index = 0
item = 'Facebook'
lst.insert(index, item)

print(lst)
```

#### 3.删除元素


```python
# 通过索引删除元素（只删除一个元素）
lst = ['Facebook', 'Google']

index = 1
del lst[index]     # del语句可以删除整个列表或列表中的特定元素。如果要删除特定元素，需要使用其索引。

print(lst)
```


```python
# 通过索引删除元素（只删除一个元素）
lst = ['Facebook', 'Google']

index = 1
e = lst.pop(index)  # 通过索引删除元素，返回元素值。

print(lst)
```


```python
# 通过元素值删除元素，（只删除一个元素）
lst = ['Facebook', 'Google', 'Google', 'Google', 'Google']

e = 'Google'
lst.remove(e)

print(lst)
```


```python
# 通过元素值删除元素，（删除多个元素）
lst = ['Facebook', 'Google', 'Google', 'Google', 'Google']

e = 'Google'
lst = [i for i in lst if i != e]

print(lst)
```

#### 4.修改元素


```python
lst = ['Facebook', 'Google']

index = 1
lst[index] = 'value'

print(lst)
```

#### 5.查找元素


```python
d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
value = lst[index]
切片
lst[1:] # 从第二个元素开始向后截取列表
lst[:4] # 从第五个元素开始向前截取列表（不包含第五个元素）
lst[1:4]
lst[1:4:2] #Python 列表截取可以接收第三个参数，参数作用是截取的步长，在索引 1 到索引 4 的位置并设置为步长为 2（不包含索引 4 的元素）
```

#### 6.列表切片


```python
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(lst[1:]) # 从第二个元素开始向后截取列表（包含第二个元素）

print(lst[:4]) # 从第五个元素开始向前截取列表（不包含第五个元素）

print(lst[1:4]) # 截取第二个元素到第五个元素（包含第二个元素，不包含第五个元素）

#Python 列表截取可以接收第三个参数，参数作用是截取的步长，在索引 1 到索引 4 的位置并设置为步长为 2（不包含索引 4 的元素）
print(lst[1:4:2])
```

#### 列表常用函数


```python
len(list) # 列表元素个数
max(list) # 返回列表元素最大值
min(list) # 返回列表元素最小值

list.append(obj) # 在列表末尾添加新的对象obj
list.count(obj) # 统计某个元素obj在列表中出现的次数
list.extend(seq) # 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
list.index(obj) # 从列表中找出某个值obj第一个匹配项的索引位置
list.insert(index, obj) # 将对象obj插入列表
list.pop([index=-1]) # 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
list.remove(obj) # 移除列表中某个值的第一个匹配项
list.reverse() # 反向列表中元素
list.sort(cmp=None, key=None, reverse=False)    #对原列表进行排序
```

### Python字典

----------

- dict 作为 Python 的关键字和内置函数，变量名不建议命名为 dict.
- 键一般是唯一的，如果重复最后的一个键值对会替换前面的，值不需要唯一。
- 键必须不可变，所以可以用数字，字符串或元组充当，所以用列表就不行        注意：元组可以作为键，列表不行


#### 1.创建字典


```python
# 创建空字典
dt = {}
```


```python
# 创建字典
dt = {'a': 1, 'b': 2, 'b': '3'}

print(dt)
```

#### 2.增加元素


```python
# 方式 1：
dt = {'a': 1, 'b': 2}

dt['key'] = 'value' 

print(dt)
```


```python
# 方式 2：
dt = {'a': 1, 'b': 2}

dt.update({'key': 'value'})

print(dt)
```

#### 3.删除元素


```python
# 方式一： pop(key[, default])   移除并返回字典中指定键的值，如果键不存在则返回默认值None。
dt = {'a': 1, 'b': 2, 'c': 3}

key = 100
value = dt.pop(key, -999) # 移除并返回字典中指定键的值，如果键不存在则返回 -999。
print(value)

key = 'a'
value = dt.pop(key, None)
print(value)
```


```python
# 方式二： del
dt = {'a': 1, 'b': 2, 'c': 3}

key = 'a'
del dt[key]     # 删除键是'key'的条目

print(dt)
```


```python
# 清空字典
# 方式一： dict.clear()
dt = {'a': 1, 'b': 2, 'c': 3}

dt.clear()      # 清空字典所有条目

print(dt)
```


```python
# 清空字典
dt = {'a': 1, 'b': 2, 'c': 3}

del dt          # 删除字典

try:
    print(dt)
except Exception as e:
    print(f'发生错误：{e}')
```

#### 4.修改元素

与添加元素一模一样


```python
# 方式 1：
dt = {'a': 1, 'b': 2}

dt['a'] = 'value' 

print(dt)
```


```python
# 方式 2：
dt = {'a': 1, 'b': 2}

dt.update({'a': 'value'})

print(dt)
```

#### 5.查找元素


```python
value = dict['key'] # 如果用字典里没有的键访问数据，会报错
dict.get('key', default=None)     #根据key找对应value，找不到这返回default的值
dict.keys()      # 输出所有键   输出为[key1, key2,……]
dict.values()    # 输出所有值   输出为[value1, value2,……]
```

#### 字典常用函数


```python
len(dict) # 计算字典元素个数，即键的总数。

dict.copy() # 返回一个字典的浅复制

dict.has_key(key) # 如果键在字典dict里返回true，否则返回false。Python3 不支持。

dict.fromkeys(seq[, val]) # 创建一个新字典，以序列 seq 中元素做字典的键，val 为字典所有键对应的初始值

dict.items() # 以列表返回可遍历的(键, 值) 元组数组
dict.keys() # 以列表返回一个字典所有的键
dict.values() # 以列表返回字典中的所有值

dict.get(key, default=None) # 返回指定键的值，如果值不在字典中返回default值
dict.setdefault(key, default=None) # 和get()类似, 但如果键不存在于字典中，将会添加键并将值设为default

dict.update(dict2) # 把字典dict2的键/值对更新到dict里
dict.pop(key[,default]) # 删除字典给定键 key 所对应的值，返回值为被删除的值。key值必须给出。 否则，返回default值。
dict.popitem() # 返回并删除字典中的最后一对键和值。
dict.clear() # 删除字典内所有元素
```

### Python双端队列

-------------

append和pop操作的是尾部（右边），appendleft和popleft操作的是头部（左边），别搞混啦

#### 1.创建双端队列


```python
from collections import deque
# 创建一个空的deque
dq = deque()
```

#### 2.增加元素

单个元素的添加


```python
# 头部（左侧）添加元素
dq = deque(['a', 'b'])
dq.appendleft('c')
print("输出结果：",end='')
print(dq)
```


```python
# 尾部（右侧）添加元素
dq = deque()
dq.append('a')
dq.append('b')
print("输出结果：",end='')
print(dq)
```

多个元素的批量添加


```python
# 头部（左侧）批量添加（注意顺序会逆序插入）
dq = deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
dq.extendleft([-2, -1]) 
print("输出结果：",end='')
print(dq)
```


```python
# 尾部（右侧）批量添加
dq = deque([0, 1, 2, 3, 4, 5, 6])
dq.extend([7, 8, 9])
print("输出结果：",end='')
print(dq)
```

#### 3.删除元素


```python
# 头部（左侧）移除元素
dq = deque([0, 1, 2, 3, 4, 5, 6])
left_item = dq.popleft()
print(left_item)
print(dq)
```


```python
# 尾部（右侧）移除元素
dq = deque([0, 1, 2, 3, 4, 5, 6])
right_item = dq.pop()
print(right_item)
print(dq)
```


```python
# 删除指定元素（如果存在多个，仅删除第一个）
dq = deque([1, 2, 3, 3, 4])
dq.remove(3)  # 删除值为 3 的元素（如果存在多个，仅删除第一个）
print(dq)
```


```python
# 清空双端队列
dq = deque([1, 2, 3, 4, 5])
dq.clear()
print(dq)
```

#### 4.修改元素


```python
dq = deque([1, 2, 3, 3, 4])
dq[2] = 100  # 修改索引 2 位置的元素
print(dq)
```

#### 5.查找元素


```python
dq = deque([1, 2, 3, 3, 4])
head = dq[0] # 表示头部元素
rear = dq[-1] # 表示尾部元素
print(dq)
print(head)
print(rear)

# 插入元素后再查找
dq.appendleft(0)
dq.append(5)
head = dq[0] # 表示头部元素
rear = dq[-1] # 表示尾部元素
print('两端插入元素后')
print(dq)
print(head)
print(rear)
```

#### 6.旋转元素


```python
# 往右转两步
dq = deque([1, 2, 3, 4, 5])
dq.rotate(2)
print(dq)  # deque([4, 5, 1, 2, 3])
```


```python
# 往左转三步
dq = deque([1, 2, 3, 4, 5])
dq.rotate(-3)
print(dq)  # deque([2, 3, 4, 5, 1])
```


```python
# 翻转双端队列
dq = deque([0, 1, 2, 3, 4, 5, 6, 7, 8])
dq.reverse()
print(dq)
```

### Python集合

-----------

没有重复元素

#### 1.创建集合


```python
#1.用{}创建set集合
person ={"student","teacher","babe",123,321,123} #同样各种类型嵌套,可以赋值重复数据，但是存储会去重
print(person) #但是显示出来则是去重的   输出：  {321, 'teacher', 'student', 'babe', 123}
```


```python
#2.空set集合用set()函数表示
person1 = set() #表示空set，不能用person1={}
```


```python
#3.用set()函数创建set集合
person2 = set(("hello","jerry",133,11,133,"jerru")) #只能传入一个参数，可以是list,tuple等 类型
print(len(person2)) # 输出： 5
print(person2)  # 输出： {133, 'jerry', 11, 'jerru', 'hello'}
```

#### 2.增加元素


```python
add()不会拆分；update()会拆分
person ={"student","teacher","babe",123,321,123}
person.add("student") #如果元素已经存在，则不报错，也不会添加,不会将字符串拆分成多个元素，区别update
print(person)   # 输出：{321, 'babe', 'teacher', 'student', 123}
person.add((1,23,"hello")) #可以添加元组，但不能是list
print(person)   # 输出： {(1, 23, 'hello'), 321, 'babe', 'teacher', 'student', 123}
```


```python
person.update((1,3)) #可以使用update添加一些元组列表，字典等。但不能是字符串，否则会拆分
print(person)   # 输出：{321, 1, 3, 'teacher', (1, 23, 'hello'), 'babe', 'student', 123}
person.update("abc")
print(person)  #会将字符串拆分成a,b，c三个元素    输出： {321, 1, 3, 'b', 'c', 'teacher', (1, 23, 'hello'), 'a', 'babe', 'student', 123}
```

#### 3.删除元素


```python
person.remove("student")#按元素去删除
print(person)  # 如果不存在 ，会报错。
```


```python
person.discard("student")#功能和remove一样，好处是没有的话，不会报错
person.pop() #在list里默认删除最后一个，在set里随机删除一个。
```


```python
# 直接清空set
person.clear()
```

#### 4.修改元素

更新set中某个元素,因为是无序的，所以不能用角标。
一般更新都是使用remove,然后在add。


#### 5.查找元素


```python
# 查询是否存在，无法返回索引，使用in判断
if "teacher" in person:
    print("true")
else:
    print("不存在")
```

#### 6.其他操作

交集∩


```python
set1 & set2
```

并集∪


```python
set1 | set2
```

差集


```python
set1 - set2
```

对称差  获取两个集合中不重叠的元素。


```python
set1 ^ set2
```

### math模块


```python
import math
math.abs(x)  # 返回数字的绝对值，如abs(-10) 返回 10
math.ceil(x) # 返回数字的上入整数，如math.ceil(4.1) 返回 5
math.exp(x)	# 返回e的x次幂(ex),如math.exp(1) 返回2.718281828459045
math.floor(x)	# 返回数字的下舍整数，如math.floor(4.9)返回 4
math.log(x)	# 如math.log(math.e)返回1.0,math.log(100,10)返回2.0
math.log10(x)	# 返回以10为基数的x的对数，如math.log10(100)返回 2.0
math.max(x1, x2,...)	# 返回给定参数的最大值，参数可以为序列。
math.min(x1, x2,...)	# 返回给定参数的最小值，参数可以为序列。
math.modf(x)	# 返回x的整数部分与小数部分，两部分的数值符号与x相同，整数部分以浮点型表示。
math.pow(x, y)	# x**y 运算后的值。
math.round(x [,n])	# 返回浮点数x的四舍五入值，如给出n值，则代表舍入到小数点后的位数。
math.sqrt(x)	# 返回数字x的平方根

math.choice(seq)	# 从序列的元素中随机挑选一个元素，比如random.choice(range(10))，从0到9中随机挑选一个整数。
math.randrange ([start,] stop [,step])	# 从指定范围内，按指定基数递增的集合中获取一个随机数，基数默认值为 1
math.random()	# 随机生成下一个实数，它在[0,1)范围内。
math.seed([x])	# 改变随机数生成器的种子seed。如果你不了解其原理，你不必特别去设定seed，Python会帮你选择seed。
math.shuffle(list)	# 将序列的所有元素随机排序
math.uniform(x, y)	# 随机生成下一个实数，它在[x,y]范围内。
```

### 代数
- 代数的核心思想 是用符号（变量）代替数，进行方程求解和数学运算。
- 汉语中的“代数” 是从西方数学翻译而来的，意思是用符号（变量）代替数值进行运算，所以称为“代数”。

### 泰勒级数
- 使用泰勒级数近似计算 f(x) 的值


```python
import math
import numpy as np
import sympy as sp

def taylor_expansion(f, a, n, x_val):
    """ 使用泰勒级数近似计算 f(x) 在 x=a 附近的值 """
    
    # 1. 定义符号变量 x（SymPy 用于符号计算）
    x = sp.Symbol('x')  
    
    # 2. 将输入函数 f(x) 转换为 sympy 表达式
    f_sym = f(x)  
    
    # 3. 初始化泰勒展开的结果
    result = 0  

    # 4. 计算 0 到 n-1 阶的泰勒展开
    for i in range(n):  
        # 4.1 计算 f(x) 的 i 阶导数，并在 x=a 处求值
        derivative = sp.diff(f_sym, x, i).subs(x, a)  
        
        # 4.2 计算泰勒展开的第 i 项
        term = (derivative / math.factorial(i)) * (x_val - a) ** i  
        
        # 4.3 将该项累加到最终结果
        result += term  

    # 5. 返回计算后的近似值，并转换为 float 以兼容 NumPy
    return float(result)  

# 测试 e^x 在 x=0 处的 5 阶展开，估算 e^0.5
f = lambda x: sp.exp(x)  # 定义 e^x
approx_value = taylor_expansion(f, 0, 5, 0.5)

# 输出计算结果
print(f"泰勒展开近似值: {approx_value}")
print(f"e^0.5 的真实值: {np.exp(0.5)}")  # 对比真实值
```

### 输入输出 两数求和


```python
import sys
line = sys.stdin.readline().strip()
array = line.split()
print(int(array[0]) + int(array[1]))
```

### 查找算法

-----------

#### 简单查找
- 时间复杂度：O(n)


```python
def simple_search(lst, e):
    for index, item in enumerate(lst):
        if item == e:
            return index
    return None
```

#### 二分查找
- 要求列表有序
- 时间复杂度：O(logn)


```python
def binary_search(lst, e):
    low = 0
    high = len(lst) - 1
    while low <= high:
        cur = (low + high) // 2
        if lst[cur] == e:
            return cur
        elif lst[cur] > e:
            high = cur - 1
        elif lst[cur] < e:
            low = cur + 1
    return None
```

#### 二叉树查找


```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert(self.root, key)

    def _insert(self, node, key):
        # 如果 key 小于当前节点的值，则插入到左子树
        if key < node.key:
            if node.left is None:
                node.left = Node(key)
            else:
                self._insert(node.left, key)
        # 如果 key 大于当前节点的值，则插入到右子树
        elif key > node.key:
            if node.right is None:
                node.right = Node(key)
            else:
                self._insert(node.right, key)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        # 如果节点为空或找到了目标值，返回结果
        if node is None or node.key == key:
            return node
        # 如果目标值小于当前节点的值，则在左子树查找
        elif key < node.key:
            return self._search(node.left, key)
        # 如果目标值大于当前节点的值，则在右子树查找
        else:
            return self._search(node.right, key)

    def print_inorder(self):
        self._inorder(self.root)

    def _inorder(self, node):
        if node:
            self._inorder(node.left)
            print(node.key, end=" ")
            self._inorder(node.right)

# 使用示例
bst = BinarySearchTree()
bst.insert(50)
bst.insert(30)
bst.insert(20)
bst.insert(40)
bst.insert(70)
bst.insert(60)
bst.insert(80)

# 查找
result = bst.search(40)
if result:
    print("Found:", result.key)
else:
    print("Not Found")

# 打印树的中序遍历（升序）
bst.print_inorder()  # 输出：20 30 40 50 60 70 80
```

    Found: 40
    20 30 40 50 60 70 80 

#### 哈希表查找


```python
class HashTable:
    def __init__(self, size):
        self.size = size  # 哈希表大小
        self.table = [None] * size  # 初始化哈希表为大小为 size 的列表

    def hash_function(self, key):
        """简单的哈希函数：key 对哈希表大小取模"""
        return key % self.size

    def insert(self, key, value):
        """插入 key-value 对"""
        index = self.hash_function(key)  # 计算哈希值
        if self.table[index] is None:
            self.table[index] = [(key, value)]  # 如果该位置为空，直接插入
        else:
            # 处理哈希冲突：如果该位置已经有数据，采用链表法（在链表中插入新元素）
            for i in range(len(self.table[index])):
                if self.table[index][i][0] == key:  # 如果已有相同的 key，更新值
                    self.table[index][i] = (key, value)
                    return
            self.table[index].append((key, value))  # 没有相同 key，追加到链表

    def search(self, key):
        """查找 key 对应的值"""
        index = self.hash_function(key)
        if self.table[index] is None:
            return None  # 如果该位置为空，表示 key 不存在
        else:
            # 遍历链表，查找是否有对应的 key
            for k, v in self.table[index]:
                if k == key:
                    return v
        return None  # 没找到对应的 key

    def delete(self, key):
        """删除 key 对应的元素"""
        index = self.hash_function(key)
        if self.table[index] is None:
            return None  # 如果该位置为空，表示 key 不存在
        else:
            for i, (k, v) in enumerate(self.table[index]):
                if k == key:
                    del self.table[index][i]  # 删除 key-value 对
                    return
        return None  # 没有找到对应的 key

    def print_table(self):
        """打印哈希表"""
        for index, bucket in enumerate(self.table):
            print(f"Index {index}: {bucket}")


# 使用哈希表的示例
hash_table = HashTable(10)  # 创建一个大小为 10 的哈希表

# 插入数据
hash_table.insert(10, "Value10")
hash_table.insert(20, "Value20")
hash_table.insert(15, "Value15")
hash_table.insert(25, "Value25")
hash_table.insert(5, "Value5")

# 查找数据
print("Search for key 15:", hash_table.search(15))  # 输出: Value15
print("Search for key 30:", hash_table.search(30))  # 输出: None

# 删除数据
hash_table.delete(15)
print("After deleting key 15:")
hash_table.print_table()

# 查找删除后的数据
print("Search for key 15 after deletion:", hash_table.search(15))  # 输出: None
```

    Search for key 15: Value15
    Search for key 30: None
    After deleting key 15:
    Index 0: [(10, 'Value10'), (20, 'Value20')]
    Index 1: None
    Index 2: None
    Index 3: None
    Index 4: None
    Index 5: [(25, 'Value25'), (5, 'Value5')]
    Index 6: None
    Index 7: None
    Index 8: None
    Index 9: None
    Search for key 15 after deletion: None


### 排序算法

---------


| 排序算法   | 最好时间复杂度 | 平均时间复杂度  | 最坏时间复杂度 | 空间复杂度 | 是否稳定 | 适用场景 |
|------------|--------------|---------------|---------------|------------|----------|---------|
| **冒泡排序** | \(O(n)\) | \(O(n^2)\) | \(O(n^2)\) | \(O(1)\) | ✅ 稳定 | 适用于数据量小、基本有序的情况 |
| **选择排序** | \(O(n^2)\) | \(O(n^2)\) | \(O(n^2)\) | \(O(1)\) | ❌ 不稳定 | 适用于数据量小，对稳定性无要求 |
| **插入排序** | \(O(n)\) | \(O(n^2)\) | \(O(n^2)\) | \(O(1)\) | ✅ 稳定 | 适用于数据基本有序的情况 |
| **希尔排序** | \(O(n \log n)\) | 依赖 gap 选择 | \(O(n^2)\) | \(O(1)\) | ❌ 不稳定 | 适用于中等规模数据，无额外空间需求 |
| **归并排序** | \(O(n \log n)\) | \(O(n \log n)\) | \(O(n \log n)\) | \(O(n)\) | ✅ 稳定 | 适用于大规模数据，需要稳定性 |
| **快速排序** | \(O(n \log n)\) | \(O(n \log n)\) | \(O(n^2)\)（最坏） | \(O(\log n)\)（递归栈） | ❌ 不稳定 | 适用于一般场景，是最快的排序算法之一 |
| **堆排序** | \(O(n \log n)\) | \(O(n \log n)\) | \(O(n \log n)\) | \(O(1)\) | ❌ 不稳定 | 适用于大规模数据、需要节省空间 |
| **计数排序** | \(O(n + k)\) | \(O(n + k)\) | \(O(n + k)\) | \(O(k)\) | ✅ 稳定 | 适用于整数、小范围数据 |
| **基数排序** | \(O(nk)\) | \(O(nk)\) | \(O(nk)\) | \(O(n + k)\) | ✅ 稳定 | 适用于定长整数、字符串排序 |
| **桶排序** | \(O(n + k)\) | \(O(n + k)\) | \(O(n^2)\) | \(O(n + k)\) | ✅ 稳定 | 适用于数据分布均匀的情况 |

---

**如何选择合适的排序算法？**
1. **数据量小（< 1,000）**
   - **已基本有序** → **插入排序**
   - **随机数据** → **选择排序** 或 **冒泡排序**

2. **数据量中等（1,000 ~ 100,000）**
   - **对稳定性要求高** → **归并排序**
   - **对空间复杂度要求高** → **快速排序（最常用）**
   - **内存受限** → **堆排序**

3. **数据量很大（> 100,000）**
   - **整数范围较小** → **计数排序 / 基数排序 / 桶排序**
   - **通用排序** → **快速排序 / 归并排序**

快速排序通常是 **通用情况下最快的排序算法**，但当需要稳定排序或处理大规模数据时，可以选择 **归并排序、、堆排序或基数排序**


#### 选择排序

- 思想：数组分为已排序和未排序两部分，每次从未排序部分选择最小（或最大）的元素，放到已排序部分的末尾，直到所有元素都被排序。
- 步骤：
1. 从数组中找到最小的元素，将其与数组的第一个元素交换位置。
2. 在剩下的未排序部分中，找到最小的元素，将其与第二个位置的元素交换。
3. 重复上述步骤，直到所有元素都排好序。
- 时间复杂度：
    - 最好情况：O(n<sup>2</sup>)
    - 最坏情况：O(n<sup>2</sup>)
    - 平均时间复杂度：O(n<sup>2</sup>)
- 空间复杂度：原地排序，不占额外内存， O(1)
- 不稳定排序


```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_index = i
        # 在未排序部分寻找最小值
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        # 交换最小值与当前位置
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

# 示例
arr = [5, 2, 9, 1, 5, 6]
sorted_arr = selection_sort(arr)
print(sorted_arr)
```

    [1, 2, 5, 5, 6, 9]


#### 插入排序

- 思想：数组分为已排序和未排序两部分，每次从未排序部分取一个元素，插入到已排序部分的正确位置，直到所有元素都被插入。
- 时间复杂度
    - 最好情况（数组本身已排序）：O(n)
    - 最坏情况（数组是逆序的）：O(n<sup>2</sup>)
    - 平均情况：O(n<sup>2</sup>)
- 空间复杂度：原地排序，不需要额外的数组存储， O(1)
- 稳定排序（相同元素的相对顺序不会改变）



```python
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        # 将比 key 大的元素后移
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        # 插入 key
        arr[j + 1] = key
    return arr

# 示例
arr = [5, 2, 9, 1, 5, 6]
sorted_arr = selection_sort(arr)
print(sorted_arr)
```

    [1, 2, 5, 5, 6, 9]


#### 希尔排序

- 希尔排序（Shell Sort）是插入排序的改进版，通过分组减少逆序对，加快排序速度。
- 步骤：
1. 选择一个步长（gap），将数组按这个间隔分组，对每组进行插入排序。
2. 逐步缩小步长，直到 gap = 1 时，进行最后一次插入排序，使整个数组有序。
3. 由于步长逐渐减小，数据在最后几轮排序时已接近有序，因此最终的插入排序速度快。
- 时间复杂度（取决于步长序列（gap））：
    - 最好情况：O(n<sup>2</sup>)
    - 最好情况：O(nlogn)
    - 平均情况：通常优于O(n<sup>2</sup>)，但具体复杂外的数组存储）

- 空间复杂：原地排序，O(1)
- 不稳定排序


```python
def shell_sort(arr):
    n = len(arr)
    gap = n // 2  # 初始步长

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            # 对当前分组使用插入排序
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2  # 缩小步长

    return arr

# 示例
arr = [12, 34, 54, 2, 3]
sorted_arr = shell_sort(arr)
print(sorted_arr)
```

    [2, 3, 12, 34, 54]


#### 冒泡排序

- 思想：不断比较相邻的元素并交换，使较大的元素逐步“冒泡”到数组的末尾，直到整个数组有序。
- 时间复杂度：
    - 最好情况：（已经有序，优化版）：O(n)
    - 最坏情况（完全逆序）：O(n<sup>2</sup>)
    - 平均情况：O(n<sup>2</sup>)
- 空间复杂度：O(1)
- 稳定排序，相等的元素不会交换位置。


```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):  # 需要进行 n-1 轮遍历
        for j in range(n - 1 - i):  # 每轮遍历，最大元素会“浮”到最后
            if arr[j] > arr[j + 1]:  # 交换相邻元素
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# 示例
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = bubble_sort(arr)
print(sorted_arr)
```

    [11, 12, 22, 25, 34, 64, 90]


#### 基数排序

- 思想：基数排序（Radix Sort） 是一种非比较排序算法，适用于整数排序或固定长度的字符串排序。它的核心思想是按位分组，逐位排序，从最低位到最高位（或反向）进行多轮稳定排序，使整个数组变得有序。
- 时间复杂度  
    - 最好情况：O(nk)  （k 为最大数的位数）
    - 最坏情况：O(nk) 
    - 平均时间复杂度：O(nk) （通常k<<n，接近O(n)）
- 空间复杂度
    - 额外存储桶或计数数组，通常为 O(n+k)
- 稳定排序
- 特点：
    - 适用于整数和定长字符串
    - ❌ 额外空间消耗较大（需要额外存储桶或数组）
    - ❌ 不适用于小数、负数（需要额外处理）
    - 适用于大规模数据排序（如电话号、身份证号排序），但在一般情况下，快速排序或归并排序更常用。


```python
def counting_sort(arr, exp):
    """对数组按照 exp 位进行计数排序"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10  # 0-9 数字的计数

    # 统计每个数字出现的次数
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    # 累加，计算每个数字的最终位置
    for i in range(1, 10):
        count[i] += count[i - 1]

    # 逆序填充 output，保证稳定性
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    # 复制回原数组
    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    """基数排序"""
    max_num = max(arr)  # 找到最大数，确定排序轮数
    exp = 1  # 逐位排序（1, 10, 100, ...）
    
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= 10  # 进位

# 示例
arr = [170, 45, 75, 90, 802, 24, 2, 66]
radix_sort(arr)
print(arr)
```

    [2, 24, 45, 66, 75, 90, 170, 802]


#### 快速排序

- 思想：快速排序是一种分治算法，通过选择一个“基准元素（pivot）”，将数组分为两部分，使得左侧所有元素小于基准，右侧所有元素大于基准，然后对两部分递归排序。
- 时间复杂度  
    - 最好情况：O(nlogn) 处理的列表本身有序且基准值总选择为列表第len(lst)//2项
    - 最坏情况：O(n<sup>2</sup>) 处理的列表本身有序且基准值总选择为列表第一项
    - 平均时间复杂度：O(nlogn)
- 空间复杂度
    - 递归版本需要额外空间存储子数组，最坏情况下空间复杂度为 O(n)。
    - 原地快排（in-place） 仅需 O(logn) 额外栈空间（递归调用）。 
- 不稳定排序
- 实现快速排序时，请随机地选择用作基准值的元素。
- 大O表示法中的常量有时候事关重大，这就是快速排序比合并排序快的原因所在。


```python
def quick_sort(lst):
    if len(lst) < 2:
        return lst
    else:
        pivot = lst[0]
        less = []
        greater = []
        for i in range(1, len(lst)):
            if lst[i] <= pivot:
                less.append(lst[i])
            else:
                greater.append(lst[i])
        return quick_sort(less) + [pivot] + quick_sort(greater)
```

#### 归并排序
- 思想：归并排序是一种分治算法（Divide and Conquer），它的基本思想是：（1）拆分：递归地将数组分成两半，直到每个子数组的长度为 1。
（2）合并：将两个有序子数组合并成一个有序数组。
- 时间复杂度
    - 最好情况：O(nlogn)
    - 最坏情况：O(nlogn)
    - 平均时间复杂度：O(nlogn)
- 空间复杂度 O(n)
- 稳定排序
- 适用于大规模数据排序
- 相比快速排序，常数因子较大，实际运行速度可能较慢（尤其是在小数据集上）


```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr  # 递归终止条件，数组长度为 1

    # 1. 拆分数组
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])  # 排序左半部分
    right = merge_sort(arr[mid:])  # 排序右半部分

    # 2. 合并两个有序数组
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # 保持稳定性
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # 处理剩余元素
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# 示例
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr = merge_sort(arr)
print(sorted_arr)
```

    [3, 9, 10, 27, 38, 43, 82]


#### 堆排序

- 思想：堆排序 是一种 基于二叉堆（Binary Heap） 的排序算法，它的核心思想是：（1）构建最大堆（大顶堆），使得堆顶元素是整个数组的最大值。（2）不断交换堆顶元素和末尾元素，然后调整堆，直到排序完成。
- 时间复杂度  
    - 构造堆：O(n) 
    - 堆排序：O(nlogn) 
    - 总体时间复杂度：O(nlogn)
- 空间复杂度：原地排序，O(1)
- 不稳定排序
- 特点：
  - 常数开销较大，速度一般比快速排序慢
  - 堆排序适用于 需要稳定 O(nlogn) 时间复杂度，且不能使用额外空间的情况，如 优先队列、操作系统调度等。


```python
def heapify(arr, n, i):
    """维护堆的性质"""
    largest = i  # 假设当前节点是最大值
    left = 2 * i + 1  # 左子节点
    right = 2 * i + 2  # 右子节点

    # 如果左子节点比当前节点大，更新 largest
    if left < n and arr[left] > arr[largest]:
        largest = left

    # 如果右子节点比当前节点大，更新 largest
    if right < n and arr[right] > arr[largest]:
        largest = right

    # 如果 largest 发生变化，则交换并继续调整
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)  # 递归调整子树

def heap_sort(arr):
    """堆排序"""
    n = len(arr)

    # 1. 构建最大堆（从最后一个非叶子节点开始调整）
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 2. 交换堆顶元素和末尾元素，并调整堆
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # 交换最大元素到数组末尾
        heapify(arr, i, 0)  # 调整剩余堆结构

# 示例
arr = [4, 10, 3, 5, 1]
heap_sort(arr)
print(arr)  # 输出 [1, 3, 4, 5, 10]

```

    [1, 3, 4, 5, 10]

### Python 库手册:*heapq* 最小堆模块

```python
import heapq
def lastStoneWeight(stones) -> int:
    h = [-1*n for n in stones]
    heapq.heapify(h)
    while len(stones) > 1:
        y = heapq.heappop(h) * -1
        x = heapq.heappop(h) * -1
        if x != y:
            heapq.heappush(h, (y-x))
    return heapq.heappop(h)*-1 if len(h) == 1 else 0

print(lastStoneWeight([1]))

```

### 树算法

-----------

#### 平衡二叉树


```python



```

#### 哈夫曼编码


- 哈夫曼编码（Huffman Coding） 是一种用于数据压缩的算法，它通过使用不同长度的二进制编码来表示不同的字符，以便最常见的字符用较短的编码表示，而不常见的字符用较长的编码表示，从而实现数据压缩。




```python
import heapq
from collections import defaultdict

# 哈夫曼树节点类
class Node:
    def __init__(self, char, freq):
        self.char = char  # 字符
        self.freq = freq  # 字符频率
        self.left = None  # 左子树
        self.right = None  # 右子树

    # 为了让堆排序时比较节点频率
    def __lt__(self, other):
        return self.freq < other.freq

# 构建哈夫曼编码
def build_huffman_tree(text):
    # 统计字符频率
    frequency = defaultdict(int)
    for char in text:
        frequency[char] += 1
    
    # 创建最小堆（优先队列）
    heap = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)
    
    # 构建哈夫曼树
    while len(heap) > 1:
        # 取出两个频率最小的节点
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # 创建新节点，频率为左右子节点之和
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        # 将新节点插入堆中
        heapq.heappush(heap, merged)
    
    # 返回哈夫曼树的根节点
    return heap[0]

# 为哈夫曼树生成编码
def generate_huffman_codes(root, prefix='', codebook={}):
    if root is None:
        return codebook
    
    # 如果是叶子节点（字符节点），将字符和编码保存
    if root.char is not None:
        codebook[root.char] = prefix
    
    # 递归地生成左子树和右子树的编码
    generate_huffman_codes(root.left, prefix + '0', codebook)
    generate_huffman_codes(root.right, prefix + '1', codebook)
    
    return codebook

# 对文本进行哈夫曼编码
def huffman_encode(text):
    root = build_huffman_tree(text)  # 构建哈夫曼树
    codes = generate_huffman_codes(root)  # 生成哈夫曼编码
    encoded_text = ''.join(codes[char] for char in text)  # 将原文本编码
    return encoded_text, codes

# 对编码后的文本进行解码
def huffman_decode(encoded_text, codes):
    # 反向构建解码表
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_text = ''
    prefix = ''
    
    # 遍历编码文本，逐步匹配解码
    for bit in encoded_text:
        prefix += bit
        if prefix in reverse_codes:
            decoded_text += reverse_codes[prefix]
            prefix = ''  # 重置prefix
    
    return decoded_text

# 使用示例
text = "this is an example for huffman encoding"
encoded_text, codes = huffman_encode(text)
print("原始文本:", text)
print("哈夫曼编码:", encoded_text)
print("哈夫曼编码表:", codes)

# 解码示例
decoded_text = huffman_decode(encoded_text, codes)
print("解码后的文本:", decoded_text)

```

    原始文本: this is an example for huffman encoding
    哈夫曼编码: 0101001001001001010110010010101111100010111100111011110011100000110111101011101110010110010101001100011011101001111110001011110000011111100101011100100010001
    哈夫曼编码表: {'n': '000', 's': '0010', 'm': '0011', 'h': '0100', 't': '01010', 'd': '01011', 'r': '01100', 'l': '01101', 'x': '01110', 'c': '01111', 'p': '10000', 'g': '10001', 'i': '1001', ' ': '101', 'u': '11000', 'o': '11001', 'f': '1101', 'e': '1110', 'a': '1111'}
    解码后的文本: this is an example for huffman encoding


### 图算法

------

#### 深度优先搜索（DFS）


```python
from collections import deque
def dfs(graph, root):
    visited = set()
    dq = deque([root])
    while dq:
        vertex = dq.pop()     #DFS使用栈，先进后出
        if vertex not in visited:
            visited.add(vertex)
            print(vertex, end=' ')
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                dq.append(neighbour)

# 有向图的邻接表表示
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

dfs(graph, 'A') 
```

    A C F B E D 

#### 广度优先搜索（BFS）


```python
from collections import deque
def bfs(graph, root):
    visited = set()
    dq = deque([root])
    while dq:
        vertex = dq.popleft()    # BFS使用队列，先进先出
        if vertex not in visited:
            visited.add(vertex)
            print(vertex, end=" ")
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                dq.append(neighbour)

# 有向图的邻接表表示
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

bfs(graph, 'A')
```

    A B C D E F 

#### 最小生成树：Prim算法

- 最小生成树是图中连接所有顶点的边的集合，且这组边的总权值最小。
- Prim算法是一种用于求解最小生成树（MST，Minimum Spanning Tree）的问题的贪心算法。
- 思想：Prim算法通过从一个起始顶点开始，每次从队列中取出权重最小的边，如果这条边连接的节点中有一个已经访问过而另一个没有访问过，则将该边添加到生成树中，并将未访问的节点标记为已访问，直到所有顶点都被包含


```python
import heapq  # 导入堆（优先队列）

class Graph:
    def __init__(self, vertices):
        self.V = vertices  # 顶点数
        self.graph = {}  # 图的邻接表
    
    def add_edge(self, u, v, weight):
        """向图中添加边，边的格式是(u, v, weight)"""
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))
    
    def prim(self, start):
        """运行Prim算法，构造最小生成树"""
        mst = []  # 存储最小生成树的边
        visited = set()  # 记录已访问的顶点
        min_heap = []  # 最小堆（优先队列）

        # 从起始顶点开始
        visited.add(start)
        for neighbor, weight in self.graph[start]:
            heapq.heappush(min_heap, (weight, start, neighbor))  # 将所有与起始顶点相连的边加入堆
        
        while min_heap:
            weight, u, v = heapq.heappop(min_heap)  # 弹出权重最小的边
            if v not in visited:  # 如果v尚未访问
                visited.add(v)  # 标记v为已访问
                mst.append((u, v, weight))  # 将这条边加入最小生成树
                for neighbor, w in self.graph[v]:  # 遍历与v相连的边
                    if neighbor not in visited:
                        heapq.heappush(min_heap, (w, v, neighbor))  # 将未访问的边加入堆
        
        return mst

# 使用示例
g = Graph(5)  # 创建一个包含5个顶点的图

# 添加边：格式为 (u, v, weight)
g.add_edge(0, 1, 2)
g.add_edge(0, 3, 6)
g.add_edge(1, 3, 8)
g.add_edge(1, 2, 3)
g.add_edge(3, 4, 7)
g.add_edge(2, 4, 5)

# 运行Prim算法，选择顶点0作为起始点
mst = g.prim(0)

# 输出最小生成树
print("最小生成树的边及权重：")
for u, v, weight in mst:
    print(f"({u}, {v}) -> 权重: {weight}")
```

    最小生成树的边及权重：
    (0, 1) -> 权重: 2
    (1, 2) -> 权重: 3
    (2, 4) -> 权重: 5
    (0, 3) -> 权重: 6


#### 最小生成树：克鲁斯卡尔（Kruskal）算法

- 克鲁斯卡尔算法（Kruskal's Algorithm） 是一种用于求解最小生成树（MST，Minimum Spanning Tree）问题的贪心算法。
- 与 Prim 算法不同，克鲁斯卡尔算法并不是从一个起始顶点开始构建生成树，而是通过边的权重来进行排序。
- 思想：如果最小权值边连接的两个顶点不在同一个连通块中，则将这条边加入生成树，并将这两个顶点合并到同一连通块。


```python
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))  # parent[i] 表示节点i的父节点
        self.rank = [0] * n  # rank[i] 用于优化合并操作

    def find(self, u):
        """查找并查集中的根节点"""
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # 路径压缩
        return self.parent[u]

    def union(self, u, v):
        """合并两个集合"""
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u != root_v:
            # 按秩合并（小的树合并到大的树下面）
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

class Kruskal:
    def __init__(self, vertices):
        self.V = vertices  # 顶点数
        self.edges = []  # 存储图的边

    def add_edge(self, u, v, weight):
        """添加边到图中"""
        self.edges.append((weight, u, v))

    def kruskal(self):
        """运行克鲁斯卡尔算法，返回最小生成树"""
        # 1. 对边按权重排序
        self.edges.sort()  # 按照边的权重进行升序排序
        
        # 2. 初始化并查集
        dsu = DisjointSet(self.V)
        
        mst = []  # 存储最小生成树的边
        
        # 3. 遍历所有边
        for weight, u, v in self.edges:
            # 4. 判断u和v是否在同一连通块中
            if dsu.find(u) != dsu.find(v):
                mst.append((u, v, weight))  # 将这条边加入最小生成树
                dsu.union(u, v)  # 合并u和v所在的连通块

        return mst

# 使用示例
kruskal = Kruskal(6)  # 创建一个包含6个顶点的图

# 添加边：格式为 (u, v, weight)
kruskal.add_edge(0, 1, 4)
kruskal.add_edge(0, 2, 4)
kruskal.add_edge(1, 2, 2)
kruskal.add_edge(1, 3, 5)
kruskal.add_edge(2, 3, 3)
kruskal.add_edge(3, 4, 6)
kruskal.add_edge(2, 4, 7)
kruskal.add_edge(4, 5, 7)

# 运行克鲁斯卡尔算法
mst = kruskal.kruskal()

# 输出最小生成树
print("最小生成树的边及权重：")
for u, v, weight in mst:
    print(f"({u}, {v}) -> 权重: {weight}")
```

    最小生成树的边及权重：
    (1, 2) -> 权重: 2
    (2, 3) -> 权重: 3
    (0, 1) -> 权重: 4
    (3, 4) -> 权重: 6
    (4, 5) -> 权重: 7


#### 最短距离：dijkstra算法


```python
def vis_parents(parents):
    from collections import deque
    d = deque()
    node = parents['final']
    while node != 'start':
        d.append(node)
        node = parents[node]
    str = 'start'
    while len(d) != 0:
        node = d.pop()
        str = str + ' -> ' + node
    return str + '-> final'

def find_lowest_cost_node(costs, processed):
    lowest_node = None
    lowest_value = float('inf')
    for n in costs.keys():
        if n in processed:
            continue
        if costs[n] < lowest_value:
            lowest_value = costs[n]
            lowest_node = n
    return lowest_node

def dijkstra(graph, costs, parents):
    processed = []
    node = find_lowest_cost_node(costs, processed)
    while node is not None:
        cost = costs[node]
        neighbors = graph[node]
        for n in neighbors.keys():
            if neighbors[n] + cost < costs[n]:
                costs[n] = neighbors[n] + cost
                parents[n] = node
        processed.append(node)
        node = find_lowest_cost_node(costs, processed)
    return vis_parents(parents)

# 构建图
graph = {}
graph['start'] = {
    'A': 6,
    'B': 2
}
graph['A'] = {
    'final': 1
}

graph['B'] = {
    'A': 9,
    'final': 5
}
graph['final'] = {}

# 构建开销表
infinity = float('inf')
costs = {
    'A': 6,
    'B': 2,
    'final': infinity
}

# 构建父节点表
parents = {
    'A': 'start',
    'B': 'start',
    'final': None
}

str = dijkstra(graph, costs, parents)
print(str)
```

#### 最短距离：floyd算法

- 用于解决**所有顶点**对最短路径问题的经典算法。该算法通过动态规划的思想，计算图中所有顶点之间的最短路径。
- 时间复杂度：O(V³)，其中V是图中的顶点数。每次都需要遍历所有顶点进行更新。
- 空间复杂度：O(V²)，用于存储距离矩阵。


```python
INF = float('inf')  # 用来表示无穷大

def floyd_warshall(graph, V):
    # 初始化距离矩阵
    dist = [[INF] * V for _ in range(V)]
    
    # 将图的边权值放入距离矩阵中
    for i in range(V):
        dist[i][i] = 0  # 从一个顶点到自己的距离为0
    for u in range(V):
        for v, weight in graph.get(u, []):
            dist[u][v] = weight  # 设置已知边的权重

    # 动态规划计算最短路径
    for k in range(V):
        for i in range(V):
            for j in range(V):
                # 更新最短路径
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

# 使用示例
graph = {
    0: [(1, 4), (2, 1)],
    1: [(2, 2), (3, 5)],
    2: [(3, 1)],
    3: []
}

V = 4  # 图中有4个顶点

# 运行Floyd-Warshall算法
dist = floyd_warshall(graph, V)

# 输出最短路径矩阵
print("从每个顶点到其他顶点的最短路径：")
for i in range(V):
    for j in range(V):
        if dist[i][j] == INF:
            print(f"从 {i} 到 {j} 没有路径")
        else:
            print(f"从 {i} 到 {j} 的最短路径为: {dist[i][j]}")



```

#### 拓扑排序AOV

- 由于DFS的回溯特性，结果列表中的节点顺序正好是拓扑排序的逆序。


```python
from collections import deque, defaultdict
def dfs_topological_sort(V, graph):
    visited = [False] * V  # 标记节点是否已访问
    stack = []  # 用于存储拓扑排序结果
    on_stack = [False] * V  # 用于检测图中是否有环
    
    def dfs(u):
        if on_stack[u]:  # 如果当前节点已经在栈中，说明图中有环
            raise ValueError("图中有环，无法进行拓扑排序")
        if visited[u]:  # 如果节点已经访问过，直接返回
            return
        visited[u] = True
        on_stack[u] = True
        for v in graph[u]:
            dfs(v)
        on_stack[u] = False
        stack.append(u)  # 将当前节点加入栈
    
    for u in range(V):
        if not visited[u]:
            dfs(u)
    
    # 最后返回栈中逆序的结果即为拓扑排序
    return stack[::-1]

# 使用示例
V = 6
graph = defaultdict(list)

# 添加边（有向边）
graph[5].append(2)
graph[5].append(0)
graph[4].append(0)
graph[4].append(1)
graph[2].append(3)
graph[3].append(1)

# 运行DFS算法
try:
    result = dfs_topological_sort(V, graph)
    print("拓扑排序结果:", result)
except ValueError as e:
    print(e)



```

    拓扑排序结果: [5, 4, 2, 3, 1, 0]


#### 关键路径AOE


```python



```
