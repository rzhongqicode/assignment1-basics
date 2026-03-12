from collections import Counter

# a = "abc".encode("utf-8")
# b = "def".encode("utf-8")
# print(a)
# print(b)
# print(a<b)
# print(a+b)
# a = [1,2,3]
# print(a[:-1])

counter = Counter()
counter["apple"] = 2
counter["banana"] += 1
# counter["apple"] -= 2
del counter["apple"]
del counter["orange"]
counter["apple"] = 1
counter2 = Counter({"apple": 3, "pie": 4})
print(counter)
# print(counter.items())
print(counter2)
print(counter - counter2)
# a = tuple("仁".encode("utf-8"))
# b = "仁".encode("utf-8")
# print(a)
# print(b)
# # print(bytes([11])+bytes([12]))
# # for byte in
# print(a[0])
# print(type(b[0]))
# print(type(a[0]))
