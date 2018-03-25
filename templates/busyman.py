t=int(raw_input())
for i in range(0,t):
  n=int(raw_input())
  a=[[]]
  for j in range(0,n):
    a[j]=[int(k) for k in raw_input().split()]
  print a
