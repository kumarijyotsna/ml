t=int(raw_input())
for _ in range(0,t):
  n=int(raw_input())
  a=[int(i) for i in raw_input().split()]
  j=0
  flag=0
  d=0
  '''while a[j]!=0:
    a[j]-=1
    a[j+1]-=1
    j+=1
    if j==n-1:
       j=0'''
  #print a
  for k in range(0,n):
    d = a[k] - d

  if d==0:
     print "YES"
  else:
     print "NO"
