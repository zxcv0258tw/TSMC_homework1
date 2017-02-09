



my_list=[]
for i in range(0,10):  #"""//for(i=0;i<10;i++)"""
    my_list.append(i+1)

for i in my_list: #"""//for(i=0;i<my_list.length();i++)"""
    print i #"""//cout<<my_list[i]"""

if my_list[0]==1 and len(my_list)<10:
    my_list[0]+=1
    print "1 state"

elif (11 in my_list) or (len(my_list)==1):
    print "2 state"
    print "range(i,j) is i~j-1"

else:
    print "3 state"  
    print "none of above"

