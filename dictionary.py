



passwd={'Mars':00000,'Mark':56680}
passwd['Happy']=9999     
passwd['Smile']=123456

del passwd['Mars']
passwd['Mark']=passwd['Mark']+1

print passwd
print passwd.keys()
print passwd.values()
print passwd.get('Tony')
