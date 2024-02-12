import numpy as np
from matplotlib import pyplot as plt

def sum_mean(disrectory, file ):
    
    f=open(disrectory+"/D1","r")
    w=f.readlines()
    f.close()
    e0=0
    e2=0
    e4=0
    e6=0
    e10=0
    e15=0
    e20=0
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        c=float(i)
        if(c<2.0):
            e0+=1
        if(c>=2.0 and c<4.0):
            e2+=1
        if(c>=4.0 and c<6.0):
            e4+=1
        if(c>=6.0 and c<10.0):
            e6+=1
        if(c>=10.0 and c<15.0):
            e10+=1
        if(c>=15 and c<20.0):
            e15+=1
        if(c>=20):
            e20+=1
        x+=float(i)
        a+=1
    x=x/a
    
    plt.figure(figsize=(10, 4))
    
    errors=[e0, e2,  e4, e6, e10, e15,  e20]
    e=["0-2", "2-4", "4-6", "6-10", "10-15", "15-20", "20-inf"]
    plt.subplot(121)
    plt.bar(e, errors)
    plt.title("D1")
    plt.xlabel("D1 [%]")
    plt.ylabel("photos")
    
    f=open(file, "a")
    f.write("D1:"+ "\n")
    f.write("0-2:" + str(e0)+ "\n")
    f.write("2-4:" + str(e2)+ "\n")
    f.write("4-6:" + str(e4)+ "\n")
    f.write("6-10:" + str(e6)+ "\n")
    f.write("10-15:" + str(e10)+ "\n")
    f.write("15-20:" + str(e15)+ "\n")
    f.write("20-inf:" + str(e20)+ "\n")
    e0=(np.round(e0/a*100,2))
    e2=(np.round(e2/a*100,2))
    e4=(np.round(e4/a*100,2))
    e6=np.round(e6/a*100,2)
    e10=(np.round(e10/a*100,2))
    e15=(np.round(e15/a*100,2))
    e20=(np.round(e20/a*100,2))
    f.write("D1 [%]:"+ "\n")
    f.write("0-2:" + str(e0)+ "\n")
    f.write("2-4:" + str(e2)+ "\n")
    f.write("4-6:" + str(e4)+ "\n")
    f.write("6-10:" + str(e6)+ "\n")
    f.write("10-15:" + str(e10)+ "\n")
    f.write("15-20:" + str(e15)+ "\n")
    f.write("20-inf:" + str(e20)+ "\n")
    x=np.round(x,2)
    f.write("All: "+ str(x)+ "\n")
    f.close()
    f=open(disrectory+"/mdre","r")
    w=f.readlines()
    f.close()
    f0=open(disrectory+"/E_0_2","a")
    f2=open(disrectory+"/E_2_4","a")
    f4=open(disrectory+"/E_4_6","a")
    f6=open(disrectory+"/E_6_10","a")
    f10=open(disrectory+"/E_10_15","a")
    f15=open(disrectory+"/E_15_20","a")
    f20=open(disrectory+"/E_20_inf","a")
    e0=0
    e2=0
    e4=0
    e6=0
    e10=0
    e15=0
    e20=0
    e3=0
    e34=0
    a=0
    x=0
    for ci in w:
        i=ci.split(" ")[1]
        c=float(i)
        if(c<2.0):
            e0+=1
            f0.write(ci)
        if(c>=3.0 and c<4.0):
            e34+=1
        if(c>=2.0 and c<3.0):
            e3+=1
        if(c>=2.0 and c<4.0):
            e2+=1
            f2.write(ci)
        if(c>=4.0 and c<6.0):
            e4+=1
            f4.write(ci)
        if(c>=6.0 and c<10.0):
            e6+=1
            f6.write(ci)
        if(c>=10.0 and c<15.0):
            e10+=1
            f10.write(ci)
        if(c>=15 and c<20.0):
            e15+=1
            f15.write(ci)
        if(c>=20):
            e20+=1
            f20.write(ci)
        x+=float(i)
        a+=1
    x=x/a
    f0.close()
    f2.close()
    f4.close()
    f6.close()
    f10.close()
    f15.close()
    f20.close()
    errors=[e0, e2,  e4, e6, e10, e15,  e20]
    e=["0-2", "2-4", "4-6", "6-10", "10-15", "15-20", "20-inf"]
    plt.subplot(122)
    plt.bar(e, errors)
    plt.title("MDRE")
    plt.xlabel("MDRE")
    plt.ylabel("photos")
    plt.suptitle('PSMNet')
    plt.savefig(disrectory+"/E.png")
    
    f=open(file, "a")
    f.write("mdre:"+ "\n")
    f.write("0-2:" + str(e0)+ "\n")
    f.write("2-3:" + str(e3) +"\n")
    f.write("3-4:" + str(e34) +"\n")
    f.write("2-4:" + str(e2)+ "\n")
    f.write("4-6:" + str(e4)+ "\n")
    f.write("6-10:" + str(e6)+ "\n")
    f.write("10-15:" + str(e10)+ "\n")
    f.write("15-20:" + str(e15)+ "\n")
    f.write("20-inf:" + str(e20)+ "\n")
    e0=(np.round(e0/a*100,2))
    e2=(np.round(e2/a*100,2))
    e4=(np.round(e4/a*100,2))
    e6=np.round(e6/a*100,2)
    e10=(np.round(e10/a*100,2))
    e15=(np.round(e15/a*100,2))
    e20=(np.round(e20/a*100,2))
    f.write("mdre [%]:"+ "\n")
    f.write("0-2:" + str(e0)+ "\n")
    f.write("2-4:" + str(e2)+ "\n")
    f.write("4-6:" + str(e4)+ "\n")
    f.write("6-10:" + str(e6)+ "\n")
    f.write("10-15:" + str(e10)+ "\n")
    f.write("15-20:" + str(e15)+ "\n")
    f.write("20-inf:" + str(e20)+ "\n")
    x=np.round(x,2)
    f.write("All: "+str(x)+"\n")
    f.close()
    mdre=x
    f=open(disrectory+"/bmp","r")
    w=f.readlines()
    f.close()
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        x+=float(i)
        a+=1
    x=x/a
    q=np.round(x,2)
    f=open(file, "a")
    f.write("bmp: "+str(q)+"\n")
    f.close()
    
    f=open(disrectory+"/bmpre","r")
    w=f.readlines()
    f.close()
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        x+=float(i)
        a+=1
    x=x/a
    q=np.round(x,2)
    q=np.round(x,2)
    f=open(file, "a")
    f.write("bmpre: "+str(q)+"\n")
    f.close()
    
    f=open(disrectory+"/mre","r")
    w=f.readlines()
    f.close()
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        x+=float(i)
        a+=1
    x=x/a
    q=np.round(x,2)
    q=np.round(x,2)
    f=open(file, "a")
    f.write("mre: "+str(q)+"\n")
    f.close()
    
    f=open(disrectory+"/mse","r")
    w=f.readlines()
    f.close()
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        x+=float(i)
        a+=1
    x=x/a
    q=np.round(x,2)
    q=np.round(x,2)
    f=open(file, "a")
    f.write("mse: "+str(q)+"\n")
    f.close()
    f=open(disrectory+"/sze","r")
    w=f.readlines()
    f.close()
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        x+=float(i)
        a+=1
    x=x/a
    q=np.round(x,2)
    q=np.round(x,2)
    f=open(file, "a")
    f.write("sze: "+str(q)+"\n")
    f.close()
    return mdre
def sum_mean_excel(disrectory, file ):
    
    f=open(disrectory+"/D1","r")
    w=f.readlines()
    f.close()
    e0=0
    e2=0
    e4=0
    e6=0
    e10=0
    e15=0
    e20=0
    a=0
    x=0
    D1=0
    for i in w:
        i=i.split(" ")[1]
        c=float(i)
        if(c<2.0):
            e0+=1
        if(c>=2.0 and c<4.0):
            e2+=1
        if(c>=4.0 and c<6.0):
            e4+=1
        if(c>=6.0 and c<10.0):
            e6+=1
        if(c>=10.0 and c<15.0):
            e10+=1
        if(c>=15 and c<20.0):
            e15+=1
        if(c>=20):
            e20+=1
        x+=float(i)
        a+=1
    D1=x/a
    D1=np.round(D1,2)
    
    plt.figure(figsize=(10, 4))
    
    errors=[e0, e2,  e4, e6, e10, e15,  e20]
    e=["0-2", "2-4", "4-6", "6-10", "10-15", "15-20", "20-inf"]
    plt.subplot(121)
    plt.bar(e, errors)
    plt.title("D1")
    plt.xlabel("D1 [%]")
    plt.ylabel("photos")
    
    f=open(file, "a")
    f.write("D1:"+ "\n")
    f.write( str(e0)+ "\n")
    f.write( str(e2)+ "\n")
    f.write(str(e4)+ "\n")
    f.write( str(e6)+ "\n")
    f.write( str(e10)+ "\n")
    f.write(str(e15)+ "\n")
    f.write( str(e20)+ "\n")
    e0=(np.round(e0/a*100,2))
    e2=(np.round(e2/a*100,2))
    e4=(np.round(e4/a*100,2))
    e6=np.round(e6/a*100,2)
    e10=(np.round(e10/a*100,2))
    e15=(np.round(e15/a*100,2))
    e20=(np.round(e20/a*100,2))
    f.write("D1 [%]:"+ "\n")
    f.write( str(e0)+ "\n")
    f.write( str(e2)+ "\n")
    f.write( str(e4)+ "\n")
    f.write(str(e6)+ "\n")
    f.write( str(e10)+ "\n")
    f.write( str(e15)+ "\n")
    f.write( str(e20)+ "\n")
    f.close()
    f=open(disrectory+"/mdre","r")
    w=f.readlines()
    f.close()
    e0=0
    e2=0
    e4=0
    e6=0
    e10=0
    e15=0
    e20=0
    a=0
    x=0
    MDRE=0
    for ci in w:
        i=ci.split(" ")[1]
        c=float(i)
        if(c<2.0):
            e0+=1
        if(c>=2.0 and c<4.0):
            e2+=1
        if(c>=4.0 and c<6.0):
            e4+=1
        if(c>=6.0 and c<10.0):
            e6+=1
        if(c>=10.0 and c<15.0):
            e10+=1
        if(c>=15 and c<20.0):
            e15+=1
        if(c>=20):
            e20+=1
        x+=float(i)
        a+=1
    MDRE=x/a
    MDRE=np.round(MDRE,2)
    errors=[e0, e2,  e4, e6, e10, e15,  e20]
    e=["0-2", "2-4", "4-6", "6-10", "10-15", "15-20", "20-inf"]
    plt.subplot(122)
    plt.bar(e, errors)
    plt.title("MDRE")
    plt.xlabel("MDRE")
    plt.ylabel("photos")
    plt.suptitle('PSMNet')
    plt.savefig(disrectory+"/E.png")
    
    f=open(file, "a")
    f.write("mdre:"+ "\n")
    f.write(str(e0)+ "\n")
    f.write(str(e2)+ "\n")
    f.write(str(e4)+ "\n")
    f.write(str(e6)+ "\n")
    f.write( str(e10)+ "\n")
    f.write( str(e15)+ "\n")
    f.write( str(e20)+ "\n")
    e0=(np.round(e0/a*100,2))
    e2=(np.round(e2/a*100,2))
    e4=(np.round(e4/a*100,2))
    e6=np.round(e6/a*100,2)
    e10=(np.round(e10/a*100,2))
    e15=(np.round(e15/a*100,2))
    e20=(np.round(e20/a*100,2))
    f.write("mdre [%]:"+ "\n")
    f.write(str(e0)+ "\n")
    f.write(str(e2)+ "\n")
    f.write(str(e4)+ "\n")
    f.write( str(e6)+ "\n")
    f.write(str(e10)+ "\n")
    f.write(str(e15)+ "\n")
    f.write( str(e20)+ "\n")
    f.close()
    
    f=open(disrectory+"/bmp","r")
    w=f.readlines()
    f.close()
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        x+=float(i)
        a+=1
    x=x/a
    q=np.round(x,2)
    f=open(file, "a")
    f.write(str(D1).replace(".",",")+"\n")
    f.write(str(MDRE).replace(".",",")+"\n")
    f.write(str(q)+"\n")
    f.close()
    
    f=open(disrectory+"/bmpre","r")
    w=f.readlines()
    f.close()
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        x+=float(i)
        a+=1
    x=x/a
    q=np.round(x,2)
    q=np.round(x,2)
    f=open(file, "a")
    f.write(str(q)+"\n")
    f.close()
    
    f=open(disrectory+"/mre","r")
    w=f.readlines()
    f.close()
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        x+=float(i)
        a+=1
    x=x/a
    q=np.round(x,2)
    q=np.round(x,2)
    f=open(file, "a")
    f.write(str(q)+"\n")
    f.close()
    
    f=open(disrectory+"/mse","r")
    w=f.readlines()
    f.close()
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        x+=float(i)
        a+=1
    x=x/a
    q=np.round(x,2)
    q=np.round(x,2)
    f=open(file, "a")
    f.write(str(q)+"\n")
    f.close()
    f=open(disrectory+"/sze","r")
    w=f.readlines()
    f.close()
    a=0
    x=0
    for i in w:
        i=i.split(" ")[1]
        x+=float(i)
        a+=1
    x=x/a
    q=np.round(x,2)
    q=np.round(x,2)
    f=open(file, "a")
    f.write(str(q)+"\n")
    f.close()
if __name__ == '__main__':
    fol="/home/developer/PSMNet_lighting/Errors_metrics/Histogram_HSV"
    sum_mean(fol,fol+"/Error")
    sum_mean_excel(fol,fol+"/Error EXCEL")
