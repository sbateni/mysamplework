#!/usr/bin/python





import sys
import re

class Point(object):
    """A point in a two dimensional space"""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return '(' + "{0:.2f}".format(self.x) + ',' + "{0:.2f}".format(self.y) + ')'


class Line(object):
    """A line between two points"""
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        if dst.x-src.x ==0:
            self.slope='inf'
        else:
            self.slope=round((dst.y-src.y)/(dst.x-src.x),4)
        self.length= ((dst.y-src.y)**2+(dst.x-src.x)**2)
       
    def __repr__(self):
        return '['+ str(self.src) + '-->' + str(self.dst) + ']'
    
class Street(object):
    """Contains each street name, coordinates, and sections"""
    def __init__(self, rawdata):
        self.name=rawdata[0]
        self.cordinates=[]
        self.sections=[]
        for points in rawdata[1:]:
            cama=points.find(',')
            self.cordinates.append (Point(float(points[0:cama]),float(points[cama+1:])))
            if len(self.cordinates)>= 2:
                self.sections.append (Line(self.cordinates[-2],self.cordinates[-1]))
        


def parse_line(line):

    line.strip()
    com1 =(re.search('[^ ]',line))
    com2=(re.search('"',line))
    p=line.find('(')
    q1=line.find('"')
    q2=line.rfind('"')
    if not com1: 
        raise Exception('Error: no valid command')
    comm=com1.group(0)
    name=re.search(r'\"(.+?)\"',line)
    cord=re.findall(r'\((.+?)\)',line)
    stripline=re.sub(r'\"(.+?)\"','',line)
    stripline=re.sub(r'\((.+?)\)','',stripline)
    stripline=re.sub(comm,'',stripline)

    info=[]
    
    if comm != 'a' and comm != 'c' and comm != 'r' and comm !='g':
       raise Exception('Error: unknown command')
    if comm!= 'g':
       if not name:
           raise Exception('Error: no valid street name') 
       elif com1.start(0) >= com2.start(0):
           raise Exception('Error: no valid command')
       elif re.search('[^ ]',stripline[:-1]):
           raise Exception('Error: nonvalid text in the input')
       elif com1.start(0) == q1-1:
           raise Exception('Error: no space between command and street name')
       elif p == q2+1:
           raise Exception('Error: no space between street name and coordinates')
         


       info.append(name.group(1).lower())
       info.extend(cord)
       
    elif re.search('[^ ]',line[com1.start(0)+1:-1]):
       raise Exception("Error: nothing must follow 'g' command")    
       
     

    return [comm,info]

def line_intersect(l1,l2):

    x1, y1 = l1.src.x, l1.src.y
    x2, y2 = l1.dst.x, l1.dst.y
    x3, y3 = l2.src.x, l2.src.y
    x4, y4 = l2.dst.x, l2.dst.y
    
    s1_x, s1_y = x2 - x1, y2 - y1     
    s2_x, s2_y = x4 - x3, y4 - y3

    
    if (-s2_x * s1_y + s1_x * s2_y )== 0:
        
        return False

    s = (-s1_y * (x1 - x3) + s1_x * (y1 - y3)) / (-s2_x * s1_y + s1_x * s2_y)
    t = ( s2_x * (y1 - y3) - s2_y * (x1 - x3)) / (-s2_x * s1_y + s1_x * s2_y)

    if  s >= 0 and s <= 1 and t >= 0 and t <= 1:
        x_int= x1 + (t * s1_x)
        y_int= y1 + (t * s1_y)
        Pint=Point(x_int, y_int)
        
        return Pint
    else:
        return False
 
def graph (database):
    vertex=set()
    intersect=set()
    edge=set()
    for i in range(len(database)):
        for j in range(i+1,len(database)):
            for seg1 in database.values()[i].sections:
                for seg2 in database.values()[j].sections:
                    
                    if  seg1.slope == seg2.slope:
                        segshort,seglong=seg1,seg2
                        if(seg1.length>seg2.length):
                            segshort,seglong=seg2,seg1
                        eps=(segshort.src,segshort.dst)
                        epl=(seglong.src,seglong.dst)    
                        for k in range(2):
                            if isPonI(eps[k],seglong):
                                vertex=vertex.union(set([seg1.src,seg1.dst,seg2.src,seg2.dst]))
                                intersect.add(eps[k])
                                if repr(Line(seglong.src,eps[k])) not in set2Sset(edge):
                                    edge.add(Line(eps[k],seglong.src))
                                if repr(Line(seglong.dst,eps[k])) not in set2Sset(edge):
                                    edge.add(Line(eps[k],seglong.dst))     
                            if isPonI(epl[k],segshort):
                                intersect.add(epl[k])
                                if repr(Line(segshort.src,epl[k])) not in set2Sset(edge):
                                    edge.add(Line(epl[k],segshort.src))
                                if repr(Line(segshort.dst,epl[k])) not in set2Sset(edge):
                                    edge.add(Line(epl[k],segshort.dst))                  

                                           
                               
                    elif line_intersect(seg1,seg2)!=False :
                        Pint=line_intersect(seg1,seg2)
                        intersect.add(Pint)
                        if repr(Line(seg1.src,Pint)) not in set2Sset(edge):
                            edge.add(Line(Pint,seg1.src))
                        if repr(Line(seg1.dst,Pint)) not in set2Sset(edge):
                            edge.add(Line(Pint,seg1.dst))
                        if repr(Line(seg2.src,Pint)) not in set2Sset(edge):
                            edge.add(Line(Pint,seg2.src))
                        if repr(Line(seg2.dst,Pint)) not in set2Sset(edge):
                            edge.add(Line(Pint,seg2.dst))                       
                        vertex=vertex.union(set([Pint,seg1.src,seg1.dst,seg2.src,seg2.dst]))
                    else:
                        pass
        
    return [vertex,intersect,edge]

def set2dict(Set):
    Dict,i={},0
    for x in Set:
        i=i+1
        Dict[str(i)]=x
    return Dict

def set2Sset(Pset):
    Sset=set()
    i=0
    for x in Pset:
        i=i+1
        Sset.add(repr(x))
    return Sset

def Sset2Pset(Sset):
    Pset=set()
    i=0
    for x in Sset:
        cama=x.find(',') 
        Pset.add(Point(float(x[1:cama]),float(x[cama+1:-1])))
    return Pset

def Sset2Lset (Sset):
    Lset=set()
    i=0
    for x in Sset:
       cama=x.find(',')
       rcama=x.rfind(',')
       arow=x.find('>')
       pclose=x.find(')')
       Lset.add(Line(Point(float(x[2:cama]),float(x[cama+1:pclose])), Point(float(x[pclose+5:rcama]),float(x[rcama+1:-2]))))
    return Lset

def isPonI (P,I):
    ptos=Line(P,I.src)
    ptod=Line(P,I.dst)
    if repr(P)==repr(I.src) or repr(P)==repr(I.dst):
        return True
    elif ptos.slope == 'inf' or ptod.slope == 'inf' or I.slope== 'inf':
        if I.slope == ptos.slope:
        
            if P.y>= min(I.src.y, I.dst.y) and P.y<= max(I.src.y, I.dst.y):
                return True
            else:
                return False                                                                            
    elif ptos.slope == ptod.slope and ptos.slope == I.slope:
        
        if P.x>= min(I.src.x, I.dst.x) and P.x<= max(I.src.x, I.dst.x):
            return True
        else:
            return False
        

def edgefind (intersect,edge):
    t1=set()
    t2=set()
    flag=0
    for e in edge:
        if e.length==0.0:
            t1.add(e)
        else:
          
           for i in intersect:
               itos=Line(i,e.src)
               itod=Line(i,e.dst)
               stoi=Line(e.src,i)
               dtoi=Line(e.dst,i)    
               if isPonI(i,e) and repr(i)!=repr(e.src) and repr(i)!=repr(e.dst):
                    flag=1 
                    t1.add(e)
                    if repr(stoi) not in set2Sset(edge.union(t2)):
                        t2.add(itos)
                    if repr(dtoi) not in set2Sset(edge.union(t2)):
                        t2.add(itod)
                       
    edge=edge.difference(t1)
    edge=edge.union(t2)
    return [edge,flag]
                                        
                                        
def output(V,E):
    Vout=[]
    Eout=[]
    count=0
    sys.stdout.write('V = {\n') 
    for i in range(len(V)):
        Vout.append((str(i+1)+':  ')[0:3]+V[i])
        sys.stdout.write(Vout[-1]+'\n')
    sys.stdout.write('}\n')
    sys.stdout.write('E = {\n')  
    for e in E:
        count=count+1
        a=V.index(repr(e.src))
        b=V.index(repr(e.dst))
        if count == len(E):
            Eout.append('<'+str(a+1)+','+str(b+1)+'>')
        else:
            Eout.append('<'+str(a+1)+','+str(b+1)+'>,')
        sys.stdout.write(Eout[-1]+'\n')
    sys.stdout.write('}\n')    
  
    
                        
                
                         


def main():

    

               
    StreetDatabase={}
    Vertex=set()
    V={}


                       
    while(True):
       
        line = sys.stdin.readline()
        if line== '':
            break;
        try:
            [com,info]=parse_line(line)
            if info != []:                
                tempstreet=Street(info)
            if com == 'a':
                if tempstreet.name in StreetDatabase.keys():
                    raise Exception('Error: street already exists')
                elif len(info)==1:
                    raise Exception('Error: no valid coordinates')
                StreetDatabase[tempstreet.name]=tempstreet

            elif com == 'r':
                if tempstreet.name not in StreetDatabase.keys():
                    raise Exception('Error: street does not exist')
                elif len(info)!=1:
                    raise Exception("Error: no coordinates must follow 'r' command") 
                del StreetDatabase[tempstreet.name]

            elif com == 'c':
                if tempstreet.name not in StreetDatabase.keys():
                    raise Exception('Error: street does not exist')
                elif len(info)==1:
                    raise Exception('Error: no valid coordinates')  
                StreetDatabase[tempstreet.name]=tempstreet
               
            elif com == 'g':
                if len(info)!=0:
                    raise Exception("Error: nothing must follow 'g' command")   
                [Vertex,Intersect,Edge]=graph(StreetDatabase)
                Vertex=set2Sset(Vertex)
                V=list(Vertex)
                
                [E,flag]=edgefind(Intersect,Edge)
                while flag == 1:
                    [E,flag]=edgefind(Intersect,E)
                output(V,Sset2Lset(set2Sset(E)))
               
        except Exception as ex:
               
            sys.stderr.write(str(ex) + '\n')
                
    sys.exit(0)
   
if __name__ == '__main__':
    main()
