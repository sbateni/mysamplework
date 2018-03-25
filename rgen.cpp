// an example of reading random numbers from /dev/urandom
// https://stackoverflow.com/questions/35726331/c-extracting-random-numbers-from-dev-urandom
#include <iostream>
#include <fstream>
#include <math.h>
#include <unistd.h>

int rand_range(int start,int end){
//generates an integer between 0 and range-1
//  // open /dev/urandom to read
      std::ifstream urandom("/dev/urandom");

          // check that it did not fail
       if (urandom.fail()) {
       // std::cerr << "Error: cannot open /dev/urandom\n";
         return -1;
       }
   if (start>end){
   int tmp=end;
   end=start;
   start=tmp;
  }
   else if (start == end) {return start;}
   unsigned int range=end-start+1;                               
   unsigned int num;
   urandom.read((char*)&num, sizeof(int));
   urandom.close();
   int remain=(int)(num%range);

//   std::cout<<"remain="<<remain;
//   std::cout<<"range="<<range;
   
   return remain+start;
}

int sign(int x){
if( x>0) return 1;
else if (x<0) return -1;
return 0;
}
class Point{
    //A point in a two dimensional space"""
    
        int x ;
        int y ;
   public:
   Point(){}
   Point(int X,int Y):x(X),y(Y){}
   friend class Line;
   friend class Street;
    bool operator ==(const Point& rhs){
         return this->x==rhs.x && this->y==rhs.y;
        }

};
    
        


class Line{
    //A line between two points"""
        
        Point src; 
        Point dst;
        
        public:
        double length;
        Line(){}
        Line(Point s, Point d):src(s),dst(d){length=pow((dst.y-src.y),2)+pow((dst.x-src.x),2);}
        friend class Street;
        bool operator ==(const Line& rhs){
         return this->src==rhs.src && this->dst==rhs.dst;
        }
};
       
    
       
    
class Street{
    //Contains each street name, coordinates, and sections"""
        public:
        char name;
        int size;
	Line *segment;

        Street(){size=0;}
        void SetSize(int s){
        size=s;
        segment=new Line[size];
        }      
        void add(){
        std::cout<<"a \""<<name<<"\" ";
        for (int i=0;i<size;i++){
          if(i==0){
          std::cout<<"("<<std::to_string(segment[i].src.x)<<","<<std::to_string(segment[i].src.y)<<") ";
          }
          std::cout<<"("<<std::to_string(segment[i].dst.x)<<","<<std::to_string(segment[i].dst.y)<<") ";
        }
         std::cout<<std::endl;
        }

       void remove(){
       std::cout<<"r \""<<name<<"\""<<std::endl;
       }  
        
}; 
        
bool pars_segment(Line seg,Street *database,int snum,int lsnum){
if(seg.length ==0) {
return false;
}
for (int i=0;i<=snum;i++){
 int segnum=database[i].size;
 if (i==snum){segnum=lsnum;}
  for (int j=0;j<segnum;j++){
   if(database[i].segment[j]==seg){
    return false;
   }
  }
}
return true;
}              
        

int main(int argc,char** argv) {

int num_s=0;
int max_num_s=10;
int max_num_seg=5;
int max_cord=20;
int max_wait=5;



int clarg;
opterr=0; 
int value; 
 while ((clarg = getopt (argc, argv, "s:n:l:c:")) != -1)
        switch (clarg)
        {
        case 's':
            value=atoi(optarg);
            if (value>=2){
            max_num_s=value;
            break;
            }
            std::cerr <<"Error: invalid value for argument 's'"<<std::endl;
            return 1;


        case 'n':
            value=atoi(optarg);
            if (value>=1){
            max_num_seg=value;
            break;
            }
            std::cerr <<"Error: invalid value for argument 'n'"<<std::endl;
            return 1;
           
             
         case 'l':
            value=atoi(optarg);
            if (value>=5){
            max_wait=value;
            break;
            }
            std::cerr <<"Error: invalid value for argument 'l'"<<std::endl;
            return 1;

         case 'c':
            value=atoi(optarg);
            if (value>=1){
            max_cord=value;
            break;
            }
            std::cerr <<"Error: invalid value for argument 'c'"<<std::endl;
            return 1;
         case '?':
            
                std::cerr << "Error: invalid command line options" << std::endl;
                          
            return 1;     

         }




//std::cerr<<"max_num_s="<<max_num_s<<std::endl;
//std::cerr<<"max_num_seg="<<max_num_seg<<std::endl;
//std::cerr<<"max_wait="<<max_wait<<std::endl;
//std::cerr<<"max_cord="<<max_cord<<std::endl;


char names[]="abcdefghijklmnopqrstuvwxyz";
int cordx,cordy,cordx_new,cordy_new;
int xlimit,ylimit;
int limitx,limity;
int w=rand_range(5,max_wait);
int errorcount=0;

//Line newseg;
//for(int i=0;i<10;i++){
//    std::cout<<rand_range(0,20)<<std::endl;
//}

while(true){




//adding new streets


num_s=rand_range(2,max_num_s);


Street *StreetDatabase=new Street[num_s];


for (int i=0;i<num_s;i++){
StreetDatabase[i].name=names[i];

int num_seg=rand_range(1,max_num_seg);
StreetDatabase[i].SetSize(num_seg);

cordx=rand_range(-max_cord,max_cord);
cordy=rand_range(-max_cord,max_cord);
Point firstpoint(cordx,cordy);
if (cordx>=0 && cordy>=0){ xlimit=-max_cord; ylimit=-max_cord;}
else if (cordx>=0 && cordy<=0){ xlimit=-max_cord; ylimit=max_cord;}
else if (cordx<=0 && cordy>=0){ xlimit=max_cord; ylimit=-max_cord;}
else if (cordx<=0 && cordy<=0){ xlimit=max_cord; ylimit=max_cord;}
for (int j=0;j<num_seg;j++){
Point oldpoint(cordx,cordy);
limitx=cordx+(xlimit-cordx)/(num_seg-j);
limity=cordy+(ylimit-cordy)/(num_seg-j);
cordx_new=rand_range(limitx,cordx+sign(xlimit));
cordy_new=rand_range(limity,cordy+sign(ylimit));
Point newpoint(cordx_new,cordy_new);

//std::cout<<"newpoint"<<cordx_new<<","<<cordy_new<<std::endl;
Line newseg(oldpoint,newpoint);

while (!pars_segment(newseg,StreetDatabase,i,j)){
cordx_new=rand_range(limitx,cordx+sign(xlimit));
cordy_new=rand_range(limity,cordy+sign(ylimit));
Point newpoint(cordx_new,cordy_new);
errorcount=errorcount+1;
  if(errorcount == 25){
           std::cerr <<"Error: failed to generate valid input for 25 simultaneous attempts"<<std::endl;
           return 1;
  }
}


StreetDatabase[i].segment[j]=Line(oldpoint,newpoint);


cordx=cordx_new;cordy=cordy_new;
}
}
for (int i=0;i<num_s;i++){
 StreetDatabase[i].add();
}
//issuing g commanf
std::cout<<"g"<<std::endl;

//wating for w seconds
sleep(w);

//removing database
if(num_s!=0){
 for (int r=0;r<num_s;r++){
 StreetDatabase[r].remove();
 }
 delete[] StreetDatabase;
 }

}
return 0;



}                       
