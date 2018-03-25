//a============================================================================
// Name        : hw.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================


#include "defs.hpp"
#include <vector>
#include <memory>
// defines Var and Lit
#include "minisat/core/SolverTypes.h"
// // defines Solver
#include "minisat/core/Solver.h"
#include <algorithm>
#include <pthread.h>
#include <time.h>

#define handle_error(msg) \
               do { perror(msg); exit(EXIT_FAILURE); } while (0)
#define handle_error_en(en, msg) \
               do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

//global variables
typedef struct solver_arg {
    int V;
    std::vector <int> edge;
    } solver_arg;

 std::vector<int> graphcover1,graphcover2,graphcover3;
 
 struct timespec ts1,ts2,ts3;
// a function for finding the most frequent element of a vector 
int most_incident(std::vector<int> edge){
 int max=0,mostvalue=edge[0];
 int i,co;
    for(i=0;i<(int)edge.size();i++)
    {
        co = (int)std::count(edge.begin(), edge.end(), edge[i]);
        if(co > max)
        {       max = co;
                mostvalue = edge[i];
        }
    } 
return mostvalue;
}
// this function removes edges connected to a vertice
void remove_edge(std::vector<int> &edge,int v){
  std::vector<int>::iterator it=find(edge.begin(),edge.end(),v);
  while(it!= edge.end())
    {
         
       int i=it-edge.begin();
       if (i%2==0)
       edge.erase(it,it+2);
       else
       edge.erase(it-1,it+1);
       
     
       it=find(edge.begin(),edge.end(),v);

    }
}  
// CNF-SAT method
void *CNFSATVC(void *arg){
    graphcover1.clear();
    clockid_t cid;
    solver_arg *arg1 = (solver_arg *)arg;
    int V=arg1->V;
    std::vector<int> edge=arg1->edge;
    std::unique_ptr<Minisat::Solver> solver(new Minisat::Solver());
    Minisat::vec<Minisat::Lit> edgeclause;
    //std::vector<int> graphcover;
    int esize=edge.size()/2;
 for (int kmin=1;kmin<=V;kmin++){
     
  
    Minisat::vec<Minisat::Lit> *vars=new Minisat::vec<Minisat::Lit>[kmin];
    for(int k=0;k<kmin;k++){
     for (int v=0;v<V;v++){
         Minisat::Lit x=Minisat::mkLit(solver->newVar());
         vars[k].push(x);
     }
     solver->addClause(vars[k]);
    }
    

    for (int m=0;m<V;m++){
     for (int p=0;p<kmin;p++){
      for (int q=p+1;q<kmin;q++){
       solver->addClause(~vars[p][m],~vars[q][m]);
      }
     }
    } 
   

   for (int m=0;m<kmin;m++){
     for (int p=0;p<V;p++){
      for (int q=p+1;q<V;q++){
       solver->addClause(~vars[m][p],~vars[m][q]);
      }
     }
   }
  

  for (int e=0;e<esize;e++){
    
    edgeclause.clear();
    for (int k=0;k<kmin;k++){
    edgeclause.push(vars[k][edge[2*e]]);
    edgeclause.push(vars[k][edge[2*e+1]]);
    }
    solver->addClause(edgeclause);
  }
 

 if( solver->solve()){
 graphcover1.clear();
 for (int k=0;k<kmin;k++){
  for(int v=0;v<V;v++){
   if (solver->modelValue(vars[k][v])== Minisat::l_True){
   graphcover1.push_back(v);
   }
  }
 }
 

 std::sort(graphcover1.begin(),graphcover1.end());

  
  solver.reset ();
  delete[] vars;
  
//running time computation
int s = pthread_getcpuclockid(pthread_self(), &cid);
                if (s != 0)
                handle_error_en(s, "pthread_getcpuclockid");
                if (clock_gettime(cid, &ts1) == -1)
                handle_error("clock_gettime");
 
pthread_exit(NULL);

 }
 solver.reset (new Minisat::Solver());
 delete[] vars;    
}        

}

//APPROX-1 method
void *APPROXVC1(void *arg){
    clockid_t cid;
    graphcover2.clear();
    solver_arg *arg1 = (solver_arg *)arg;
    //int V=arg1->V;
    std::vector<int> edge=arg1->edge;
    while(!edge.empty()){        
   
      int v=most_incident(edge);
     // std::cout <<"mi="<<v<<std::endl;
      graphcover2.push_back(v);
      remove_edge(edge,v); 
      
     }
std::sort(graphcover2.begin(),graphcover2.end());
 int s = pthread_getcpuclockid(pthread_self(), &cid);
                if (s != 0)
                handle_error_en(s, "pthread_getcpuclockid");
   if (clock_gettime(cid, &ts2) == -1)
               handle_error("clock_gettime");

pthread_exit(NULL);
}

//APPROX-2 method
void *APPROXVC2(void *arg){
    clockid_t cid; 
    graphcover3.clear();
    solver_arg *arg1 = (solver_arg *)arg;
    //int V=arg1->V;
    std::vector<int> edge=arg1->edge;
    while(!edge.empty()){
      int esize=edge.size()/2;
      int v=rand()%esize;
      int temp1=edge[2*v],temp2=edge[2*v+1]; 
      graphcover3.push_back(temp1);
      graphcover3.push_back(temp2);
               
      remove_edge(edge,temp1);
      remove_edge(edge,temp2);
     }
std::sort(graphcover3.begin(),graphcover3.end());

  int s = pthread_getcpuclockid(pthread_self(), &cid);
                if (s != 0)
                handle_error_en(s, "pthread_getcpuclockid");
   if (clock_gettime(cid, &ts3) == -1)
               handle_error("clock_gettime");

pthread_exit(NULL);
}

//this function ouputs the rsults of three methods
void output_result(){
              std::cout <<"CNF-SAT-VC: ";
              for (int v :graphcover1) {
                 std::cout << v;
                 if(v!=graphcover1[graphcover1.size()-1])
                    std::cout << ",";

              }
              std::cout <<std::endl;
              
              std::cout <<"APPROX-VC-1: ";
              for (int v :graphcover2) {
                  std::cout << v ;
                  if(v!=graphcover2[graphcover2.size()-1])
                     std::cout << ",";
 
              }
              std::cout <<std::endl;
              
              std::cout <<"APPROX-VC-2: "; 
              for (int v :graphcover3) {
                   std::cout << v ;
                   if(v!=graphcover3[graphcover3.size()-1])
                      std::cout << ",";
  
              }
              std::cout <<std::endl;
}

//IO thread
void * IO(void *arg) {
         
          
          

	 std::string line;

	 char cmd;
	 int V,first,second;
	 
	 Node *tmp;
	 
	 
	 List *argE=new List;
         std::vector<int> edge;
	 char state='V';
         solver_arg solverarg;
         
         pthread_t solver[3];
           

	while (true) {

	std::getline(std::cin, line);

        if(std::cin.eof()){break;}
        
	    List *arg=new List;

	    std::string err_msg;
	    if (parse_line(line, cmd, arg, err_msg,state)) {
	    switch (cmd) {

	    case 'V':


	    	V=arg->get_head()->Data();
	    	delete arg;
	         state='E';
	        break;

	    case 'E':
                
                edge.clear();
	    	delete argE;
	    	argE=new List;
	    	tmp=arg->get_head();
	    	if(!tmp){state='V'; break;}
	    	do{

	    	    first=tmp->Data();
	    	    argE->Append(first);
	    	    tmp=tmp->Next();
	    		second=tmp->Data();
	    		argE->Append(second);
	    		tmp=tmp->Next();
	    		 if (first>=V || second>=V){
	    			 std::cerr << "Error: " << "indice exceeds vertice numbers" <<std::endl;
	    			 delete arg;
	    			 break;
	    		 }

                 if (first!=second){
                         edge.push_back(first);
                         edge.push_back(second); 
	    		
                 }
	    		 arg->Delete(first);
	    		 arg->Delete(second);
	    		 state='V';
	    }while (tmp);
                solverarg.V=V;solverarg.edge=edge;
                
                //creating threads for each method
                pthread_create(&solver[0],NULL,CNFSATVC,(void *)&solverarg);
                pthread_create(&solver[1],NULL,APPROXVC1,(void *)&solverarg);
                pthread_create(&solver[2],NULL,APPROXVC2,(void *)&solverarg);
                
                               


                //joining threads
                pthread_join(solver[0], NULL);
                pthread_join(solver[1], NULL);
                pthread_join(solver[2], NULL);
             
                  
               //results are ready   
                output_result();
               

	    	break;

	    

        }
        }
	    else {
	               std::cerr << "Error: " << err_msg <<std::endl;
	           }
	}

        


	pthread_exit(NULL);
}
int main(){
//create IO thread
 pthread_t IO_thread;
 pthread_create(&IO_thread, NULL,IO,NULL);
//join IO thread to main thread
 pthread_join(IO_thread, NULL);
  

return 0;
}  

