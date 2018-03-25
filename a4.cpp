//============================================================================
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

// this function solves the minimal Vertex Cover problem using minisat and prints the result to standard output	  
void vertexcover(int V,std::vector<int> edge){
    std::unique_ptr<Minisat::Solver> solver(new Minisat::Solver());
    Minisat::vec<Minisat::Lit> edgeclause;
    std::vector<int> graphcover;
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
 graphcover.clear();
 for (int k=0;k<kmin;k++){
  for(int v=0;v<V;v++){
   if (solver->modelValue(vars[k][v])== Minisat::l_True){
   graphcover.push_back(v);
   }
  }
 }
 

 std::sort(graphcover.begin(),graphcover.end());
 for (int v : graphcover) {
        std::cout << v << " ";
 }
  std::cout <<std::endl;
  solver.reset ();
  delete[] vars;

 return;

 }
 solver.reset (new Minisat::Solver());
 delete[] vars;    
}        

}

int main() {
         
          
          

	 std::string line;

	 char cmd;
	 int V,first,second;
	 
	 Node *tmp;
	 
	 
	 List *argE=new List;
         std::vector<int> edge;
	 char state='V';

    

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
                
                vertexcover(V,edge);
	    	break;

	    

        }
        }
	    else {
	               std::cerr << "Error: " << err_msg <<std::endl;
	           }
	}




	return 0;
}
