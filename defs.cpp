/*
 * defs.cpp
 *
 *  Created on: Oct 18, 2017
 *      Author: ehsan
 */
#include "defs.hpp"
bool parse_line (const std::string& line,
                 char& cmd, List* arg, std::string& err_msg,char&state) {

    std::istringstream input(line);

    // remove whitespace
    ws(input);


    char ch;
    input >> ch;



    int num;
    cmd=ch;
    char sc;
    if (ch == 'E' || ch=='V' || ch=='s') {
    	switch(ch){
    	case 'V':
                
    		if (state=='E'){err_msg= "Expected E command"; return false;}


    		break;
    	case 'E':
                
                ws(input);
    		if (state!='E'){err_msg="Expected V or s command"; return false;}
    		while(true){
    		    		        		while(sc!='<' && !input.fail()){

    		    		        		input >> sc;
    		    		        		}
    		    		        		if(input.fail()){break;}
    		    		        		input >> num;
    		    		        		arg->Append(num);
    		    		        		input >>sc;
    		    		        		input >> num;
    		    		        		arg->Append(num);
    		    		        		input >>sc;

    		    		        	}
    		return true;
    		break;
    	case 's':
    		if (state !='s'){err_msg="Expected V or E command";return false;}
    		break;

    	}

        // remove whitespace


        while(true){
               input >> num;
               if(input.fail()){
               	break;
               }
               arg->Append(num);
        }
        if(!arg->get_head()){err_msg="no arg follows command";return false;}
        return true;
    }

    else {
        err_msg = "Unknown command";
        return false;
    }

}


