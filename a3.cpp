// based on the example from
// https://stackoverflow.com/questions/13041416/redirect-stdout-of-two-processes-to-another-processs-stdin-in-linux-c
#include<vector>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include<iostream>

/// Entry point of process A
int procA(void) {
    // Process A writing to C
   // std::cerr << "A is running"<<std::endl;

    char* argv[2];
    argv[0]=(char*)"a1.py";
    argv[1]=NULL;
    execv ("a1.py", argv);

   
   return 0;
 
}

/// Entry point of process B
int procB(void) {
    // Process B writing to C
   // std::cerr << "B is running"<<std::endl;

    while (!std::cin.eof()) {
        // read a line of input until EOL and store in a string
        std::string line;
        std::getline(std::cin, line);
        if (line.size () > 0)
            std::cout << line << std::endl;
    }
 //   std::cout << "[B] saw EOF" << std::endl;
    return 0;
}

/// Entry point of process C
int procC(void) {
    // Process C reading from both A and B
   //  std::cerr << "C is running"<<std::endl;

    char* argv[2];
    argv[0]=(char*)"a2";
    argv[1]=NULL;
    execv ("a2", argv);

   
    return 0;
}

int procD(char** arg) {
   // writing to A
   // std::cout<<"starting procD(rgen)"<<std::endl;
   //  std::cerr << "D is running "<<std::endl;

     execv ("rgen", arg);     
                
                         
                                                        
    return 0;
                                                            
}                                                          
int main(int argc,char **argv)
{

             
     
   // create a pipe
     std::vector<pid_t> kids;
    // create a pipe
     int ABtoC[2],DtoA[2];
     pipe(ABtoC);
     pipe(DtoA); 


    pid_t child_pid;
    child_pid = fork ();
    if (child_pid == 0)
    {
        // redirect stdout to the pipe
        dup2(ABtoC[1], STDOUT_FILENO);
        close(ABtoC[0]);
        close(ABtoC[1]);     // Close this too!
        // redirect stdin from another pipe
        dup2(DtoA[0], STDIN_FILENO);
        close(DtoA[1]); 
        // start process A
        return procA();
    }
    else if (child_pid < 0) {
        std::cerr << "Error: could not fork"<<std::endl;
        return 1;
    }

    kids.push_back(child_pid);

    child_pid = fork();
    if (child_pid == 0)
    {
        // redirect stdin from the pipe
        dup2(ABtoC[0], STDIN_FILENO);
        close(ABtoC[1]);
        close(ABtoC[0]);

        // start process C
        return procC();
    }
    else if (child_pid < 0) {
        std::cerr << "Error: could not fork"<<std::endl;
        return 1;
    }

    kids.push_back(child_pid);
    

    // redirect stdout to the pipe
    
    
    child_pid= fork();
    if (child_pid == 0)
    {
      dup2(DtoA[1], STDOUT_FILENO);
        close(DtoA[0]);
        close(DtoA[1]);

        return procD(argv);
    }
    else if (child_pid < 0) {
        std::cerr << "Error: could not fork"<<std::endl;
        return 1;
    }
    kids.push_back(child_pid);
    child_pid=0;
    dup2(ABtoC[1], STDOUT_FILENO);
    close(ABtoC[0]);
    close(ABtoC[1]);

// start process B
    int res =  procB();

    // send kill signal to all children
    for (pid_t k : kids) {
        int status;
        kill (k, SIGTERM);
        waitpid(k, &status, 0);
    }

  // exit with return code of process B
   return res;

}
