/*
 * main.hpp
 *
 *  Created on: Oct 16, 2017
 *      Author: ehsan
 */

#ifndef DEFS_HPP_
#define DEFS_HPP_
#include <iostream>
#include <sstream>




// Node class
class Node {
    int data;
    Node* next;

  public:
    Node() {};
    void SetData(int aData) { data = aData; };
    void SetNext(Node* aNext) { next = aNext; };
    int Data() { return data; };
    Node* Next() { return next; };
};

// List class
class List {
    Node *head;
  public:
    List() { head = NULL; };

Node *get_head() {return head;}



bool IsinList(int data) {
	  // Create a temp pointer
	    Node *tmp = head;

	    // No nodes
	    if ( tmp == NULL )
	    return 0;



	    // Parse thru the nodes

	    do {
	        if ( tmp->Data() == data ) return 1;
	       	tmp = tmp->Next();
	    } while ( tmp != NULL );
	    return 0;
}
	    void Print() {


	        Node *tmp = head;


	        if ( tmp == NULL ) {
	     //   std::cout << "EMPTY" << "\n";
	        return;
	        }


	        if ( tmp->Next() == NULL ) {
	      //  std::cout << tmp->Data();
	      //  std::cout << " --> ";
	      //  std::cout << "NULL" << "\n";
	        }
	        else {

	        do {
	            std::cout << tmp->Data();

	            tmp = tmp->Next();
	            if (tmp != NULL)  std::cout << "-";
	        }
	        while ( tmp != NULL );

	        std::cout <<"\n";
	        }
	    }

/**
 * Append a node to the linked list
 */
void Append(int data) {


	    Node* newNode = new Node();
	    newNode->SetData(data);
	    newNode->SetNext(NULL);


	    Node *tmp = head;

	    if ( tmp != NULL ) {

	    while ( tmp->Next() != NULL ) {
	        tmp = tmp->Next();

	    }


	    tmp->SetNext(newNode);
	    }
	    else {

	    head = newNode;
	    }
	}
//Inserts a node to the begining of a list
void Insert(int data) {
       Node *entry = new Node();
       Node *temp=head;
	   head=entry;
	   head->SetData(data);
       head->SetNext(temp);


   }


/**
 * Delete a node from the list
 */
void Delete(int data) {


    Node *tmp = head;

    // No nodes
    if ( tmp == NULL )
    return;


    if ( tmp->Next() == NULL ) {
    delete tmp;
    head = NULL;
    }
    else {

    Node *prev=head;
    do {
        if ( tmp->Data() == data ) {head=tmp->Next();break;}

        prev = tmp;
        tmp = tmp->Next();
    } while ( tmp != NULL );



    prev->SetNext(tmp->Next());


    delete tmp;
    }
}
};

bool parse_line (const std::string& line,
                 char& cmd, List* arg, std::string& err_msg,char&state);



#endif /* DEFS_HPP_ */
