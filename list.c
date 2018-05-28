/*
 ============================================================================
 Name        : list.c
 Author      : Ehsan Bateni
 Version     :
 Copyright   : Your copyright notice
 Description : List.c in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include "list.h"

/*Concrete list structure based on singly linked-list data type with a nested structure for node */
typedef struct node {
	unsigned int data;
	struct node* next;
} node;

struct _List {
	node* head;
	node* tail;
	pthread_mutex_t mtx; /* Mutex for making each list object thread-safe when necessary*/
};

/****************************************************
 --FUNCTION	::	CreateList
 --ACTION	::	Create an empty list, returning the head pointer
 --@param	::	NONE
 --@return	::	The head pointer to the new list or NULL if the list can't be created
 --***************************************************/
List* CreateList() {
	List* new_list = (List*) malloc(sizeof(List));
	if (new_list == NULL) {
		printf("Error: List could not be created\n");
		return NULL;
	}
	/* Initializing an empty list */
	new_list->head = NULL;
	new_list->tail = NULL;
	new_list->mtx = PTHREAD_MUTEX_INITIALIZER;
	printf("A new empty list created\n");
	return new_list;
}

/****************************************************
 --FUNCTION	::	find
 --ACTION	::	Checks if an element exists in the list
 --@param	::	list The list to be searched
 --@param	::	value The value to be searched for in the list
 --@return	::	True if the value exists in the list and False otherwise (including an empty list case)
 --***************************************************/
bool find(List *list, unsigned int value) {
	node* iterator = list->head;
	while (iterator != NULL) {
		if (iterator->data == value)
			return true;
		iterator = iterator->next;
	}
	return false;
}

/****************************************************
 --FUNCTION	::	DestroyList
 --ACTION	::	Destroys an existing list
 --@param	::	list The list to query
 --@return	::	NONE
 --***************************************************/
void DestroyList(List *list) {
	node* iterator = list->head;
	node* tmp;
	/* Emptying the list */
	while (iterator != NULL) {
		tmp = iterator->next;
		free(iterator);
		iterator = tmp;
	}

	free(list); /* Destroying the empty list */
	printf("List destroyed \n");
}

/****************************************************
 --FUNCTION	::	getSize
 --ACTION	::	Get the number of elements in the list
 --@param	::	list The list to query
 --@return	::	The number of elements in the list
 --***************************************************/
unsigned int getSize(List *list) {
	pthread_mutex_lock(&list->mtx); /* Making the function thread-safe in list structure level */
	node *iterator = list->head;
	unsigned int count = 0;
	while (iterator != NULL) {
		count++;
		iterator = iterator->next;
	}
	pthread_mutex_unlock(&list->mtx);
	return count;
}

/****************************************************
 --FUNCTION	::	addValue
 --ACTION	::	Adds a value to the end of the list
 --@param	::	list The list to query
 --@param   ::  value The value to add
 --@return	::	NONE
 --***************************************************/
void addValue(List *list, unsigned int value) {

	node* new_node = (node*) malloc(sizeof(node)); /* Creating a new node for new value */
	new_node->data = value;
	new_node->next = NULL;

	pthread_mutex_lock(&list->mtx); /* Making the function thread-safe in list structure level */
	/* If it's the first element it's both the head and tail of the list */
	if (list->head == NULL) {
		list->head = new_node;
		list->tail = new_node;
		/* adding the new element to the end of the list */
	} else {
		list->tail->next = new_node;
		list->tail = new_node;
	}
	pthread_mutex_unlock(&list->mtx);
}

/****************************************************
 --FUNCTION	::	consumeValue
 --ACTION	::	Returns a value from the head of the list, removing it from the list at the same time. Values are returned in the same order they were added via calls
 to 'addValue'
 --@param	::	list The list to query
 --@param   ::  A pointer to store the value retrieved from the list
 --@return 0::  On success
 --@return-1::  If there is no value to retrieve from the list.
 I.e: The list is empty.
 --***************************************************/
int consumeValue(List *list, unsigned int *value) {

	pthread_mutex_lock(&list->mtx); /* Making the function thread-safe in list structure level */
	node* front = list->head;
	if (front == NULL) {
		printf("Error: List is empty\n");
		return -1;
	}

	list->head = list->head->next;
	*value = front->data;
	free(front);
	pthread_mutex_unlock(&list->mtx);

	return 0;

}

/****************************************************
 --FUNCTION	::	intersectList
 --ACTION	::	Computes the intersection of two lists
 --@param   ::  list1 The first list in the intersection
 --@param   ::  list2 The second list in the intersection
 --@return	::	A new list that is the intersection of list1 and list2
 --***************************************************/
List* intersectList(List *list1, List *list2) {
	List* intersectlist = CreateList();
	node* iterator = list1->head;

	while (iterator != NULL) {
		/* If the element is on the second list and is not already in the intersection list, adds it to intersection list */
		if (!find(intersectlist, iterator->data) && find(list2, iterator->data))
			addValue(intersectlist, iterator->data);
		iterator = iterator->next;
	}
	return intersectlist;
}

/****************************************************
 --FUNCTION	::	unionList
 --ACTION	::	Create a list of unique entries
 --@param   ::  list1 The first list in the union
 --@param   ::  list2 The second list in the union
 --@return	::	A new list that is the union of list1 and list2
 --***************************************************/
List* unionList(List *list1, List *list2) {
	List* unionlist = CreateList();
	node* iterator[2] = { list1->head, list2->head };
	/* iterate through both lists simultanously */
	while (iterator[0] != NULL && iterator[1] != NULL) {
		if (!find(unionlist, iterator[0]->data))
			addValue(unionlist, iterator[0]->data);
		if (!find(unionlist, iterator[1]->data))
			addValue(unionlist, iterator[1]->data);
		iterator[0] = iterator[0]->next;
		iterator[1] = iterator[1]->next;
	}
	if (iterator[0] == NULL)
		iterator[0] = iterator[1];
	/* Iterate through the rest of longer list */
	while (iterator[0] != NULL) {
		if (!find(unionlist, iterator[0]->data))
			addValue(unionlist, iterator[0]->data);
		iterator[0] = iterator[0]->next;
	}

	return unionlist;
}

/****************************************************
 --FUNCTION	::	uniqueList
 --ACTION	::	Create a list of unique entries
 --@param	::	list The source list used to compose the set
 --@return	::	A new list that is the set created from the specified list
 --***************************************************/
List* uniqueList(List *list) {

	List* uniquelist = intersectList(list, list); /* Uniquelist is the intersection of a list with itself */

	return uniquelist;

}

/****************************************************
 --FUNCTION	::	printList
 --ACTION	::	Prints a list
 --@param	::	list The list to be printed
 --@return	::	NONE
 --***************************************************/
void printList(List *list) {
	node* iterator = list->head;

	while (iterator != NULL) {
		printf("%d->", iterator->data);
		iterator = iterator->next;
	}
	printf("NULL\n");
}

