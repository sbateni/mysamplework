#ifndef __LIST_H__
#define __LIST_H__

/**
 * Opaque reference to a list node.
 *
 * TODO Define the concrete list structure in your implementation
 */
typedef struct _List List;

/**
 * Create an empty list, returning the head pointer.
 *
 * @return The head pointer to the new list or NULL if the
 *         list can't be created
 */
List* CreateList();

/**
 * Destroys an existing list.
 *
 * The list does not have to be empty.
 *
 * @param head The head pointer to the list
 */
void DestroyList(List *list);

/**
 * Get the number of elements in the list
 *
 * @note This may be called at any time, from any thread.
 *
 * @param list The list to query
 *
 * @return The number of elements in the list
 */
unsigned int getSize(List *list);

/**
 * Adds a value to the list.
 *
 * @note This may be called at any time, from any thread.
 *
 * @param list The list to modify
 * @param value The value to add
 */
void addValue(List *list, unsigned int value);

/**
 * Returns a value from the list, removing it from the list
 * at the same time.
 *
 * Values are returned in the same order they were added via calls
 * to 'addValue'
 *
 * @note This may be called at any time, from any thread.
 *
 * @param list The list to query
 * @param value A pointer to store the value retrieved from the list
 *
 * @return 0    On success
 * @return -1   If there is no value to retrieve from the list.
 *              I.e: The list is empty.
 */
int consumeValue(List *list, unsigned int *value);

/**
 * Computes the intersection of two lists.
 *
 * A new list containing the intersection of the
 * two arguments is returned.
 *
 * @param list1 The first list in the intersection
 * @param list2 The second list in the interesection
 *
 * @return A new list that is the intersection of list1 and list2
 */
List* intersectList(List *list1, List *list2);

/**
 * Computes the union of two lists.
 *
 * A new list containing the union of the
 * two arguments is returned.
 *
 * @param list1 The first list in the union
 * @param list2 The second list in the union
 *
 * @return A new list that is the union of list1 and list2
 */
List* unionList(List *list1, List *list2);

/**
 * Create a list of unique entries
 *
 * A new list containing a single entry for
 * each unique value in the specified parameter list.
 *
 * @param list The source list used to compose the set
 *
 * @return A new list that is the set created from the specified list
 */
List* uniqueList(List *list);


#endif
