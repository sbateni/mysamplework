#include <stdio.h>

//implementing Hoare partition scheme
int partition(int*arr , int left, int right) {
   int pivot = arr[left];
   
   
   --left;
   ++right;

   while(1) {
      while(arr[++left] < pivot) {
      }
		
      while(arr[--right] > pivot) {
      }

      if(left >= right) {
          return right;
      }
             
 
         int temp = arr[left];
         arr[left] = arr[right];
         arr[right] = temp;     
      
   }
	
  
}
void quickSort(int *arr, int left, int right) {
   if(right-left <10) {    //for partitions with length no more than 10 do the insertion sort
      insertionSort(arr,left,right);
      return;   
   }
   
      int p = partition(arr, left, right);
              
      quickSort(arr,left,p);
      quickSort(arr,p+1,right);
           
         
} 
void insertionSort(int *arr, int left, int right)
{
   int i, x, j;
   for (i =left+1; i < right+1; i++)
   {
       x = arr[i];
       j = i-1;
while (j >= left && arr[j] > x)
       {
           arr[j+1] = arr[j];
           j = j-1;
       }
       arr[j+1] = x;
   }
}


void sort(int* arr_ptr, int arr_sz)
{
 quickSort(arr_ptr,0,arr_sz-1);
}




   
