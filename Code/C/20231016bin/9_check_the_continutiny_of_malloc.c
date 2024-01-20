#include <stdio.h>
#include<stdlib.h>
int main()
{   
    void *q;
    void *p = 0;
    p = malloc(8);
    q = malloc(8);
    printf("%p\n",p);
    printf("%p\n",q);
    system("pause"); 
    return 0;
}
