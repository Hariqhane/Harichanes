#include <stdio.h>
int main()
{   
    int ai[]={1,5,3,4,5,6,7,8,};
    int *q1 =&ai[1];
    int *q = ai;
    printf("q1-q=%d\n",q1-q);//表示：两个地址之间的元素差；
    printf("*q1-*q=%d\n",*q1-*q);
    getchar(); 
    return 0;
}
