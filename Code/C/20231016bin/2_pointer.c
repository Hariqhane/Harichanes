#include <stdio.h>
void f(int *p);
void g(int k);
int main()
{
    int i = 5;
    g(i);
    printf("main中等于%d\n",i);
    f(&i);//此处i并不在函数f中
    printf("f后main中i等于%d\n",i);
    getchar();
    return 0;
}
void f(int *p)
{
    *p+=10;
    printf("f中i等于%d\n",*p);
}

void g(int k)
{
    k+=1;
    printf("g中i等于%d\n",k);
}
int function1(int a[]);
int function1(int*a);
int sum(int *ar,int n);
int sum(int *,int);
int sum(int ar[],int n);
int sum(int [],int);
