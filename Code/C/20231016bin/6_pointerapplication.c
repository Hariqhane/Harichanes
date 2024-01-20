#include <stdio.h>
int divide(int a, int b, int *result);
int main()
{   
    int a,b;
    scanf("%d %d",&a,&b);
    int c;
    divide(a,b,&c);
    printf("%d",c);
    system("pause"); 
    return 0;
}
int divide(int a, int b,int *result)
{
    int ret = 1;
    if( b == 0)
    {
        ret = 0;
    }
    else
        *result = a/b;
        return ret;
}
