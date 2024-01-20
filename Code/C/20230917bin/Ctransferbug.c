#include <stdio.h>
void swap(int a,int b)
{
    int t = a;
    a = b;
    b = t;
}
int main()
{   
    int a = 5;
    int b = 6;
    swap(a,b);
    printf("%d,%d\n",a,b);
    swap(5,6);
    system("pause");
    return 0;

}