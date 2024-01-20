#include <stdio.h>
int main()
{
    int n;//目标
    double i;//循环变量
    double sum=0;//求和计数器
    double sign=1;
    scanf("%d",&n);
    for(i=1;;i++){
        sum+=sign/i;
        printf("i=%f,%f\n",i,sum);
        sign = -sign;
    }
    system("pause");
    return 0;
}