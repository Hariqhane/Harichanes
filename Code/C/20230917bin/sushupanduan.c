#include <stdio.h>
int main()
{
    int x;
    int count =2;
    int a;
    int cdf=1;
    printf("请输入要判断的数字：");
    scanf("%d",&x);
    if(x==1||x==0){
        printf("%d既不是素数,也不是合数\n",x);
    }
    else{

    while(count<x){
        a=x%count;
        cdf*=a;
        count++;
    }
    if(cdf==0){
        printf("该数为合数\n");
        }
    else{
        printf("该数为素数\n");
    }
        }
    system("pause");
    return 0;
}