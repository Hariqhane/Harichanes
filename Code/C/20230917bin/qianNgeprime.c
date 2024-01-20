#include <stdio.h>
int main()
{
    int aim;//目标数量
    int count=0;//计数器
    int x;//循环控制变量2   
    int i;//循环控制变量1 
    printf("请输入要输出素数的个数：");
    scanf("%d",&aim);
    for(x=2,count=1;count<=aim;x++){   
            int isPrime=1;
            for(i=2;i<x;i++){
                if(x%i==0){
                    isPrime=0;
                    break;
                              }
                             }
        if(isPrime==1){
            printf("第%d个素数:%d\n",count,x);
            count++;
           // if(count==aim){
               // printf("第%d个素数为:%d\n",aim,x);
            //}
        }                       
    }
    system("pause");
    return 0;
}